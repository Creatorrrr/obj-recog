from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from obj_recog.blend_scene_loader import load_blend_scene_manifest
from obj_recog.blender_worker import (
    BlenderFrameRequest,
    BlenderSceneBuildRequest,
    BlenderWorkerClient,
    build_realtime_blender_worker_command,
)
from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.sim_planner import OpenAILivingRoomPlanner, build_planner_context
from obj_recog.sim_protocol import (
    ActionPrimitive,
    ActionSchedule,
    ActionStep,
    EpisodePhase,
    EpisodeReport,
    HiddenWorldState,
    OperatorSceneState,
    RobotPose,
    SensorFrame,
)
from obj_recog.sim_scene import build_interior_test_tv_scene_spec, build_living_room_scene_spec, pose_distance_to_goal


SCENARIO_SPECS = {
    "living_room_navigation_v1": build_living_room_scene_spec(),
    "interior_test_tv_navigation_v1": build_interior_test_tv_scene_spec(),
}

_ROBOT_COLLISION_RADIUS_M = 0.22


@dataclass(frozen=True, slots=True)
class CameraRigSpec:
    image_width: int
    image_height: int
    fps: float
    horizontal_fov_deg: float
    near_plane_m: float
    far_plane_m: float
    camera_height_m: float = 1.25

    @classmethod
    def from_config(cls, config: AppConfig) -> CameraRigSpec:
        return cls(
            image_width=int(config.width),
            image_height=int(config.height),
            fps=float(config.sim_camera_fps),
            horizontal_fov_deg=float(config.sim_camera_fov_deg),
            near_plane_m=float(config.sim_camera_near),
            far_plane_m=float(config.sim_camera_far),
        )


class LivingRoomSensorBackend(Protocol):
    def build_scene(self, scene_spec) -> None:
        ...

    def render_frame(self, *, world_state: HiddenWorldState, frame_index: int, timestamp_sec: float) -> SensorFrame:
        ...

    def close(self) -> None:
        ...


class BlenderLivingRoomSensorBackend:
    def __init__(
        self,
        *,
        config: AppConfig,
        camera_rig: CameraRigSpec,
        worker_client: BlenderWorkerClient,
    ) -> None:
        self._config = config
        self._camera_rig = camera_rig
        self._worker_client = worker_client
        self._started = False

    def build_scene(self, scene_spec) -> None:
        if not self._started:
            self._worker_client.start()
            self._started = True
        self._worker_client.build_scene(
            BlenderSceneBuildRequest(
                scene_spec=scene_spec,
                image_width=self._camera_rig.image_width,
                image_height=self._camera_rig.image_height,
                horizontal_fov_deg=self._camera_rig.horizontal_fov_deg,
                near_plane_m=self._camera_rig.near_plane_m,
                far_plane_m=self._camera_rig.far_plane_m,
            ),
            timeout_sec=15.0,
        )

    def render_frame(self, *, world_state: HiddenWorldState, frame_index: int, timestamp_sec: float) -> SensorFrame:
        pose_world = _pose_matrix(world_state.robot_pose)
        response = self._worker_client.request_frame(
            BlenderFrameRequest(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                robot_pose=world_state.robot_pose,
                camera_pose_world=pose_world,
            ),
            timeout_sec=10.0,
        )
        return SensorFrame(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            frame_bgr=np.asarray(_load_sensor_array(response.rgb_path), dtype=np.uint8),
            depth_map=np.asarray(_load_sensor_array(response.depth_path), dtype=np.float32),
            semantic_mask=np.asarray(_load_sensor_array(response.semantic_mask_path), dtype=np.uint8),
            instance_mask=np.asarray(_load_sensor_array(response.instance_mask_path), dtype=np.uint8),
            camera_pose_world=np.asarray(response.camera_pose_world, dtype=np.float32),
            intrinsics=dict(response.intrinsics),
            render_time_ms=response.render_time_ms,
            metadata={"worker_state": response.worker_state},
        )

    def close(self) -> None:
        self._worker_client.close()


class LivingRoomEpisodeRunner:
    def __init__(
        self,
        *,
        config: AppConfig,
        report_path: str | Path,
        planner,
        sensor_backend: LivingRoomSensorBackend,
        scene_spec=None,
        camera_rig: CameraRigSpec | None = None,
    ) -> None:
        self._config = config
        self._scene_spec = scene_spec or SCENARIO_SPECS.get(str(config.scenario), build_living_room_scene_spec())
        self._camera_rig = camera_rig or CameraRigSpec.from_config(config)
        self._planner = planner
        self._sensor_backend = sensor_backend
        self._report_dir = Path(report_path).parent
        self._report_dir.mkdir(parents=True, exist_ok=True)
        self._sensor_backend.build_scene(self._scene_spec)

        self._state = HiddenWorldState(
            scene_spec=self._scene_spec,
            robot_pose=self._scene_spec.start_pose,
        )
        self._selfcal_actions = (
            ActionStep(ActionPrimitive.CAMERA_PAN_LEFT, 6.0),
            ActionStep(ActionPrimitive.CAMERA_PAN_RIGHT, 6.0),
            ActionStep(ActionPrimitive.TURN_LEFT, 6.0),
            ActionStep(ActionPrimitive.TURN_RIGHT, 6.0),
            ActionStep(ActionPrimitive.MOVE_FORWARD, 0.12),
        )
        self._action_history: list[str] = []
        self._planner_turn_logs: list[dict[str, object]] = []
        self._current_schedule: ActionSchedule | None = None
        self._latest_planner_context = None
        self._closed = False
        self._terminal_frame_emitted = False
        self._write_selfcalibration_artifact()
        self._write_planner_turns_artifact()

    @property
    def current_phase(self) -> EpisodePhase:
        return self._state.phase

    @property
    def current_schedule(self) -> ActionSchedule | None:
        return self._current_schedule

    @staticmethod
    def _tracking_safe_step_limit(primitive: ActionPrimitive) -> float | None:
        if primitive in {
            ActionPrimitive.MOVE_FORWARD,
            ActionPrimitive.MOVE_BACKWARD,
            ActionPrimitive.STRAFE_LEFT,
            ActionPrimitive.STRAFE_RIGHT,
        }:
            return 0.12
        if primitive in {
            ActionPrimitive.TURN_LEFT,
            ActionPrimitive.TURN_RIGHT,
            ActionPrimitive.CAMERA_PAN_LEFT,
            ActionPrimitive.CAMERA_PAN_RIGHT,
        }:
            return 6.0
        return None

    def next_frame(self, *, timeout_sec: float | None = 1.0) -> FramePacket | None:
        _ = timeout_sec
        if self._closed:
            return None
        if self._state.phase in {EpisodePhase.SUCCEEDED, EpisodePhase.FAILED}:
            if self._terminal_frame_emitted:
                self._finalize_report()
                return None
            self._terminal_frame_emitted = True

        frame_index = int(self._state.frame_index)
        timestamp_sec = frame_index / max(self._camera_rig.fps, 1e-6)
        sensor_frame = self._sensor_backend.render_frame(
            world_state=self._state,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )
        return FramePacket(
            frame_bgr=np.asarray(sensor_frame.frame_bgr, dtype=np.uint8),
            timestamp_sec=float(sensor_frame.timestamp_sec),
            depth_map=np.asarray(sensor_frame.depth_map, dtype=np.float32),
            semantic_mask=np.asarray(sensor_frame.semantic_mask, dtype=np.uint8),
            instance_mask=np.asarray(sensor_frame.instance_mask, dtype=np.uint8),
            pose_world_gt=np.asarray(sensor_frame.camera_pose_world, dtype=np.float32),
            intrinsics_gt=CameraIntrinsics(**sensor_frame.intrinsics),
            scenario_state=OperatorSceneState(
                scene_spec=self._scene_spec,
                robot_pose=self._state.robot_pose,
                phase=self._state.phase,
                semantic_target_class=self._scene_spec.semantic_target_class,
            ),
            planner_context=self._latest_planner_context,
            calibration_source="self_calibration/converged"
            if self._state.selfcal_step_index >= len(self._selfcal_actions)
            else "self_calibration/pending",
        )

    def record_runtime_observation(self, *, frame_packet: FramePacket, artifacts) -> None:
        if self._state.phase in {EpisodePhase.SUCCEEDED, EpisodePhase.FAILED}:
            self._state.frame_index += 1
            self._state.elapsed_sec += 1.0 / max(self._camera_rig.fps, 1e-6)
            self._finalize_report()
            return

        if self._state.phase == EpisodePhase.SELF_CALIBRATING:
            self._advance_self_calibration()
            if self._state.phase == EpisodePhase.PERCEIVE_AND_PLAN:
                self._plan_next_schedule(artifacts=artifacts)
        elif self._state.phase in {EpisodePhase.PERCEIVE_AND_PLAN, EpisodePhase.REASSESS}:
            self._plan_next_schedule(artifacts=artifacts)
        elif self._state.phase == EpisodePhase.EXECUTING_SCHEDULE:
            self._execute_one_step()

        if pose_distance_to_goal(self._scene_spec, self._state.robot_pose) <= 0.5:
            self._state.phase = EpisodePhase.SUCCEEDED
            self._state.mission_succeeded = True
        elif self._state.elapsed_sec >= float(self._config.eval_budget_sec):
            self._state.phase = EpisodePhase.FAILED

        self._state.frame_index += 1
        self._state.elapsed_sec += 1.0 / max(self._camera_rig.fps, 1e-6)
        self._write_selfcalibration_artifact()
        self._write_planner_turns_artifact()
        if self._state.phase in {EpisodePhase.SUCCEEDED, EpisodePhase.FAILED}:
            self._finalize_report()

    def close(self) -> None:
        self._closed = True
        backend_close = getattr(self._sensor_backend, "close", None)
        if callable(backend_close):
            backend_close()
        self._finalize_report()

    def _advance_self_calibration(self) -> None:
        if self._state.selfcal_step_index < len(self._selfcal_actions):
            step = self._selfcal_actions[self._state.selfcal_step_index]
            self._apply_step(step)
            self._action_history.append(f"{step.primitive.value}:{step.value}")
            self._state.selfcal_step_index += 1
        if self._state.selfcal_step_index >= len(self._selfcal_actions):
            self._state.phase = EpisodePhase.PERCEIVE_AND_PLAN

    def _plan_next_schedule(self, *, artifacts) -> None:
        context = build_planner_context(
            phase=self._state.phase,
            frame_index=self._state.frame_index,
            goal_description=self._scene_spec.goal_description,
            detections=list(getattr(artifacts, "detections", []) or []),
            scene_graph_snapshot=getattr(artifacts, "scene_graph_snapshot", None),
            reconstruction_summary={
                "mesh_vertices": int(np.asarray(getattr(artifacts, "mesh_vertices_xyz", np.empty((0, 3)))).shape[0]),
                "tracked_points": int(np.asarray(getattr(artifacts, "mesh_vertices_xyz", np.empty((0, 3)))).shape[0]),
            },
            depth_summary={
                "min_depth_m": float(np.nanmin(np.asarray(getattr(artifacts, "depth_map", np.ones((1, 1)))))),
                "median_depth_m": float(np.nanmedian(np.asarray(getattr(artifacts, "depth_map", np.ones((1, 1)))))),
            },
            recent_actions=tuple(self._action_history[-8:]),
            calibration_status=(
                "converged" if self._state.selfcal_step_index >= len(self._selfcal_actions) else "pending"
            ),
            tracking_status=str(getattr(artifacts, "slam_tracking_state", "UNKNOWN")),
        )
        schedule = self._planner.plan(context=context)
        self._latest_planner_context = context
        if not schedule.steps:
            schedule = ActionSchedule(
                steps=(ActionStep(ActionPrimitive.PAUSE, 0.5),),
                rationale="planner returned no safe actions; pause and reassess",
                model=getattr(self._planner, "model", "fallback"),
                issued_at_frame=self._state.frame_index,
            )
        self._current_schedule = schedule
        self._state.schedule_cursor = 0
        self._state.phase = EpisodePhase.EXECUTING_SCHEDULE
        self._state.planner_turn_count += 1
        self._planner_turn_logs.append(
            {
                "frame_index": self._state.frame_index,
                "phase": context.phase.value,
                "prompt": context,
                "schedule": {
                    "rationale": schedule.rationale,
                    "model": schedule.model,
                    "steps": [
                        {"primitive": step.primitive.value, "value": step.value}
                        for step in schedule.steps
                    ],
                },
            }
        )

    def _execute_one_step(self) -> None:
        if self._current_schedule is None or self._state.schedule_cursor >= len(self._current_schedule.steps):
            self._state.phase = EpisodePhase.REASSESS
            return
        step = self._next_tracking_safe_step()
        if step is None:
            self._state.phase = EpisodePhase.REASSESS
            return
        self._apply_step(step)
        self._action_history.append(f"{step.primitive.value}:{step.value}")
        if self._state.schedule_cursor >= len(self._current_schedule.steps):
            self._state.phase = EpisodePhase.REASSESS

    def _next_tracking_safe_step(self) -> ActionStep | None:
        if self._current_schedule is None or self._state.schedule_cursor >= len(self._current_schedule.steps):
            return None
        cursor = int(self._state.schedule_cursor)
        step = self._current_schedule.steps[cursor]
        limit = self._tracking_safe_step_limit(step.primitive)
        if limit is None or abs(float(step.value)) <= limit + 1e-9:
            self._state.schedule_cursor += 1
            return step

        chunk_value = math.copysign(limit, float(step.value))
        residual_value = float(step.value) - chunk_value
        updated_steps = list(self._current_schedule.steps)
        updated_steps[cursor] = ActionStep(step.primitive, residual_value)
        self._current_schedule = ActionSchedule(
            steps=tuple(updated_steps),
            rationale=self._current_schedule.rationale,
            model=self._current_schedule.model,
            issued_at_frame=self._current_schedule.issued_at_frame,
        )
        return ActionStep(step.primitive, chunk_value)

    def _apply_step(self, step: ActionStep) -> None:
        pose = self._state.robot_pose
        x = float(pose.x)
        y = float(pose.y)
        z = float(pose.z)
        yaw_deg = float(pose.yaw_deg)
        camera_pan_deg = float(pose.camera_pan_deg)

        if step.primitive in {
            ActionPrimitive.MOVE_FORWARD,
            ActionPrimitive.MOVE_BACKWARD,
            ActionPrimitive.STRAFE_LEFT,
            ActionPrimitive.STRAFE_RIGHT,
        }:
            x, z = self._translated_ground_position(
                x=x,
                z=z,
                yaw_deg=yaw_deg,
                primitive=step.primitive,
                distance_m=float(step.value),
            )
        elif step.primitive == ActionPrimitive.TURN_LEFT:
            yaw_deg += float(step.value)
        elif step.primitive == ActionPrimitive.TURN_RIGHT:
            yaw_deg -= float(step.value)
        elif step.primitive == ActionPrimitive.CAMERA_PAN_LEFT:
            camera_pan_deg += float(step.value)
        elif step.primitive == ActionPrimitive.CAMERA_PAN_RIGHT:
            camera_pan_deg -= float(step.value)

        candidate_pose = RobotPose(
            x=x,
            y=y,
            z=z,
            yaw_deg=yaw_deg,
            camera_pan_deg=max(-60.0, min(60.0, camera_pan_deg)),
        )
        if step.primitive in {
            ActionPrimitive.MOVE_FORWARD,
            ActionPrimitive.MOVE_BACKWARD,
            ActionPrimitive.STRAFE_LEFT,
            ActionPrimitive.STRAFE_RIGHT,
        } and self._pose_collides(candidate_pose):
            return
        self._state.robot_pose = candidate_pose

    def _translated_ground_position(
        self,
        *,
        x: float,
        z: float,
        yaw_deg: float,
        primitive: ActionPrimitive,
        distance_m: float,
    ) -> tuple[float, float]:
        yaw_rad = math.radians(float(yaw_deg))
        forward_x = math.sin(yaw_rad)
        forward_z = math.cos(yaw_rad)
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)
        if primitive == ActionPrimitive.MOVE_FORWARD:
            return x + (forward_x * distance_m), z + (forward_z * distance_m)
        if primitive == ActionPrimitive.MOVE_BACKWARD:
            return x - (forward_x * distance_m), z - (forward_z * distance_m)
        if primitive == ActionPrimitive.STRAFE_LEFT:
            return x - (right_x * distance_m), z - (right_z * distance_m)
        if primitive == ActionPrimitive.STRAFE_RIGHT:
            return x + (right_x * distance_m), z + (right_z * distance_m)
        return x, z

    def _pose_collides(self, pose: RobotPose) -> bool:
        half_width = max((float(self._scene_spec.room_size_xyz[0]) * 0.5) - _ROBOT_COLLISION_RADIUS_M, 0.1)
        half_depth = max((float(self._scene_spec.room_size_xyz[2]) * 0.5) - _ROBOT_COLLISION_RADIUS_M, 0.1)
        if abs(float(pose.x)) > half_width or abs(float(pose.z)) > half_depth:
            return True

        pose_x = float(pose.x)
        pose_z = float(pose.z)
        for item in self._scene_spec.objects:
            if not bool(item.collider):
                continue
            local_x, local_z = self._object_local_ground_position(
                pose_x=pose_x,
                pose_z=pose_z,
                object_center_x=float(item.center_xyz[0]),
                object_center_z=float(item.center_xyz[2]),
                object_yaw_deg=float(item.yaw_deg),
            )
            half_x = (float(item.size_xyz[0]) * 0.5) + _ROBOT_COLLISION_RADIUS_M
            half_z = (float(item.size_xyz[2]) * 0.5) + _ROBOT_COLLISION_RADIUS_M
            if abs(local_x) <= half_x and abs(local_z) <= half_z:
                return True
        return False

    @staticmethod
    def _object_local_ground_position(
        *,
        pose_x: float,
        pose_z: float,
        object_center_x: float,
        object_center_z: float,
        object_yaw_deg: float,
    ) -> tuple[float, float]:
        rel_x = pose_x - object_center_x
        rel_z = pose_z - object_center_z
        yaw_rad = math.radians(float(object_yaw_deg))
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        return (
            (rel_x * cos_yaw) - (rel_z * sin_yaw),
            (rel_x * sin_yaw) + (rel_z * cos_yaw),
        )

    def _write_selfcalibration_artifact(self) -> None:
        payload = {
            "phase": self._state.phase.value,
            "completed": self._state.selfcal_step_index >= len(self._selfcal_actions),
            "completed_steps": int(self._state.selfcal_step_index),
            "total_steps": len(self._selfcal_actions),
        }
        (self._report_dir / "self_calibration.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _write_planner_turns_artifact(self) -> None:
        turns_path = self._report_dir / "planner_turns.jsonl"
        lines = []
        for item in self._planner_turn_logs:
            prompt = item["prompt"]
            lines.append(
                json.dumps(
                    {
                        "frame_index": item["frame_index"],
                        "phase": item["phase"],
                        "prompt": {
                            "phase": prompt.phase.value,
                            "frame_index": prompt.frame_index,
                            "goal_description": prompt.goal_description,
                            "visible_detections": list(prompt.perception.visible_detections),
                            "visible_segments": list(prompt.perception.visible_segments),
                            "visible_graph_relations": list(prompt.perception.visible_graph_relations),
                            "reconstruction_summary": dict(prompt.perception.reconstruction_summary),
                            "depth_summary": dict(prompt.perception.depth_summary),
                            "calibration_status": prompt.perception.calibration_status,
                            "tracking_status": prompt.perception.tracking_status,
                            "recent_actions": list(prompt.recent_actions),
                        },
                        "schedule": item["schedule"],
                    },
                    ensure_ascii=False,
                )
            )
        turns_path.write_text("\n".join(lines), encoding="utf-8")

    def _finalize_report(self) -> None:
        report = EpisodeReport(
            scenario_id=self._scene_spec.scene_id,
            success=bool(self._state.mission_succeeded),
            final_phase=self._state.phase,
            total_frames=int(self._state.frame_index),
            planner_turns=int(self._state.planner_turn_count),
            self_calibration_completed=bool(self._state.selfcal_step_index >= len(self._selfcal_actions)),
            final_distance_to_goal_m=pose_distance_to_goal(self._scene_spec, self._state.robot_pose),
            report_dir=str(self._report_dir),
        )
        (self._report_dir / "episode_report.json").write_text(
            json.dumps(
                {
                    "scenario_id": report.scenario_id,
                    "success": report.success,
                    "final_phase": report.final_phase.value,
                    "total_frames": report.total_frames,
                    "planner_turns": report.planner_turns,
                    "self_calibration_completed": report.self_calibration_completed,
                    "final_distance_to_goal_m": report.final_distance_to_goal_m,
                    "report_dir": report.report_dir,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


SimulationFrameSource = LivingRoomEpisodeRunner


@dataclass(frozen=True, slots=True)
class LivingRoomSimulationRuntime:
    config: AppConfig
    report_path: str | Path
    planner: object | None = None
    sensor_backend_factory: object | None = None
    blender_worker_client_factory: object | None = None

    def create_frame_source(self) -> LivingRoomEpisodeRunner:
        camera_rig = CameraRigSpec.from_config(self.config)
        scene_spec = _resolve_scene_spec(self.config)
        planner = self.planner or OpenAILivingRoomPlanner(
            model=self.config.sim_planner_model,
            timeout_sec=self.config.sim_planner_timeout_sec,
        )
        sensor_backend = (
            self.sensor_backend_factory(self.config, camera_rig)
            if self.sensor_backend_factory is not None
            else _build_blender_sensor_backend(
                config=self.config,
                camera_rig=camera_rig,
                report_path=self.report_path,
                scene_spec=scene_spec,
                worker_client_factory=self.blender_worker_client_factory,
            )
        )
        return LivingRoomEpisodeRunner(
            config=self.config,
            report_path=self.report_path,
            planner=planner,
            sensor_backend=sensor_backend,
            scene_spec=scene_spec,
            camera_rig=camera_rig,
        )


SimulationRuntime = LivingRoomSimulationRuntime


def _build_blender_sensor_backend(
    *,
    config: AppConfig,
    camera_rig: CameraRigSpec,
    report_path: str | Path,
    scene_spec,
    worker_client_factory=None,
) -> BlenderLivingRoomSensorBackend:
    if not config.blender_exec:
        raise RuntimeError("sim input requires --blender-exec for the living room renderer")
    worker_client = (
        worker_client_factory(config=config, camera_rig=camera_rig)
        if worker_client_factory is not None
        else _default_blender_worker_client(config=config, report_path=report_path, scene_spec=scene_spec)
    )
    return BlenderLivingRoomSensorBackend(
        config=config,
        camera_rig=camera_rig,
        worker_client=worker_client,
    )


def _default_blender_worker_client(*, config: AppConfig, report_path: str | Path, scene_spec) -> BlenderWorkerClient:
    repo_root = Path(__file__).resolve().parents[2]
    command = build_realtime_blender_worker_command(
        blender_exec=str(config.blender_exec),
        repo_root=repo_root,
        output_root=Path(report_path).parent,
        blend_file=scene_spec.blend_file_path,
    )
    return BlenderWorkerClient(command=command)


def _resolve_scene_spec(config: AppConfig):
    scenario_id = str(config.scenario)
    base_scene = SCENARIO_SPECS.get(scenario_id, build_living_room_scene_spec())
    if scenario_id != "interior_test_tv_navigation_v1":
        return base_scene
    if not base_scene.blend_file_path or not config.blender_exec:
        return base_scene
    manifest = load_blend_scene_manifest(
        blend_file_path=base_scene.blend_file_path,
        blender_exec=str(config.blender_exec),
    )
    return build_interior_test_tv_scene_spec(manifest)


def _pose_matrix(pose: RobotPose) -> np.ndarray:
    yaw_rad = math.radians(float(pose.yaw_deg))
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = cos_yaw
    matrix[0, 2] = sin_yaw
    matrix[2, 0] = -sin_yaw
    matrix[2, 2] = cos_yaw
    matrix[0, 3] = float(pose.x)
    matrix[1, 3] = float(pose.y)
    matrix[2, 3] = float(pose.z)
    return matrix


def _load_sensor_array(path: str | Path) -> np.ndarray:
    array_path = Path(path)
    if array_path.suffix == ".npy":
        return np.load(array_path)
    raise RuntimeError(f"unsupported sensor artifact: {array_path}")
