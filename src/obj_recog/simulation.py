from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.sim_planner import OpenAILivingRoomPlanner, build_planner_context
from obj_recog.sim_protocol import (
    ActionPrimitive,
    ActionSchedule,
    ActionStep,
    EpisodePhase,
    EpisodeReport,
    HiddenWorldState,
    RobotPose,
    SensorFrame,
    UnityActionCommand,
)
from obj_recog.sim_scene import build_living_room_scene_spec
from obj_recog.unity_rgb import UnityRgbClient, command_from_step


SCENARIO_SPECS = {
    "living_room_navigation_v1": build_living_room_scene_spec(),
}


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


class RgbSimulationBackend(Protocol):
    def reset_episode(self, *, scene_spec) -> SensorFrame:
        ...

    def apply_action(self, command: UnityActionCommand) -> SensorFrame:
        ...

    def close(self) -> None:
        ...


class UnityRgbSensorBackend:
    def __init__(
        self,
        *,
        client: UnityRgbClient,
    ) -> None:
        self._client = client
        self._frame_counter = 0

    def reset_episode(self, *, scene_spec) -> SensorFrame:
        self._frame_counter = 0
        frame = self._client.reset_episode(scenario_id=str(scene_spec.scene_id))
        return SensorFrame(
            frame_index=0,
            timestamp_sec=float(frame.timestamp_sec),
            frame_bgr=np.asarray(frame.frame_bgr, dtype=np.uint8),
        )

    def apply_action(self, command: UnityActionCommand) -> SensorFrame:
        self._frame_counter += 1
        frame = self._client.apply_action(command)
        return SensorFrame(
            frame_index=int(self._frame_counter),
            timestamp_sec=float(frame.timestamp_sec),
            frame_bgr=np.asarray(frame.frame_bgr, dtype=np.uint8),
        )

    def close(self) -> None:
        self._client.close()


class LivingRoomEpisodeRunner:
    def __init__(
        self,
        *,
        config: AppConfig,
        report_path: str | Path,
        planner,
        sensor_backend: RgbSimulationBackend,
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
        self._latest_sensor_frame = self._sensor_backend.reset_episode(scene_spec=self._scene_spec)

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
        self._executed_schedule_steps = 0
        self._write_selfcalibration_artifact()
        self._write_planner_turns_artifact()
        self._write_episode_report()

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

    @staticmethod
    def _normalize_label(label: object) -> str:
        return str(label or "").strip().lower().replace(" ", "_")

    @classmethod
    def _label_matches_target(cls, *, detection_label: object, target_label: str) -> bool:
        normalized_detection = cls._normalize_label(detection_label)
        normalized_target = cls._normalize_label(target_label)
        if not normalized_detection or not normalized_target:
            return False
        return (
            normalized_detection == normalized_target
            or normalized_target in normalized_detection
            or normalized_detection in normalized_target
        )

    @staticmethod
    def _bbox_depth_m(depth_map: np.ndarray, xyxy: tuple[int, int, int, int]) -> float | None:
        depth = np.asarray(depth_map, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            return None
        height, width = depth.shape[:2]
        x1, y1, x2, y2 = (int(value) for value in xyxy)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        crop = depth[y1:y2, x1:x2]
        valid = np.isfinite(crop) & (crop > 0.05)
        if not np.any(valid):
            return None
        return float(np.nanmedian(crop[valid]))

    def _goal_visually_reached(self, *, artifacts) -> bool:
        target_label = getattr(self._scene_spec, "semantic_target_class", "")
        detections = list(getattr(artifacts, "detections", []) or [])
        depth_map = np.asarray(getattr(artifacts, "depth_map", np.empty((0, 0))), dtype=np.float32)
        frame_shape = np.asarray(getattr(artifacts, "frame_bgr", np.empty((0, 0, 3)))).shape
        if len(frame_shape) < 2:
            return False
        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        if frame_height <= 0 or frame_width <= 0:
            return False

        for detection in detections:
            if not self._label_matches_target(
                detection_label=getattr(detection, "label", ""),
                target_label=target_label,
            ):
                continue
            x1, y1, x2, y2 = (int(value) for value in getattr(detection, "xyxy", (0, 0, 0, 0)))
            bbox_width = max(1, x2 - x1)
            bbox_height = max(1, y2 - y1)
            area_ratio = float(bbox_width * bbox_height) / float(max(1, frame_width * frame_height))
            bbox_center_x = x1 + (bbox_width * 0.5)
            center_offset_ratio = abs(bbox_center_x - (frame_width * 0.5)) / float(max(1.0, frame_width * 0.5))
            median_depth_m = self._bbox_depth_m(depth_map, (x1, y1, x2, y2))
            if median_depth_m is not None:
                if median_depth_m <= 2.5 and area_ratio >= 0.08 and center_offset_ratio <= 0.35:
                    return True
                continue
            if area_ratio >= 0.22 and center_offset_ratio <= 0.25:
                return True
        return False

    def next_frame(self, *, timeout_sec: float | None = 1.0) -> FramePacket | None:
        _ = timeout_sec
        if self._closed or self._latest_sensor_frame is None:
            return None
        sensor_frame = self._latest_sensor_frame
        return FramePacket(
            frame_bgr=np.asarray(sensor_frame.frame_bgr, dtype=np.uint8),
            timestamp_sec=float(sensor_frame.timestamp_sec),
            planner_context=self._latest_planner_context,
            calibration_source="self_calibration/converged"
            if self._state.selfcal_step_index >= len(self._selfcal_actions)
            else "self_calibration/pending",
        )

    def record_runtime_observation(self, *, frame_packet: FramePacket, artifacts) -> None:
        _ = frame_packet
        if self._closed or self._latest_sensor_frame is None:
            return
        if self._goal_visually_reached(artifacts=artifacts):
            self._mark_goal_completed()
            return

        next_command = self._select_next_command(artifacts=artifacts)
        self._latest_sensor_frame = self._sensor_backend.apply_action(next_command)
        self._state.frame_index += 1
        self._state.elapsed_sec += 1.0 / max(self._camera_rig.fps, 1e-6)
        self._write_selfcalibration_artifact()
        self._write_planner_turns_artifact()
        self._write_episode_report()

    def close(self) -> None:
        self._closed = True
        backend_close = getattr(self._sensor_backend, "close", None)
        if callable(backend_close):
            backend_close()
        self._write_episode_report()

    def _mark_goal_completed(self) -> None:
        self._state.mission_succeeded = True
        self._state.phase = EpisodePhase.SUCCEEDED
        self._latest_sensor_frame = None
        self._write_episode_report()

    def _select_next_command(self, *, artifacts) -> UnityActionCommand:
        if self._state.phase == EpisodePhase.SELF_CALIBRATING:
            step = self._advance_self_calibration()
        elif self._state.phase in {EpisodePhase.PERCEIVE_AND_PLAN, EpisodePhase.REASSESS}:
            self._plan_next_schedule(artifacts=artifacts)
            step = self._consume_schedule_step()
        elif self._state.phase == EpisodePhase.EXECUTING_SCHEDULE:
            step = self._consume_schedule_step()
        else:
            step = ActionStep(ActionPrimitive.PAUSE, 0.5)
        self._action_history.append(f"{step.primitive.value}:{step.value}")
        return command_from_step(step.primitive, step.value)

    def _advance_self_calibration(self) -> ActionStep:
        if self._state.selfcal_step_index >= len(self._selfcal_actions):
            self._state.phase = EpisodePhase.PERCEIVE_AND_PLAN
            return ActionStep(ActionPrimitive.PAUSE, 0.5)
        step = self._selfcal_actions[self._state.selfcal_step_index]
        self._state.selfcal_step_index += 1
        if self._state.selfcal_step_index >= len(self._selfcal_actions):
            self._state.phase = EpisodePhase.PERCEIVE_AND_PLAN
        return step

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
        try:
            schedule = self._planner.plan(context=context)
        except Exception as exc:
            schedule = ActionSchedule(
                steps=(ActionStep(ActionPrimitive.PAUSE, 0.5),),
                rationale=f"planner unavailable; pause and reassess ({exc})",
                model="planner_error",
                issued_at_frame=self._state.frame_index,
            )
        self._latest_planner_context = context
        max_schedule_steps = max(1, int(getattr(self._config, "sim_action_batch_size", 1)))
        truncated_steps = tuple(schedule.steps[:max_schedule_steps])
        if truncated_steps != schedule.steps:
            schedule = ActionSchedule(
                steps=truncated_steps,
                rationale=schedule.rationale,
                model=schedule.model,
                issued_at_frame=schedule.issued_at_frame,
            )
        if not schedule.steps:
            schedule = ActionSchedule(
                steps=(ActionStep(ActionPrimitive.PAUSE, 0.5),),
                rationale="planner returned no safe actions; pause and reassess",
                model=getattr(self._planner, "model", "fallback"),
                issued_at_frame=self._state.frame_index,
            )
        self._current_schedule = schedule
        self._state.schedule_cursor = 0
        self._executed_schedule_steps = 0
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

    def _consume_schedule_step(self) -> ActionStep:
        step = self._next_tracking_safe_step()
        if step is None:
            self._current_schedule = None
            self._executed_schedule_steps = 0
            self._state.phase = EpisodePhase.REASSESS
            return ActionStep(ActionPrimitive.PAUSE, 0.5)
        self._executed_schedule_steps += 1
        max_schedule_steps = max(1, int(getattr(self._config, "sim_action_batch_size", 1)))
        if self._current_schedule is None or self._state.schedule_cursor >= len(self._current_schedule.steps):
            self._current_schedule = None
            self._executed_schedule_steps = 0
            self._state.phase = EpisodePhase.REASSESS
        elif self._executed_schedule_steps >= max_schedule_steps:
            self._current_schedule = None
            self._state.schedule_cursor = 0
            self._executed_schedule_steps = 0
            self._state.phase = EpisodePhase.REASSESS
        else:
            self._state.phase = EpisodePhase.EXECUTING_SCHEDULE
        return step

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

    def _write_episode_report(self) -> None:
        report = EpisodeReport(
            scenario_id=self._scene_spec.scene_id,
            success=True if self._state.mission_succeeded else None,
            final_phase=self._state.phase,
            total_frames=int(self._state.frame_index),
            planner_turns=int(self._state.planner_turn_count),
            self_calibration_completed=bool(self._state.selfcal_step_index >= len(self._selfcal_actions)),
            final_distance_to_goal_m=None,
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
                    "offline_evaluation_required": True,
                    "sim_interface_mode": str(self._config.sim_interface_mode),
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
    unity_client_factory: object | None = None

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
            else _build_unity_rgb_sensor_backend(
                config=self.config,
                camera_rig=camera_rig,
                unity_client_factory=self.unity_client_factory,
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


def _build_unity_rgb_sensor_backend(
    *,
    config: AppConfig,
    camera_rig: CameraRigSpec,
    unity_client_factory=None,
) -> UnityRgbSensorBackend:
    _ = camera_rig
    client = (
        unity_client_factory(config=config)
        if unity_client_factory is not None
        else UnityRgbClient(
            host=str(config.unity_host),
            port=int(config.unity_port),
            timeout_sec=30.0,
            unity_player_path=config.unity_player_path,
        )
    )
    return UnityRgbSensorBackend(client=client)


def _resolve_scene_spec(config: AppConfig):
    return SCENARIO_SPECS.get(str(config.scenario), build_living_room_scene_spec())
