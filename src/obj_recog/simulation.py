from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.metric_stabilization import MetricCorrectionResult, MetricDepthCalibrator
from obj_recog.reconstruct import CameraIntrinsics, back_project_pixels, intrinsics_for_frame
from obj_recog.sim_planner import OpenAILivingRoomPlanner, build_planner_context, planner_prompt_from_context
from obj_recog.sim_protocol import (
    ActionSchedule,
    CommandKind,
    ExecutedMacroCommand,
    EpisodePhase,
    EpisodeReport,
    HiddenWorldState,
    MotionCommand,
    PlannerActionEffectSummary,
    PlannerCameraState,
    PlannerGoalCompletion,
    PlannerNavigationAffordances,
    PlannerNavigationSectorObservation,
    PlannerReconstructionBrief,
    PlannerSafetyFlags,
    PlannerSearchOutcome,
    RigCapabilities,
    SensorFrame,
    UnityRigDeltaCommand,
)
from obj_recog.sim_scene import build_living_room_scene_spec
from obj_recog.unity_rgb import UnityRgbClient, rig_capabilities_from_metadata
from obj_recog.unity_vendor_check import validate_unity_vendor_setup


SCENARIO_SPECS = {
    "living_room_navigation_v1": build_living_room_scene_spec(),
}

_BODY_MOTION_TRACKING_STATES = {"TRACKING", "RELOCALIZED"}
_TARGET_EVIDENCE_REPLAN_STATES = {"appeared", "stronger"}
_DEFAULT_MICROSTEP_LIMITS = {
    "translate_distance_m": 0.08,
    "body_yaw_deg": 4.0,
    "camera_yaw_deg": 4.0,
    "camera_pitch_deg": 4.0,
}
_DEFAULT_RIG_CAPABILITIES = RigCapabilities(
    move_speed_mps=1.6,
    turn_speed_deg_per_sec=100.0,
    camera_yaw_speed_deg_per_sec=90.0,
    camera_pitch_speed_deg_per_sec=90.0,
    camera_yaw_limit_deg=70.0,
    camera_pitch_limit_deg=55.0,
)
_INITIALIZING_RECOVERY_FRAME_THRESHOLD = 6


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


@dataclass(slots=True)
class _ActiveMacroExecution:
    command: MotionCommand
    start_state: dict[str, object]
    requested_translation_m: float = 0.0
    requested_yaw_deg: float = 0.0
    target_camera_yaw_deg: float | None = None
    target_camera_pitch_deg: float | None = None
    requested_pause_sec: float = 0.0
    sent_translation_m: float = 0.0
    sent_yaw_deg: float = 0.0
    sent_camera_yaw_deg: float = 0.0
    sent_camera_pitch_deg: float = 0.0
    microstep_count: int = 0
    allow_untracked_completion: bool = False
    soft_block_streak: int = 0
    recent_step_observations: list[_MicrostepProgressObservation] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _MacroMotionProgress:
    commanded_progress_m: float | None
    vision_progress_m: float | None
    fused_progress_m: float | None
    progress_source: str
    commanded_yaw_deg: float | None = None
    vision_yaw_deg: float | None = None
    fused_yaw_deg: float | None = None


@dataclass(frozen=True, slots=True)
class _MicrostepProgressObservation:
    clearance_delta_m: float | None
    scene_change_score: float
    front_clearance_m: float | None
    hard_blocked: bool
    soft_blocked: bool


class RgbSimulationBackend(Protocol):
    def reset_episode(self, *, scene_spec) -> SensorFrame:
        ...

    def apply_action(self, command: UnityRigDeltaCommand) -> SensorFrame:
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
        self._action_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unity-rgb")
        self._pending_frame_future: Future[SensorFrame] | None = None

    def reset_episode(self, *, scene_spec) -> SensorFrame:
        self._clear_pending_frame_future()
        self._frame_counter = 0
        frame = self._client.reset_episode(scenario_id=str(scene_spec.scene_id))
        capabilities = rig_capabilities_from_metadata(frame.metadata)
        if capabilities is None:
            raise RuntimeError(
                "Unity RGB player did not advertise rig capabilities during reset_episode. "
                "This build is likely stale and predates the rig_delta motion protocol. "
                "Rebuild the Unity player from the current source and try again."
            )
        return SensorFrame(
            frame_index=0,
            timestamp_sec=float(frame.timestamp_sec),
            frame_bgr=np.asarray(frame.frame_bgr, dtype=np.uint8),
            metadata=dict(frame.metadata or {}),
        )

    def apply_action(self, command: UnityRigDeltaCommand) -> SensorFrame:
        self._frame_counter += 1
        frame = self._client.apply_action(command)
        return SensorFrame(
            frame_index=int(self._frame_counter),
            timestamp_sec=float(frame.timestamp_sec),
            frame_bgr=np.asarray(frame.frame_bgr, dtype=np.uint8),
            metadata=dict(frame.metadata or {}),
        )

    def submit_action(self, command: UnityRigDeltaCommand) -> None:
        if self.is_waiting_for_frame():
            raise RuntimeError("Unity RGB backend already has an in-flight action")
        self._frame_counter += 1
        frame_index = int(self._frame_counter)
        self._pending_frame_future = self._action_executor.submit(
            self._request_sensor_frame,
            command,
            frame_index,
        )

    def poll_action_frame(self, *, timeout_sec: float | None = 0.0) -> SensorFrame | None:
        future = self._pending_frame_future
        if future is None:
            return None
        resolved_timeout = 0.0 if timeout_sec is None else max(0.0, float(timeout_sec))
        try:
            frame = future.result(timeout=resolved_timeout)
        except FutureTimeoutError:
            return None
        except Exception as exc:
            self._pending_frame_future = None
            raise RuntimeError("Unity RGB action request failed") from exc
        self._pending_frame_future = None
        return frame

    def is_waiting_for_frame(self) -> bool:
        return self._pending_frame_future is not None

    def close(self) -> None:
        try:
            self._clear_pending_frame_future()
            self._client.close()
        finally:
            self._action_executor.shutdown(wait=False, cancel_futures=True)

    def _request_sensor_frame(self, command: UnityRigDeltaCommand, frame_index: int) -> SensorFrame:
        frame = self._client.apply_action(command)
        return SensorFrame(
            frame_index=int(frame_index),
            timestamp_sec=float(frame.timestamp_sec),
            frame_bgr=np.asarray(frame.frame_bgr, dtype=np.uint8),
            metadata=dict(frame.metadata or {}),
        )

    def _clear_pending_frame_future(self) -> None:
        future = self._pending_frame_future
        if future is None:
            return
        if not future.done():
            future.cancel()
        self._pending_frame_future = None


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
        self._rig_capabilities = (
            rig_capabilities_from_metadata(self._latest_sensor_frame.metadata) or _DEFAULT_RIG_CAPABILITIES
        )

        self._state = HiddenWorldState(
            scene_spec=self._scene_spec,
            robot_pose=self._scene_spec.start_pose,
        )
        self._selfcal_actions = (
            MotionCommand(kind=CommandKind.AIM_CAMERA, yaw_deg=-12.0, pitch_deg=0.0, intent="self-calibrate left"),
            MotionCommand(kind=CommandKind.AIM_CAMERA, yaw_deg=12.0, pitch_deg=0.0, intent="self-calibrate right"),
            MotionCommand(kind=CommandKind.AIM_CAMERA, yaw_deg=0.0, pitch_deg=8.0, intent="self-calibrate up"),
            MotionCommand(
                kind=CommandKind.ROTATE_BODY,
                direction="left",
                mode="angle_deg",
                value=16.0,
                intent="self-calibrate turn left",
            ),
            MotionCommand(
                kind=CommandKind.TRANSLATE,
                direction="forward",
                mode="distance_m",
                value=0.32,
                intent="self-calibrate translation forward",
            ),
            MotionCommand(
                kind=CommandKind.TRANSLATE,
                direction="right",
                mode="distance_m",
                value=0.24,
                intent="self-calibrate lateral translation",
            ),
        )
        self._current_camera_state = PlannerCameraState(
            yaw_deg=float(getattr(self._scene_spec.start_pose, "camera_pan_deg", 0.0)),
            pitch_deg=float(getattr(self._scene_spec.start_pose, "camera_pitch_deg", 0.0)),
        )
        self._initializing_streak_frames = 0
        self._action_history: list[str] = []
        self._planner_turn_logs: list[dict[str, object]] = []
        self._completed_search_history: list[PlannerSearchOutcome] = []
        self._recent_action_effects: list[PlannerActionEffectSummary] = []
        self._current_schedule: ActionSchedule | None = None
        self._active_schedule_start_frame: int | None = None
        self._active_schedule_start_observation: dict[str, object] | None = None
        self._active_schedule_executed_commands: list[str] = []
        self._previous_observation_state: dict[str, object] | None = None
        self._active_macro_execution: _ActiveMacroExecution | None = None
        self._latest_planner_context = None
        self._latest_planner_schedule: ActionSchedule | None = None
        self._closed = False
        self._debug_microstep_trace: list[dict[str, object]] = []
        self._metric_depth_calibrator = MetricDepthCalibrator()
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
    def _tracking_state_allows_body_motion(tracking_status: str) -> bool:
        return str(tracking_status or "").upper() in _BODY_MOTION_TRACKING_STATES

    def _bootstrap_body_motion_allowed(self, observation_state: dict[str, object]) -> bool:
        tracking_status = str(observation_state.get("tracking_status", "UNKNOWN")).upper()
        if tracking_status != "INITIALIZING":
            return False
        if self._state.selfcal_step_index < len(self._selfcal_actions):
            return False
        streak_frames = int(observation_state.get("initializing_streak_frames", self._initializing_streak_frames))
        return streak_frames >= _INITIALIZING_RECOVERY_FRAME_THRESHOLD

    def _body_motion_allowed_for_observation_state(self, observation_state: dict[str, object]) -> bool:
        return self._tracking_state_allows_body_motion(str(observation_state.get("tracking_status", "UNKNOWN"))) or (
            self._bootstrap_body_motion_allowed(observation_state)
        )

    def _update_tracking_recovery_state(self, current_state: dict[str, object]) -> dict[str, object]:
        tracking_status = str(current_state.get("tracking_status", "UNKNOWN")).upper()
        if tracking_status == "INITIALIZING":
            self._initializing_streak_frames += 1
        else:
            self._initializing_streak_frames = 0
        current_state["initializing_streak_frames"] = int(self._initializing_streak_frames)
        current_state["bootstrap_body_motion_allowed"] = self._bootstrap_body_motion_allowed(current_state)
        return current_state

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
    def _camera_intrinsics_for_artifacts(artifacts) -> CameraIntrinsics:
        intrinsics = getattr(artifacts, "intrinsics", None)
        if isinstance(intrinsics, CameraIntrinsics):
            return intrinsics
        frame_shape = np.asarray(getattr(artifacts, "frame_bgr", np.empty((0, 0, 3)))).shape
        if len(frame_shape) >= 2 and int(frame_shape[0]) > 0 and int(frame_shape[1]) > 0:
            return intrinsics_for_frame(int(frame_shape[1]), int(frame_shape[0]))
        return intrinsics_for_frame(640, 360)

    @staticmethod
    def _clamp_camera_state(
        yaw_deg: float,
        pitch_deg: float,
        rig_capabilities: RigCapabilities,
    ) -> PlannerCameraState:
        return PlannerCameraState(
            yaw_deg=float(np.clip(yaw_deg, -rig_capabilities.camera_yaw_limit_deg, rig_capabilities.camera_yaw_limit_deg)),
            pitch_deg=float(
                np.clip(
                    pitch_deg,
                    -rig_capabilities.camera_pitch_limit_deg,
                    rig_capabilities.camera_pitch_limit_deg,
                )
            ),
        )

    @staticmethod
    def _format_motion_command(command: MotionCommand) -> str:
        if command.kind in {CommandKind.TRANSLATE, CommandKind.ROTATE_BODY}:
            return (
                f"{command.kind.value}:{command.direction or ''}:"
                f"{command.mode or ''}:{0.0 if command.value is None else float(command.value):.2f}"
            )
        if command.kind is CommandKind.AIM_CAMERA:
            return (
                f"{command.kind.value}:yaw={0.0 if command.yaw_deg is None else float(command.yaw_deg):.1f}:"
                f"pitch={0.0 if command.pitch_deg is None else float(command.pitch_deg):.1f}"
            )
        return f"{command.kind.value}:{0.0 if command.duration_sec is None else float(command.duration_sec):.2f}"

    def _microstep_limit_for_command(self, command: MotionCommand) -> float:
        if command.kind is CommandKind.TRANSLATE:
            return float(_DEFAULT_MICROSTEP_LIMITS["translate_distance_m"])
        if command.kind is CommandKind.ROTATE_BODY:
            return float(_DEFAULT_MICROSTEP_LIMITS["body_yaw_deg"])
        if command.kind is CommandKind.AIM_CAMERA:
            yaw_remaining = 0.0 if command.yaw_deg is None else abs(float(command.yaw_deg) - self._current_camera_state.yaw_deg)
            pitch_remaining = 0.0 if command.pitch_deg is None else abs(float(command.pitch_deg) - self._current_camera_state.pitch_deg)
            if yaw_remaining >= pitch_remaining:
                return float(_DEFAULT_MICROSTEP_LIMITS["camera_yaw_deg"])
            return float(_DEFAULT_MICROSTEP_LIMITS["camera_pitch_deg"])
        return 0.0

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

    @staticmethod
    def _camera_yaw_deg(camera_pose_world: np.ndarray | None) -> float | None:
        pose = np.asarray(camera_pose_world, dtype=np.float32)
        if pose.shape != (4, 4):
            return None
        yaw_rad = math.atan2(float(pose[0, 2]), float(pose[2, 2]))
        return float(math.degrees(yaw_rad))

    @staticmethod
    def _pose_progress_m(previous_pose: np.ndarray | None, current_pose: np.ndarray | None) -> float | None:
        previous = np.asarray(previous_pose, dtype=np.float32)
        current = np.asarray(current_pose, dtype=np.float32)
        if previous.shape != (4, 4) or current.shape != (4, 4):
            return None
        previous_xyz = previous[:3, 3]
        current_xyz = current[:3, 3]
        return float(np.linalg.norm(current_xyz - previous_xyz))

    @staticmethod
    def _angle_delta_deg(previous_yaw_deg: float | None, current_yaw_deg: float | None) -> float | None:
        if previous_yaw_deg is None or current_yaw_deg is None:
            return None
        delta = float(current_yaw_deg) - float(previous_yaw_deg)
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        return delta

    @staticmethod
    def _sector_clearance(
        depth_map: np.ndarray,
        *,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
    ) -> float | None:
        depth = np.asarray(depth_map, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            return None
        height, width = depth.shape[:2]
        x1 = int(max(0, min(width - 1, round(width * float(x_bounds[0])))))
        x2 = int(max(x1 + 1, min(width, round(width * float(x_bounds[1])))))
        y1 = int(max(0, min(height - 1, round(height * float(y_bounds[0])))))
        y2 = int(max(y1 + 1, min(height, round(height * float(y_bounds[1])))))
        crop = depth[y1:y2, x1:x2]
        valid = crop[np.isfinite(crop) & (crop > 0.05)]
        if valid.size == 0:
            return None
        return float(np.nanpercentile(valid, 30))

    @staticmethod
    def _depth_percentiles(depth_map: np.ndarray) -> tuple[float, float, float]:
        depth = np.asarray(depth_map, dtype=np.float32)
        valid = depth[np.isfinite(depth) & (depth > 0.05)]
        if valid.size == 0:
            return (0.0, 0.0, 0.0)
        return (
            float(np.percentile(valid, 10.0)),
            float(np.percentile(valid, 50.0)),
            float(np.percentile(valid, 90.0)),
        )

    def prepare_runtime_artifacts(self, *, artifacts) -> None:
        if bool(getattr(artifacts, "metric_depth_prepared", False)):
            return
        raw_depth_map = np.asarray(
            getattr(artifacts, "raw_depth_map", getattr(artifacts, "depth_map", np.empty((0, 0)))),
            dtype=np.float32,
        )
        if raw_depth_map.ndim != 2:
            setattr(artifacts, "metric_depth_prepared", True)
            return
        artifacts.raw_depth_map = raw_depth_map.copy()
        correction = self._metric_depth_calibrator.correct(
            raw_depth_map,
            intrinsics=self._camera_intrinsics_for_artifacts(artifacts),
            segments=list(getattr(artifacts, "segments", []) or ()),
            camera_height_m=float(self._camera_rig.camera_height_m),
            camera_pitch_deg=float(self._current_camera_state.pitch_deg),
        )
        if self._maybe_update_motion_anchor(artifacts=artifacts, correction=correction):
            correction = self._metric_depth_calibrator.apply(raw_depth_map)
        artifacts.depth_map = np.asarray(correction.depth_map, dtype=np.float32)
        artifacts.metric_depth_prepared = True
        depth_diagnostics = getattr(artifacts, "depth_diagnostics", None)
        if depth_diagnostics is not None:
            artifacts.depth_diagnostics = replace(
                depth_diagnostics,
                normalized_distance_percentiles=self._depth_percentiles(artifacts.depth_map),
                metric_scale_factor=float(correction.scale_factor),
                metric_confidence=float(correction.confidence),
                anchor_count=int(correction.anchor_count),
                correction_state=str(correction.correction_state),
            )

    def _maybe_update_motion_anchor(
        self,
        *,
        artifacts,
        correction: MetricCorrectionResult,
    ) -> bool:
        execution = self._active_macro_execution
        if execution is None or execution.command.kind is not CommandKind.TRANSLATE:
            return False
        if self._previous_observation_state is None:
            return False
        if self._state.phase != EpisodePhase.SELF_CALIBRATING or self._current_schedule is not None:
            return False
        previous_raw_depth = np.asarray(self._previous_observation_state.get("raw_depth_map", np.empty((0, 0))), dtype=np.float32)
        current_raw_depth = np.asarray(getattr(artifacts, "raw_depth_map", np.empty((0, 0))), dtype=np.float32)
        if previous_raw_depth.shape != current_raw_depth.shape or previous_raw_depth.ndim != 2:
            return False
        if self._previous_observation_state.get("target_detection") is not None:
            return False
        if self._target_visible_in_artifacts(artifacts=artifacts):
            return False
        current_forward_clearance = self._sector_clearance(
            correction.depth_map,
            x_bounds=(0.35, 0.65),
            y_bounds=(0.45, 0.95),
        )
        if current_forward_clearance is None or float(current_forward_clearance) < 1.2:
            return False
        if float(current_forward_clearance) < 0.45:
            return False
        last_step_m = min(
            float(_DEFAULT_MICROSTEP_LIMITS["translate_distance_m"]),
            max(0.0, float(execution.sent_translation_m)),
        )
        if last_step_m <= 1e-6:
            return False
        return self._metric_depth_calibrator.update_motion_anchor(
            previous_raw_depth,
            current_raw_depth,
            direction=str(execution.command.direction or "forward"),
            commanded_delta_m=last_step_m,
        )

    @staticmethod
    def _direction_from_action_text(action_text: str) -> str | None:
        text = str(action_text or "")
        if ":left" in text:
            return "left"
        if ":right" in text:
            return "right"
        if ":backward" in text:
            return "rear"
        if ":forward" in text:
            return "front"
        return None

    def _recently_failed_directions(self) -> tuple[str, ...]:
        directions: list[str] = []
        for effect in self._recent_action_effects[-4:]:
            if not effect.likely_blocked:
                continue
            direction = self._direction_from_action_text(effect.action)
            if direction:
                directions.append(direction)
        for search in self._completed_search_history[-4:]:
            if not search.likely_blocked:
                continue
            for action in search.executed_actions:
                direction = self._direction_from_action_text(action)
                if direction:
                    directions.append(direction)
                    break
        return tuple(dict.fromkeys(directions))

    def _navigation_sector_map(
        self,
        *,
        depth_map: np.ndarray,
        rear_clearance_m: float | None,
        recently_failed_directions: tuple[str, ...],
    ) -> tuple[PlannerNavigationSectorObservation, ...]:
        failed = set(recently_failed_directions)
        sector_specs = (
            ("left", (0.00, 0.22), (0.45, 0.95)),
            ("front-left", (0.18, 0.42), (0.42, 0.90)),
            ("front", (0.35, 0.65), (0.42, 0.95)),
            ("front-right", (0.58, 0.82), (0.42, 0.90)),
            ("right", (0.78, 1.00), (0.45, 0.95)),
        )
        sectors: list[PlannerNavigationSectorObservation] = []
        for sector_name, x_bounds, y_bounds in sector_specs:
            clearance_m = self._sector_clearance(depth_map, x_bounds=x_bounds, y_bounds=y_bounds)
            traversable = clearance_m is not None and clearance_m >= 0.95
            obstacle_likelihood = 1.0
            frontier_score = 0.0
            if clearance_m is not None:
                obstacle_likelihood = float(np.clip(1.0 - (float(clearance_m) / 2.5), 0.0, 1.0))
                frontier_score = float(np.clip(float(clearance_m) / 2.5, 0.0, 1.0))
            sectors.append(
                PlannerNavigationSectorObservation(
                    sector=sector_name,
                    clearance_m=clearance_m,
                    traversable=bool(traversable),
                    obstacle_likelihood=obstacle_likelihood,
                    frontier_score=frontier_score,
                    recently_failed=sector_name in failed or (
                        sector_name.startswith("front") and "front" in failed
                    ),
                )
            )
        for sector_name in ("rear-left", "rear", "rear-right"):
            sectors.append(
                PlannerNavigationSectorObservation(
                    sector=sector_name,
                    clearance_m=rear_clearance_m,
                    traversable=rear_clearance_m is not None and rear_clearance_m >= 0.95,
                    obstacle_likelihood=(
                        1.0
                        if rear_clearance_m is None
                        else float(np.clip(1.0 - (float(rear_clearance_m) / 2.5), 0.0, 1.0))
                    ),
                    frontier_score=(
                        0.0
                        if rear_clearance_m is None
                        else float(np.clip(float(rear_clearance_m) / 2.5, 0.0, 1.0))
                    ),
                    recently_failed=sector_name in failed or "rear" in failed,
                )
            )
        return tuple(sectors)

    def _navigation_affordances_summary(self, *, artifacts) -> PlannerNavigationAffordances:
        depth_map = np.asarray(getattr(artifacts, "depth_map", np.empty((0, 0))), dtype=np.float32)
        forward_clearance_m = self._sector_clearance(depth_map, x_bounds=(0.35, 0.65), y_bounds=(0.45, 0.95))
        left_clearance_m = self._sector_clearance(depth_map, x_bounds=(0.0, 0.35), y_bounds=(0.45, 0.95))
        right_clearance_m = self._sector_clearance(depth_map, x_bounds=(0.65, 1.0), y_bounds=(0.45, 0.95))
        previous_affordances = None
        if self._previous_observation_state is not None:
            previous_affordances = self._previous_observation_state.get("navigation_affordances")
        rear_clearance_m = None
        if isinstance(previous_affordances, PlannerNavigationAffordances):
            rear_clearance_m = previous_affordances.forward_clearance_m
        recently_failed_directions = self._recently_failed_directions()
        direction_scores = {
            "front": -1.0 if forward_clearance_m is None else float(forward_clearance_m),
            "left": -1.0 if left_clearance_m is None else float(left_clearance_m),
            "right": -1.0 if right_clearance_m is None else float(right_clearance_m),
            "rear": -1.0 if rear_clearance_m is None else float(rear_clearance_m),
        }
        candidate_open_directions = tuple(
            direction
            for direction, clearance in sorted(direction_scores.items(), key=lambda item: item[1], reverse=True)
            if clearance >= 1.20
        )
        front_blocked = forward_clearance_m is not None and forward_clearance_m < 0.70
        side_clearances = [value for value in (left_clearance_m, right_clearance_m) if value is not None]
        side_best = max(side_clearances) if side_clearances else None
        dead_end_likelihood = 0.15
        if front_blocked and (side_best is None or side_best < 0.85):
            dead_end_likelihood = 0.9
        elif front_blocked:
            dead_end_likelihood = 0.55
        elif not candidate_open_directions:
            dead_end_likelihood = 0.45
        best_exploration_direction = None
        if candidate_open_directions:
            best_exploration_direction = candidate_open_directions[0]
        else:
            best_exploration_direction = max(direction_scores.items(), key=lambda item: item[1])[0]
        sector_map = self._navigation_sector_map(
            depth_map=depth_map,
            rear_clearance_m=rear_clearance_m,
            recently_failed_directions=recently_failed_directions,
        )
        return PlannerNavigationAffordances(
            forward_clearance_m=forward_clearance_m,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            rear_clearance_m=rear_clearance_m,
            front_blocked=bool(front_blocked),
            candidate_open_directions=candidate_open_directions,
            dead_end_likelihood=float(np.clip(dead_end_likelihood, 0.0, 1.0)),
            best_exploration_direction=best_exploration_direction,
            sector_map=sector_map,
            recently_failed_directions=recently_failed_directions,
        )

    def _observation_state_from_artifacts(self, *, artifacts) -> dict[str, object]:
        if not bool(getattr(artifacts, "metric_depth_prepared", False)):
            self.prepare_runtime_artifacts(artifacts=artifacts)
        navigation_affordances = self._navigation_affordances_summary(artifacts=artifacts)
        target_detection = self._target_detection_summary(artifacts=artifacts)
        scene_graph_snapshot = getattr(artifacts, "scene_graph_snapshot", None)
        visible_graph_relations = tuple(
            f"{edge.source}->{edge.relation}->{edge.target}"
            for edge in ((scene_graph_snapshot.visible_edges if scene_graph_snapshot is not None else ()) or ())
        )
        return {
            "camera_pose_world": np.asarray(
                getattr(artifacts, "camera_pose_world", np.empty((0, 0))),
                dtype=np.float32,
            ).copy(),
            "raw_depth_map": np.asarray(
                getattr(artifacts, "raw_depth_map", getattr(artifacts, "depth_map", np.empty((0, 0)))),
                dtype=np.float32,
            ).copy(),
            "corrected_depth_map": np.asarray(getattr(artifacts, "depth_map", np.empty((0, 0))), dtype=np.float32).copy(),
            "yaw_deg": self._camera_yaw_deg(getattr(artifacts, "camera_pose_world", None)),
            "target_detection": target_detection,
            "visible_detection_labels": tuple(
                str(getattr(item, "label", "")) for item in list(getattr(artifacts, "detections", []) or [])
            ),
            "visible_segment_labels": tuple(
                str(getattr(item, "label", "")) for item in list(getattr(artifacts, "segments", []) or [])
            ),
            "visible_graph_relations": visible_graph_relations,
            "navigation_affordances": navigation_affordances,
            "mesh_vertex_count": int(np.asarray(getattr(artifacts, "mesh_vertices_xyz", np.empty((0, 3)))).shape[0]),
            "mesh_triangle_count": int(np.asarray(getattr(artifacts, "mesh_triangles", np.empty((0, 3)))).shape[0]),
            "tracking_status": str(getattr(artifacts, "slam_tracking_state", "UNKNOWN")),
            "tracked_feature_count": int(getattr(artifacts, "tracked_feature_count", 0)),
            "median_reprojection_error": getattr(artifacts, "median_reprojection_error", None),
            "current_camera_state": self._current_camera_state,
            "camera_intrinsics": self._camera_intrinsics_for_artifacts(artifacts),
        }

    def _replan_frame_budget(self) -> int:
        interval_sec = float(getattr(self._config, "sim_replan_interval_sec", 1.0))
        fps = float(max(self._camera_rig.fps, 1e-6))
        return max(1, int(math.ceil(interval_sec * fps)))

    def _target_detection_summary(self, *, artifacts) -> dict[str, float | str | None] | None:
        target_label = getattr(self._scene_spec, "semantic_target_class", "")
        detections = list(getattr(artifacts, "detections", []) or [])
        depth_map = np.asarray(getattr(artifacts, "depth_map", np.empty((0, 0))), dtype=np.float32)
        intrinsics = self._camera_intrinsics_for_artifacts(artifacts)
        frame_shape = np.asarray(getattr(artifacts, "frame_bgr", np.empty((0, 0, 3)))).shape
        if len(frame_shape) < 2:
            return None
        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        if frame_height <= 0 or frame_width <= 0:
            return None

        best_summary = None
        best_score = (-1.0, -1.0)
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
            if bbox_center_x < (frame_width * 0.4):
                horizontal_position = "left"
            elif bbox_center_x > (frame_width * 0.6):
                horizontal_position = "right"
            else:
                horizontal_position = "center"
            confidence = float(getattr(detection, "confidence", 0.0))
            bbox_center_y = y1 + (bbox_height * 0.5)
            median_depth_m = self._bbox_depth_m(depth_map, (x1, y1, x2, y2))
            relative_xyz = None
            center_ray_xyz = None
            if median_depth_m is not None:
                center_xy = np.asarray([[bbox_center_x, bbox_center_y]], dtype=np.float32)
                projected = back_project_pixels(
                    center_xy,
                    np.asarray([float(median_depth_m)], dtype=np.float32),
                    intrinsics,
                )
                if projected.shape == (1, 3):
                    center_ray_xyz = tuple(float(value) for value in projected[0])
                    relative_xyz = center_ray_xyz
            summary = {
                "label": str(getattr(detection, "label", "")),
                "confidence": confidence,
                "area_ratio": area_ratio,
                "center_offset_ratio": center_offset_ratio,
                "horizontal_position": horizontal_position,
                "median_depth_m": median_depth_m,
                "relative_xyz": relative_xyz,
                "center_ray_xyz": center_ray_xyz,
            }
            score = (area_ratio, confidence)
            if score > best_score:
                best_score = score
                best_summary = summary
        return best_summary

    def _goal_visually_reached(self, *, artifacts) -> bool:
        target_detection = self._target_detection_summary(artifacts=artifacts)
        if target_detection is None:
            return False
        median_depth_m = target_detection.get("median_depth_m")
        area_ratio_value = target_detection.get("area_ratio", 0.0)
        center_offset_value = target_detection.get("center_offset_ratio", 1.0)
        area_ratio = 0.0 if area_ratio_value is None else float(area_ratio_value)
        center_offset_ratio = 1.0 if center_offset_value is None else float(center_offset_value)
        if median_depth_m is not None:
            return bool(
                float(median_depth_m) <= 2.5
                and area_ratio >= 0.08
                and center_offset_ratio <= 0.35
            )
        return bool(area_ratio >= 0.22 and center_offset_ratio <= 0.25)

    @staticmethod
    def _schedule_reports_goal_reached(schedule: ActionSchedule | None) -> bool:
        goal_completion = None if schedule is None else getattr(schedule, "goal_completion", None)
        return bool(goal_completion is not None and bool(getattr(goal_completion, "reached", False)))

    def _visual_goal_completion_assessment(self, *, artifacts) -> PlannerGoalCompletion | None:
        target_detection = self._target_detection_summary(artifacts=artifacts)
        if target_detection is None or not self._goal_visually_reached(artifacts=artifacts):
            return None
        area_ratio = float(target_detection.get("area_ratio", 0.0) or 0.0)
        center_offset_ratio = float(target_detection.get("center_offset_ratio", 1.0) or 1.0)
        median_depth_m = target_detection.get("median_depth_m")
        depth_text = "unknown"
        if median_depth_m is not None:
            depth_text = f"{float(median_depth_m):.2f}m"
        return PlannerGoalCompletion(
            reached=True,
            confidence=0.65,
            rationale=(
                "visual fallback: target detection remained large and centered "
                + f"(area_ratio={area_ratio:.3f}, center_offset_ratio={center_offset_ratio:.3f}, depth={depth_text})"
            ),
        )

    def _target_visible_in_artifacts(self, *, artifacts) -> bool:
        target_label = getattr(self._scene_spec, "semantic_target_class", "")
        for detection in list(getattr(artifacts, "detections", []) or []):
            if self._label_matches_target(
                detection_label=getattr(detection, "label", ""),
                target_label=target_label,
            ):
                return True
        return False

    def _reconstruction_brief(self, *, observation_state: dict[str, object]) -> PlannerReconstructionBrief:
        previous_state = self._previous_observation_state
        pose_delta_m = None
        yaw_delta_deg = None
        mesh_growth_delta = 0
        if previous_state is not None:
            pose_delta_m = self._pose_progress_m(
                previous_state.get("camera_pose_world"),
                observation_state.get("camera_pose_world"),
            )
            yaw_delta_deg = self._angle_delta_deg(
                previous_state.get("yaw_deg"),
                observation_state.get("yaw_deg"),
            )
            mesh_growth_delta = int(observation_state["mesh_vertex_count"]) - int(previous_state.get("mesh_vertex_count", 0))
        navigation_affordances = observation_state.get("navigation_affordances")
        frontier_directions = ()
        if isinstance(navigation_affordances, PlannerNavigationAffordances):
            frontier_directions = tuple(navigation_affordances.candidate_open_directions[:3])
            explored_directions = tuple(
                item.sector for item in navigation_affordances.sector_map if item.clearance_m is not None
            )
            unexplored_directions = tuple(
                item.sector
                for item in navigation_affordances.sector_map
                if item.clearance_m is None or item.frontier_score >= 0.45
            )
            recently_failed_directions = tuple(navigation_affordances.recently_failed_directions)
        else:
            explored_directions = ()
            unexplored_directions = ()
            recently_failed_directions = ()
        return PlannerReconstructionBrief(
            pose_delta_m=pose_delta_m,
            yaw_delta_deg=yaw_delta_deg,
            mesh_vertex_count=int(observation_state["mesh_vertex_count"]),
            mesh_triangle_count=int(observation_state["mesh_triangle_count"]),
            mesh_growth_delta=int(mesh_growth_delta),
            frontier_directions=frontier_directions,
            explored_directions=explored_directions,
            unexplored_directions=unexplored_directions,
            recently_failed_directions=recently_failed_directions,
            tracked_feature_count=int(observation_state.get("tracked_feature_count", 0) or 0),
            median_reprojection_error=(
                None
                if observation_state.get("median_reprojection_error") is None
                else float(observation_state["median_reprojection_error"])
            ),
        )

    @staticmethod
    def _clearance_change_text(
        previous_affordances: PlannerNavigationAffordances | None,
        current_affordances: PlannerNavigationAffordances | None,
    ) -> str:
        if previous_affordances is None or current_affordances is None:
            return "unknown"
        previous_clearance = previous_affordances.forward_clearance_m
        current_clearance = current_affordances.forward_clearance_m
        if previous_clearance is None or current_clearance is None:
            return "unknown"
        delta = float(current_clearance) - float(previous_clearance)
        if abs(delta) < 0.08:
            return "forward:stable"
        return f"forward:{delta:+.2f}m"

    @staticmethod
    def _target_evidence_change_text(
        previous_target_detection: dict[str, float | str | None] | None,
        current_target_detection: dict[str, float | str | None] | None,
    ) -> str:
        if previous_target_detection is None and current_target_detection is None:
            return "none"
        if previous_target_detection is None and current_target_detection is not None:
            return "appeared"
        if previous_target_detection is not None and current_target_detection is None:
            return "lost"
        previous_area = float(previous_target_detection.get("area_ratio") or 0.0)
        current_area = float(current_target_detection.get("area_ratio") or 0.0)
        if current_area >= previous_area + 0.01:
            return "stronger"
        if current_area + 0.01 < previous_area:
            return "weaker"
        return "stable"

    def _macro_motion_progress(
        self,
        *,
        execution: _ActiveMacroExecution,
        start_state: dict[str, object],
        current_state: dict[str, object],
    ) -> _MacroMotionProgress:
        vision_translation_m = self._pose_progress_m(
            start_state.get("camera_pose_world"),
            current_state.get("camera_pose_world"),
        )
        vision_yaw_deg = self._angle_delta_deg(
            start_state.get("yaw_deg"),
            current_state.get("yaw_deg"),
        )
        commanded_progress_m = None
        commanded_yaw_deg = None
        fused_progress_m = None
        fused_yaw_deg = None
        progress_source = "none"
        if execution.command.kind is CommandKind.TRANSLATE:
            commanded_progress_m = float(execution.sent_translation_m)
            fused_progress_m = float(execution.sent_translation_m)
            progress_source = "commanded"
        elif execution.command.kind is CommandKind.ROTATE_BODY:
            commanded_yaw_deg = float(execution.sent_yaw_deg)
            fused_yaw_deg = float(execution.sent_yaw_deg)
            progress_source = "commanded"
        elif execution.command.kind in {CommandKind.AIM_CAMERA, CommandKind.PAUSE}:
            vision_translation_m = 0.0
            vision_yaw_deg = 0.0
            fused_progress_m = 0.0
            fused_yaw_deg = 0.0
            progress_source = "clamped"
        return _MacroMotionProgress(
            commanded_progress_m=commanded_progress_m,
            vision_progress_m=None if vision_translation_m is None else float(vision_translation_m),
            fused_progress_m=fused_progress_m,
            progress_source=progress_source,
            commanded_yaw_deg=commanded_yaw_deg,
            vision_yaw_deg=None if vision_yaw_deg is None else abs(float(vision_yaw_deg)),
            fused_yaw_deg=fused_yaw_deg,
        )

    @staticmethod
    def _scene_change_score(previous_depth_map: np.ndarray, current_depth_map: np.ndarray) -> float:
        previous = np.asarray(previous_depth_map, dtype=np.float32)
        current = np.asarray(current_depth_map, dtype=np.float32)
        if previous.shape != current.shape or previous.ndim != 2:
            return 0.0
        height, width = previous.shape[:2]
        x1 = int(round(width * 0.3))
        x2 = int(round(width * 0.7))
        y1 = int(round(height * 0.5))
        y2 = int(round(height * 0.9))
        previous_roi = previous[y1:y2, x1:x2]
        current_roi = current[y1:y2, x1:x2]
        valid = (
            np.isfinite(previous_roi)
            & np.isfinite(current_roi)
            & (previous_roi > 0.05)
            & (current_roi > 0.05)
        )
        if not np.any(valid):
            return 0.0
        return float(np.nanmedian(np.abs(previous_roi[valid] - current_roi[valid])))

    def _microstep_progress_observation(
        self,
        *,
        execution: _ActiveMacroExecution,
        previous_state: dict[str, object] | None,
        current_state: dict[str, object],
    ) -> _MicrostepProgressObservation:
        if execution.command.kind is not CommandKind.TRANSLATE or previous_state is None:
            return _MicrostepProgressObservation(
                clearance_delta_m=None,
                scene_change_score=0.0,
                front_clearance_m=None,
                hard_blocked=False,
                soft_blocked=False,
            )
        previous_affordances = previous_state.get("navigation_affordances")
        current_affordances = current_state.get("navigation_affordances")
        previous_front_clearance = (
            None
            if not isinstance(previous_affordances, PlannerNavigationAffordances)
            else previous_affordances.forward_clearance_m
        )
        current_front_clearance = (
            None
            if not isinstance(current_affordances, PlannerNavigationAffordances)
            else current_affordances.forward_clearance_m
        )
        clearance_delta_m = None
        if previous_front_clearance is not None and current_front_clearance is not None:
            clearance_delta_m = float(current_front_clearance) - float(previous_front_clearance)
        scene_change_score = self._scene_change_score(
            np.asarray(previous_state.get("corrected_depth_map", np.empty((0, 0))), dtype=np.float32),
            np.asarray(current_state.get("corrected_depth_map", np.empty((0, 0))), dtype=np.float32),
        )
        hard_blocked = current_front_clearance is not None and float(current_front_clearance) < 0.45
        soft_blocked = bool(
            not hard_blocked
            and clearance_delta_m is not None
            and abs(float(clearance_delta_m)) < 0.08
            and scene_change_score < 0.06
            and previous_front_clearance is not None
            and current_front_clearance is not None
            and float(current_front_clearance) <= float(previous_front_clearance) + 0.04
        )
        return _MicrostepProgressObservation(
            clearance_delta_m=clearance_delta_m,
            scene_change_score=scene_change_score,
            front_clearance_m=current_front_clearance,
            hard_blocked=bool(hard_blocked),
            soft_blocked=bool(soft_blocked),
        )

    def _record_macro_effect(
        self,
        *,
        executed_macro: ExecutedMacroCommand,
        start_state: dict[str, object],
        current_state: dict[str, object],
    ) -> None:
        estimated_motion_parts: list[str] = []
        if executed_macro.commanded_progress_m is not None:
            estimated_motion_parts.append(f"commanded={float(executed_macro.commanded_progress_m):.2f}m")
        if executed_macro.vision_progress_m is not None:
            estimated_motion_parts.append(f"vision={float(executed_macro.vision_progress_m):.2f}m")
        if executed_macro.fused_progress_m is not None:
            estimated_motion_parts.append(f"fused={float(executed_macro.fused_progress_m):.2f}m")
        if executed_macro.measured_yaw_deg is not None:
            estimated_motion_parts.append(f"yaw={float(executed_macro.measured_yaw_deg):+.1f}deg")
        estimated_motion_parts.append(f"source={str(executed_macro.progress_source)}")
        if executed_macro.command.kind is CommandKind.AIM_CAMERA:
            estimated_motion_parts.append(
                f"camera=yaw{self._current_camera_state.yaw_deg:+.1f}/pitch{self._current_camera_state.pitch_deg:+.1f}"
            )
        if executed_macro.command.kind is CommandKind.PAUSE:
            estimated_motion_parts.append(f"pause={float(executed_macro.command.duration_sec or 0.0):.2f}s")
        if not estimated_motion_parts:
            estimated_motion_parts.append("pose_unavailable")
        effect = PlannerActionEffectSummary(
            action=self._format_motion_command(executed_macro.command),
            estimated_motion=", ".join(estimated_motion_parts),
            clearance_change=self._clearance_change_text(
                start_state.get("navigation_affordances"),
                current_state.get("navigation_affordances"),
            ),
            target_evidence_change=executed_macro.target_evidence_change,
            likely_blocked=bool(not executed_macro.completed and executed_macro.aborted),
            aborted=bool(executed_macro.aborted),
            commanded_progress_m=executed_macro.commanded_progress_m,
            vision_progress_m=executed_macro.vision_progress_m,
            fused_progress_m=executed_macro.fused_progress_m,
            progress_source=str(executed_macro.progress_source),
        )
        self._recent_action_effects.append(effect)
        self._recent_action_effects = self._recent_action_effects[-4:]
        self._action_history.append(effect.action)
        self._action_history = self._action_history[-8:]
        self._active_schedule_executed_commands.append(effect.action)

    def _finalize_active_macro_execution(
        self,
        *,
        current_state: dict[str, object],
        completed: bool,
        aborted: bool,
    ) -> ExecutedMacroCommand | None:
        execution = self._active_macro_execution
        if execution is None:
            return None
        progress = self._macro_motion_progress(
            execution=execution,
            start_state=execution.start_state,
            current_state=current_state,
        )
        target_evidence_change = self._target_evidence_change_text(
            execution.start_state.get("target_detection"),
            current_state.get("target_detection"),
        )
        executed_macro = ExecutedMacroCommand(
            command=execution.command,
            measured_translation_m=progress.fused_progress_m,
            measured_yaw_deg=progress.fused_yaw_deg,
            completed=bool(completed),
            aborted=bool(aborted),
            microstep_count=int(execution.microstep_count),
            target_evidence_change=target_evidence_change,
            pose_progress_m=progress.vision_progress_m,
            intent=str(execution.command.intent or ""),
            commanded_progress_m=progress.commanded_progress_m,
            vision_progress_m=progress.vision_progress_m,
            fused_progress_m=progress.fused_progress_m,
            progress_source=str(progress.progress_source),
        )
        self._record_macro_effect(
            executed_macro=executed_macro,
            start_state=execution.start_state,
            current_state=current_state,
        )
        self._active_macro_execution = None
        return executed_macro

    def _finalize_completed_search(self, *, artifacts) -> None:
        if self._active_schedule_start_frame is None:
            return
        current_state = self._observation_state_from_artifacts(artifacts=artifacts)
        start_state = self._active_schedule_start_observation or {}
        pose_progress_m = self._pose_progress_m(
            start_state.get("camera_pose_world"),
            current_state.get("camera_pose_world"),
        )
        start_signature = {
            "detections": set(start_state.get("visible_detection_labels", ())),
            "segments": set(start_state.get("visible_segment_labels", ())),
            "relations": set(start_state.get("visible_graph_relations", ())),
        }
        end_signature = {
            "detections": set(current_state.get("visible_detection_labels", ())),
            "segments": set(current_state.get("visible_segment_labels", ())),
            "relations": set(current_state.get("visible_graph_relations", ())),
        }
        entered_new_view = bool(
            end_signature["detections"] - start_signature["detections"]
            or end_signature["segments"] - start_signature["segments"]
            or end_signature["relations"] - start_signature["relations"]
            or (pose_progress_m is not None and pose_progress_m >= 0.08)
        )
        evidence_gained = bool(
            self._target_evidence_change_text(
                start_state.get("target_detection"),
                current_state.get("target_detection"),
            )
            in {"appeared", "stronger"}
            or entered_new_view
        )
        likely_blocked = any(item.likely_blocked for item in self._recent_action_effects[-4:])
        self._completed_search_history.append(
            PlannerSearchOutcome(
                start_frame=int(self._active_schedule_start_frame),
                end_frame=int(self._state.frame_index),
                executed_actions=tuple(self._active_schedule_executed_commands),
                target_visible_after_search=self._target_visible_in_artifacts(artifacts=artifacts),
                tracking_status=str(getattr(artifacts, "slam_tracking_state", "UNKNOWN")),
                entered_new_view=entered_new_view,
                pose_progress_m=pose_progress_m,
                likely_blocked=likely_blocked,
                evidence_gained=evidence_gained,
            )
        )
        self._completed_search_history = self._completed_search_history[-4:]
        self._active_schedule_start_frame = None
        self._active_schedule_start_observation = None
        self._active_schedule_executed_commands = []

    def next_frame(self, *, timeout_sec: float | None = 1.0) -> FramePacket | None:
        if self._latest_sensor_frame is None and self.is_waiting_for_frame():
            self._promote_pending_sensor_frame(timeout_sec=timeout_sec)
        _ = timeout_sec
        if self._closed or self._latest_sensor_frame is None:
            return None
        sensor_frame = self._latest_sensor_frame
        latest_tracking_status = (
            "UNKNOWN"
            if self._previous_observation_state is None
            else str(self._previous_observation_state.get("tracking_status", "UNKNOWN"))
        )
        return FramePacket(
            frame_bgr=np.asarray(sensor_frame.frame_bgr, dtype=np.uint8),
            timestamp_sec=float(sensor_frame.timestamp_sec),
            intrinsics_gt=intrinsics_for_frame(
                int(np.asarray(sensor_frame.frame_bgr).shape[1]),
                int(np.asarray(sensor_frame.frame_bgr).shape[0]),
            ),
            planner_context=self._latest_planner_context,
            planner_schedule=self._latest_planner_schedule,
            tracking_state=latest_tracking_status,
            calibration_source="self_calibration/converged"
            if self._state.selfcal_step_index >= len(self._selfcal_actions)
            else "self_calibration/pending",
        )

    def record_runtime_observation(self, *, frame_packet: FramePacket, artifacts) -> None:
        _ = frame_packet
        if self._closed or self._latest_sensor_frame is None:
            return
        current_state = self._observation_state_from_artifacts(artifacts=artifacts)
        current_state = self._update_tracking_recovery_state(current_state)
        self._update_active_macro_execution(current_state=current_state)
        next_command = self._select_next_command(artifacts=artifacts, current_state=current_state)
        self._previous_observation_state = current_state
        if self._state.phase == EpisodePhase.SUCCEEDED:
            return
        submit_action = getattr(self._sensor_backend, "submit_action", None)
        if callable(submit_action):
            self._latest_sensor_frame = None
            submit_action(next_command)
            return
        self._commit_sensor_frame(self._sensor_backend.apply_action(next_command))

    def close(self) -> None:
        self._closed = True
        backend_close = getattr(self._sensor_backend, "close", None)
        if callable(backend_close):
            backend_close()
        self._write_episode_report()

    def _mark_goal_completed(self) -> None:
        self._state.mission_succeeded = True
        self._state.phase = EpisodePhase.SUCCEEDED
        self._current_schedule = None
        self._active_macro_execution = None
        self._latest_sensor_frame = None
        self._write_episode_report()

    def _record_planner_turn(self, *, context, schedule: ActionSchedule) -> None:
        self._state.planner_turn_count += 1
        self._planner_turn_logs.append(
            {
                "frame_index": self._state.frame_index,
                "phase": context.phase.value,
                "prompt": context,
                "schedule": {
                    "rationale": schedule.rationale,
                    "situation_summary": schedule.situation_summary,
                    "behavior_mode": schedule.behavior_mode,
                    "model": schedule.model,
                    "goal_hypothesis": None
                    if schedule.goal_hypothesis is None
                    else {
                        "status": schedule.goal_hypothesis.status,
                        "bearing_hint": schedule.goal_hypothesis.bearing_hint,
                        "distance_hint": schedule.goal_hypothesis.distance_hint,
                        "evidence": list(schedule.goal_hypothesis.evidence_sources),
                        "confidence": float(schedule.goal_hypothesis.confidence),
                    },
                    "goal_completion": None
                    if schedule.goal_completion is None
                    else {
                        "reached": bool(schedule.goal_completion.reached),
                        "confidence": float(schedule.goal_completion.confidence),
                        "rationale": schedule.goal_completion.rationale,
                    },
                    "safety_flags": None
                    if schedule.safety_flags is None
                    else {
                        "front_blocked": bool(schedule.safety_flags.front_blocked),
                        "dead_end_risk": float(schedule.safety_flags.dead_end_risk),
                        "tracking_risk": schedule.safety_flags.tracking_risk,
                        "replan_reason": schedule.safety_flags.replan_reason,
                    },
                    "confidence": schedule.confidence,
                    "commands": [
                        self._serialize_motion_command(command)
                        for command in schedule.commands
                    ],
                },
            }
        )

    @staticmethod
    def _serialize_motion_command(command: MotionCommand) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": command.kind.value,
            "intent": str(command.intent or ""),
        }
        if command.kind in {CommandKind.TRANSLATE, CommandKind.ROTATE_BODY}:
            payload["direction"] = command.direction
            payload["mode"] = command.mode
            payload["value"] = command.value
        elif command.kind is CommandKind.AIM_CAMERA:
            payload["yaw_deg"] = command.yaw_deg
            payload["pitch_deg"] = command.pitch_deg
        else:
            payload["duration_sec"] = command.duration_sec
        return payload

    @staticmethod
    def _contains_contradictory_commands(commands: tuple[MotionCommand, ...]) -> bool:
        rotate_directions = {
            str(command.direction)
            for command in commands
            if command.kind is CommandKind.ROTATE_BODY and str(command.direction or "").strip()
        }
        if {"left", "right"}.issubset(rotate_directions):
            return True
        aim_signs = set()
        for command in commands:
            if command.kind is not CommandKind.AIM_CAMERA or command.yaw_deg is None:
                continue
            if float(abs(command.yaw_deg)) < 4.0:
                continue
            aim_signs.add("right" if float(command.yaw_deg) > 0.0 else "left")
        return len(aim_signs) > 1

    @staticmethod
    def _commands_include_body_motion(commands: tuple[MotionCommand, ...]) -> bool:
        return any(command.kind in {CommandKind.TRANSLATE, CommandKind.ROTATE_BODY} for command in commands)

    @staticmethod
    def _tracking_risk(tracking_status: str) -> str:
        normalized = str(tracking_status or "").upper()
        if normalized in {"TRACKING", "RELOCALIZED"}:
            return "low"
        if normalized in {"LOST", "UNKNOWN"}:
            return "high"
        return "medium"

    def _bootstrap_tracking_recovery_commands(
        self,
        *,
        observation_state: dict[str, object],
    ) -> tuple[MotionCommand, ...]:
        affordances = observation_state.get("navigation_affordances")
        best_direction = None
        left_clearance = None
        right_clearance = None
        front_clearance = None
        rear_clearance = None
        if isinstance(affordances, PlannerNavigationAffordances):
            best_direction = affordances.best_exploration_direction
            left_clearance = affordances.left_clearance_m
            right_clearance = affordances.right_clearance_m
            front_clearance = affordances.forward_clearance_m
            rear_clearance = affordances.rear_clearance_m

        translate_direction = "forward"
        translate_value = 0.32
        if best_direction in {"left", "front-left", "rear-left"} and (left_clearance is None or left_clearance >= 0.7):
            translate_direction = "left"
            translate_value = 0.24
        elif best_direction in {"right", "front-right", "rear-right"} and (
            right_clearance is None or right_clearance >= 0.7
        ):
            translate_direction = "right"
            translate_value = 0.24
        elif front_clearance is not None and front_clearance >= 0.9:
            translate_direction = "forward"
            translate_value = 0.32
        elif left_clearance is not None and right_clearance is not None:
            if left_clearance >= right_clearance and left_clearance >= 0.65:
                translate_direction = "left"
                translate_value = 0.24
            elif right_clearance >= 0.65:
                translate_direction = "right"
                translate_value = 0.24
            elif rear_clearance is not None and rear_clearance >= 0.7:
                translate_direction = "backward"
                translate_value = 0.24

        if "right" in str(best_direction or "") or translate_direction == "right":
            rotate_direction = "right"
            aim_yaw = 14.0
        else:
            rotate_direction = "left"
            aim_yaw = -14.0

        return (
            MotionCommand(
                kind=CommandKind.TRANSLATE,
                direction=translate_direction,
                mode="distance_m",
                value=translate_value,
                intent="create monocular parallax for tracking bootstrap",
            ),
            MotionCommand(
                kind=CommandKind.ROTATE_BODY,
                direction=rotate_direction,
                mode="angle_deg",
                value=16.0,
                intent="add body yaw change while tracking bootstraps",
            ),
            MotionCommand(
                kind=CommandKind.AIM_CAMERA,
                yaw_deg=aim_yaw,
                pitch_deg=0.0,
                intent="stabilize view along the bootstrap motion corridor",
            ),
        )

    def _escape_commands(self, *, observation_state: dict[str, object]) -> tuple[MotionCommand, ...]:
        affordances = observation_state.get("navigation_affordances")
        best_direction = None
        if isinstance(affordances, PlannerNavigationAffordances):
            best_direction = affordances.best_exploration_direction
        if not self._body_motion_allowed_for_observation_state(observation_state):
            return (
                MotionCommand(
                    kind=CommandKind.AIM_CAMERA,
                    yaw_deg=-12.0 if best_direction != "right" else 12.0,
                    pitch_deg=0.0,
                    intent="scan safer corridor while tracking recovers",
                ),
                MotionCommand(kind=CommandKind.PAUSE, duration_sec=0.4, intent="tracking unstable; reassess"),
            )
        translate_direction = (
            "left"
            if best_direction == "left"
            else "right"
            if best_direction == "right"
            else "backward"
        )
        rotate_direction = "left" if best_direction != "right" else "right"
        aim_yaw = -12.0 if best_direction != "right" else 12.0
        return (
            MotionCommand(
                kind=CommandKind.TRANSLATE,
                direction=translate_direction,
                mode="distance_m",
                value=0.16,
                intent="escape blocked pose",
            ),
            MotionCommand(
                kind=CommandKind.ROTATE_BODY,
                direction=rotate_direction,
                mode="angle_deg",
                value=12.0,
                intent="reorient toward safer view",
            ),
            MotionCommand(
                kind=CommandKind.AIM_CAMERA,
                yaw_deg=aim_yaw,
                pitch_deg=0.0,
                intent="confirm escape corridor",
            ),
        )

    def _sanitize_command(self, command: MotionCommand, *, observation_state: dict[str, object]) -> MotionCommand | None:
        intent = str(command.intent or "")
        allow_body_motion = self._body_motion_allowed_for_observation_state(observation_state)
        if command.kind is CommandKind.TRANSLATE:
            if not allow_body_motion:
                return None
            direction = str(command.direction or "")
            mode = str(command.mode or "")
            if direction not in {"forward", "backward", "left", "right"}:
                return None
            if mode not in {"distance_m", "hold_sec"}:
                return None
            if mode == "distance_m":
                value = min(abs(float(command.value or 0.0)), 0.72)
            else:
                max_hold_sec = 0.72 / max(float(self._rig_capabilities.move_speed_mps), 1e-6)
                value = min(abs(float(command.value or 0.0)), max_hold_sec)
            if value <= 1e-6:
                return None
            return MotionCommand(kind=CommandKind.TRANSLATE, direction=direction, mode=mode, value=value, intent=intent)
        if command.kind is CommandKind.ROTATE_BODY:
            if not allow_body_motion:
                return None
            direction = str(command.direction or "")
            mode = str(command.mode or "")
            if direction not in {"left", "right"}:
                return None
            if mode not in {"angle_deg", "hold_sec"}:
                return None
            if mode == "angle_deg":
                value = min(abs(float(command.value or 0.0)), 36.0)
            else:
                max_hold_sec = 36.0 / max(float(self._rig_capabilities.turn_speed_deg_per_sec), 1e-6)
                value = min(abs(float(command.value or 0.0)), max_hold_sec)
            if value <= 1e-6:
                return None
            return MotionCommand(kind=CommandKind.ROTATE_BODY, direction=direction, mode=mode, value=value, intent=intent)
        if command.kind is CommandKind.AIM_CAMERA:
            yaw_deg = float(np.clip(float(command.yaw_deg or 0.0), -70.0, 70.0))
            pitch_deg = float(np.clip(float(command.pitch_deg or 0.0), -55.0, 55.0))
            return MotionCommand(kind=CommandKind.AIM_CAMERA, yaw_deg=yaw_deg, pitch_deg=pitch_deg, intent=intent)
        duration_sec = max(0.0, float(command.duration_sec or 0.0))
        return MotionCommand(kind=CommandKind.PAUSE, duration_sec=duration_sec, intent=intent)

    def _validated_schedule(
        self,
        *,
        schedule: ActionSchedule,
        observation_state: dict[str, object],
        context,
    ) -> ActionSchedule:
        if self._schedule_reports_goal_reached(schedule):
            return replace(schedule, commands=())
        affordances = observation_state.get("navigation_affordances")
        front_blocked = bool(
            isinstance(affordances, PlannerNavigationAffordances) and affordances.front_blocked
        )
        dead_end_risk = float(
            0.0
            if not isinstance(affordances, PlannerNavigationAffordances)
            else affordances.dead_end_likelihood
        )
        tracking_risk = self._tracking_risk(observation_state.get("tracking_status", "UNKNOWN"))
        commands = tuple(
            command
            for command in (
                self._sanitize_command(command, observation_state=observation_state)
                for command in tuple(schedule.commands or ())
            )
            if command is not None
        )[:3]
        if self._contains_contradictory_commands(commands):
            return ActionSchedule(
                commands=self._escape_commands(observation_state=observation_state),
                rationale="planner returned contradictory actions; use escape fallback",
                model=f"{schedule.model}/validated",
                issued_at_frame=schedule.issued_at_frame,
                situation_summary=schedule.situation_summary or "Contradictory actions rejected; executing escape fallback.",
                behavior_mode="escape",
                goal_hypothesis=schedule.goal_hypothesis or context.goal_estimate,
                goal_completion=schedule.goal_completion,
                safety_flags=PlannerSafetyFlags(
                    front_blocked=front_blocked,
                    dead_end_risk=dead_end_risk,
                    tracking_risk=tracking_risk,
                    replan_reason="contradictory_actions",
                ),
                confidence=0.0 if schedule.confidence is None else float(schedule.confidence),
            )
        if tracking_risk == "high" and any(
            command.kind in {CommandKind.TRANSLATE, CommandKind.ROTATE_BODY}
            for command in tuple(schedule.commands or ())
        ):
            return ActionSchedule(
                commands=(
                    MotionCommand(
                        kind=CommandKind.AIM_CAMERA,
                        yaw_deg=float(self._current_camera_state.yaw_deg),
                        pitch_deg=float(self._current_camera_state.pitch_deg),
                        intent="hold camera while tracking recovers",
                    ),
                    MotionCommand(kind=CommandKind.PAUSE, duration_sec=0.5, intent="tracking unstable; pause"),
                ),
                rationale="tracking risk too high for aggressive translation",
                model=f"{schedule.model}/validated",
                issued_at_frame=schedule.issued_at_frame,
                situation_summary=schedule.situation_summary or "Tracking is unstable; pausing to reassess.",
                behavior_mode="scan",
                goal_hypothesis=schedule.goal_hypothesis or context.goal_estimate,
                goal_completion=schedule.goal_completion,
                safety_flags=PlannerSafetyFlags(
                    front_blocked=front_blocked,
                    dead_end_risk=dead_end_risk,
                    tracking_risk=tracking_risk,
                    replan_reason="tracking_risk",
                ),
                confidence=0.0 if schedule.confidence is None else float(schedule.confidence),
            )
        if self._bootstrap_body_motion_allowed(observation_state) and not self._commands_include_body_motion(commands):
            bootstrap_commands = self._bootstrap_tracking_recovery_commands(observation_state=observation_state)
            return ActionSchedule(
                commands=bootstrap_commands,
                rationale="tracking is still initializing after self-calibration; force bootstrap parallax motion",
                model=f"{schedule.model}/validated",
                issued_at_frame=schedule.issued_at_frame,
                situation_summary=(
                    schedule.situation_summary
                    or "Tracking remained in INITIALIZING after self-calibration; executing bootstrap motion."
                ),
                behavior_mode="escape",
                goal_hypothesis=schedule.goal_hypothesis or context.goal_estimate,
                goal_completion=schedule.goal_completion,
                safety_flags=PlannerSafetyFlags(
                    front_blocked=front_blocked,
                    dead_end_risk=dead_end_risk,
                    tracking_risk=tracking_risk,
                    replan_reason="tracking_bootstrap",
                ),
                confidence=0.0 if schedule.confidence is None else float(schedule.confidence),
            )
        if not commands:
            commands = (MotionCommand(kind=CommandKind.PAUSE, duration_sec=0.5, intent="empty schedule fallback"),)
        if schedule.safety_flags is None:
            schedule = ActionSchedule(
                commands=commands,
                rationale=schedule.rationale,
                model=schedule.model,
                issued_at_frame=schedule.issued_at_frame,
                situation_summary=schedule.situation_summary,
                behavior_mode=schedule.behavior_mode,
                goal_hypothesis=schedule.goal_hypothesis or context.goal_estimate,
                goal_completion=schedule.goal_completion,
                safety_flags=PlannerSafetyFlags(
                    front_blocked=front_blocked,
                    dead_end_risk=dead_end_risk,
                    tracking_risk=tracking_risk,
                    replan_reason="model_response",
                ),
                confidence=schedule.confidence,
            )
        elif commands != tuple(schedule.commands):
            schedule = ActionSchedule(
                commands=commands,
                rationale=schedule.rationale,
                model=schedule.model,
                issued_at_frame=schedule.issued_at_frame,
                situation_summary=schedule.situation_summary,
                behavior_mode=schedule.behavior_mode,
                goal_hypothesis=schedule.goal_hypothesis or context.goal_estimate,
                goal_completion=schedule.goal_completion,
                safety_flags=schedule.safety_flags,
                confidence=schedule.confidence,
            )
        return schedule

    def _start_macro_execution(
        self,
        *,
        command: MotionCommand,
        current_state: dict[str, object],
    ) -> _ActiveMacroExecution:
        execution = _ActiveMacroExecution(command=command, start_state=dict(current_state))
        tracking_status = str(current_state.get("tracking_status", "UNKNOWN"))
        execution.allow_untracked_completion = (
            command.kind in {CommandKind.TRANSLATE, CommandKind.ROTATE_BODY}
            and not self._tracking_state_allows_body_motion(tracking_status)
            and self._body_motion_allowed_for_observation_state(current_state)
        )
        if command.kind is CommandKind.TRANSLATE:
            execution.requested_translation_m = (
                float(command.value or 0.0) * float(self._rig_capabilities.move_speed_mps)
                if str(command.mode) == "hold_sec"
                else float(command.value or 0.0)
            )
        elif command.kind is CommandKind.ROTATE_BODY:
            execution.requested_yaw_deg = (
                float(command.value or 0.0) * float(self._rig_capabilities.turn_speed_deg_per_sec)
                if str(command.mode) == "hold_sec"
                else float(command.value or 0.0)
            )
        elif command.kind is CommandKind.AIM_CAMERA:
            target_state = self._clamp_camera_state(
                float(command.yaw_deg or 0.0),
                float(command.pitch_deg or 0.0),
                self._rig_capabilities,
            )
            execution.target_camera_yaw_deg = target_state.yaw_deg
            execution.target_camera_pitch_deg = target_state.pitch_deg
        else:
            execution.requested_pause_sec = float(command.duration_sec or 0.0)
        return execution

    def _append_microstep_trace(self, *, command: MotionCommand, rig_delta: UnityRigDeltaCommand) -> None:
        self._debug_microstep_trace.append(
            {
                "frame_index": int(self._state.frame_index),
                "command": self._serialize_motion_command(command),
                "rig_delta": {
                    "translate_forward_m": float(rig_delta.translate_forward_m),
                    "translate_right_m": float(rig_delta.translate_right_m),
                    "body_yaw_deg": float(rig_delta.body_yaw_deg),
                    "camera_yaw_delta_deg": float(rig_delta.camera_yaw_delta_deg),
                    "camera_pitch_delta_deg": float(rig_delta.camera_pitch_delta_deg),
                    "pause_sec": float(rig_delta.pause_sec),
                },
            }
        )
        self._debug_microstep_trace = self._debug_microstep_trace[-128:]

    def _next_microstep_delta_for_active_macro(self) -> UnityRigDeltaCommand | None:
        execution = self._active_macro_execution
        if execution is None:
            return None
        command = execution.command
        rig_delta = UnityRigDeltaCommand()
        if command.kind is CommandKind.TRANSLATE:
            remaining_m = max(0.0, float(execution.requested_translation_m) - float(execution.sent_translation_m))
            if remaining_m <= 1e-6:
                return None
            chunk_m = min(float(_DEFAULT_MICROSTEP_LIMITS["translate_distance_m"]), remaining_m)
            execution.sent_translation_m += chunk_m
            if command.direction == "forward":
                rig_delta = UnityRigDeltaCommand(translate_forward_m=chunk_m)
            elif command.direction == "backward":
                rig_delta = UnityRigDeltaCommand(translate_forward_m=-chunk_m)
            elif command.direction == "left":
                rig_delta = UnityRigDeltaCommand(translate_right_m=-chunk_m)
            else:
                rig_delta = UnityRigDeltaCommand(translate_right_m=chunk_m)
        elif command.kind is CommandKind.ROTATE_BODY:
            remaining_deg = max(0.0, float(execution.requested_yaw_deg) - float(execution.sent_yaw_deg))
            if remaining_deg <= 1e-6:
                return None
            chunk_deg = min(float(_DEFAULT_MICROSTEP_LIMITS["body_yaw_deg"]), remaining_deg)
            execution.sent_yaw_deg += chunk_deg
            rig_delta = UnityRigDeltaCommand(body_yaw_deg=chunk_deg if command.direction == "left" else -chunk_deg)
        elif command.kind is CommandKind.AIM_CAMERA:
            target_yaw_deg = float(
                execution.target_camera_yaw_deg
                if execution.target_camera_yaw_deg is not None
                else self._current_camera_state.yaw_deg
            )
            target_pitch_deg = float(
                execution.target_camera_pitch_deg
                if execution.target_camera_pitch_deg is not None
                else self._current_camera_state.pitch_deg
            )
            yaw_remaining_deg = target_yaw_deg - float(self._current_camera_state.yaw_deg)
            pitch_remaining_deg = target_pitch_deg - float(self._current_camera_state.pitch_deg)
            yaw_step_deg = (
                0.0
                if abs(yaw_remaining_deg) <= 0.25
                else math.copysign(
                    min(float(_DEFAULT_MICROSTEP_LIMITS["camera_yaw_deg"]), abs(yaw_remaining_deg)),
                    yaw_remaining_deg,
                )
            )
            pitch_step_deg = (
                0.0
                if abs(pitch_remaining_deg) <= 0.25
                else math.copysign(
                    min(float(_DEFAULT_MICROSTEP_LIMITS["camera_pitch_deg"]), abs(pitch_remaining_deg)),
                    pitch_remaining_deg,
                )
            )
            if abs(yaw_step_deg) <= 1e-6 and abs(pitch_step_deg) <= 1e-6:
                return None
            execution.sent_camera_yaw_deg += abs(yaw_step_deg)
            execution.sent_camera_pitch_deg += abs(pitch_step_deg)
            self._current_camera_state = self._clamp_camera_state(
                self._current_camera_state.yaw_deg + yaw_step_deg,
                self._current_camera_state.pitch_deg + pitch_step_deg,
                self._rig_capabilities,
            )
            rig_delta = UnityRigDeltaCommand(
                camera_yaw_delta_deg=yaw_step_deg,
                camera_pitch_delta_deg=pitch_step_deg,
            )
        else:
            if execution.microstep_count > 0:
                return None
            rig_delta = UnityRigDeltaCommand(pause_sec=float(execution.requested_pause_sec))
        execution.microstep_count += 1
        self._append_microstep_trace(command=command, rig_delta=rig_delta)
        return rig_delta

    def _update_active_macro_execution(self, *, current_state: dict[str, object]) -> None:
        execution = self._active_macro_execution
        if execution is None:
            return
        is_self_calibration_macro = self._state.phase == EpisodePhase.SELF_CALIBRATING and self._current_schedule is None
        target_evidence_change = self._target_evidence_change_text(
            execution.start_state.get("target_detection"),
            current_state.get("target_detection"),
        )
        body_motion_allowed_now = self._body_motion_allowed_for_observation_state(current_state)
        completed = False
        aborted = False
        replan_after_finalize = False
        if execution.command.kind is CommandKind.TRANSLATE:
            step_observation = self._microstep_progress_observation(
                execution=execution,
                previous_state=self._previous_observation_state,
                current_state=current_state,
            )
            execution.recent_step_observations.append(step_observation)
            execution.recent_step_observations = execution.recent_step_observations[-2:]
            if step_observation.soft_blocked:
                execution.soft_block_streak += 1
            else:
                execution.soft_block_streak = 0
            hard_blocked = bool(step_observation.hard_blocked)
            soft_blocked = execution.soft_block_streak >= 2
            evidence_replan_allowed = target_evidence_change in _TARGET_EVIDENCE_REPLAN_STATES
            if str(execution.command.direction or "") == "forward":
                committed_distance_m = max(0.24, 0.35 * float(execution.requested_translation_m))
                evidence_replan_allowed = evidence_replan_allowed and (
                    float(execution.sent_translation_m) >= committed_distance_m - 1e-6
                )
            else:
                evidence_replan_allowed = evidence_replan_allowed and execution.microstep_count > 0
            if not body_motion_allowed_now and not (is_self_calibration_macro or execution.allow_untracked_completion):
                aborted = True
            elif hard_blocked or soft_blocked:
                aborted = True
            elif execution.sent_translation_m >= execution.requested_translation_m - 1e-6:
                completed = True
                replan_after_finalize = bool(evidence_replan_allowed)
            elif evidence_replan_allowed:
                completed = True
                replan_after_finalize = True
        elif execution.command.kind is CommandKind.ROTATE_BODY:
            if not body_motion_allowed_now and not (is_self_calibration_macro or execution.allow_untracked_completion):
                aborted = True
            elif execution.sent_yaw_deg >= execution.requested_yaw_deg - 1e-6:
                completed = True
            elif target_evidence_change in _TARGET_EVIDENCE_REPLAN_STATES and execution.microstep_count > 0:
                completed = True
                replan_after_finalize = True
        elif execution.command.kind is CommandKind.AIM_CAMERA:
            completed = (
                abs(float(self._current_camera_state.yaw_deg) - float(execution.target_camera_yaw_deg or 0.0)) <= 0.25
                and abs(float(self._current_camera_state.pitch_deg) - float(execution.target_camera_pitch_deg or 0.0)) <= 0.25
            )
        else:
            completed = execution.microstep_count > 0
        if not completed and not aborted:
            return
        self._finalize_active_macro_execution(current_state=current_state, completed=completed, aborted=aborted)
        if is_self_calibration_macro:
            if self._state.selfcal_step_index >= len(self._selfcal_actions):
                self._state.phase = EpisodePhase.PERCEIVE_AND_PLAN
            else:
                self._state.phase = EpisodePhase.SELF_CALIBRATING
            return
        if aborted or replan_after_finalize:
            self._current_schedule = None
            self._state.schedule_cursor = 0
            self._state.phase = EpisodePhase.REASSESS
            return
        if self._current_schedule is None or self._state.schedule_cursor >= len(self._current_schedule.commands):
            self._current_schedule = None
            self._state.schedule_cursor = 0
            self._state.phase = EpisodePhase.REASSESS
            return
        self._state.phase = EpisodePhase.EXECUTING_SCHEDULE

    def _advance_self_calibration(self, *, current_state: dict[str, object]) -> bool:
        while self._state.selfcal_step_index < len(self._selfcal_actions):
            command = self._selfcal_actions[self._state.selfcal_step_index]
            self._state.selfcal_step_index += 1
            self._active_macro_execution = self._start_macro_execution(command=command, current_state=current_state)
            return True
        self._state.phase = EpisodePhase.PERCEIVE_AND_PLAN
        return False

    def _start_next_schedule_macro(self, *, current_state: dict[str, object]) -> bool:
        if self._current_schedule is None or self._state.schedule_cursor >= len(self._current_schedule.commands):
            self._current_schedule = None
            self._state.schedule_cursor = 0
            self._state.phase = EpisodePhase.REASSESS
            return False
        command = self._current_schedule.commands[int(self._state.schedule_cursor)]
        self._state.schedule_cursor += 1
        self._active_macro_execution = self._start_macro_execution(command=command, current_state=current_state)
        return True

    def _select_next_command(self, *, artifacts, current_state: dict[str, object]) -> UnityRigDeltaCommand:
        for _ in range(12):
            if self._state.phase == EpisodePhase.SUCCEEDED:
                return UnityRigDeltaCommand(pause_sec=0.0)
            if self._state.phase == EpisodePhase.SELF_CALIBRATING:
                if self._active_macro_execution is None and not self._advance_self_calibration(current_state=current_state):
                    continue
            elif self._state.phase in {EpisodePhase.PERCEIVE_AND_PLAN, EpisodePhase.REASSESS} or self._current_schedule is None:
                self._plan_next_schedule(artifacts=artifacts, observation_state=current_state)
                continue
            elif self._active_macro_execution is None and not self._start_next_schedule_macro(current_state=current_state):
                continue
            next_delta = self._next_microstep_delta_for_active_macro()
            if next_delta is None:
                self._finalize_active_macro_execution(current_state=current_state, completed=True, aborted=False)
                continue
            return next_delta
        return UnityRigDeltaCommand(pause_sec=0.25)

    def _plan_next_schedule(self, *, artifacts, observation_state: dict[str, object] | None = None) -> None:
        self._finalize_completed_search(artifacts=artifacts)
        observation_state = observation_state or self._observation_state_from_artifacts(artifacts=artifacts)
        reconstruction_brief = self._reconstruction_brief(observation_state=observation_state)
        max_commands_per_schedule = min(3, max(1, int(getattr(self._config, "sim_action_batch_size", 3))))
        allowed_command_kinds = (
            tuple(item.value for item in CommandKind)
            if self._body_motion_allowed_for_observation_state(observation_state)
            else (CommandKind.AIM_CAMERA.value, CommandKind.PAUSE.value)
        )
        depth_map = np.asarray(getattr(artifacts, "depth_map", np.ones((1, 1))), dtype=np.float32)
        context = build_planner_context(
            phase=self._state.phase,
            frame_index=self._state.frame_index,
            goal_description=self._scene_spec.goal_description,
            detections=list(getattr(artifacts, "detections", []) or []),
            scene_graph_snapshot=getattr(artifacts, "scene_graph_snapshot", None),
            reconstruction_summary={
                "mesh_vertices": int(reconstruction_brief.mesh_vertex_count),
                "mesh_triangles": int(reconstruction_brief.mesh_triangle_count),
                "tracked_points": int(reconstruction_brief.mesh_vertex_count),
                "mesh_growth_delta": int(reconstruction_brief.mesh_growth_delta),
                "tracked_feature_count": int(reconstruction_brief.tracked_feature_count),
                "median_reprojection_error": reconstruction_brief.median_reprojection_error,
                "initializing_streak_frames": int(observation_state.get("initializing_streak_frames", 0)),
            },
            depth_summary={
                "min_depth_m": float(np.nanmin(depth_map)),
                "median_depth_m": float(np.nanmedian(depth_map)),
            },
            recent_actions=tuple(self._action_history[-8:]),
            recent_searches=tuple(self._completed_search_history),
            target_label=str(self._scene_spec.semantic_target_class),
            calibration_status=(
                "converged" if self._state.selfcal_step_index >= len(self._selfcal_actions) else "pending"
            ),
            tracking_status=str(getattr(artifacts, "slam_tracking_state", "UNKNOWN")),
            target_detection=observation_state.get("target_detection"),
            segments=list(getattr(artifacts, "segments", []) or []),
            depth_map=depth_map,
            frame_shape=tuple(np.asarray(getattr(artifacts, "frame_bgr", np.empty((0, 0, 3)))).shape[:2]),
            camera_pose_world=np.asarray(getattr(artifacts, "camera_pose_world", np.empty((0, 0))), dtype=np.float32),
            camera_intrinsics=observation_state.get("camera_intrinsics"),
            current_camera_state=self._current_camera_state,
            navigation_affordances=observation_state.get("navigation_affordances"),
            reconstruction_brief=reconstruction_brief,
            recent_action_effects=tuple(self._recent_action_effects),
            allowed_command_kinds=allowed_command_kinds,
            max_commands_per_schedule=max_commands_per_schedule,
            execution_capabilities={
                "move_speed_mps": float(self._rig_capabilities.move_speed_mps),
                "turn_speed_deg_per_sec": float(self._rig_capabilities.turn_speed_deg_per_sec),
                "camera_yaw_speed_deg_per_sec": float(self._rig_capabilities.camera_yaw_speed_deg_per_sec),
                "camera_pitch_speed_deg_per_sec": float(self._rig_capabilities.camera_pitch_speed_deg_per_sec),
                "camera_yaw_limit_deg": float(self._rig_capabilities.camera_yaw_limit_deg),
                "camera_pitch_limit_deg": float(self._rig_capabilities.camera_pitch_limit_deg),
                "max_translate_distance_m": 0.72,
                "max_rotate_body_deg": 36.0,
            },
            microstep_limits=dict(_DEFAULT_MICROSTEP_LIMITS),
        )
        try:
            try:
                schedule = self._planner.plan(
                    context=context,
                    frame_bgr=np.asarray(getattr(artifacts, "frame_bgr", np.empty((0, 0, 3))), dtype=np.uint8),
                )
            except TypeError:
                schedule = self._planner.plan(context=context)
        except Exception as exc:
            schedule = ActionSchedule(
                commands=(MotionCommand(kind=CommandKind.PAUSE, duration_sec=0.5, intent="planner fallback"),),
                rationale=f"planner unavailable; pause and reassess ({exc})",
                model="planner_error",
                issued_at_frame=self._state.frame_index,
                situation_summary=f"Planner unavailable: {exc}",
                behavior_mode="scan",
                safety_flags=PlannerSafetyFlags(
                    front_blocked=False,
                    dead_end_risk=0.0,
                    tracking_risk="unknown",
                    replan_reason="planner_error",
                ),
                confidence=0.0,
            )
        self._latest_planner_context = context
        schedule = self._validated_schedule(schedule=schedule, observation_state=observation_state, context=context)
        truncated_commands = tuple(schedule.commands[:max_commands_per_schedule])
        if truncated_commands != schedule.commands:
            schedule = replace(schedule, commands=truncated_commands)
        if schedule.goal_completion is None:
            visual_goal_completion = self._visual_goal_completion_assessment(artifacts=artifacts)
            if visual_goal_completion is not None:
                schedule = replace(schedule, goal_completion=visual_goal_completion)
        if self._schedule_reports_goal_reached(schedule):
            self._latest_planner_schedule = schedule
            self._record_planner_turn(context=context, schedule=schedule)
            self._mark_goal_completed()
            return
        if not schedule.commands:
            schedule = ActionSchedule(
                commands=(MotionCommand(kind=CommandKind.PAUSE, duration_sec=0.5, intent="empty schedule fallback"),),
                rationale="planner returned no safe actions; pause and reassess",
                model=getattr(self._planner, "model", "fallback"),
                issued_at_frame=self._state.frame_index,
                situation_summary="Planner returned no safe actions.",
                behavior_mode="scan",
                goal_hypothesis=context.goal_estimate,
                goal_completion=schedule.goal_completion,
                safety_flags=PlannerSafetyFlags(
                    front_blocked=bool(
                        context.perception.navigation_affordances is not None
                        and context.perception.navigation_affordances.front_blocked
                    ),
                    dead_end_risk=float(
                        0.0
                        if context.perception.navigation_affordances is None
                        else context.perception.navigation_affordances.dead_end_likelihood
                    ),
                    tracking_risk="high" if context.perception.tracking_status != "TRACKING" else "low",
                    replan_reason="empty_schedule",
                ),
                confidence=0.0,
            )
        self._latest_planner_schedule = schedule
        self._current_schedule = schedule
        self._active_schedule_start_frame = int(self._state.frame_index)
        self._active_schedule_start_observation = observation_state
        self._active_schedule_executed_commands = []
        self._active_macro_execution = None
        self._state.schedule_cursor = 0
        self._state.phase = EpisodePhase.EXECUTING_SCHEDULE
        self._record_planner_turn(context=context, schedule=schedule)

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
            prompt_payload = json.loads(planner_prompt_from_context(prompt))
            lines.append(
                json.dumps(
                    {
                        "frame_index": item["frame_index"],
                        "phase": item["phase"],
                        "prompt": prompt_payload,
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

    def is_waiting_for_frame(self) -> bool:
        waiting_getter = getattr(self._sensor_backend, "is_waiting_for_frame", None)
        return bool(callable(waiting_getter) and waiting_getter())

    def _promote_pending_sensor_frame(self, *, timeout_sec: float | None) -> None:
        poll_action_frame = getattr(self._sensor_backend, "poll_action_frame", None)
        if not callable(poll_action_frame):
            return
        sensor_frame = poll_action_frame(timeout_sec=timeout_sec)
        if sensor_frame is None:
            return
        self._commit_sensor_frame(sensor_frame)

    def _commit_sensor_frame(self, sensor_frame: SensorFrame) -> None:
        self._latest_sensor_frame = sensor_frame
        self._rig_capabilities = rig_capabilities_from_metadata(sensor_frame.metadata) or self._rig_capabilities
        self._state.frame_index += 1
        self._state.elapsed_sec += 1.0 / max(self._camera_rig.fps, 1e-6)
        self._write_selfcalibration_artifact()
        self._write_planner_turns_artifact()
        self._write_episode_report()


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
    validate_unity_vendor_setup()
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
