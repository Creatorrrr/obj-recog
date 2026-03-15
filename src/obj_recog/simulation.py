from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.sim_planner import OpenAILivingRoomPlanner, build_planner_context, planner_prompt_from_context
from obj_recog.sim_protocol import (
    ActionPrimitive,
    ActionSchedule,
    ActionStep,
    EpisodePhase,
    EpisodeReport,
    HiddenWorldState,
    PlannerActionEffectSummary,
    PlannerNavigationAffordances,
    PlannerNavigationSectorObservation,
    PlannerReconstructionBrief,
    PlannerSafetyFlags,
    PlannerSearchOutcome,
    RobotPose,
    SensorFrame,
    UnityActionCommand,
)
from obj_recog.types import PanopticSegment
from obj_recog.sim_scene import build_living_room_scene_spec
from obj_recog.unity_rgb import UnityRgbClient, command_from_step
from obj_recog.unity_vendor_check import validate_unity_vendor_setup


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
        self._action_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unity-rgb")
        self._pending_frame_future: Future[SensorFrame] | None = None

    def reset_episode(self, *, scene_spec) -> SensorFrame:
        self._clear_pending_frame_future()
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

    def submit_action(self, command: UnityActionCommand) -> None:
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

    def _request_sensor_frame(self, command: UnityActionCommand, frame_index: int) -> SensorFrame:
        frame = self._client.apply_action(command)
        return SensorFrame(
            frame_index=int(frame_index),
            timestamp_sec=float(frame.timestamp_sec),
            frame_bgr=np.asarray(frame.frame_bgr, dtype=np.uint8),
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
        self._completed_search_history: list[PlannerSearchOutcome] = []
        self._recent_action_effects: list[PlannerActionEffectSummary] = []
        self._current_schedule: ActionSchedule | None = None
        self._active_schedule_start_frame: int | None = None
        self._active_schedule_start_observation: dict[str, object] | None = None
        self._active_schedule_executed_steps: list[str] = []
        self._previous_observation_state: dict[str, object] | None = None
        self._pending_action_step: ActionStep | None = None
        self._latest_planner_context = None
        self._latest_planner_schedule: ActionSchedule | None = None
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
    def _format_action_step(step: ActionStep) -> str:
        return f"{step.primitive.value}:{step.value}"

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
    def _direction_from_action_text(action_text: str) -> str | None:
        text = str(action_text or "")
        if "strafe_left" in text or "turn_left" in text or "camera_pan_left" in text:
            return "left"
        if "strafe_right" in text or "turn_right" in text or "camera_pan_right" in text:
            return "right"
        if "move_backward" in text:
            return "rear"
        if "move_forward" in text:
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
            if clearance >= 1.15
        )
        front_blocked = forward_clearance_m is not None and forward_clearance_m < 0.8
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
        }

    def _replan_frame_budget(self) -> int:
        interval_sec = float(getattr(self._config, "sim_replan_interval_sec", 1.0))
        fps = float(max(self._camera_rig.fps, 1e-6))
        return max(1, int(math.ceil(interval_sec * fps)))

    def _target_detection_summary(self, *, artifacts) -> dict[str, float | str | None] | None:
        target_label = getattr(self._scene_spec, "semantic_target_class", "")
        detections = list(getattr(artifacts, "detections", []) or [])
        depth_map = np.asarray(getattr(artifacts, "depth_map", np.empty((0, 0))), dtype=np.float32)
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
            summary = {
                "label": str(getattr(detection, "label", "")),
                "confidence": confidence,
                "area_ratio": area_ratio,
                "center_offset_ratio": center_offset_ratio,
                "horizontal_position": horizontal_position,
                "median_depth_m": self._bbox_depth_m(depth_map, (x1, y1, x2, y2)),
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

    def _likely_blocked(
        self,
        *,
        step: ActionStep,
        previous_state: dict[str, object],
        current_state: dict[str, object],
    ) -> bool:
        translation_progress = self._pose_progress_m(
            previous_state.get("camera_pose_world"),
            current_state.get("camera_pose_world"),
        )
        yaw_delta_deg = self._angle_delta_deg(
            previous_state.get("yaw_deg"),
            current_state.get("yaw_deg"),
        )
        previous_affordances = previous_state.get("navigation_affordances")
        current_affordances = current_state.get("navigation_affordances")
        if step.primitive in {
            ActionPrimitive.MOVE_FORWARD,
            ActionPrimitive.MOVE_BACKWARD,
            ActionPrimitive.STRAFE_LEFT,
            ActionPrimitive.STRAFE_RIGHT,
        }:
            previous_forward = (
                None
                if not isinstance(previous_affordances, PlannerNavigationAffordances)
                else previous_affordances.forward_clearance_m
            )
            current_forward = (
                None
                if not isinstance(current_affordances, PlannerNavigationAffordances)
                else current_affordances.forward_clearance_m
            )
            if translation_progress is not None and translation_progress < 0.025:
                if previous_forward is None or previous_forward < 1.0:
                    return True
                if current_forward is not None and current_forward <= previous_forward + 0.05:
                    return True
        if step.primitive in {ActionPrimitive.TURN_LEFT, ActionPrimitive.TURN_RIGHT}:
            return yaw_delta_deg is not None and abs(float(yaw_delta_deg)) < 2.0
        return False

    def _record_recent_action_effect(self, *, current_state: dict[str, object]) -> None:
        if self._pending_action_step is None or self._previous_observation_state is None:
            return
        previous_state = self._previous_observation_state
        translation_progress = self._pose_progress_m(
            previous_state.get("camera_pose_world"),
            current_state.get("camera_pose_world"),
        )
        yaw_delta_deg = self._angle_delta_deg(
            previous_state.get("yaw_deg"),
            current_state.get("yaw_deg"),
        )
        estimated_motion_parts = []
        if translation_progress is not None:
            estimated_motion_parts.append(f"translation={translation_progress:.2f}m")
        if yaw_delta_deg is not None:
            estimated_motion_parts.append(f"yaw={yaw_delta_deg:+.1f}deg")
        if not estimated_motion_parts:
            estimated_motion_parts.append("pose_unavailable")
        effect = PlannerActionEffectSummary(
            action=self._format_action_step(self._pending_action_step),
            estimated_motion=", ".join(estimated_motion_parts),
            clearance_change=self._clearance_change_text(
                previous_state.get("navigation_affordances"),
                current_state.get("navigation_affordances"),
            ),
            target_evidence_change=self._target_evidence_change_text(
                previous_state.get("target_detection"),
                current_state.get("target_detection"),
            ),
            likely_blocked=self._likely_blocked(
                step=self._pending_action_step,
                previous_state=previous_state,
                current_state=current_state,
            ),
        )
        self._recent_action_effects.append(effect)
        self._recent_action_effects = self._recent_action_effects[-4:]
        self._pending_action_step = None

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
                executed_actions=tuple(self._active_schedule_executed_steps),
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
        self._active_schedule_executed_steps = []

    def next_frame(self, *, timeout_sec: float | None = 1.0) -> FramePacket | None:
        if self._latest_sensor_frame is None and self.is_waiting_for_frame():
            self._promote_pending_sensor_frame(timeout_sec=timeout_sec)
        _ = timeout_sec
        if self._closed or self._latest_sensor_frame is None:
            return None
        sensor_frame = self._latest_sensor_frame
        return FramePacket(
            frame_bgr=np.asarray(sensor_frame.frame_bgr, dtype=np.uint8),
            timestamp_sec=float(sensor_frame.timestamp_sec),
            planner_context=self._latest_planner_context,
            planner_schedule=self._latest_planner_schedule,
            calibration_source="self_calibration/converged"
            if self._state.selfcal_step_index >= len(self._selfcal_actions)
            else "self_calibration/pending",
        )

    def record_runtime_observation(self, *, frame_packet: FramePacket, artifacts) -> None:
        _ = frame_packet
        if self._closed or self._latest_sensor_frame is None:
            return
        current_state = self._observation_state_from_artifacts(artifacts=artifacts)
        self._record_recent_action_effect(current_state=current_state)
        if self._goal_visually_reached(artifacts=artifacts):
            self._previous_observation_state = current_state
            self._mark_goal_completed()
            return

        next_command = self._select_next_command(artifacts=artifacts)
        self._previous_observation_state = current_state
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
        self._latest_sensor_frame = None
        self._write_episode_report()

    @staticmethod
    def _contains_contradictory_steps(steps: tuple[ActionStep, ...]) -> bool:
        primitives = {step.primitive for step in steps}
        contradictory_pairs = (
            {ActionPrimitive.TURN_LEFT, ActionPrimitive.TURN_RIGHT},
            {ActionPrimitive.CAMERA_PAN_LEFT, ActionPrimitive.CAMERA_PAN_RIGHT},
            {ActionPrimitive.STRAFE_LEFT, ActionPrimitive.STRAFE_RIGHT},
        )
        return any(pair.issubset(primitives) for pair in contradictory_pairs)

    @staticmethod
    def _tracking_risk(tracking_status: str) -> str:
        normalized = str(tracking_status or "").upper()
        if normalized in {"TRACKING", "RELOCALIZED"}:
            return "low"
        if normalized in {"LOST", "UNKNOWN"}:
            return "high"
        return "medium"

    def _escape_steps(self, *, observation_state: dict[str, object]) -> tuple[ActionStep, ...]:
        affordances = observation_state.get("navigation_affordances")
        best_direction = None
        if isinstance(affordances, PlannerNavigationAffordances):
            best_direction = affordances.best_exploration_direction
        translation_step = (
            ActionStep(ActionPrimitive.STRAFE_LEFT, 0.12, intent="escape toward left opening")
            if best_direction == "left"
            else ActionStep(ActionPrimitive.STRAFE_RIGHT, 0.12, intent="escape toward right opening")
            if best_direction == "right"
            else ActionStep(ActionPrimitive.MOVE_BACKWARD, 0.12, intent="back out of blocked state")
        )
        turn_step = (
            ActionStep(ActionPrimitive.TURN_LEFT, 6.0, intent="reorient toward safer view")
            if best_direction != "right"
            else ActionStep(ActionPrimitive.TURN_RIGHT, 6.0, intent="reorient toward safer view")
        )
        pan_step = (
            ActionStep(ActionPrimitive.CAMERA_PAN_LEFT, 6.0, intent="confirm escape corridor")
            if best_direction != "right"
            else ActionStep(ActionPrimitive.CAMERA_PAN_RIGHT, 6.0, intent="confirm escape corridor")
        )
        return (translation_step, turn_step, pan_step)

    def _validated_schedule(
        self,
        *,
        schedule: ActionSchedule,
        observation_state: dict[str, object],
        context,
    ) -> ActionSchedule:
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
        if self._contains_contradictory_steps(tuple(schedule.steps)):
            return ActionSchedule(
                steps=self._escape_steps(observation_state=observation_state),
                rationale="planner returned contradictory actions; use escape fallback",
                model=f"{schedule.model}/validated",
                issued_at_frame=schedule.issued_at_frame,
                situation_summary=schedule.situation_summary or "Contradictory actions rejected; executing escape fallback.",
                behavior_mode="escape",
                goal_hypothesis=schedule.goal_hypothesis or context.goal_estimate,
                safety_flags=PlannerSafetyFlags(
                    front_blocked=front_blocked,
                    dead_end_risk=dead_end_risk,
                    tracking_risk=tracking_risk,
                    replan_reason="contradictory_actions",
                ),
                confidence=0.0 if schedule.confidence is None else float(schedule.confidence),
            )
        if tracking_risk == "high" and any(
            step.primitive in {ActionPrimitive.MOVE_FORWARD, ActionPrimitive.STRAFE_LEFT, ActionPrimitive.STRAFE_RIGHT}
            for step in schedule.steps
        ):
            return ActionSchedule(
                steps=(ActionStep(ActionPrimitive.PAUSE, 0.5, intent="tracking unstable; pause"),),
                rationale="tracking risk too high for aggressive translation",
                model=f"{schedule.model}/validated",
                issued_at_frame=schedule.issued_at_frame,
                situation_summary=schedule.situation_summary or "Tracking is unstable; pausing to reassess.",
                behavior_mode="scan",
                goal_hypothesis=schedule.goal_hypothesis or context.goal_estimate,
                safety_flags=PlannerSafetyFlags(
                    front_blocked=front_blocked,
                    dead_end_risk=dead_end_risk,
                    tracking_risk=tracking_risk,
                    replan_reason="tracking_risk",
                ),
                confidence=0.0 if schedule.confidence is None else float(schedule.confidence),
            )
        if schedule.safety_flags is None:
            schedule = ActionSchedule(
                steps=tuple(schedule.steps),
                rationale=schedule.rationale,
                model=schedule.model,
                issued_at_frame=schedule.issued_at_frame,
                situation_summary=schedule.situation_summary,
                behavior_mode=schedule.behavior_mode,
                goal_hypothesis=schedule.goal_hypothesis or context.goal_estimate,
                safety_flags=PlannerSafetyFlags(
                    front_blocked=front_blocked,
                    dead_end_risk=dead_end_risk,
                    tracking_risk=tracking_risk,
                    replan_reason="model_response",
                ),
                confidence=schedule.confidence,
            )
        return schedule

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
        self._action_history.append(self._format_action_step(step))
        self._pending_action_step = step
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
        self._finalize_completed_search(artifacts=artifacts)
        observation_state = self._observation_state_from_artifacts(artifacts=artifacts)
        reconstruction_brief = self._reconstruction_brief(observation_state=observation_state)
        batch_step_limit = max(1, int(getattr(self._config, "sim_action_batch_size", 1)))
        replan_frame_budget = self._replan_frame_budget()
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
            navigation_affordances=observation_state.get("navigation_affordances"),
            reconstruction_brief=reconstruction_brief,
            recent_action_effects=tuple(self._recent_action_effects),
            batch_step_limit=batch_step_limit,
            replan_frame_budget=replan_frame_budget,
            tracking_safe_limits={
                primitive.value: self._tracking_safe_step_limit(primitive)
                for primitive in ActionPrimitive
                if self._tracking_safe_step_limit(primitive) is not None
            },
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
                steps=(ActionStep(ActionPrimitive.PAUSE, 0.5),),
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
        truncated_steps = tuple(schedule.steps[:batch_step_limit])
        if truncated_steps != schedule.steps:
            schedule = ActionSchedule(
                steps=truncated_steps,
                rationale=schedule.rationale,
                model=schedule.model,
                issued_at_frame=schedule.issued_at_frame,
                situation_summary=schedule.situation_summary,
                behavior_mode=schedule.behavior_mode,
                goal_hypothesis=schedule.goal_hypothesis,
                safety_flags=schedule.safety_flags,
                confidence=schedule.confidence,
            )
        if not schedule.steps:
            schedule = ActionSchedule(
                steps=(ActionStep(ActionPrimitive.PAUSE, 0.5),),
                rationale="planner returned no safe actions; pause and reassess",
                model=getattr(self._planner, "model", "fallback"),
                issued_at_frame=self._state.frame_index,
                situation_summary="Planner returned no safe actions.",
                behavior_mode="scan",
                goal_hypothesis=context.goal_estimate,
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
        self._active_schedule_executed_steps = []
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
                    "safety_flags": None
                    if schedule.safety_flags is None
                    else {
                        "front_blocked": bool(schedule.safety_flags.front_blocked),
                        "dead_end_risk": float(schedule.safety_flags.dead_end_risk),
                        "tracking_risk": schedule.safety_flags.tracking_risk,
                        "replan_reason": schedule.safety_flags.replan_reason,
                    },
                    "confidence": schedule.confidence,
                    "steps": [
                        {"primitive": step.primitive.value, "value": step.value, "intent": step.intent}
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
        self._active_schedule_executed_steps.append(self._format_action_step(step))
        self._executed_schedule_steps += 1
        max_schedule_steps = max(1, int(getattr(self._config, "sim_action_batch_size", 1)))
        replan_frame_budget = self._replan_frame_budget()
        if self._current_schedule is None or self._state.schedule_cursor >= len(self._current_schedule.steps):
            self._current_schedule = None
            self._executed_schedule_steps = 0
            self._state.phase = EpisodePhase.REASSESS
        elif self._executed_schedule_steps >= max_schedule_steps:
            self._current_schedule = None
            self._state.schedule_cursor = 0
            self._executed_schedule_steps = 0
            self._state.phase = EpisodePhase.REASSESS
        elif self._executed_schedule_steps >= replan_frame_budget:
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
        updated_steps[cursor] = ActionStep(step.primitive, residual_value, intent=step.intent)
        self._current_schedule = ActionSchedule(
            steps=tuple(updated_steps),
            rationale=self._current_schedule.rationale,
            model=self._current_schedule.model,
            issued_at_frame=self._current_schedule.issued_at_frame,
            situation_summary=self._current_schedule.situation_summary,
            behavior_mode=self._current_schedule.behavior_mode,
            goal_hypothesis=self._current_schedule.goal_hypothesis,
            safety_flags=self._current_schedule.safety_flags,
            confidence=self._current_schedule.confidence,
        )
        return ActionStep(step.primitive, chunk_value, intent=step.intent)

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
