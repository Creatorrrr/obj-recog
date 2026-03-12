from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from obj_recog.auto_calibration import create_approximate_calibration, refine_focal_lengths
from obj_recog.config import AppConfig, SIM_SCENARIO_CHOICES
from obj_recog.frame_source import FramePacket
from obj_recog.opencv_runtime import load_cv2
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.slam_bridge import KeyframeObservation, TRACKING_OK_STATES
from obj_recog.types import Detection, FrameArtifacts


_TARGET_LABEL = "target"
_CAMERA_HEIGHT_METERS = 1.4
_ROOM_WIDTH_METERS = 6.0
_ROOM_DEPTH_METERS = 8.0
_ROOM_HEIGHT_METERS = 3.0
_GOAL_SELECTOR_SYSTEM_INSTRUCTIONS = (
    "You are a camera navigation goal selector. "
    "Return JSON only with keys target_label, desired_bearing, desired_distance_band, reason, confidence. "
    "desired_bearing must be a number in radians relative to the current camera forward direction. "
    "desired_distance_band must be a two-item array [near, far] in meters. "
    "Do not return coordinates, speed commands, or extra keys."
)
_GOAL_SELECTOR_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "name": "navigation_goal",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_label": {"type": "string"},
            "desired_bearing": {"type": "number"},
            "desired_distance_band": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": [
            "target_label",
            "desired_bearing",
            "desired_distance_band",
            "reason",
            "confidence",
        ],
    },
}


def _response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(response, "output", None) or []
    chunks: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
    return "\n".join(chunks).strip()


@dataclass(frozen=True, slots=True)
class NavigationGoal:
    target_label: str
    desired_bearing: float
    desired_distance_band: tuple[float, float]
    reason: str
    confidence: float
    source: str = "heuristic"


@dataclass(frozen=True, slots=True)
class CameraRigSpec:
    image_width: int
    image_height: int
    fps: float
    horizontal_fov_deg: float
    near_plane_m: float
    far_plane_m: float
    enable_distortion: bool
    depth_noise_std: float
    motion_blur: float
    yaw_rate_limit_deg: float
    linear_velocity_limit_mps: float

    @classmethod
    def from_config(cls, config: AppConfig) -> CameraRigSpec:
        return cls(
            image_width=int(config.width),
            image_height=int(config.height),
            fps=float(config.sim_camera_fps),
            horizontal_fov_deg=float(config.sim_camera_fov_deg),
            near_plane_m=float(config.sim_camera_near),
            far_plane_m=float(config.sim_camera_far),
            enable_distortion=bool(config.sim_enable_distortion),
            depth_noise_std=max(0.0, float(config.sim_depth_noise_std)),
            motion_blur=float(np.clip(config.sim_motion_blur, 0.0, 0.95)),
            yaw_rate_limit_deg=float(config.sim_yaw_rate_limit_deg),
            linear_velocity_limit_mps=float(config.sim_linear_velocity_limit),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    room_width_m: float
    room_depth_m: float
    room_height_m: float
    start_x: float = 0.0
    start_z: float = 0.0
    start_yaw_deg: float = -25.0
    landmark_points: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class StaticObjectSpec:
    label: str
    center_world: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    color_bgr: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class DynamicActorSpec:
    actor_id: str
    actor_type: str
    label: str
    size_xyz: tuple[float, float, float]
    color_bgr: tuple[int, int, int]
    base_center_world: tuple[float, float, float]
    waypoints_world: tuple[tuple[float, float, float], ...]
    loop_duration_sec: float
    start_time_sec: float = 0.0
    phase_offset_sec: float = 0.0


@dataclass(frozen=True, slots=True)
class MissionSpec:
    target_label: str = _TARGET_LABEL
    verify_required_streak: int = 3
    center_tolerance_x_ratio: float = 0.12
    center_tolerance_y_ratio: float = 0.3
    min_area_ratio: float = 0.03
    depth_band_m: tuple[float, float] = (3.2, 5.0)
    eval_budget_sec_override: float | None = None


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    scene_id: str
    scenario_family: str
    difficulty_level: int
    environment: EnvironmentSpec
    static_objects: tuple[StaticObjectSpec, ...]
    dynamic_actors: tuple[DynamicActorSpec, ...]
    mission: MissionSpec


@dataclass(frozen=True, slots=True)
class SimulationScenarioState:
    scene_id: str
    difficulty_level: int
    phase: str
    step_index: int
    elapsed_sec: float
    selfcal_converged: bool
    rig_x: float
    rig_z: float
    yaw_deg: float
    visible_labels: tuple[str, ...]
    active_goal: NavigationGoal | None
    target_motion_state: str
    render_backend: str


@dataclass(frozen=True, slots=True)
class _SceneObject:
    label: str
    center_world: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    color_bgr: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _VisibleObject:
    label: str
    bbox: tuple[int, int, int, int]
    depth_m: float
    bearing_rad: float
    area_pixels: int
    color_bgr: tuple[int, int, int]


@dataclass(slots=True)
class _RuntimeObservation:
    timestamp_sec: float | None
    visible_objects: list[_VisibleObject]
    tracking_state: str
    pose_error_m: float | None


@dataclass(frozen=True, slots=True)
class _RenderOutput:
    frame_bgr: np.ndarray
    depth_map: np.ndarray
    visible_objects: list[_VisibleObject]
    backend: str


def _static_object(
    label: str,
    center_world: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    color_bgr: tuple[int, int, int],
) -> StaticObjectSpec:
    return StaticObjectSpec(
        label=label,
        center_world=center_world,
        size_xyz=size_xyz,
        color_bgr=color_bgr,
    )


def _dynamic_actor(
    actor_id: str,
    actor_type: str,
    label: str,
    base_center_world: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    color_bgr: tuple[int, int, int],
    waypoints_world: tuple[tuple[float, float, float], ...],
    loop_duration_sec: float,
    *,
    start_time_sec: float = 0.0,
    phase_offset_sec: float = 0.0,
) -> DynamicActorSpec:
    return DynamicActorSpec(
        actor_id=actor_id,
        actor_type=actor_type,
        label=label,
        base_center_world=base_center_world,
        size_xyz=size_xyz,
        color_bgr=color_bgr,
        waypoints_world=waypoints_world,
        loop_duration_sec=loop_duration_sec,
        start_time_sec=start_time_sec,
        phase_offset_sec=phase_offset_sec,
    )


SCENARIO_SPECS: dict[str, ScenarioSpec] = {
    "studio_open_v1": ScenarioSpec(
        scene_id="studio_open_v1",
        scenario_family="studio",
        difficulty_level=1,
        environment=EnvironmentSpec(
            room_width_m=6.0,
            room_depth_m=8.0,
            room_height_m=3.0,
        ),
        static_objects=(
            _static_object("table", (0.0, 0.45, 3.1), (1.5, 0.9, 0.8), (95, 145, 190)),
            _static_object("chair", (-1.1, 0.55, 4.4), (0.7, 1.1, 0.7), (60, 180, 120)),
            _static_object("plant", (1.8, 0.7, 4.8), (0.6, 1.4, 0.6), (55, 155, 85)),
            _static_object(_TARGET_LABEL, (1.5, 0.45, 5.4), (0.8, 0.9, 0.8), (40, 80, 225)),
        ),
        dynamic_actors=(),
        mission=MissionSpec(),
    ),
    "office_clutter_v1": ScenarioSpec(
        scene_id="office_clutter_v1",
        scenario_family="office",
        difficulty_level=2,
        environment=EnvironmentSpec(
            room_width_m=6.5,
            room_depth_m=8.5,
            room_height_m=3.0,
        ),
        static_objects=(
            _static_object("desk", (-0.8, 0.45, 3.2), (1.4, 0.9, 0.75), (90, 150, 190)),
            _static_object("desk", (1.0, 0.45, 4.0), (1.5, 0.9, 0.8), (90, 150, 190)),
            _static_object("chair", (-1.7, 0.55, 4.5), (0.7, 1.1, 0.7), (60, 180, 120)),
            _static_object("cabinet", (0.7, 0.85, 4.8), (0.7, 1.7, 0.6), (120, 120, 145)),
            _static_object("box", (-0.2, 0.35, 5.0), (0.7, 0.7, 0.7), (150, 120, 90)),
            _static_object(_TARGET_LABEL, (1.7, 0.45, 5.7), (0.8, 0.9, 0.8), (40, 80, 225)),
        ),
        dynamic_actors=(),
        mission=MissionSpec(),
    ),
    "lab_corridor_v1": ScenarioSpec(
        scene_id="lab_corridor_v1",
        scenario_family="lab",
        difficulty_level=3,
        environment=EnvironmentSpec(
            room_width_m=5.5,
            room_depth_m=9.5,
            room_height_m=3.0,
        ),
        static_objects=(
            _static_object("partition", (-1.2, 1.1, 4.6), (0.5, 2.2, 2.0), (110, 110, 120)),
            _static_object("partition", (0.8, 1.1, 5.1), (0.5, 2.2, 2.3), (110, 110, 120)),
            _static_object("cart", (-0.2, 0.55, 3.8), (0.9, 1.1, 0.7), (65, 160, 180)),
            _static_object("pillar", (1.9, 1.0, 5.8), (0.45, 2.0, 0.45), (125, 125, 135)),
            _static_object(_TARGET_LABEL, (2.1, 0.45, 6.6), (0.8, 0.9, 0.8), (40, 80, 225)),
        ),
        dynamic_actors=(),
        mission=MissionSpec(),
    ),
    "showroom_occlusion_v1": ScenarioSpec(
        scene_id="showroom_occlusion_v1",
        scenario_family="showroom",
        difficulty_level=4,
        environment=EnvironmentSpec(
            room_width_m=6.8,
            room_depth_m=8.8,
            room_height_m=3.2,
        ),
        static_objects=(
            _static_object("display", (-1.6, 0.75, 4.9), (0.9, 1.5, 0.9), (55, 90, 205)),
            _static_object("display", (-0.3, 0.75, 5.4), (0.9, 1.5, 0.9), (70, 110, 210)),
            _static_object("display", (0.6, 0.75, 5.0), (0.9, 1.5, 0.9), (58, 95, 208)),
            _static_object("pedestal", (2.0, 0.5, 4.6), (0.7, 1.0, 0.7), (165, 150, 110)),
            _static_object(_TARGET_LABEL, (1.4, 0.45, 5.7), (0.85, 0.95, 0.85), (40, 80, 225)),
        ),
        dynamic_actors=(
            _dynamic_actor(
                "showroom-occluder",
                "occluder",
                "occluder",
                (0.5, 0.8, 5.15),
                (1.0, 1.6, 0.9),
                (180, 180, 190),
                ((0.1, 0.8, 5.0), (1.6, 0.8, 5.2)),
                4.0,
                start_time_sec=8.0,
                phase_offset_sec=0.8,
            ),
        ),
        mission=MissionSpec(),
    ),
    "office_crossflow_v1": ScenarioSpec(
        scene_id="office_crossflow_v1",
        scenario_family="office",
        difficulty_level=5,
        environment=EnvironmentSpec(
            room_width_m=7.0,
            room_depth_m=9.0,
            room_height_m=3.0,
        ),
        static_objects=(
            _static_object("desk", (-1.5, 0.45, 3.6), (1.3, 0.9, 0.8), (90, 150, 190)),
            _static_object("desk", (1.2, 0.45, 4.2), (1.4, 0.9, 0.8), (90, 150, 190)),
            _static_object("doorframe", (0.0, 1.1, 5.0), (0.8, 2.2, 0.35), (135, 125, 110)),
            _static_object("cabinet", (2.1, 0.85, 6.0), (0.8, 1.7, 0.6), (120, 120, 145)),
            _static_object(_TARGET_LABEL, (1.7, 0.45, 6.3), (0.8, 0.9, 0.8), (40, 80, 225)),
        ),
        dynamic_actors=(
            _dynamic_actor(
                "crossflow-left",
                "distractor",
                "distractor",
                (-1.8, 0.6, 4.9),
                (0.7, 1.2, 0.7),
                (80, 190, 225),
                ((-2.1, 0.6, 4.7), (0.6, 0.6, 4.7)),
                5.5,
                start_time_sec=8.0,
            ),
            _dynamic_actor(
                "crossflow-right",
                "distractor",
                "distractor",
                (1.8, 0.6, 5.6),
                (0.75, 1.2, 0.75),
                (130, 185, 75),
                ((1.9, 0.6, 5.8), (-0.8, 0.6, 5.2)),
                6.0,
                start_time_sec=8.0,
                phase_offset_sec=1.2,
            ),
        ),
        mission=MissionSpec(),
    ),
    "warehouse_moving_target_v1": ScenarioSpec(
        scene_id="warehouse_moving_target_v1",
        scenario_family="warehouse",
        difficulty_level=6,
        environment=EnvironmentSpec(
            room_width_m=8.0,
            room_depth_m=10.0,
            room_height_m=3.5,
        ),
        static_objects=(
            _static_object("shelf", (-2.0, 1.0, 5.0), (0.8, 2.0, 2.8), (110, 130, 170)),
            _static_object("shelf", (2.3, 1.0, 5.6), (0.8, 2.0, 2.8), (110, 130, 170)),
            _static_object("pillar", (-0.2, 1.0, 6.0), (0.45, 2.0, 0.45), (135, 135, 145)),
            _static_object("box", (1.0, 0.4, 4.6), (0.8, 0.8, 0.8), (155, 120, 90)),
            _static_object("crate", (-1.1, 0.45, 6.9), (0.9, 0.9, 0.9), (145, 105, 82)),
        ),
        dynamic_actors=(
            _dynamic_actor(
                "warehouse-target",
                "moving_target",
                _TARGET_LABEL,
                (1.8, 0.45, 5.8),
                (0.8, 0.9, 0.8),
                (40, 80, 225),
                ((1.8, 0.45, 5.8), (0.7, 0.45, 6.4), (-0.4, 0.45, 5.8), (0.9, 0.45, 5.1)),
                6.5,
            ),
            _dynamic_actor(
                "warehouse-occluder",
                "occluder",
                "occluder",
                (0.2, 0.8, 5.1),
                (0.95, 1.6, 0.85),
                (180, 180, 190),
                ((-0.6, 0.8, 5.0), (1.4, 0.8, 5.2)),
                4.5,
                start_time_sec=8.0,
                phase_offset_sec=0.4,
            ),
            _dynamic_actor(
                "warehouse-distractor-a",
                "distractor",
                "distractor",
                (-1.6, 0.55, 4.8),
                (0.75, 1.1, 0.75),
                (80, 190, 225),
                ((-1.8, 0.55, 4.5), (0.4, 0.55, 4.8)),
                5.0,
                start_time_sec=8.0,
            ),
            _dynamic_actor(
                "warehouse-distractor-b",
                "distractor",
                "distractor",
                (1.9, 0.55, 6.6),
                (0.75, 1.1, 0.75),
                (130, 185, 75),
                ((2.1, 0.55, 6.7), (-0.5, 0.55, 6.0)),
                6.0,
                start_time_sec=8.0,
                phase_offset_sec=1.0,
            ),
        ),
        mission=MissionSpec(),
    ),
}

if tuple(SCENARIO_SPECS) != tuple(SIM_SCENARIO_CHOICES):
    raise RuntimeError("scenario registry does not match configured CLI scenario choices")


class GoalSelector(Protocol):
    def select_goal(self, *, visible_objects: list[_VisibleObject], target_label: str) -> NavigationGoal | None:
        ...


class SceneRenderer(Protocol):
    backend_name: str

    def render(
        self,
        *,
        pose_world: np.ndarray,
        intrinsics: CameraIntrinsics,
        scene_objects: list[_SceneObject],
        previous_frame_bgr: np.ndarray | None,
    ) -> _RenderOutput:
        ...


def _pose_world_matrix(x: float, z: float, yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )
    pose[:3, 3] = np.array([x, _CAMERA_HEIGHT_METERS, z], dtype=np.float32)
    return pose


def _object_corners(item: _SceneObject) -> np.ndarray:
    center = np.asarray(item.center_world, dtype=np.float32)
    half = np.asarray(item.size_xyz, dtype=np.float32) * 0.5
    corners: list[np.ndarray] = []
    for dx in (-half[0], half[0]):
        for dy in (-half[1], half[1]):
            for dz in (-half[2], half[2]):
                corners.append(center + np.array([dx, dy, dz], dtype=np.float32))
    return np.asarray(corners, dtype=np.float32)


def _project_points(
    points_world: np.ndarray,
    *,
    pose_world: np.ndarray,
    intrinsics: CameraIntrinsics,
    near_plane_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(np.asarray(pose_world, dtype=np.float32))
    homogeneous = np.concatenate(
        (points_world.astype(np.float32), np.ones((points_world.shape[0], 1), dtype=np.float32)),
        axis=1,
    )
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    valid = camera[:, 2] > max(float(near_plane_m), 1e-3)
    projected = np.empty((points_world.shape[0], 2), dtype=np.float32)
    projected[:, 0] = intrinsics.fx * (camera[:, 0] / np.maximum(camera[:, 2], 1e-6)) + intrinsics.cx
    projected[:, 1] = intrinsics.fy * (-camera[:, 1] / np.maximum(camera[:, 2], 1e-6)) + intrinsics.cy
    return projected, valid


def _visible_object(
    item: _SceneObject,
    *,
    pose_world: np.ndarray,
    intrinsics: CameraIntrinsics,
    rig: CameraRigSpec,
) -> _VisibleObject | None:
    corners = _object_corners(item)
    projected, valid = _project_points(
        corners,
        pose_world=pose_world,
        intrinsics=intrinsics,
        near_plane_m=rig.near_plane_m,
    )
    if not np.any(valid):
        return None

    center = np.asarray(item.center_world, dtype=np.float32).reshape(1, 3)
    center_camera = (np.linalg.inv(pose_world) @ np.append(center[0], 1.0).astype(np.float32))[:3]
    if center_camera[2] <= max(float(rig.near_plane_m), 1e-3):
        return None

    valid_projected = projected[valid]
    width = rig.image_width
    height = rig.image_height
    x1 = max(0, int(math.floor(float(np.min(valid_projected[:, 0])))))
    y1 = max(0, int(math.floor(float(np.min(valid_projected[:, 1])))))
    x2 = min(width, int(math.ceil(float(np.max(valid_projected[:, 0])))))
    y2 = min(height, int(math.ceil(float(np.max(valid_projected[:, 1])))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    bearing = math.atan2(float(center_camera[0]), float(center_camera[2]))
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return _VisibleObject(
        label=item.label,
        bbox=(x1, y1, x2, y2),
        depth_m=float(center_camera[2]),
        bearing_rad=bearing,
        area_pixels=area,
        color_bgr=item.color_bgr,
    )


def _visible_objects_for_scene(
    *,
    scene_objects: list[_SceneObject],
    pose_world: np.ndarray,
    intrinsics: CameraIntrinsics,
    rig: CameraRigSpec,
) -> list[_VisibleObject]:
    candidates = [
        item
        for item in (_visible_object(obj, pose_world=pose_world, intrinsics=intrinsics, rig=rig) for obj in scene_objects)
        if item is not None
    ]
    return _filter_occluded_visible_objects(candidates)


def _bbox_intersection_area(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    x1 = max(int(left[0]), int(right[0]))
    y1 = max(int(left[1]), int(right[1]))
    x2 = min(int(left[2]), int(right[2]))
    y2 = min(int(left[3]), int(right[3]))
    return max(0, x2 - x1) * max(0, y2 - y1)


def _filter_occluded_visible_objects(visible_objects: list[_VisibleObject]) -> list[_VisibleObject]:
    if len(visible_objects) < 2:
        return sorted(visible_objects, key=lambda item: float(item.depth_m), reverse=True)

    kept_near_first: list[_VisibleObject] = []
    for candidate in sorted(visible_objects, key=lambda item: float(item.depth_m)):
        occluded = False
        for closer in kept_near_first:
            overlap_area = _bbox_intersection_area(candidate.bbox, closer.bbox)
            overlap_ratio = overlap_area / max(float(candidate.area_pixels), 1.0)
            if overlap_ratio >= 0.6 and float(closer.depth_m) + 0.15 < float(candidate.depth_m):
                occluded = True
                break
        if not occluded:
            kept_near_first.append(candidate)
    kept_near_first.sort(key=lambda item: float(item.depth_m), reverse=True)
    return kept_near_first


def _lerp_point(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    return (
        float(start[0] + ((end[0] - start[0]) * alpha)),
        float(start[1] + ((end[1] - start[1]) * alpha)),
        float(start[2] + ((end[2] - start[2]) * alpha)),
    )


def _resolve_waypoint_loop_position(actor: DynamicActorSpec, *, elapsed_sec: float) -> tuple[float, float, float]:
    if not actor.waypoints_world:
        return actor.base_center_world
    if len(actor.waypoints_world) == 1:
        return actor.waypoints_world[0]
    effective_elapsed = max(0.0, float(elapsed_sec) - float(actor.start_time_sec)) + float(actor.phase_offset_sec)
    loop_duration = max(float(actor.loop_duration_sec), 1e-6)
    segment_duration = loop_duration / float(len(actor.waypoints_world))
    phase_time = effective_elapsed % loop_duration
    segment_index = int(phase_time / segment_duration) % len(actor.waypoints_world)
    next_index = (segment_index + 1) % len(actor.waypoints_world)
    alpha = (phase_time - (segment_index * segment_duration)) / max(segment_duration, 1e-6)
    return _lerp_point(
        actor.waypoints_world[segment_index],
        actor.waypoints_world[next_index],
        float(alpha),
    )


def _scene_object_from_specs(
    *,
    label: str,
    center_world: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    color_bgr: tuple[int, int, int],
) -> _SceneObject:
    return _SceneObject(
        label=label,
        center_world=center_world,
        size_xyz=size_xyz,
        color_bgr=color_bgr,
    )


def _get_scenario_spec(scene_id: str) -> ScenarioSpec:
    if scene_id not in SCENARIO_SPECS:
        raise ValueError(f"unsupported scenario: {scene_id}")
    return SCENARIO_SPECS[scene_id]


class OpenAINavigationGoalSelector:
    def __init__(
        self,
        *,
        model: str,
        timeout_sec: float,
        api_key: str | None = None,
        client=None,
    ) -> None:
        self._model = model
        self._timeout_sec = float(timeout_sec)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = client
        self._client_disabled = False

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if self._client_disabled or not self._api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError:  # pragma: no cover - depends on local install.
            self._client_disabled = True
            return None
        try:
            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout_sec)
        except Exception:
            self._client_disabled = True
            return None
        return self._client

    def select_goal(self, *, visible_objects: list[_VisibleObject], target_label: str) -> NavigationGoal | None:
        client = self._ensure_client()
        if client is None:
            return None
        summary = {
            "target_label": target_label,
            "visible_objects": [
                {
                    "label": item.label,
                    "bearing_rad": round(float(item.bearing_rad), 4),
                    "depth_m": round(float(item.depth_m), 4),
                    "area_pixels": int(item.area_pixels),
                    "bbox": list(item.bbox),
                }
                for item in visible_objects[:8]
            ],
        }
        request_kwargs = dict(
            model=self._model,
            instructions=_GOAL_SELECTOR_SYSTEM_INSTRUCTIONS,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": json.dumps(summary, ensure_ascii=False)},
                    ],
                }
            ],
            max_output_tokens=180,
            timeout=self._timeout_sec,
            text={"format": dict(_GOAL_SELECTOR_RESPONSE_SCHEMA), "verbosity": "low"},
        )
        if str(self._model).startswith("gpt-5"):
            request_kwargs["reasoning"] = {"effort": "minimal"}

        try:
            response = client.responses.create(**request_kwargs)
            payload = json.loads(_response_text(response))
        except Exception:
            return None

        try:
            distance_band = tuple(float(value) for value in payload["desired_distance_band"])
            if len(distance_band) != 2:
                return None
            return NavigationGoal(
                target_label=str(payload["target_label"]),
                desired_bearing=float(payload["desired_bearing"]),
                desired_distance_band=(distance_band[0], distance_band[1]),
                reason=str(payload["reason"]),
                confidence=float(payload["confidence"]),
                source="llm",
            )
        except Exception:
            return None


class HeuristicGoalSelector:
    def select_goal(self, *, visible_objects: list[_VisibleObject], target_label: str) -> NavigationGoal | None:
        for item in visible_objects:
            if item.label != target_label:
                continue
            return NavigationGoal(
                target_label=target_label,
                desired_bearing=0.0,
                desired_distance_band=(3.1, 3.6),
                reason="Keep the target centered and close enough for verification.",
                confidence=0.98,
            )
        return None


class AnalyticSceneRenderer:
    backend_name = "analytic"

    def __init__(self, *, rig: CameraRigSpec, rng=None, environment: EnvironmentSpec | None = None) -> None:
        self._rig = rig
        self._rng = rng or np.random.default_rng(0)
        self._environment = environment

    def render(
        self,
        *,
        pose_world: np.ndarray,
        intrinsics: CameraIntrinsics,
        scene_objects: list[_SceneObject],
        previous_frame_bgr: np.ndarray | None,
    ) -> _RenderOutput:
        frame = np.zeros((self._rig.image_height, self._rig.image_width, 3), dtype=np.uint8)
        horizon = max(1, frame.shape[0] // 2)
        frame[:horizon, :, :] = np.array([118, 96, 78], dtype=np.uint8)
        frame[horizon:, :, :] = np.array([48, 42, 38], dtype=np.uint8)
        depth_map = np.full(frame.shape[:2], float(self._rig.far_plane_m), dtype=np.float32)

        visible_objects = _visible_objects_for_scene(
            scene_objects=scene_objects,
            pose_world=pose_world,
            intrinsics=intrinsics,
            rig=self._rig,
        )
        for item in visible_objects:
            x1, y1, x2, y2 = item.bbox
            frame[y1:y2, x1:x2] = np.asarray(item.color_bgr, dtype=np.uint8)
            frame[y1:y2, x1 : min(x1 + 2, x2)] = 255
            frame[y1:y2, max(x2 - 2, x1):x2] = 255
            frame[y1 : min(y1 + 2, y2), x1:x2] = 255
            frame[max(y2 - 2, y1):y2, x1:x2] = 255
            noise = self._rng.normal(0.0, float(self._rig.depth_noise_std), size=(y2 - y1, x2 - x1))
            depth_map[y1:y2, x1:x2] = np.clip(
                float(item.depth_m) + noise.astype(np.float32),
                float(self._rig.near_plane_m),
                float(self._rig.far_plane_m),
            )

        if previous_frame_bgr is not None and float(self._rig.motion_blur) > 0.0:
            alpha = float(self._rig.motion_blur)
            frame = np.clip(
                ((1.0 - alpha) * frame.astype(np.float32)) + (alpha * previous_frame_bgr.astype(np.float32)),
                0.0,
                255.0,
            ).astype(np.uint8)

        return _RenderOutput(
            frame_bgr=frame,
            depth_map=depth_map,
            visible_objects=visible_objects,
            backend=self.backend_name,
        )


class Open3DSceneRenderer:
    backend_name = "open3d"

    def __init__(
        self,
        *,
        rig: CameraRigSpec,
        environment: EnvironmentSpec | None = None,
        o3d_module=None,
    ) -> None:
        if o3d_module is None:
            import open3d as o3d
        else:
            o3d = o3d_module
        self._o3d = o3d
        self._rig = rig
        self._environment = environment or EnvironmentSpec(
            room_width_m=_ROOM_WIDTH_METERS,
            room_depth_m=_ROOM_DEPTH_METERS,
            room_height_m=_ROOM_HEIGHT_METERS,
        )
        rendering = getattr(getattr(o3d, "visualization", None), "rendering", None)
        offscreen_renderer = None if rendering is None else getattr(rendering, "OffscreenRenderer", None)
        if rendering is None or offscreen_renderer is None:
            raise RuntimeError("open3d rendering.OffscreenRenderer is required for open3d sim rendering")
        self._rendering = rendering
        self._renderer = offscreen_renderer(rig.image_width, rig.image_height)

    def render(
        self,
        *,
        pose_world: np.ndarray,
        intrinsics: CameraIntrinsics,
        scene_objects: list[_SceneObject],
        previous_frame_bgr: np.ndarray | None,
    ) -> _RenderOutput:
        scene = self._renderer.scene
        clear_geometry = getattr(scene, "clear_geometry", None)
        if callable(clear_geometry):
            clear_geometry()
        set_background = getattr(scene, "set_background", None)
        if callable(set_background):
            set_background(np.array([0.08, 0.08, 0.10, 1.0], dtype=np.float32))

        material = self._rendering.MaterialRecord()
        material.shader = "defaultLit"
        self._add_room_geometry(scene, material)
        for index, item in enumerate(scene_objects):
            mesh = self._create_box_mesh(item)
            color_bgr = np.asarray(item.color_bgr, dtype=np.float32) / 255.0
            mesh.paint_uniform_color(color_bgr[::-1].tolist())
            scene.add_geometry(f"obj-{index}-{item.label}", mesh, material)

        eye = np.asarray(pose_world[:3, 3], dtype=np.float32)
        forward = np.asarray(pose_world[:3, 2], dtype=np.float32)
        center = eye + forward * 3.0
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        vertical_fov_deg = math.degrees(
            2.0 * math.atan((self._rig.image_height * 0.5) / max(float(intrinsics.fy), 1e-6))
        )
        self._renderer.setup_camera(float(vertical_fov_deg), center, eye, up)

        color_image = np.asarray(self._renderer.render_to_image())
        if color_image.ndim == 2:
            color_image = np.repeat(color_image[:, :, None], 3, axis=2)
        if color_image.shape[-1] == 4:
            color_image = color_image[:, :, :3]
        frame = np.asarray(color_image[:, :, ::-1], dtype=np.uint8).copy()
        depth_image = np.asarray(self._renderer.render_to_depth_image(z_in_view_space=True), dtype=np.float32)
        depth_map = np.where(np.isfinite(depth_image), depth_image, float(self._rig.far_plane_m)).astype(np.float32)
        depth_map = np.clip(depth_map, float(self._rig.near_plane_m), float(self._rig.far_plane_m))
        if previous_frame_bgr is not None and float(self._rig.motion_blur) > 0.0:
            alpha = float(self._rig.motion_blur)
            frame = np.clip(
                ((1.0 - alpha) * frame.astype(np.float32)) + (alpha * previous_frame_bgr.astype(np.float32)),
                0.0,
                255.0,
            ).astype(np.uint8)

        visible_objects = _visible_objects_for_scene(
            scene_objects=scene_objects,
            pose_world=pose_world,
            intrinsics=intrinsics,
            rig=self._rig,
        )
        return _RenderOutput(
            frame_bgr=frame,
            depth_map=depth_map,
            visible_objects=visible_objects,
            backend=self.backend_name,
        )

    def _create_box_mesh(self, item: _SceneObject):
        mesh = self._o3d.geometry.TriangleMesh.create_box(
            width=float(item.size_xyz[0]),
            height=float(item.size_xyz[1]),
            depth=float(item.size_xyz[2]),
        )
        mesh.translate(
            np.array(
                [
                    float(item.center_world[0] - item.size_xyz[0] * 0.5),
                    float(item.center_world[1] - item.size_xyz[1] * 0.5),
                    float(item.center_world[2] - item.size_xyz[2] * 0.5),
                ],
                dtype=np.float64,
            )
        )
        compute_normals = getattr(mesh, "compute_vertex_normals", None)
        if callable(compute_normals):
            compute_normals()
        return mesh

    def _add_room_geometry(self, scene, material) -> None:
        environment = self._environment
        room_surfaces = [
            ("floor", (environment.room_width_m, 0.05, environment.room_depth_m), (0.0, -0.025, environment.room_depth_m * 0.5), (0.35, 0.35, 0.38)),
            ("back-wall", (environment.room_width_m, environment.room_height_m, 0.05), (0.0, environment.room_height_m * 0.5, environment.room_depth_m), (0.45, 0.42, 0.40)),
            ("left-wall", (0.05, environment.room_height_m, environment.room_depth_m), (-environment.room_width_m * 0.5, environment.room_height_m * 0.5, environment.room_depth_m * 0.5), (0.42, 0.40, 0.38)),
            ("right-wall", (0.05, environment.room_height_m, environment.room_depth_m), (environment.room_width_m * 0.5, environment.room_height_m * 0.5, environment.room_depth_m * 0.5), (0.42, 0.40, 0.38)),
        ]
        for name, size, center, color in room_surfaces:
            mesh = self._o3d.geometry.TriangleMesh.create_box(*map(float, size))
            mesh.translate(
                np.array(
                    [
                        float(center[0] - size[0] * 0.5),
                        float(center[1] - size[1] * 0.5),
                        float(center[2] - size[2] * 0.5),
                    ],
                    dtype=np.float64,
                )
            )
            mesh.paint_uniform_color(list(color))
            compute_normals = getattr(mesh, "compute_vertex_normals", None)
            if callable(compute_normals):
                compute_normals()
            scene.add_geometry(name, mesh, material)


class ExternalManifestFrameSource:
    def __init__(self, *, manifest_path: str | Path, cv2_module=None) -> None:
        self._manifest_path = Path(manifest_path)
        self._root = self._manifest_path.parent
        self._cv2_module = cv2_module
        self._frames = list((json.loads(self._manifest_path.read_text(encoding="utf-8"))).get("frames") or [])
        self._index = 0

    def next_frame(self, *, timeout_sec: float | None = None) -> FramePacket | None:
        _ = timeout_sec
        if self._index >= len(self._frames):
            return None
        frame_entry = dict(self._frames[self._index])
        self._index += 1

        frame_bgr = self._load_array_or_image(frame_entry["rgb_path"])
        depth_path = frame_entry.get("depth_path")
        depth_map = None if depth_path is None else np.asarray(self._load_array_or_image(depth_path), dtype=np.float32)
        pose_world_gt = frame_entry.get("pose_world_gt")
        intrinsics_gt = frame_entry.get("intrinsics_gt")
        detections = [
            Detection(
                xyxy=tuple(int(value) for value in detection["xyxy"]),
                class_id=int(detection["class_id"]),
                label=str(detection["label"]),
                confidence=float(detection["confidence"]),
                color=tuple(int(value) for value in detection.get("color", (0, 255, 0))),
            )
            for detection in frame_entry.get("detections") or []
        ] or None

        return FramePacket(
            frame_bgr=np.asarray(frame_bgr, dtype=np.uint8),
            timestamp_sec=None if frame_entry.get("timestamp_sec") is None else float(frame_entry["timestamp_sec"]),
            depth_map=depth_map,
            pose_world_gt=None if pose_world_gt is None else np.asarray(pose_world_gt, dtype=np.float32),
            intrinsics_gt=None if intrinsics_gt is None else CameraIntrinsics(
                fx=float(intrinsics_gt["fx"]),
                fy=float(intrinsics_gt["fy"]),
                cx=float(intrinsics_gt["cx"]),
                cy=float(intrinsics_gt["cy"]),
            ),
            detections=detections,
            scenario_state=frame_entry.get("scenario_state"),
            calibration_source=frame_entry.get("calibration_source"),
        )

    def _load_array_or_image(self, relative_or_absolute_path: str) -> np.ndarray:
        path = Path(relative_or_absolute_path)
        if not path.is_absolute():
            path = self._root / path
        if path.suffix == ".npy":
            return np.load(path)
        cv2 = load_cv2(self._cv2_module)
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"failed to load external sim asset: {path}")
        return np.asarray(frame)

    def close(self) -> None:
        return None


class SimulationFrameSource:
    def __init__(
        self,
        *,
        config: AppConfig,
        report_path: str | Path,
        goal_selector: GoalSelector | None = None,
        fallback_goal_selector: GoalSelector | None = None,
        camera_rig: CameraRigSpec | None = None,
        refine_intrinsics=refine_focal_lengths,
        renderer: SceneRenderer | None = None,
        external_frame_source: ExternalManifestFrameSource | None = None,
    ) -> None:
        self._config = config
        self._report_path = Path(report_path)
        self._camera_rig = camera_rig or CameraRigSpec.from_config(config)
        self._scenario = _get_scenario_spec(str(config.scenario))
        self._mission = self._scenario.mission
        self._goal_selector = goal_selector or HeuristicGoalSelector()
        self._fallback_goal_selector = fallback_goal_selector
        if self._fallback_goal_selector is None and goal_selector is not None and not isinstance(goal_selector, HeuristicGoalSelector):
            self._fallback_goal_selector = HeuristicGoalSelector()
        self._refine_intrinsics = refine_intrinsics
        self._rng = np.random.default_rng(int(config.sim_seed))
        self._frame_index = 0
        self._phase = "BOOTSTRAP_SELF_CAL"
        self._warmup_start_x = float(self._scenario.environment.start_x)
        self._warmup_start_z = float(self._scenario.environment.start_z)
        self._warmup_start_yaw_rad = math.radians(float(self._scenario.environment.start_yaw_deg))
        self._x = self._warmup_start_x
        self._z = self._warmup_start_z
        self._yaw_rad = self._warmup_start_yaw_rad
        self._active_goal: NavigationGoal | None = None
        self._selfcal_converged = False
        self._report_emitted = False
        self._mission_success = False
        self._first_valid_view_sec: float | None = None
        self._goal_request_count = 0
        self._llm_goal_accept_count = 0
        self._fallback_count = 0
        self._verify_streak = 0
        self._failure_reason: str | None = None
        self._target_seen_once = False
        self._target_missing_after_acquire_frames = 0
        self._current_intrinsics = create_approximate_calibration(
            image_width=self._camera_rig.image_width,
            image_height=self._camera_rig.image_height,
        )
        self._true_intrinsics = self._build_true_intrinsics()
        self._warmup_keyframe_poses: dict[int, np.ndarray] = {}
        self._warmup_observations: list[KeyframeObservation] = []
        self._last_keyframe_capture_sec = -999.0
        self._last_visible_objects: list[_VisibleObject] = []
        self._last_runtime_observation: _RuntimeObservation | None = None
        self._previous_frame_bgr: np.ndarray | None = None
        self._current_timestamp_sec: float | None = None
        self._runtime_frame_count = 0
        self._runtime_tracking_ok_frames = 0
        self._runtime_pose_error_sum = 0.0
        self._runtime_pose_error_count = 0
        self._render_backend = ""
        self._static_scene_objects = self._build_scene_objects()
        self._active_scene_objects = list(self._static_scene_objects)
        self._landmark_points = self._build_landmark_points()
        self._target_motion_state = "moving" if self._scenario_has_dynamic_actor("moving_target") else "static"
        self._eval_budget_sec = float(
            self._mission.eval_budget_sec_override
            if self._mission.eval_budget_sec_override is not None
            else config.eval_budget_sec
        )
        self._renderer = renderer or AnalyticSceneRenderer(
            rig=self._camera_rig,
            rng=self._rng,
            environment=self._scenario.environment,
        )
        self._external_frame_source = external_frame_source

    @property
    def report_path(self) -> Path:
        return self._report_path

    def close(self) -> None:
        if not self._report_emitted:
            if not self._mission_success and self._failure_reason is None:
                self._failure_reason = "aborted"
            self._emit_report()
        if self._external_frame_source is not None:
            self._external_frame_source.close()

    def next_frame(self, *, timeout_sec: float | None = None) -> FramePacket | None:
        _ = timeout_sec
        if self._phase == "REPORT":
            if self._report_emitted:
                return None
            packet = self._render_packet()
            if packet is None:
                self._emit_report()
                return None
            self._emit_report()
            return packet

        packet = self._render_packet()
        if packet is None:
            if not self._mission_success and self._failure_reason is None:
                self._failure_reason = "external_stream_end" if self._external_frame_source is not None else "aborted"
            self._emit_report()
            return None
        self._post_step(packet)
        self._frame_index += 1
        return packet

    def record_runtime_observation(self, *, frame_packet: FramePacket, artifacts: FrameArtifacts) -> None:
        visible_objects = self._runtime_visible_objects_from_artifacts(artifacts)
        pose_error_m = None
        if frame_packet.pose_world_gt is not None:
            gt_pose = np.asarray(frame_packet.pose_world_gt, dtype=np.float32)
            est_pose = np.asarray(artifacts.camera_pose_world, dtype=np.float32)
            pose_error_m = float(np.linalg.norm(est_pose[:3, 3] - gt_pose[:3, 3]))
            self._runtime_pose_error_sum += pose_error_m
            self._runtime_pose_error_count += 1
        tracking_state = str(artifacts.slam_tracking_state)
        self._runtime_frame_count += 1
        if tracking_state in TRACKING_OK_STATES:
            self._runtime_tracking_ok_frames += 1
        target_visible = next((item for item in visible_objects if item.label == self._mission.target_label), None)
        if self._first_valid_view_sec is None and target_visible is not None and self._is_valid_view(target_visible):
            self._first_valid_view_sec = frame_packet.timestamp_sec
        self._last_runtime_observation = _RuntimeObservation(
            timestamp_sec=frame_packet.timestamp_sec,
            visible_objects=visible_objects,
            tracking_state=tracking_state,
            pose_error_m=pose_error_m,
        )

    def _build_scene_objects(self) -> list[_SceneObject]:
        return [
            _scene_object_from_specs(
                label=item.label,
                center_world=item.center_world,
                size_xyz=item.size_xyz,
                color_bgr=item.color_bgr,
            )
            for item in self._scenario.static_objects
        ]

    def _build_landmark_points(self) -> dict[int, np.ndarray]:
        points: dict[int, np.ndarray] = {}
        point_id = 1
        environment = self._scenario.environment
        room_points = list(environment.landmark_points) or [
            (-environment.room_width_m / 2.0, 0.2, max(1.8, environment.room_depth_m * 0.25)),
            (environment.room_width_m / 2.0, 0.2, max(1.8, environment.room_depth_m * 0.25)),
            (-environment.room_width_m / 2.0, 1.8, environment.room_depth_m * 0.8),
            (environment.room_width_m / 2.0, 1.8, environment.room_depth_m * 0.8),
        ]
        for point in room_points:
            points[point_id] = np.asarray(point, dtype=np.float32)
            point_id += 1
        for item in self._static_scene_objects:
            center = np.asarray(item.center_world, dtype=np.float32)
            half = np.asarray(item.size_xyz, dtype=np.float32) / 2.0
            for dx in (-half[0], half[0]):
                for dy in (-half[1], half[1]):
                    for dz in (-half[2], half[2]):
                        points[point_id] = center + np.array([dx, dy, dz], dtype=np.float32)
                        point_id += 1
        return points

    def _build_true_intrinsics(self) -> CameraIntrinsics:
        width = float(self._camera_rig.image_width)
        height = float(self._camera_rig.image_height)
        fov_rad = math.radians(float(self._camera_rig.horizontal_fov_deg))
        focal = (width * 0.5) / max(math.tan(fov_rad * 0.5), 1e-6)
        return CameraIntrinsics(
            fx=focal,
            fy=focal,
            cx=width * 0.5,
            cy=height * 0.5,
        )

    def _pose_world(self) -> np.ndarray:
        return _pose_world_matrix(self._x, self._z, self._yaw_rad)

    def _render_packet(self) -> FramePacket | None:
        if self._external_frame_source is not None:
            return self._render_external_packet()
        self._update_dynamic_actors()
        pose_world = self._pose_world()
        self._current_timestamp_sec = None
        render_output = self._renderer.render(
            pose_world=pose_world,
            intrinsics=self._true_intrinsics,
            scene_objects=self._active_scene_objects,
            previous_frame_bgr=self._previous_frame_bgr,
        )
        if isinstance(render_output, tuple):
            frame_bgr, depth_map, visible_objects = render_output
            render_output = _RenderOutput(
                frame_bgr=np.asarray(frame_bgr, dtype=np.uint8),
                depth_map=np.asarray(depth_map, dtype=np.float32),
                visible_objects=list(visible_objects),
                backend=getattr(self._renderer, "backend_name", "custom"),
            )
        self._render_backend = render_output.backend
        self._previous_frame_bgr = render_output.frame_bgr.copy()
        self._last_visible_objects = list(render_output.visible_objects)
        detections = [
            Detection(
                xyxy=item.bbox,
                class_id=index,
                label=item.label,
                confidence=0.99,
                color=item.color_bgr,
            )
            for index, item in enumerate(sorted(render_output.visible_objects, key=lambda value: (value.label, value.depth_m)))
        ]
        scenario_state = SimulationScenarioState(
            scene_id=self._scenario.scene_id,
            difficulty_level=self._scenario.difficulty_level,
            phase=self._phase,
            step_index=self._frame_index,
            elapsed_sec=self._elapsed_sec,
            selfcal_converged=self._selfcal_converged,
            rig_x=float(self._x),
            rig_z=float(self._z),
            yaw_deg=math.degrees(self._yaw_rad),
            visible_labels=tuple(item.label for item in render_output.visible_objects),
            active_goal=self._active_goal,
            target_motion_state=self._target_motion_state,
            render_backend=self._render_backend,
        )
        return FramePacket(
            frame_bgr=np.asarray(render_output.frame_bgr, dtype=np.uint8),
            timestamp_sec=self._elapsed_sec,
            depth_map=np.asarray(render_output.depth_map, dtype=np.float32),
            pose_world_gt=pose_world,
            intrinsics_gt=self._intrinsics_estimate(),
            detections=detections,
            scenario_state=scenario_state,
            tracking_state="TRACKING",
            keyframe_inserted=(self._frame_index % max(1, int(round(self._camera_rig.fps // 2))) == 0),
            keyframe_id=self._frame_index,
            calibration_source="auto" if self._selfcal_converged else "disabled/approx",
        )

    @property
    def _elapsed_sec(self) -> float:
        if self._current_timestamp_sec is not None:
            return float(self._current_timestamp_sec)
        fps = max(float(self._camera_rig.fps), 1.0)
        return float(self._frame_index) / fps

    def _render_external_packet(self) -> FramePacket | None:
        assert self._external_frame_source is not None
        source_packet = self._external_frame_source.next_frame(timeout_sec=1.0)
        if source_packet is None:
            return None
        self._current_timestamp_sec = (
            None if source_packet.timestamp_sec is None else float(source_packet.timestamp_sec)
        )
        pose_world = None if source_packet.pose_world_gt is None else np.asarray(source_packet.pose_world_gt, dtype=np.float32)
        if pose_world is not None:
            self._sync_pose_from_external(pose_world)
        intrinsics = source_packet.intrinsics_gt or self._intrinsics_estimate()
        visible_objects = self._visible_objects_from_frame_packet(
            packet=source_packet,
            intrinsics=intrinsics,
        )
        self._render_backend = "external-manifest"
        self._previous_frame_bgr = np.asarray(source_packet.frame_bgr, dtype=np.uint8).copy()
        self._last_visible_objects = visible_objects
        scenario_state = SimulationScenarioState(
            scene_id=self._scenario.scene_id,
            difficulty_level=self._scenario.difficulty_level,
            phase=self._phase,
            step_index=self._frame_index,
            elapsed_sec=self._elapsed_sec,
            selfcal_converged=self._selfcal_converged,
            rig_x=float(self._x),
            rig_z=float(self._z),
            yaw_deg=math.degrees(self._yaw_rad),
            visible_labels=tuple(item.label for item in visible_objects),
            active_goal=self._active_goal,
            target_motion_state=self._target_motion_state,
            render_backend=self._render_backend,
        )
        detections = list(source_packet.detections or [])
        if not detections:
            detections = [
                Detection(
                    xyxy=item.bbox,
                    class_id=index,
                    label=item.label,
                    confidence=0.99,
                    color=item.color_bgr,
                )
                for index, item in enumerate(visible_objects)
            ]
        return replace(
            source_packet,
            frame_bgr=np.asarray(source_packet.frame_bgr, dtype=np.uint8).copy(),
            timestamp_sec=self._elapsed_sec,
            depth_map=(
                None
                if source_packet.depth_map is None
                else np.asarray(source_packet.depth_map, dtype=np.float32)
            ),
            pose_world_gt=pose_world,
            intrinsics_gt=intrinsics,
            detections=detections,
            scenario_state=scenario_state,
            calibration_source=source_packet.calibration_source or ("auto" if self._selfcal_converged else "disabled/approx"),
        )

    def _intrinsics_estimate(self) -> CameraIntrinsics:
        matrix = np.asarray(self._current_intrinsics.camera_matrix, dtype=np.float32)
        return CameraIntrinsics(
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
        )

    def _sync_pose_from_external(self, pose_world: np.ndarray) -> None:
        self._x = float(pose_world[0, 3])
        self._z = float(pose_world[2, 3])
        self._yaw_rad = math.atan2(float(pose_world[0, 2]), float(pose_world[2, 2]))

    def _visible_objects_from_frame_packet(
        self,
        *,
        packet: FramePacket,
        intrinsics: CameraIntrinsics,
    ) -> list[_VisibleObject]:
        visible_objects: list[_VisibleObject] = []
        detections = list(packet.detections or [])
        for detection in detections:
            x1, y1, x2, y2 = (int(value) for value in detection.xyxy)
            if x2 <= x1 or y2 <= y1:
                continue
            if packet.depth_map is None:
                depth_m = float(self._camera_rig.far_plane_m)
            else:
                crop = np.asarray(packet.depth_map[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)], dtype=np.float32)
                valid_depth = crop[np.isfinite(crop) & (crop > 0.0)]
                depth_m = (
                    float(self._camera_rig.far_plane_m)
                    if valid_depth.size == 0
                    else float(np.median(valid_depth))
                )
            center_x = (x1 + x2) * 0.5
            bearing_rad = math.atan2((center_x - float(intrinsics.cx)) / max(float(intrinsics.fx), 1e-6), 1.0)
            visible_objects.append(
                _VisibleObject(
                    label=detection.label,
                    bbox=(x1, y1, x2, y2),
                    depth_m=depth_m,
                    bearing_rad=bearing_rad,
                    area_pixels=max(0, x2 - x1) * max(0, y2 - y1),
                    color_bgr=detection.color,
                )
            )
        visible_objects.sort(key=lambda item: float(item.depth_m), reverse=True)
        return visible_objects

    def _control_visible_objects(self) -> list[_VisibleObject]:
        if self._last_runtime_observation is not None:
            return list(self._last_runtime_observation.visible_objects)
        return list(self._last_visible_objects)

    def _scenario_has_dynamic_actor(self, actor_type: str) -> bool:
        return any(item.actor_type == actor_type for item in self._scenario.dynamic_actors)

    def _update_dynamic_actors(self) -> None:
        active_scene_objects = list(self._static_scene_objects)
        target_motion_state = "moving" if self._scenario_has_dynamic_actor("moving_target") else "static"
        for actor in self._scenario.dynamic_actors:
            if self._elapsed_sec < float(actor.start_time_sec):
                center_world = actor.base_center_world
            else:
                center_world = _resolve_waypoint_loop_position(actor, elapsed_sec=self._elapsed_sec)
            active_scene_objects.append(
                _scene_object_from_specs(
                    label=actor.label,
                    center_world=center_world,
                    size_xyz=actor.size_xyz,
                    color_bgr=actor.color_bgr,
                )
            )
        self._active_scene_objects = active_scene_objects
        self._target_motion_state = target_motion_state

    def _post_step(self, packet: FramePacket) -> None:
        if self._elapsed_sec >= float(self._eval_budget_sec) or self._frame_index >= int(self._config.sim_max_steps) - 1:
            if not self._mission_success:
                self._failure_reason = self._failure_reason or self._derive_timeout_reason()
                self._phase = "REPORT"
            return

        if self._phase == "BOOTSTRAP_SELF_CAL":
            if packet.pose_world_gt is not None:
                self._record_warmup_observations(np.asarray(packet.pose_world_gt, dtype=np.float32))
            self._apply_warmup_motion()
            if self._elapsed_sec >= 8.0:
                self._finalize_self_calibration()
                self._phase = "EXPLORE"
            return

        if self._phase == "EXPLORE":
            target_visible = self._target_visible_object()
            if target_visible is not None:
                self._target_seen_once = True
                self._target_missing_after_acquire_frames = 0
                self._active_goal = self._request_navigation_goal()
                if self._active_goal is not None:
                    self._phase = "NAVIGATE_TO_VIEW"
                    return
            self._drive_explore()
            return

        if self._phase == "NAVIGATE_TO_VIEW":
            target_visible = self._target_visible_object()
            if target_visible is None:
                if self._target_seen_once:
                    self._target_missing_after_acquire_frames += 1
                self._active_goal = None
                self._phase = "EXPLORE"
                return
            self._target_seen_once = True
            self._target_missing_after_acquire_frames = 0
            if self._is_valid_view(target_visible):
                if self._first_valid_view_sec is None:
                    self._first_valid_view_sec = self._elapsed_sec
                self._verify_streak = 1
                self._phase = "VERIFY_VIEW"
                return
            self._drive_toward_goal(target_visible)
            return

        if self._phase == "VERIFY_VIEW":
            target_visible = self._target_visible_object()
            if target_visible is None:
                if self._target_seen_once:
                    self._target_missing_after_acquire_frames += 1
                self._verify_streak = 0
                self._phase = "EXPLORE"
                return
            self._target_seen_once = True
            self._target_missing_after_acquire_frames = 0
            if self._is_valid_view(target_visible):
                if self._first_valid_view_sec is None:
                    self._first_valid_view_sec = self._elapsed_sec
                self._verify_streak += 1
                if self._verify_streak >= int(self._mission.verify_required_streak):
                    self._mission_success = True
                    self._phase = "REPORT"
                    return
            else:
                self._verify_streak = 0
                self._phase = "NAVIGATE_TO_VIEW"
                return
            self._drive_toward_goal(target_visible)

    def _record_warmup_observations(self, pose_world: np.ndarray) -> None:
        if self._external_frame_source is not None:
            return
        if self._elapsed_sec - self._last_keyframe_capture_sec < 0.75:
            return
        keyframe_id = len(self._warmup_keyframe_poses) + 1
        self._warmup_keyframe_poses[keyframe_id] = pose_world.copy()
        self._last_keyframe_capture_sec = self._elapsed_sec
        points_world = np.asarray(list(self._landmark_points.values()), dtype=np.float32)
        projected, valid = _project_points(
            points_world,
            pose_world=pose_world,
            intrinsics=self._true_intrinsics,
            near_plane_m=self._camera_rig.near_plane_m,
        )
        width = float(self._camera_rig.image_width)
        height = float(self._camera_rig.image_height)
        for offset, point_id in enumerate(self._landmark_points):
            if not bool(valid[offset]):
                continue
            u = float(projected[offset, 0])
            v = float(projected[offset, 1])
            if not (0.0 <= u < width and 0.0 <= v < height):
                continue
            xyz = self._landmark_points[point_id]
            self._warmup_observations.append(
                KeyframeObservation(
                    keyframe_id=keyframe_id,
                    point_id=int(point_id),
                    u=u,
                    v=v,
                    x=float(xyz[0]),
                    y=float(xyz[1]),
                    z=float(xyz[2]),
                )
            )

    def _finalize_self_calibration(self) -> None:
        approx_matrix = np.asarray(self._current_intrinsics.camera_matrix, dtype=np.float32)
        refined_fx, refined_fy, _poses, _points = self._refine_intrinsics(
            initial_fx=float(approx_matrix[0, 0]),
            initial_fy=float(approx_matrix[1, 1]),
            cx=float(approx_matrix[0, 2]),
            cy=float(approx_matrix[1, 2]),
            keyframe_poses=self._warmup_keyframe_poses,
            keyframe_observations=self._warmup_observations,
        )
        refined = approx_matrix.copy()
        refined[0, 0] = refined_fx
        refined[1, 1] = refined_fy
        self._current_intrinsics = type(self._current_intrinsics)(
            camera_matrix=refined,
            distortion_coefficients=np.zeros((1, 5), dtype=np.float32),
            image_width=self._camera_rig.image_width,
            image_height=self._camera_rig.image_height,
            rms_error=0.0,
        )
        true_fx = float(self._true_intrinsics.fx)
        fx_error = abs(refined_fx - true_fx) / max(true_fx, 1e-6)
        self._selfcal_converged = (
            len(self._warmup_keyframe_poses) >= 6
            and len(self._warmup_observations) >= 120
            and fx_error <= 0.35
        )

    def _apply_warmup_motion(self) -> None:
        if self._external_frame_source is not None:
            return
        t = self._elapsed_sec
        if t < 2.0:
            self._yaw_rad = self._warmup_start_yaw_rad
            self._x = self._warmup_start_x
            self._z = self._warmup_start_z
            return
        if t < 4.0:
            progress = (t - 2.0) / 2.0
            self._yaw_rad = self._warmup_start_yaw_rad + math.radians(50.0 * progress)
            return
        if t < 6.0:
            progress = (t - 4.0) / 2.0
            self._yaw_rad = self._warmup_start_yaw_rad + math.radians(50.0 - (55.0 * progress))
            self._z = self._warmup_start_z + (0.65 * progress)
            return
        if t < 8.0:
            progress = (t - 6.0) / 2.0
            self._yaw_rad = self._warmup_start_yaw_rad + math.radians(-5.0 + (30.0 * progress))
            self._z = self._warmup_start_z + 0.65 - (0.45 * progress)

    def _target_visible_object(self) -> _VisibleObject | None:
        for item in self._control_visible_objects():
            if item.label == self._mission.target_label:
                return item
        return None

    def _request_navigation_goal(self) -> NavigationGoal | None:
        self._goal_request_count += 1
        candidate = self._goal_selector.select_goal(
            visible_objects=self._control_visible_objects(),
            target_label=self._mission.target_label,
        )
        if self._is_valid_goal(candidate):
            if candidate is not None and candidate.source == "llm":
                self._llm_goal_accept_count += 1
            return candidate
        if self._fallback_goal_selector is None:
            return None
        self._fallback_count += 1
        fallback_goal = self._fallback_goal_selector.select_goal(
            visible_objects=self._control_visible_objects(),
            target_label=self._mission.target_label,
        )
        return fallback_goal if self._is_valid_goal(fallback_goal) else None

    def _is_valid_goal(self, goal: NavigationGoal | None) -> bool:
        if goal is None:
            return False
        if goal.target_label != self._mission.target_label:
            return False
        if not math.isfinite(float(goal.desired_bearing)):
            return False
        near_m, far_m = goal.desired_distance_band
        if not (math.isfinite(float(near_m)) and math.isfinite(float(far_m))):
            return False
        return 0.0 < float(near_m) < float(far_m)

    def _drive_explore(self) -> None:
        if self._external_frame_source is not None:
            return
        target = next(item for item in self._active_scene_objects if item.label == self._mission.target_label)
        target_xy = np.array([target.center_world[0], target.center_world[2]], dtype=np.float32)
        camera_xy = np.array([self._x, self._z], dtype=np.float32)
        delta = target_xy - camera_xy
        desired_yaw = math.atan2(float(delta[0]), float(delta[1]))
        yaw_delta = desired_yaw - self._yaw_rad
        yaw_delta = math.atan2(math.sin(yaw_delta), math.cos(yaw_delta))
        max_step = math.radians(float(self._camera_rig.yaw_rate_limit_deg)) / max(float(self._camera_rig.fps), 1.0)
        self._yaw_rad += float(np.clip(yaw_delta, -max_step, max_step))
        if abs(yaw_delta) < math.radians(12.0):
            forward = min(float(self._camera_rig.linear_velocity_limit_mps) * 0.5, float(np.linalg.norm(delta) - 2.6))
            if forward > 0.0:
                self._x += math.sin(self._yaw_rad) * forward / max(float(self._camera_rig.fps), 1.0)
                self._z += math.cos(self._yaw_rad) * forward / max(float(self._camera_rig.fps), 1.0)

    def _drive_toward_goal(self, target_visible: _VisibleObject) -> None:
        if self._external_frame_source is not None:
            return
        goal = self._active_goal
        if goal is None:
            self._phase = "EXPLORE"
            return
        max_yaw_step = math.radians(float(self._camera_rig.yaw_rate_limit_deg)) / max(float(self._camera_rig.fps), 1.0)
        bearing_error = float(goal.desired_bearing - target_visible.bearing_rad)
        bearing_error = math.atan2(math.sin(bearing_error), math.cos(bearing_error))
        self._yaw_rad -= float(np.clip(bearing_error, -max_yaw_step, max_yaw_step))

        desired_distance = sum(goal.desired_distance_band) * 0.5
        distance_error = float(target_visible.depth_m - desired_distance)
        max_step = float(self._camera_rig.linear_velocity_limit_mps) / max(float(self._camera_rig.fps), 1.0)
        forward_step = float(np.clip(distance_error * 0.4, -max_step, max_step))
        self._x += math.sin(self._yaw_rad) * forward_step
        self._z += math.cos(self._yaw_rad) * forward_step

    def _runtime_visible_objects_from_artifacts(self, artifacts: FrameArtifacts) -> list[_VisibleObject]:
        objects: list[_VisibleObject] = []
        intrinsics = artifacts.intrinsics
        for detection in artifacts.detections:
            x1, y1, x2, y2 = (int(value) for value in detection.xyxy)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = np.asarray(artifacts.depth_map[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)], dtype=np.float32)
            valid_depth = crop[np.isfinite(crop) & (crop > 0.0)]
            if valid_depth.size == 0:
                continue
            depth_m = float(np.median(valid_depth))
            center_x = (x1 + x2) * 0.5
            bearing_rad = math.atan2((center_x - float(intrinsics.cx)) / max(float(intrinsics.fx), 1e-6), 1.0)
            area_pixels = max(0, x2 - x1) * max(0, y2 - y1)
            objects.append(
                _VisibleObject(
                    label=detection.label,
                    bbox=(x1, y1, x2, y2),
                    depth_m=depth_m,
                    bearing_rad=bearing_rad,
                    area_pixels=area_pixels,
                    color_bgr=detection.color,
                )
            )
        objects.sort(key=lambda item: float(item.depth_m), reverse=True)
        return objects

    def _is_valid_view(self, item: _VisibleObject) -> bool:
        width = float(self._camera_rig.image_width)
        height = float(self._camera_rig.image_height)
        x1, y1, x2, y2 = item.bbox
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        centered = (
            abs(center_x - (width * 0.5)) <= (width * float(self._mission.center_tolerance_x_ratio))
            and abs(center_y - (height * 0.5)) <= (height * float(self._mission.center_tolerance_y_ratio))
        )
        large_enough = item.area_pixels >= int(width * height * float(self._mission.min_area_ratio))
        close_enough = float(self._mission.depth_band_m[0]) <= float(item.depth_m) <= float(self._mission.depth_band_m[1])
        return centered and large_enough and close_enough

    def _derive_timeout_reason(self) -> str:
        if self._target_seen_once:
            if self._scenario_has_dynamic_actor("moving_target"):
                return "moving_target_timeout"
            if self._scenario_has_dynamic_actor("occluder"):
                return "occluded_timeout"
            return "lost_after_acquire"
        return "timeout"

    def _emit_report(self) -> None:
        total_frames = max(self._frame_index, 1)
        if self._runtime_frame_count > 0:
            tracking_uptime = float(self._runtime_tracking_ok_frames) / float(self._runtime_frame_count)
        else:
            tracking_uptime = float(total_frames) / float(total_frames)
        pose_error_vs_gt = (
            0.0
            if self._runtime_pose_error_count == 0
            else float(self._runtime_pose_error_sum) / float(self._runtime_pose_error_count)
        )
        report = {
            "scenario": self._config.scenario,
            "scenario_family": self._scenario.scenario_family,
            "difficulty_level": int(self._scenario.difficulty_level),
            "seed": int(self._config.sim_seed),
            "sim_profile": self._config.sim_profile,
            "sim_perception_mode": self._config.sim_perception_mode,
            "render_backend": self._render_backend or getattr(self._renderer, "backend_name", "unknown"),
            "mission_success": bool(self._mission_success),
            "failure_reason": self._failure_reason,
            "time_to_first_valid_view": (
                None if self._first_valid_view_sec is None else round(float(self._first_valid_view_sec), 3)
            ),
            "tracking_uptime": round(tracking_uptime, 4),
            "selfcal_converged": bool(self._selfcal_converged),
            "pose_error_vs_gt": round(float(pose_error_vs_gt), 4),
            "llm_goal_accept_rate": (
                0.0 if self._goal_request_count == 0 else float(self._llm_goal_accept_count) / float(self._goal_request_count)
            ),
            "fallback_count": int(self._fallback_count),
            "dynamic_actor_count": int(len(self._scenario.dynamic_actors)),
            "target_class": self._mission.target_label,
            "steps": int(self._frame_index),
        }
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        self._report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        self._report_emitted = True


@dataclass(slots=True)
class SimulationRuntime:
    config: AppConfig
    report_path: str | Path
    goal_selector: GoalSelector | None = None
    fallback_goal_selector: GoalSelector | None = None
    refine_intrinsics: object = refine_focal_lengths
    openai_client: object | None = None
    cv2_module: object | None = None
    open3d_module: object | None = None
    renderer: SceneRenderer | None = None

    def create_frame_source(self) -> SimulationFrameSource:
        scenario = _get_scenario_spec(str(self.config.scenario))
        goal_selector = self.goal_selector
        if goal_selector is None:
            if self.config.sim_goal_selector == "llm":
                goal_selector = OpenAINavigationGoalSelector(
                    model=self.config.sim_goal_model,
                    timeout_sec=self.config.sim_goal_timeout_sec,
                    client=self.openai_client,
                )
            else:
                goal_selector = HeuristicGoalSelector()

        external_frame_source = None
        if self.config.sim_profile == "external":
            if not self.config.sim_external_manifest:
                raise RuntimeError("sim_profile=external requires --sim-external-manifest")
            external_frame_source = ExternalManifestFrameSource(
                manifest_path=self.config.sim_external_manifest,
                cv2_module=self.cv2_module,
            )

        renderer = self.renderer
        if renderer is None and external_frame_source is None:
            if self.open3d_module is not None:
                renderer = Open3DSceneRenderer(
                    rig=CameraRigSpec.from_config(self.config),
                    environment=scenario.environment,
                    o3d_module=self.open3d_module,
                )
            else:
                try:
                    renderer = Open3DSceneRenderer(
                        rig=CameraRigSpec.from_config(self.config),
                        environment=scenario.environment,
                    )
                except Exception:
                    renderer = AnalyticSceneRenderer(
                        rig=CameraRigSpec.from_config(self.config),
                        environment=scenario.environment,
                    )

        return SimulationFrameSource(
            config=self.config,
            report_path=self.report_path,
            goal_selector=goal_selector,
            fallback_goal_selector=self.fallback_goal_selector,
            camera_rig=CameraRigSpec.from_config(self.config),
            refine_intrinsics=self.refine_intrinsics,
            renderer=renderer,
            external_frame_source=external_frame_source,
        )
