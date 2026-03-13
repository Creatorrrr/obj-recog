from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ActionPrimitive(str, Enum):
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    STRAFE_LEFT = "strafe_left"
    STRAFE_RIGHT = "strafe_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    CAMERA_PAN_LEFT = "camera_pan_left"
    CAMERA_PAN_RIGHT = "camera_pan_right"
    PAUSE = "pause"


class EpisodePhase(str, Enum):
    SELF_CALIBRATING = "SELF_CALIBRATING"
    PERCEIVE_AND_PLAN = "PERCEIVE_AND_PLAN"
    EXECUTING_SCHEDULE = "EXECUTING_SCHEDULE"
    REASSESS = "REASSESS"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class RobotPose:
    x: float
    y: float
    z: float
    yaw_deg: float
    camera_pan_deg: float = 0.0


@dataclass(frozen=True, slots=True)
class LivingRoomObjectSpec:
    object_id: str
    semantic_label: str
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    yaw_deg: float
    material_key: str
    collider: bool = True


@dataclass(frozen=True, slots=True)
class LivingRoomLightSpec:
    light_id: str
    light_type: str
    location_xyz: tuple[float, float, float]
    rotation_deg_xyz: tuple[float, float, float]
    color_rgb: tuple[float, float, float]
    energy: float


@dataclass(frozen=True, slots=True)
class LivingRoomSceneSpec:
    scene_id: str
    room_size_xyz: tuple[float, float, float]
    wall_thickness_m: float
    window_wall: str
    start_pose: RobotPose
    hidden_goal_pose_xyz: tuple[float, float, float]
    objects: tuple[LivingRoomObjectSpec, ...]
    lights: tuple[LivingRoomLightSpec, ...]


@dataclass(slots=True)
class HiddenWorldState:
    scene_spec: LivingRoomSceneSpec
    robot_pose: RobotPose
    elapsed_sec: float = 0.0
    frame_index: int = 0
    phase: EpisodePhase = EpisodePhase.SELF_CALIBRATING
    selfcal_step_index: int = 0
    schedule_cursor: int = 0
    planner_turn_count: int = 0
    mission_succeeded: bool = False


@dataclass(slots=True)
class SensorFrame:
    frame_index: int
    timestamp_sec: float
    frame_bgr: np.ndarray
    depth_map: np.ndarray
    semantic_mask: np.ndarray
    instance_mask: np.ndarray
    camera_pose_world: np.ndarray
    intrinsics: dict[str, float]
    render_time_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PerceptionSnapshot:
    visible_detections: tuple[str, ...]
    visible_segments: tuple[str, ...]
    visible_graph_relations: tuple[str, ...]
    reconstruction_summary: dict[str, float | int]
    depth_summary: dict[str, float | int]
    calibration_status: str
    tracking_status: str


@dataclass(frozen=True, slots=True)
class PlannerContext:
    phase: EpisodePhase
    frame_index: int
    goal_description: str
    perception: PerceptionSnapshot
    recent_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActionStep:
    primitive: ActionPrimitive
    value: float


@dataclass(frozen=True, slots=True)
class ActionSchedule:
    steps: tuple[ActionStep, ...]
    rationale: str
    model: str
    issued_at_frame: int


@dataclass(frozen=True, slots=True)
class OperatorSceneState:
    scene_spec: LivingRoomSceneSpec
    robot_pose: RobotPose
    phase: EpisodePhase
    semantic_target_class: str = "dining_table"


@dataclass(frozen=True, slots=True)
class EpisodeReport:
    scenario_id: str
    success: bool
    final_phase: EpisodePhase
    total_frames: int
    planner_turns: int
    self_calibration_completed: bool
    final_distance_to_goal_m: float
    report_dir: str
