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
    blend_file_path: str | None = None
    goal_description: str = "Reach the front position of the dining table using only current visible evidence."
    semantic_target_class: str = "dining_table"
    scene_metadata: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


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
    render_time_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlannerObjectObservation:
    label: str
    confidence: float
    direction: str
    bbox_area_ratio: float
    depth_m: float | None
    is_target_match: bool


@dataclass(frozen=True, slots=True)
class PlannerSegmentObservation:
    label: str
    coverage_ratio: float
    dominant_direction: str
    opening_hint: str | None


@dataclass(frozen=True, slots=True)
class PlannerNavigationAffordances:
    forward_clearance_m: float | None
    left_clearance_m: float | None
    right_clearance_m: float | None
    rear_clearance_m: float | None
    front_blocked: bool
    candidate_open_directions: tuple[str, ...]
    dead_end_likelihood: float
    best_exploration_direction: str | None


@dataclass(frozen=True, slots=True)
class PlannerReconstructionBrief:
    pose_delta_m: float | None
    yaw_delta_deg: float | None
    mesh_vertex_count: int
    mesh_triangle_count: int
    mesh_growth_delta: int
    frontier_directions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlannerGoalEstimate:
    status: str
    bearing_hint: str | None
    distance_hint: str | None
    evidence_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlannerActionEffectSummary:
    action: str
    estimated_motion: str
    clearance_change: str
    target_evidence_change: str
    likely_blocked: bool


@dataclass(frozen=True, slots=True)
class PlannerConstraintSummary:
    allowed_primitives: tuple[str, ...]
    tracking_safe_limits: dict[str, float]
    batch_step_limit: int
    replan_frame_budget: int
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PerceptionSnapshot:
    visible_detections: tuple[str, ...]
    visible_segments: tuple[str, ...]
    visible_graph_relations: tuple[str, ...]
    reconstruction_summary: dict[str, float | int]
    depth_summary: dict[str, float | int]
    target_detection: dict[str, float | str | None] | None
    calibration_status: str
    tracking_status: str
    objects: tuple[PlannerObjectObservation, ...] = ()
    structural_segments: tuple[PlannerSegmentObservation, ...] = ()
    navigation_affordances: PlannerNavigationAffordances | None = None
    reconstruction_brief: PlannerReconstructionBrief | None = None


@dataclass(frozen=True, slots=True)
class PlannerMemoryObservation:
    label: str
    kind: str
    state: str
    last_seen_direction: str | None
    age_frames: int


@dataclass(frozen=True, slots=True)
class PlannerSearchOutcome:
    start_frame: int
    end_frame: int
    executed_actions: tuple[str, ...]
    target_visible_after_search: bool
    tracking_status: str
    entered_new_view: bool = False
    pose_progress_m: float | None = None
    likely_blocked: bool = False
    evidence_gained: bool = False


@dataclass(frozen=True, slots=True)
class PlannerMemorySnapshot:
    target_memory: PlannerMemoryObservation | None
    nonvisible_observations: tuple[PlannerMemoryObservation, ...]
    recent_searches: tuple[PlannerSearchOutcome, ...]


@dataclass(frozen=True, slots=True)
class PlannerContext:
    phase: EpisodePhase
    frame_index: int
    goal_description: str
    target_label: str
    perception: PerceptionSnapshot
    memory: PlannerMemorySnapshot
    goal_estimate: PlannerGoalEstimate
    recent_action_effects: tuple[PlannerActionEffectSummary, ...]
    constraints: PlannerConstraintSummary
    recent_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActionStep:
    primitive: ActionPrimitive
    value: float


@dataclass(frozen=True, slots=True)
class UnityActionCommand:
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
    success: bool | None
    final_phase: EpisodePhase
    total_frames: int
    planner_turns: int
    self_calibration_completed: bool
    final_distance_to_goal_m: float | None
    report_dir: str
