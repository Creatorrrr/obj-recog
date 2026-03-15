from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class CommandKind(str, Enum):
    TRANSLATE = "translate"
    ROTATE_BODY = "rotate_body"
    AIM_CAMERA = "aim_camera"
    PAUSE = "pause"


class TranslationDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"


class RotationDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class CommandValueMode(str, Enum):
    DISTANCE_M = "distance_m"
    HOLD_SEC = "hold_sec"
    ANGLE_DEG = "angle_deg"


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
    camera_pitch_deg: float = 0.0


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
    distance_bucket: str | None = None
    relative_xyz: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class PlannerSegmentObservation:
    label: str
    coverage_ratio: float
    dominant_direction: str
    opening_hint: str | None


@dataclass(frozen=True, slots=True)
class PlannerSceneNodeObservation:
    label: str
    kind: str
    state: str
    bearing: str | None
    distance_bucket: str | None
    confidence: float
    relative_xyz: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class PlannerSceneEdgeObservation:
    source_label: str
    target_label: str
    relation: str
    confidence: float
    source_kind: str
    distance_bucket: str | None = None


@dataclass(frozen=True, slots=True)
class PlannerNavigationSectorObservation:
    sector: str
    clearance_m: float | None
    traversable: bool
    obstacle_likelihood: float
    frontier_score: float
    recently_failed: bool = False


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
    sector_map: tuple[PlannerNavigationSectorObservation, ...] = ()
    recently_failed_directions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlannerReconstructionBrief:
    pose_delta_m: float | None
    yaw_delta_deg: float | None
    mesh_vertex_count: int
    mesh_triangle_count: int
    mesh_growth_delta: int
    frontier_directions: tuple[str, ...]
    explored_directions: tuple[str, ...] = ()
    unexplored_directions: tuple[str, ...] = ()
    recently_failed_directions: tuple[str, ...] = ()
    tracked_feature_count: int = 0
    median_reprojection_error: float | None = None


@dataclass(frozen=True, slots=True)
class PlannerGoalEstimate:
    status: str
    bearing_hint: str | None
    distance_hint: str | None
    evidence_sources: tuple[str, ...]
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class PlannerActionEffectSummary:
    action: str
    estimated_motion: str
    clearance_change: str
    target_evidence_change: str
    likely_blocked: bool
    aborted: bool = False
    commanded_progress_m: float | None = None
    vision_progress_m: float | None = None
    fused_progress_m: float | None = None
    progress_source: str = "unknown"
    frame_index: int | None = None


@dataclass(frozen=True, slots=True)
class PlannerConstraintSummary:
    allowed_command_kinds: tuple[str, ...]
    execution_capabilities: dict[str, float]
    microstep_limits: dict[str, float]
    max_commands_per_schedule: int
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlannerSafetyFlags:
    front_blocked: bool
    dead_end_risk: float
    tracking_risk: str
    replan_reason: str


@dataclass(frozen=True, slots=True)
class PlannerGoalCompletion:
    reached: bool
    confidence: float
    rationale: str


@dataclass(frozen=True, slots=True)
class PlannerCameraState:
    yaw_deg: float
    pitch_deg: float


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
    frame_size: tuple[int, int] = (0, 0)
    scene_nodes: tuple[PlannerSceneNodeObservation, ...] = ()
    scene_edges: tuple[PlannerSceneEdgeObservation, ...] = ()
    current_camera_state: PlannerCameraState | None = None


@dataclass(frozen=True, slots=True)
class PlannerMemoryObservation:
    label: str
    kind: str
    state: str
    last_seen_direction: str | None
    age_frames: int
    last_seen_frame: int | None = None
    confidence: float | None = None
    relative_xyz: tuple[float, float, float] | None = None
    distance_bucket: str | None = None


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
class PlannerGoalCompletionEvidence:
    forward_progress_since_last_seen_m: float | None
    estimated_remaining_distance_m: float | None
    recent_close_sighting: bool
    target_disappeared_after_approach: bool
    memory_freshness: str
    anchor_matches: int
    contradictions: tuple[str, ...] = ()
    last_seen_frame: int | None = None
    last_seen_direction: str | None = None
    last_seen_relative_xyz: tuple[float, float, float] | None = None
    last_seen_distance_bucket: str | None = None
    target_memory_confidence: float | None = None
    anchor_labels: tuple[str, ...] = ()
    visible_anchor_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlannerContext:
    phase: EpisodePhase
    frame_index: int
    goal_description: str
    target_label: str
    perception: PerceptionSnapshot
    memory: PlannerMemorySnapshot
    goal_estimate: PlannerGoalEstimate
    goal_completion_evidence: PlannerGoalCompletionEvidence | None
    recent_action_effects: tuple[PlannerActionEffectSummary, ...]
    constraints: PlannerConstraintSummary
    recent_actions: tuple[str, ...]
    image_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MotionCommand:
    kind: CommandKind
    intent: str = ""
    direction: str | None = None
    mode: str | None = None
    value: float | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    duration_sec: float | None = None


@dataclass(frozen=True, slots=True)
class UnityRigDeltaCommand:
    translate_forward_m: float = 0.0
    translate_right_m: float = 0.0
    body_yaw_deg: float = 0.0
    camera_yaw_delta_deg: float = 0.0
    camera_pitch_delta_deg: float = 0.0
    pause_sec: float = 0.0


@dataclass(frozen=True, slots=True)
class RigCapabilities:
    move_speed_mps: float
    turn_speed_deg_per_sec: float
    camera_yaw_speed_deg_per_sec: float
    camera_pitch_speed_deg_per_sec: float
    camera_yaw_limit_deg: float
    camera_pitch_limit_deg: float


@dataclass(frozen=True, slots=True)
class ExecutedMacroCommand:
    command: MotionCommand
    measured_translation_m: float | None
    measured_yaw_deg: float | None
    completed: bool
    aborted: bool
    microstep_count: int
    target_evidence_change: str
    pose_progress_m: float | None = None
    intent: str = ""
    commanded_progress_m: float | None = None
    vision_progress_m: float | None = None
    fused_progress_m: float | None = None
    progress_source: str = "unknown"


@dataclass(frozen=True, slots=True)
class ActionSchedule:
    commands: tuple[MotionCommand, ...]
    rationale: str
    model: str
    issued_at_frame: int
    situation_summary: str = ""
    behavior_mode: str = "scan"
    goal_hypothesis: PlannerGoalEstimate | None = None
    goal_completion: PlannerGoalCompletion | None = None
    safety_flags: PlannerSafetyFlags | None = None
    confidence: float | None = None


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
