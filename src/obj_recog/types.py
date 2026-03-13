from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from obj_recog.reconstruct import CameraIntrinsics
    from obj_recog.scene_graph import GraphEdge, GraphNode, SceneGraphSnapshot


@dataclass(frozen=True, slots=True)
class Detection:
    xyxy: tuple[int, int, int, int]
    class_id: int
    label: str
    confidence: float
    color: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class PanopticSegment:
    segment_id: int
    label_id: int
    label: str
    color_rgb: tuple[int, int, int]
    mask: np.ndarray
    area_pixels: int


@dataclass(slots=True)
class SegmentationResult:
    overlay_bgr: np.ndarray
    segment_id_map: np.ndarray
    segments: list[PanopticSegment]


@dataclass(frozen=True, slots=True)
class DepthDiagnostics:
    calibration_source: str
    profile: str
    raw_percentiles: tuple[float, float, float]
    normalizer_low_high: tuple[float, float]
    normalized_distance_percentiles: tuple[float, float, float]
    valid_depth_ratio: float
    dense_z_span: float
    mesh_z_span: float
    intrinsics_summary: tuple[float, float, float, float]
    hint: str


@dataclass(frozen=True, slots=True)
class PerceptionDiagnostics:
    perception_mode: str
    detection_source: str
    depth_source: str
    pose_source: str
    gt_target_visible: bool
    benchmark_valid: bool


@dataclass(slots=True)
class FrameArtifacts:
    frame_bgr: np.ndarray
    intrinsics: CameraIntrinsics
    detections: list[Detection]
    depth_map: np.ndarray
    points_xyz: np.ndarray
    points_rgb: np.ndarray
    dense_map_points_xyz: np.ndarray
    dense_map_points_rgb: np.ndarray
    mesh_vertices_xyz: np.ndarray
    mesh_triangles: np.ndarray
    mesh_vertex_colors: np.ndarray
    camera_pose_world: np.ndarray
    tracking_ok: bool
    is_keyframe: bool
    trajectory_xyz: np.ndarray
    segment_id: int
    slam_tracking_state: str
    keyframe_id: int | None
    sparse_map_points_xyz: np.ndarray
    loop_closure_applied: bool
    segmentation_overlay_bgr: np.ndarray
    segments: list[PanopticSegment]
    depth_diagnostics: DepthDiagnostics | None = None
    perception_diagnostics: PerceptionDiagnostics | None = None
    scene_graph_snapshot: SceneGraphSnapshot | None = None
    visible_graph_nodes: list[GraphNode] = field(default_factory=list)
    visible_graph_edges: list[GraphEdge] = field(default_factory=list)
