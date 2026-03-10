from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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
    segments: list[PanopticSegment]


@dataclass(slots=True)
class FrameArtifacts:
    frame_bgr: np.ndarray
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
    scene_graph_snapshot: SceneGraphSnapshot | None = None
    visible_graph_nodes: list[GraphNode] = field(default_factory=list)
    visible_graph_edges: list[GraphEdge] = field(default_factory=list)
