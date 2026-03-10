from __future__ import annotations

import numpy as np

from obj_recog.scene_graph import SceneGraphSnapshot
from obj_recog.types import Detection, PanopticSegment


def _display_points_for_view(points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points_xyz.size == 0:
        return points_xyz.copy()
    transformed = points_xyz.copy()
    transformed[:, 1] *= -1.0
    transformed[:, 2] *= -1.0
    return transformed


def _measure_text(cv2, text: str, font: int, scale: float, thickness: int) -> tuple[int, int]:
    get_text_size = getattr(cv2, "getTextSize", None)
    if callable(get_text_size):
        (width, height), _baseline = get_text_size(text, font, scale, thickness)
        return int(width), int(height)
    return max(1, int(round(len(text) * 7 * scale))), max(1, int(round(14 * scale)))


def _draw_segmentation_legend(cv2, canvas: np.ndarray, segments: list[PanopticSegment]) -> None:
    if not segments:
        return

    unique_segments: dict[int, PanopticSegment] = {}
    for segment in sorted(segments, key=lambda item: int(item.area_pixels), reverse=True):
        unique_segments.setdefault(int(segment.label_id), segment)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_thickness = 1
    chip_size = 10
    row_height = 18
    top_padding = 12
    right_padding = 12
    text_gap = 6
    max_entries = 6

    entries = list(unique_segments.values())[:max_entries]
    for index, segment in enumerate(entries):
        label = str(segment.label)
        text_width, text_height = _measure_text(cv2, label, font, font_scale, text_thickness)
        total_width = chip_size + text_gap + text_width
        x_right = canvas.shape[1] - right_padding
        x_left = max(0, x_right - total_width)
        y_top = top_padding + (index * row_height)
        y_baseline = y_top + max(text_height, chip_size)
        chip_y = y_top + max(0, (text_height - chip_size) // 2)
        chip_color_bgr = tuple(int(channel) for channel in segment.color_rgb[::-1])

        cv2.rectangle(
            canvas,
            (x_left, chip_y),
            (x_left + chip_size, chip_y + chip_size),
            chip_color_bgr,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x_left + chip_size + text_gap, y_baseline),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )


def _draw_scene_graph_summary(
    cv2,
    canvas: np.ndarray,
    *,
    scene_graph_snapshot: SceneGraphSnapshot | None,
    visible_graph_nodes,
    visible_graph_edges,
) -> None:
    if scene_graph_snapshot is None:
        return

    edges = list(visible_graph_edges or scene_graph_snapshot.visible_edges)
    nodes = {node.id: node for node in (visible_graph_nodes or scene_graph_snapshot.visible_nodes)}
    ego_edges = [edge for edge in edges if edge.source == "ego" and edge.target in nodes]
    if not ego_edges:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    text_thickness = 1
    row_height = 16
    top_padding = 52
    right_padding = 12
    max_entries = 6
    entries = sorted(
        ego_edges,
        key=lambda edge: (-float(edge.confidence), edge.relation, nodes[edge.target].label),
    )[:max_entries]

    for index, edge in enumerate(entries):
        node = nodes[edge.target]
        suffix = f" ({edge.distance_bucket})" if edge.distance_bucket else ""
        text = f"{edge.relation} {node.label}{suffix}"
        text_width, text_height = _measure_text(cv2, text, font, font_scale, text_thickness)
        x = max(0, canvas.shape[1] - right_padding - text_width)
        y = top_padding + (index * row_height) + text_height
        cv2.putText(
            canvas,
            text,
            (x, y),
            font,
            font_scale,
            (220, 240, 255),
            text_thickness,
            cv2.LINE_AA,
        )


def _draw_runtime_status(
    cv2,
    canvas: np.ndarray,
    *,
    slam_tracking_state: str | None,
    keyframe_id: int | None,
    mesh_triangle_count: int | None,
    mesh_vertex_count: int | None,
    graph_node_count: int | None = None,
    graph_edge_count: int | None = None,
    localized_node_count: int | None = None,
) -> None:
    if (
        slam_tracking_state is None
        and keyframe_id is None
        and mesh_triangle_count is None
        and mesh_vertex_count is None
        and graph_node_count is None
        and graph_edge_count is None
        and localized_node_count is None
    ):
        return

    lines = [
        f"SLAM {slam_tracking_state or '-'}",
        f"KF {keyframe_id if keyframe_id is not None else '-'}",
        f"Mesh {int(mesh_triangle_count or 0)}t / {int(mesh_vertex_count or 0)}v",
    ]
    if graph_node_count is not None or graph_edge_count is not None or localized_node_count is not None:
        lines.extend(
            [
                f"Graph nodes {int(graph_node_count or 0)}",
                f"Graph edges {int(graph_edge_count or 0)}",
                f"Localized {int(localized_node_count or 0)}",
            ]
        )
    x = 12
    y = max(20, canvas.shape[0] - 48)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + (index * 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    fps: float,
    *,
    segmentation_overlay_bgr: np.ndarray | None = None,
    segments: list[PanopticSegment] | None = None,
    segmentation_alpha: float = 0.35,
    tracking_ok: bool | None = None,
    is_keyframe: bool | None = None,
    segment_id: int | None = None,
    camera_name: str | None = None,
    camera_fallback_active: bool = False,
    slam_tracking_state: str | None = None,
    keyframe_id: int | None = None,
    mesh_triangle_count: int | None = None,
    mesh_vertex_count: int | None = None,
    loop_closure_applied: bool = False,
    scene_graph_snapshot: SceneGraphSnapshot | None = None,
    visible_graph_nodes=None,
    visible_graph_edges=None,
) -> np.ndarray:
    import cv2

    canvas = frame_bgr.copy()
    graph_nodes = list(visible_graph_nodes or scene_graph_snapshot.visible_nodes) if scene_graph_snapshot is not None else []
    graph_edges = list(visible_graph_edges or scene_graph_snapshot.visible_edges) if scene_graph_snapshot is not None else []
    graph_node_count = None
    graph_edge_count = None
    localized_node_count = None
    runtime_status_enabled = any(
        value is not None
        for value in (slam_tracking_state, keyframe_id, mesh_triangle_count, mesh_vertex_count)
    )
    if scene_graph_snapshot is not None and runtime_status_enabled:
        graph_node_count = sum(1 for node in graph_nodes if node.id != "ego")
        graph_edge_count = len(graph_edges)
        localized_node_count = sum(
            1 for node in graph_nodes if node.id != "ego" and node.world_centroid is not None
        )
    if (
        segmentation_overlay_bgr is not None
        and segmentation_overlay_bgr.shape == canvas.shape
        and segmentation_alpha > 0.0
    ):
        overlay = np.asarray(segmentation_overlay_bgr, dtype=np.uint8)
        active_mask = np.any(overlay != 0, axis=2)
        if np.any(active_mask):
            blended = canvas.astype(np.float32, copy=True)
            overlay_float = overlay.astype(np.float32, copy=False)
            blended[active_mask] = (
                blended[active_mask] * (1.0 - segmentation_alpha)
                + overlay_float[active_mask] * segmentation_alpha
            )
            canvas = blended.astype(np.uint8)

    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        color_bgr = tuple(int(channel) for channel in detection.color[::-1])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_bgr,
            2,
            cv2.LINE_AA,
        )

    _draw_runtime_status(
        cv2,
        canvas,
        slam_tracking_state=slam_tracking_state,
        keyframe_id=keyframe_id,
        mesh_triangle_count=mesh_triangle_count,
        mesh_vertex_count=mesh_vertex_count,
        graph_node_count=graph_node_count,
        graph_edge_count=graph_edge_count,
        localized_node_count=localized_node_count,
    )

    if segments:
        _draw_segmentation_legend(cv2, canvas, list(segments))
    _draw_scene_graph_summary(
        cv2,
        canvas,
        scene_graph_snapshot=scene_graph_snapshot,
        visible_graph_nodes=graph_nodes,
        visible_graph_edges=graph_edges,
    )
    return canvas


def highlight_detected_points(
    point_pixels: np.ndarray,
    point_colors: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    highlighted = point_colors.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        inside = (
            (point_pixels[:, 0] >= x1)
            & (point_pixels[:, 0] <= x2)
            & (point_pixels[:, 1] >= y1)
            & (point_pixels[:, 1] <= y2)
        )
        if np.any(inside):
            highlighted[inside] = np.asarray(detection.color, dtype=np.float32) / 255.0
    return highlighted


class Open3DMeshViewer:
    def __init__(self, window_name: str = "3D Reconstruction", o3d_module=None) -> None:
        if o3d_module is None:
            try:
                import open3d as o3d
            except ImportError as exc:  # pragma: no cover - depends on local install.
                raise RuntimeError("open3d is required to render the 3D mesh") from exc
        else:
            o3d = o3d_module

        self._o3d = o3d
        self._vis = o3d.visualization.Visualizer()
        window_created = self._vis.create_window(window_name=window_name, width=960, height=720)
        if not window_created:
            raise RuntimeError("failed to create Open3D window")
        self._mesh = o3d.geometry.TriangleMesh()
        self._graph_nodes = o3d.geometry.PointCloud()
        self._graph_edges = o3d.geometry.LineSet()
        self._ego_lines = o3d.geometry.LineSet()
        self._has_fitted_view = False
        self._vis.add_geometry(self._mesh)
        self._vis.add_geometry(self._graph_nodes)
        self._vis.add_geometry(self._graph_edges)
        self._vis.add_geometry(self._ego_lines)
        self._vis.poll_events()
        self._vis.update_renderer()

        render_option = self._vis.get_render_option()
        if render_option is not None:
            render_option.background_color = np.array([0.05, 0.05, 0.08], dtype=np.float64)
            show_back_face = getattr(render_option, "mesh_show_back_face", None)
            if show_back_face is not None:
                render_option.mesh_show_back_face = True

    def update(
        self,
        mesh_vertices_xyz: np.ndarray,
        mesh_triangles: np.ndarray,
        mesh_vertex_colors: np.ndarray,
        scene_graph_snapshot: SceneGraphSnapshot | None = None,
        *_unused,
    ) -> bool:
        display_vertices_xyz = _display_points_for_view(mesh_vertices_xyz)
        self._mesh.vertices = self._o3d.utility.Vector3dVector(
            display_vertices_xyz.astype(np.float64, copy=False)
        )
        self._mesh.triangles = self._o3d.utility.Vector3iVector(
            mesh_triangles.astype(np.int32, copy=False)
        )
        self._mesh.vertex_colors = self._o3d.utility.Vector3dVector(
            mesh_vertex_colors.astype(np.float64, copy=False)
        )
        compute_normals = getattr(self._mesh, "compute_vertex_normals", None)
        if callable(compute_normals):
            compute_normals()
        self._vis.update_geometry(self._mesh)
        self._update_scene_graph(scene_graph_snapshot)
        if not self._has_fitted_view and display_vertices_xyz.size > 0 and mesh_triangles.size > 0:
            self._vis.reset_view_point(True)
            self._has_fitted_view = True
        is_active = self._vis.poll_events()
        self._vis.update_renderer()
        return bool(is_active)

    def _update_scene_graph(self, scene_graph_snapshot: SceneGraphSnapshot | None) -> None:
        empty_points = np.empty((0, 3), dtype=np.float64)
        empty_lines = np.empty((0, 2), dtype=np.int32)
        if scene_graph_snapshot is None:
            self._graph_nodes.points = self._o3d.utility.Vector3dVector(empty_points)
            self._graph_nodes.colors = self._o3d.utility.Vector3dVector(empty_points)
            self._graph_edges.points = self._o3d.utility.Vector3dVector(empty_points)
            self._graph_edges.lines = self._o3d.utility.Vector2iVector(empty_lines)
            self._graph_edges.colors = self._o3d.utility.Vector3dVector(empty_points)
            self._ego_lines.points = self._o3d.utility.Vector3dVector(empty_points)
            self._ego_lines.lines = self._o3d.utility.Vector2iVector(empty_lines)
            self._ego_lines.colors = self._o3d.utility.Vector3dVector(empty_points)
            self._vis.update_geometry(self._graph_nodes)
            self._vis.update_geometry(self._graph_edges)
            self._vis.update_geometry(self._ego_lines)
            return

        nodes = {node.id: node for node in scene_graph_snapshot.visible_nodes}
        graph_points: list[np.ndarray] = []
        graph_colors: list[np.ndarray] = []
        for node in scene_graph_snapshot.visible_nodes:
            if node.id == "ego" or node.world_centroid is None:
                continue
            graph_points.append(np.asarray(node.world_centroid, dtype=np.float64))
            if node.type == "object":
                graph_colors.append(np.array([1.0, 0.6, 0.2], dtype=np.float64))
            else:
                graph_colors.append(np.array([0.7, 0.7, 0.7], dtype=np.float64))
        node_points_array = np.asarray(graph_points, dtype=np.float64).reshape(-1, 3)
        node_colors_array = np.asarray(graph_colors, dtype=np.float64).reshape(-1, 3)
        node_points_array = _display_points_for_view(node_points_array).astype(np.float64, copy=False)
        self._graph_nodes.points = self._o3d.utility.Vector3dVector(node_points_array)
        self._graph_nodes.colors = self._o3d.utility.Vector3dVector(node_colors_array)

        line_points: list[np.ndarray] = []
        line_indices: list[list[int]] = []
        line_colors: list[np.ndarray] = []
        point_index_by_node: dict[str, int] = {}

        def _point_index(node_id: str, xyz: np.ndarray) -> int:
            if node_id in point_index_by_node:
                return point_index_by_node[node_id]
            point_index_by_node[node_id] = len(line_points)
            line_points.append(np.asarray(xyz, dtype=np.float64))
            return point_index_by_node[node_id]

        relation_colors = {
            "on": np.array([0.2, 0.9, 0.2], dtype=np.float64),
            "inside": np.array([0.3, 0.8, 0.3], dtype=np.float64),
            "inside_region": np.array([0.3, 0.8, 0.3], dtype=np.float64),
            "attached_to": np.array([0.3, 0.8, 0.3], dtype=np.float64),
            "near": np.array([1.0, 0.55, 0.1], dtype=np.float64),
        }
        for edge in scene_graph_snapshot.visible_edges:
            source_xyz = (
                scene_graph_snapshot.camera_pose_world[:3, 3]
                if edge.source == "ego"
                else nodes.get(edge.source, None).world_centroid if edge.source in nodes else None
            )
            target_xyz = nodes.get(edge.target, None).world_centroid if edge.target in nodes else None
            if source_xyz is None or target_xyz is None:
                continue
            start_index = _point_index(edge.source, np.asarray(source_xyz, dtype=np.float64))
            end_index = _point_index(edge.target, np.asarray(target_xyz, dtype=np.float64))
            line_indices.append([start_index, end_index])
            line_colors.append(
                relation_colors.get(edge.relation, np.array([0.1, 0.9, 0.9], dtype=np.float64))
            )
        line_points_array = _display_points_for_view(
            np.asarray(line_points, dtype=np.float64).reshape(-1, 3)
        ).astype(np.float64, copy=False)
        self._graph_edges.points = self._o3d.utility.Vector3dVector(line_points_array)
        self._graph_edges.lines = self._o3d.utility.Vector2iVector(
            np.asarray(line_indices, dtype=np.int32).reshape(-1, 2)
        )
        self._graph_edges.colors = self._o3d.utility.Vector3dVector(
            np.asarray(line_colors, dtype=np.float64).reshape(-1, 3)
        )

        ego_points, ego_lines, ego_colors = _ego_frustum_geometry(scene_graph_snapshot.camera_pose_world)
        self._ego_lines.points = self._o3d.utility.Vector3dVector(ego_points)
        self._ego_lines.lines = self._o3d.utility.Vector2iVector(ego_lines)
        self._ego_lines.colors = self._o3d.utility.Vector3dVector(ego_colors)
        self._vis.update_geometry(self._graph_nodes)
        self._vis.update_geometry(self._graph_edges)
        self._vis.update_geometry(self._ego_lines)

    def close(self) -> None:
        self._vis.destroy_window()


Open3DPointCloudViewer = Open3DMeshViewer


def _ego_frustum_geometry(camera_pose_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.06, -0.04, 0.12],
            [0.06, -0.04, 0.12],
            [0.06, 0.04, 0.12],
            [-0.06, 0.04, 0.12],
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate(
        (local_points, np.ones((local_points.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    world_points = (np.asarray(camera_pose_world, dtype=np.float64) @ homogeneous.T).T[:, :3]
    world_points = _display_points_for_view(world_points).astype(np.float64, copy=False)
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    colors = np.tile(np.array([[0.2, 0.9, 0.9]], dtype=np.float64), (lines.shape[0], 1))
    return world_points, lines, colors
