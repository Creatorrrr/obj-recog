from __future__ import annotations

import numpy as np

from obj_recog.types import Detection, PanopticSegment


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


def _draw_runtime_status(
    cv2,
    canvas: np.ndarray,
    *,
    slam_tracking_state: str | None,
    keyframe_id: int | None,
    mesh_triangle_count: int | None,
    mesh_vertex_count: int | None,
) -> None:
    if (
        slam_tracking_state is None
        and keyframe_id is None
        and mesh_triangle_count is None
        and mesh_vertex_count is None
    ):
        return

    lines = [
        f"SLAM {slam_tracking_state or '-'}",
        f"KF {keyframe_id if keyframe_id is not None else '-'}",
        f"Mesh {int(mesh_triangle_count or 0)}t / {int(mesh_vertex_count or 0)}v",
    ]
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
) -> np.ndarray:
    import cv2

    canvas = frame_bgr.copy()
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
    )

    if segments:
        _draw_segmentation_legend(cv2, canvas, list(segments))
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
        self._has_fitted_view = False
        self._vis.add_geometry(self._mesh)
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
        *_unused,
    ) -> bool:
        self._mesh.vertices = self._o3d.utility.Vector3dVector(
            mesh_vertices_xyz.astype(np.float64, copy=False)
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
        if not self._has_fitted_view and mesh_vertices_xyz.size > 0 and mesh_triangles.size > 0:
            self._vis.reset_view_point(True)
            self._has_fitted_view = True
        is_active = self._vis.poll_events()
        self._vis.update_renderer()
        return bool(is_active)

    def close(self) -> None:
        self._vis.destroy_window()


Open3DPointCloudViewer = Open3DMeshViewer
