from __future__ import annotations

import numpy as np

from obj_recog.types import Detection


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    fps: float,
    *,
    tracking_ok: bool | None = None,
    is_keyframe: bool | None = None,
    segment_id: int | None = None,
    camera_name: str | None = None,
    camera_fallback_active: bool = False,
    slam_tracking_state: str | None = None,
    keyframe_id: int | None = None,
    loop_closure_applied: bool = False,
) -> np.ndarray:
    import cv2

    canvas = frame_bgr.copy()
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
