from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from obj_recog.blend_scene_loader import BlendSceneManifest, BlendSceneObject
from obj_recog.scene_graph import GraphEdge, GraphNode, SceneGraphSnapshot
from obj_recog.sim_protocol import EpisodePhase, OperatorSceneState
from obj_recog.sim_scene import build_interior_test_tv_scene_spec, build_living_room_scene_spec, build_scene_mesh_components
from obj_recog.types import (
    DepthDiagnostics,
    Detection,
    PanopticSegment,
    PerceptionDiagnostics,
    RenderSnapshot,
    TemporalStereoDiagnostics,
)
from obj_recog.visualization import (
    Open3DEnvironmentViewer,
    Open3DMeshViewer,
    _InteractiveViewState,
    _display_points_for_environment_view,
    _display_points_for_view,
    draw_detections,
    explanation_button_rect,
    highlight_detected_points,
    render_multiline_unicode_text,
    render_explanation_panel,
    runtime_window_position,
)


def test_highlight_detected_points_recolors_only_points_inside_detection() -> None:
    point_pixels = np.array(
        [
            [2, 2],
            [5, 5],
            [7, 1],
        ],
        dtype=np.int32,
    )
    point_colors = np.array(
        [
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
        ],
        dtype=np.float32,
    )
    detections = [
        Detection(
            xyxy=(1, 1, 6, 6),
            class_id=0,
            label="person",
            confidence=0.91,
            color=(255, 0, 0),
        )
    ]

    recolored = highlight_detected_points(point_pixels, point_colors, detections)

    assert np.allclose(recolored[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(recolored[1], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(recolored[2], point_colors[2])


class _FakeVector3dVector:
    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data)


class _FakePointCloud:
    def __init__(self) -> None:
        self.points = None
        self.colors = None


class _FakeLineSet:
    def __init__(self) -> None:
        self.points = None
        self.lines = None
        self.colors = None


class _FakeTriangleMesh:
    def __init__(self) -> None:
        self.vertices = None
        self.triangles = None
        self.vertex_colors = None
        self.normals_computed = False

    def compute_vertex_normals(self) -> None:
        self.normals_computed = True


class _FakeRenderOption:
    def __init__(self) -> None:
        self.background_color = None
        self.point_size = None
        self.mesh_show_back_face = None


class _FakeViewControl:
    def __init__(self) -> None:
        self.lookat = None
        self.front = None
        self.up = None
        self.zoom = None

    def set_lookat(self, value) -> None:
        self.lookat = np.asarray(value, dtype=np.float64)

    def set_front(self, value) -> None:
        self.front = np.asarray(value, dtype=np.float64)

    def set_up(self, value) -> None:
        self.up = np.asarray(value, dtype=np.float64)

    def set_zoom(self, value: float) -> None:
        self.zoom = float(value)


class _FakeVisualizer:
    def __init__(self) -> None:
        self.geometry = []
        self.reset_calls: list[bool] = []
        self.poll_count = 0
        self.renderer_updates = 0
        self.window_calls: list[dict[str, int | str]] = []
        self.view_control = _FakeViewControl()
        self.updated_geometries: list[object] = []
        self.render_option = _FakeRenderOption()

    def create_window(self, window_name: str, width: int, height: int, **kwargs) -> bool:
        call = {
            "window_name": window_name,
            "width": int(width),
            "height": int(height),
        }
        for key, value in kwargs.items():
            call[str(key)] = int(value)
        self.window_calls.append(call)
        return True

    def add_geometry(self, geometry) -> None:
        self.geometry.append(geometry)

    def get_render_option(self) -> _FakeRenderOption:
        return self.render_option

    def update_geometry(self, geometry) -> None:
        self.updated_geometries.append(geometry)
        return None

    def poll_events(self) -> bool:
        self.poll_count += 1
        return True

    def update_renderer(self) -> None:
        self.renderer_updates += 1

    def reset_view_point(self, reset_bounding_box: bool) -> None:
        self.reset_calls.append(reset_bounding_box)

    def get_view_control(self) -> _FakeViewControl:
        return self.view_control

    def destroy_window(self) -> None:
        return None


class _FakeIO:
    def __init__(self) -> None:
        self.read_calls: list[str] = []

    def read_triangle_mesh(self, path: str):
        self.read_calls.append(str(path))
        mesh = _FakeTriangleMesh()
        mesh.vertices = _FakeVector3dVector(np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]))
        mesh.triangles = _FakeVector3dVector(np.array([[0, 1, 2]], dtype=np.int32))
        mesh.vertex_colors = _FakeVector3dVector(np.array([[0.2, 0.2, 0.2]] * 3))
        return mesh


class _FakeO3D:
    def __init__(self) -> None:
        self.io = _FakeIO()

    class visualization:
        Visualizer = _FakeVisualizer

    class geometry:
        TriangleMesh = _FakeTriangleMesh
        PointCloud = _FakePointCloud
        LineSet = _FakeLineSet

    class utility:
        @staticmethod
        def Vector3dVector(data: np.ndarray) -> _FakeVector3dVector:
            return _FakeVector3dVector(data)

        @staticmethod
        def Vector3iVector(data: np.ndarray) -> _FakeVector3dVector:
            return _FakeVector3dVector(data)

        @staticmethod
        def Vector2iVector(data: np.ndarray) -> _FakeVector3dVector:
            return _FakeVector3dVector(data)


def _render_snapshot(
    *,
    geometry_revision: int,
    color_revision: int | None = None,
    color_value: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> RenderSnapshot:
    return RenderSnapshot(
        mesh_vertices_xyz=np.array(
            [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.2, 1.0]],
            dtype=np.float32,
        ),
        mesh_triangles=np.array([[0, 1, 2]], dtype=np.int32),
        mesh_vertex_colors=np.tile(np.array([color_value], dtype=np.float32), (3, 1)),
        mesh_geometry_revision=geometry_revision,
        mesh_color_revision=(geometry_revision if color_revision is None else color_revision),
    )


def test_draw_detections_renders_only_detection_labels() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        frame = np.zeros((12, 12, 3), dtype=np.uint8)
        draw_detections(
            frame,
            [
                Detection(
                    xyxy=(1, 1, 5, 5),
                    class_id=0,
                    label="person",
                    confidence=0.91,
                    color=(255, 0, 0),
                )
            ],
            24.0,
            tracking_ok=True,
            is_keyframe=True,
            segment_id=2,
            camera_name="iPhone",
            camera_fallback_active=True,
            slam_tracking_state="TRACKING",
            keyframe_id=7,
            loop_closure_applied=True,
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert fake_cv2.text_calls == [
        "person 0.91",
        "SLAM TRACKING",
        "KF 7",
        "Mesh 0t / 0v",
        "Stereo off",
    ]


def test_draw_detections_renders_small_runtime_status_lines() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        draw_detections(
            frame,
            [],
            24.0,
            slam_tracking_state="TRACKING",
            keyframe_id=7,
            mesh_triangle_count=123,
            mesh_vertex_count=88,
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert fake_cv2.text_calls == [
        "SLAM TRACKING",
        "KF 7",
        "Mesh 123t / 88v",
        "Stereo off",
    ]


def test_draw_detections_renders_explanation_status_line() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        draw_detections(
            frame,
            [],
            24.0,
            slam_tracking_state="TRACKING",
            keyframe_id=7,
            mesh_triangle_count=123,
            mesh_vertex_count=88,
            explanation_status="ready",
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert "Explain: ready" in fake_cv2.text_calls
    assert "Explain OFF" in fake_cv2.text_calls


def test_draw_detections_renders_explanation_toggle_on_label() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        draw_detections(
            frame,
            [],
            24.0,
            slam_tracking_state="TRACKING",
            keyframe_id=7,
            mesh_triangle_count=123,
            mesh_vertex_count=88,
            explanation_status="ready",
            explanation_auto_refresh_enabled=True,
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert "Explain ON" in fake_cv2.text_calls


def test_draw_detections_renders_depth_debug_overlay_lines() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        diagnostics = DepthDiagnostics(
            calibration_source="explicit",
            profile="balanced",
            raw_percentiles=(0.2, 0.8, 1.4),
            normalizer_low_high=(0.1, 1.6),
            normalized_distance_percentiles=(1.2, 2.1, 4.4),
            valid_depth_ratio=0.92,
            dense_z_span=2.4,
            mesh_z_span=1.7,
            intrinsics_summary=(575.0, 575.0, 320.0, 180.0),
            hint="depth normalization compression likely",
        )
        draw_detections(
            frame,
            [],
            24.0,
            slam_tracking_state="TRACKING",
            keyframe_id=7,
            mesh_triangle_count=123,
            mesh_vertex_count=88,
            depth_diagnostics=diagnostics,
            depth_debug_level="basic",
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert "Calib explicit" in fake_cv2.text_calls
    assert "Depth balanced" in fake_cv2.text_calls
    assert "Dist 1.20/2.10/4.40m" in fake_cv2.text_calls
    assert "Mesh z 1.70m" in fake_cv2.text_calls
    assert "Hint depth normalization compression likely" in fake_cv2.text_calls


def test_draw_detections_renders_detailed_depth_debug_lines() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        diagnostics = DepthDiagnostics(
            calibration_source="disabled/approx",
            profile="depthy",
            raw_percentiles=(0.1, 0.7, 1.8),
            normalizer_low_high=(0.05, 1.9),
            normalized_distance_percentiles=(1.0, 2.9, 5.4),
            valid_depth_ratio=0.85,
            dense_z_span=3.2,
            mesh_z_span=2.2,
            intrinsics_summary=(520.0, 518.0, 320.0, 180.0),
            hint="approx intrinsics in use",
        )
        draw_detections(
            frame,
            [],
            24.0,
            depth_diagnostics=diagnostics,
            depth_debug_level="detailed",
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert "Raw 0.10/0.70/1.80" in fake_cv2.text_calls
    assert "Norm 0.05/1.90" in fake_cv2.text_calls
    assert "Dense/Mesh 3.20/2.20m" in fake_cv2.text_calls
    assert "Valid 85.0%" in fake_cv2.text_calls
    assert "fx/fy 520.0/518.0" in fake_cv2.text_calls


def test_explanation_button_rect_is_bottom_right() -> None:
    rect = explanation_button_rect(frame_width=640, frame_height=480)

    assert rect[0] > 640 // 2
    assert rect[1] > 480 // 2
    assert rect[2] == 640 - 12
    assert rect[3] == 480 - 12


def test_draw_detections_blends_segmentation_overlay_before_boxes() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            return None

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        overlay = np.full((2, 2, 3), (20, 40, 80), dtype=np.uint8)
        blended = draw_detections(
            frame,
            [],
            24.0,
            segmentation_overlay_bgr=overlay,
            segmentation_alpha=0.5,
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert np.array_equal(blended[0, 0], np.array([10, 20, 40], dtype=np.uint8))


def test_draw_detections_renders_segmentation_legend_in_top_right() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[tuple[str, tuple[int, int]]] = []
            self.rectangle_calls: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            self.rectangle_calls.append((pt1, pt2, color, thickness))
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append((text, org))
            return None

        def getTextSize(self, text, font, scale, thickness):
            return ((len(text) * 7, 12), 0)

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        frame = np.zeros((80, 160, 3), dtype=np.uint8)
        segments = [
            PanopticSegment(
                segment_id=1,
                label_id=7,
                label="wall",
                color_rgb=(64, 196, 255),
                mask=np.ones((80, 160), dtype=bool),
                area_pixels=6000,
            ),
            PanopticSegment(
                segment_id=2,
                label_id=4,
                label="chair",
                color_rgb=(255, 99, 71),
                mask=np.ones((80, 160), dtype=bool),
                area_pixels=3000,
            ),
        ]
        draw_detections(
            frame,
            [],
            24.0,
            segments=segments,
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert [text for text, _org in fake_cv2.text_calls] == ["wall", "chair"]
    assert all(org[0] >= 100 for _text, org in fake_cv2.text_calls)
    assert len(fake_cv2.rectangle_calls) >= 2


def test_draw_detections_renders_scene_graph_summary_in_top_right() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[tuple[str, tuple[int, int]]] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append((text, org))
            return None

        def getTextSize(self, text, font, scale, thickness):
            return ((len(text) * 7, 12), 0)

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        frame = np.zeros((80, 200, 3), dtype=np.uint8)
        snapshot = SceneGraphSnapshot(
            frame_index=2,
            camera_pose_world=np.eye(4, dtype=np.float32),
            nodes=(
                GraphNode(
                    id="ego",
                    type="ego",
                    label="camera",
                    state="visible",
                    confidence=1.0,
                    world_centroid=np.zeros(3, dtype=np.float32),
                    last_seen_frame=2,
                    last_seen_direction="front",
                    source_track_id=None,
                ),
                GraphNode(
                    id="obj_table_1",
                    type="object",
                    label="table",
                    state="visible",
                    confidence=0.91,
                    world_centroid=np.array([0.0, 0.0, 2.0], dtype=np.float32),
                    last_seen_frame=2,
                    last_seen_direction="front",
                    source_track_id=1,
                ),
                GraphNode(
                    id="obj_cup_2",
                    type="object",
                    label="cup",
                    state="visible",
                    confidence=0.88,
                    world_centroid=np.array([0.5, 0.0, 2.2], dtype=np.float32),
                    last_seen_frame=2,
                    last_seen_direction="front-right",
                    source_track_id=2,
                ),
            ),
            edges=(
                GraphEdge(
                    source="ego",
                    target="obj_table_1",
                    relation="front",
                    confidence=0.9,
                    last_updated_frame=2,
                    distance_bucket="mid",
                    source_kind="detection",
                ),
                GraphEdge(
                    source="ego",
                    target="obj_cup_2",
                    relation="front-right",
                    confidence=0.8,
                    last_updated_frame=2,
                    distance_bucket="mid",
                    source_kind="detection",
                ),
            ),
            visible_node_ids=("ego", "obj_table_1", "obj_cup_2"),
            visible_edge_keys=(
                ("ego", "obj_table_1", "front"),
                ("ego", "obj_cup_2", "front-right"),
            ),
        )
        draw_detections(
            frame,
            [],
            24.0,
            scene_graph_snapshot=snapshot,
            visible_graph_nodes=list(snapshot.visible_nodes),
            visible_graph_edges=list(snapshot.visible_edges),
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert [text for text, _org in fake_cv2.text_calls] == [
        "front table (mid)",
        "front-right cup (mid)",
    ]
    assert all(org[0] >= 20 for _text, org in fake_cv2.text_calls)


def test_draw_detections_renders_scene_graph_debug_counts_in_runtime_status() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        snapshot = SceneGraphSnapshot(
            frame_index=4,
            camera_pose_world=np.eye(4, dtype=np.float32),
            nodes=(
                GraphNode(
                    id="ego",
                    type="ego",
                    label="camera",
                    state="visible",
                    confidence=1.0,
                    world_centroid=np.zeros(3, dtype=np.float32),
                    last_seen_frame=4,
                    last_seen_direction="front",
                    source_track_id=None,
                ),
                GraphNode(
                    id="obj_cup_1",
                    type="object",
                    label="cup",
                    state="visible",
                    confidence=0.9,
                    world_centroid=np.array([0.2, 0.0, 1.5], dtype=np.float32),
                    last_seen_frame=4,
                    last_seen_direction="front-right",
                    source_track_id=1,
                ),
                GraphNode(
                    id="seg_wall_2",
                    type="segment",
                    label="wall",
                    state="visible",
                    confidence=0.8,
                    world_centroid=None,
                    last_seen_frame=4,
                    last_seen_direction="left",
                    source_track_id=2,
                ),
            ),
            edges=(
                GraphEdge(
                    source="ego",
                    target="obj_cup_1",
                    relation="front-right",
                    confidence=0.85,
                    last_updated_frame=4,
                    distance_bucket="near",
                    source_kind="detection",
                ),
            ),
            visible_node_ids=("ego", "obj_cup_1", "seg_wall_2"),
            visible_edge_keys=(("ego", "obj_cup_1", "front-right"),),
        )
        draw_detections(
            frame,
            [],
            30.0,
            slam_tracking_state="TRACKING",
            keyframe_id=9,
            mesh_triangle_count=20,
            mesh_vertex_count=15,
            scene_graph_snapshot=snapshot,
            visible_graph_nodes=list(snapshot.visible_nodes),
            visible_graph_edges=list(snapshot.visible_edges),
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert fake_cv2.text_calls == [
        "SLAM TRACKING",
        "KF 9",
        "Mesh 20t / 15v",
        "Graph nodes 2",
        "Graph edges 1",
        "Localized 1",
        "Stereo off",
        "front-right cup (near)",
    ]


def test_render_explanation_panel_renders_status_and_truncates_wrapped_body() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

        def getTextSize(self, text, font, scale, thickness):
            return ((len(text) * 7, 12), 0)

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        panel = render_explanation_panel(
            status="ready",
            text=(
                "현재 장면 설명이 길게 이어집니다. " * 20
                + "\n핵심 객체: 컵, 탁자\n공간 관계: 컵은 탁자 위\n불확실성: 중간"
            ),
            model="gpt-4.1-mini",
            latency_ms=321.0,
            timestamp_label="12:34:56",
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

        assert panel.shape[0] >= 420
        assert fake_cv2.text_calls[0].startswith("Status: READY")
        assert any(text.startswith("Model: gpt-4.1-mini") for text in fake_cv2.text_calls)
        assert any(text.startswith("Latency: 321") for text in fake_cv2.text_calls)
        footer_lines = [text for text in fake_cv2.text_calls if text.startswith("Wheel: scroll | Lines ")]
        assert len(footer_lines) <= 1


def test_render_explanation_panel_uses_injected_cv2_module() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

    fake_cv2 = _FakeCV2()

    panel = render_explanation_panel(
        status="idle",
        text="",
        model="gpt-4.1-mini",
        latency_ms=None,
        timestamp_label="12:34:56",
        cv2_module=fake_cv2,
    )

    assert panel.shape[1] >= 320
    assert fake_cv2.text_calls[0].startswith("Status: IDLE")


def test_render_explanation_panel_uses_unicode_text_renderer_for_body_lines() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

        def getTextSize(self, text, font, scale, thickness):
            return ((len(text) * 7, 12), 0)

    unicode_calls: list[list[str]] = []

    def _fake_unicode_renderer(canvas, lines, *, origin, line_height, color):
        unicode_calls.append(list(lines))
        return canvas

    fake_cv2 = _FakeCV2()
    panel = render_explanation_panel(
        status="ready",
        text="현재 장면 설명\n핵심 객체: 컵\n공간 관계: 컵은 앞쪽\n불확실성: 낮음",
        model="gpt-5-mini",
        latency_ms=12.0,
        timestamp_label="12:34:56",
        refresh_status="updating",
        cv2_module=fake_cv2,
        unicode_text_renderer=_fake_unicode_renderer,
    )

    assert panel.shape[0] >= 240
    assert fake_cv2.text_calls[:4] == [
        "Status: READY | 12:34:56",
        "Refresh: updating",
        "Model: gpt-5-mini",
        "Latency: 12ms",
    ]
    assert unicode_calls == [[
        "현재 장면 설명",
        "핵심 객체: 컵",
        "공간 관계: 컵은 앞쪽",
        "불확실성: 낮음",
    ]]


def test_render_explanation_panel_includes_request_context_and_uses_taller_default_panel() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

        def getTextSize(self, text, font, scale, thickness):
            return ((len(text) * 7, 12), 0)

    unicode_calls: list[list[str]] = []

    def _fake_unicode_renderer(canvas, lines, *, origin, line_height, color):
        unicode_calls.append(list(lines))
        return canvas

    fake_cv2 = _FakeCV2()
    panel = render_explanation_panel(
        status="loading",
        text="response summary",
        model="gpt-5-mini",
        latency_ms=18.0,
        timestamp_label="12:34:56",
        request_context="Visible objects\n- chair\n- table",
        cv2_module=fake_cv2,
        unicode_text_renderer=_fake_unicode_renderer,
    )

    assert panel.shape[0] >= 1080
    assert "Request chars: 31" in fake_cv2.text_calls
    assert "Response chars: 16" in fake_cv2.text_calls
    assert unicode_calls
    assert unicode_calls[-1] == ["response summary"]


def test_render_explanation_panel_supports_planner_tabs_and_returns_tab_metadata() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

        def getTextSize(self, text, font, scale, thickness):
            return ((len(text) * 7, 12), 0)

    unicode_calls: list[list[str]] = []

    def _fake_unicode_renderer(canvas, lines, *, origin, line_height, color):
        unicode_calls.append(list(lines))
        return canvas

    fake_cv2 = _FakeCV2()
    panel, metadata = render_explanation_panel(
        status="ready",
        text="explanation response",
        model="gpt-5-mini",
        latency_ms=22.0,
        timestamp_label="12:34:56",
        planner_request_context='{"goal":"tv","mode":"inferred"}',
        planner_response_text='{"behavior_mode":"scan","commands":[{"kind":"rotate_body","direction":"left","mode":"angle_deg","value":12.0}]}',
        active_tab="planner_request",
        cv2_module=fake_cv2,
        unicode_text_renderer=_fake_unicode_renderer,
        return_metadata=True,
    )

    assert panel.shape[0] >= 1080
    assert metadata["active_tab"] == "planner_request"
    assert "planner_request" in metadata["tab_rects"]
    assert "planner_response" in metadata["tab_rects"]
    assert unicode_calls
    assert unicode_calls[-1][0] == "Raw request"
    assert any(text == "Raw Request" for text in fake_cv2.text_calls)


def test_render_explanation_panel_shows_ready_empty_response_message() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            return None

    unicode_calls: list[list[str]] = []

    def _fake_unicode_renderer(canvas, lines, *, origin, line_height, color):
        unicode_calls.append(list(lines))
        return canvas

    panel = render_explanation_panel(
        status="ready",
        text="",
        model="gpt-5-mini",
        latency_ms=42.0,
        timestamp_label="12:34:56",
        cv2_module=_FakeCV2(),
        unicode_text_renderer=_fake_unicode_renderer,
    )

    assert panel.shape[1] >= 320
    assert unicode_calls == [["모델이 비어 있는 설명을 반환했습니다. 다시 시도해 주세요."]]


def test_render_explanation_panel_renders_refresh_status_line() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

    fake_cv2 = _FakeCV2()
    render_explanation_panel(
        status="ready",
        text="기존 분석",
        model="gpt-5-mini",
        latency_ms=42.0,
        timestamp_label="12:34:56",
        refresh_status="failed",
        cv2_module=fake_cv2,
        unicode_text_renderer=lambda canvas, lines, **_: canvas,
    )

    assert "Refresh: failed" in fake_cv2.text_calls


def test_render_explanation_panel_supports_scroll_offset_and_returns_layout_metadata() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

        def getTextSize(self, text, font, scale, thickness):
            return ((len(text) * 7, 12), 0)

    unicode_calls: list[list[str]] = []

    def _fake_unicode_renderer(canvas, lines, *, origin, line_height, color):
        unicode_calls.append(list(lines))
        return canvas

    fake_cv2 = _FakeCV2()
    lines = [f"line {index:02d}" for index in range(1, 19)]
    panel, metadata = render_explanation_panel(
        status="ready",
        text="\n".join(lines),
        model="gpt-5-mini",
        latency_ms=12.0,
        timestamp_label="12:34:56",
        width=960,
        height=360,
        scroll_offset=4,
        cv2_module=fake_cv2,
        unicode_text_renderer=_fake_unicode_renderer,
        return_metadata=True,
    )

    assert panel.shape[1] >= 960
    assert metadata["scroll_offset"] == 4
    assert metadata["max_scroll_offset"] > 0
    assert metadata["up_rect"] is not None
    assert metadata["down_rect"] is not None
    assert metadata["scrollbar_rect"] is not None
    assert unicode_calls
    assert unicode_calls[-1][0] == "line 05"


def test_render_multiline_unicode_text_returns_canvas_when_pil_unavailable() -> None:
    canvas = np.zeros((40, 100, 3), dtype=np.uint8)

    output = render_multiline_unicode_text(
        canvas,
        ["현재 장면 설명"],
        origin=(10, 10),
        line_height=18,
        color=(255, 255, 255),
        image_module=None,
        draw_module=None,
        font_module=None,
    )

    assert output.shape == canvas.shape


def test_open3d_viewer_resets_view_on_first_non_empty_update_only() -> None:
    viewer = Open3DMeshViewer(o3d_module=_FakeO3D())
    vertices = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    viewer.update(vertices, triangles, colors)
    viewer.update(vertices, triangles, colors)

    assert viewer._vis.reset_calls == [True]
    assert len(viewer._vis.geometry) == 5


def test_open3d_viewer_applies_expected_initial_camera_on_first_mesh() -> None:
    viewer = Open3DMeshViewer(o3d_module=_FakeO3D())
    vertices = np.array(
        [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.2, 1.0], [0.2, 0.2, 1.2]],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    colors = np.array([[1.0, 0.0, 0.0]] * 4, dtype=np.float32)

    viewer.update(vertices, triangles, colors)

    np.testing.assert_allclose(viewer._vis.view_control.lookat, np.array([0.1, -0.1, -1.1]))
    np.testing.assert_allclose(viewer._vis.view_control.front, np.array([-0.58, 0.48, 0.66]))
    np.testing.assert_allclose(viewer._vis.view_control.up, np.array([0.0, 1.0, 0.0]))
    assert viewer._vis.view_control.zoom == pytest.approx(0.68)


def test_open3d_viewer_skips_mesh_upload_when_revision_is_unchanged() -> None:
    viewer = Open3DMeshViewer(o3d_module=_FakeO3D())
    vertices = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.2, 1.0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[1.0, 0.0, 0.0]] * 3, dtype=np.float32)

    viewer.update(vertices, triangles, colors, None, 4)
    viewer.update(vertices, triangles, colors, None, 4)

    mesh_updates = [geometry for geometry in viewer._vis.updated_geometries if geometry is viewer._mesh]
    assert len(mesh_updates) == 1


def test_open3d_viewer_submit_keeps_only_latest_snapshot_until_apply_due() -> None:
    viewer = Open3DMeshViewer(o3d_module=_FakeO3D(), tick_hz=30.0, apply_hz=12.0)
    first = _render_snapshot(geometry_revision=1, color_value=(1.0, 0.0, 0.0))
    second = _render_snapshot(geometry_revision=2, color_value=(0.0, 1.0, 0.0))

    viewer.submit(first)
    viewer.submit(second)

    assert viewer.apply_latest_if_due(now=0.0) is True
    assert viewer.tick(now=0.0) is True
    assert viewer._last_mesh_geometry_revision == 2
    assert np.allclose(
        viewer._mesh.vertex_colors.data,
        np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (3, 1)),
    )


def test_open3d_viewer_apply_latest_if_due_waits_for_apply_interval() -> None:
    viewer = Open3DMeshViewer(o3d_module=_FakeO3D(), tick_hz=30.0, apply_hz=12.0)

    viewer.submit(_render_snapshot(geometry_revision=1, color_value=(1.0, 0.0, 0.0)))
    assert viewer.apply_latest_if_due(now=0.0) is True
    assert viewer._last_mesh_geometry_revision == 1

    viewer.submit(_render_snapshot(geometry_revision=2, color_value=(0.0, 1.0, 0.0)))
    assert viewer.apply_latest_if_due(now=0.02) is False
    assert viewer._last_mesh_geometry_revision == 1

    assert viewer.apply_latest_if_due(now=0.09) is True
    assert viewer._last_mesh_geometry_revision == 2


def test_open3d_viewer_tick_waits_for_tick_interval() -> None:
    viewer = Open3DMeshViewer(o3d_module=_FakeO3D(), tick_hz=30.0, apply_hz=12.0)
    initial_poll_count = viewer._vis.poll_count

    assert viewer.tick(now=0.0) is True
    first_tick_poll_count = viewer._vis.poll_count
    assert first_tick_poll_count == initial_poll_count + 1

    assert viewer.tick(now=0.01) is True
    assert viewer._vis.poll_count == first_tick_poll_count

    assert viewer.tick(now=0.04) is True
    assert viewer._vis.poll_count == first_tick_poll_count + 1


def test_runtime_window_position_uses_primary_frame_size_grid() -> None:
    assert runtime_window_position("Object Recognition", primary_width=640, primary_height=360) == (32, 48)
    assert runtime_window_position("Environment Model", primary_width=640, primary_height=360) == (704, 48)
    assert runtime_window_position("Situation Explanation", primary_width=640, primary_height=360) == (32, 440)
    assert runtime_window_position("3D Reconstruction", primary_width=640, primary_height=360) == (704, 440)


def test_open3d_viewer_uses_runtime_window_layout_for_initial_position() -> None:
    viewer = Open3DMeshViewer(
        o3d_module=_FakeO3D(),
        layout_primary_width=640,
        layout_primary_height=360,
    )

    assert viewer._vis.window_calls == [
        {
            "window_name": "3D Reconstruction",
            "width": 640,
            "height": 480,
            "left": 704,
            "top": 440,
        }
    ]
    assert viewer._vis.render_option.mesh_show_back_face is False


def test_environment_viewer_uses_environment_window_layout_for_initial_position() -> None:
    viewer = Open3DEnvironmentViewer(
        o3d_module=_FakeO3D(),
        layout_primary_width=640,
        layout_primary_height=360,
    )

    assert viewer._vis.window_calls == [
        {
            "window_name": "Environment Model",
            "width": 960,
            "height": 720,
            "left": 704,
            "top": 48,
        }
    ]


def test_environment_viewer_renders_room_object_meshes_and_camera_marker() -> None:
    fake_o3d = _FakeO3D()
    viewer = Open3DEnvironmentViewer(o3d_module=fake_o3d)
    scenario_state = SimpleNamespace(
        room_width_m=6.0,
        room_depth_m=8.0,
        room_height_m=3.0,
        environment_objects=(
            {
                "label": "chair",
                "asset_id": "chair_modern",
                "center_world": (-1.1, 0.55, 4.4),
                "size_xyz": (0.7, 1.1, 0.7),
                "yaw_deg": 15.0,
                "color_bgr": (60, 180, 120),
                "preview_mesh_path": "/tmp/chair_modern.ply",
                "target_role": False,
                "visible": True,
            },
            {
                "label": "backpack",
                "asset_id": "backpack_canvas",
                "center_world": (1.5, 0.45, 5.4),
                "size_xyz": (0.8, 0.9, 0.8),
                "yaw_deg": 0.0,
                "color_bgr": (40, 80, 225),
                "preview_mesh_path": "/tmp/backpack_canvas.ply",
                "target_role": True,
                "visible": True,
            },
        ),
        rig_x=0.0,
        rig_y=1.6,
        rig_z=0.0,
        yaw_deg=-25.0,
    )

    is_active = viewer.update(scenario_state)

    assert is_active is True
    assert len(viewer._vis.geometry) == 3
    assert viewer._room_mesh.vertices.data.shape[0] > 0
    assert viewer._room_mesh.triangles.data.shape[0] > 0
    assert viewer._object_mesh.vertices.data.shape[0] > 0
    assert viewer._object_mesh.triangles.data.shape[0] > 0
    assert viewer._camera_lines.points.data.shape[0] > 0
    assert viewer._camera_lines.lines.data.shape[0] > 0
    assert viewer._vis.reset_calls == [True]
    assert fake_o3d.io.read_calls == ["/tmp/chair_modern.ply", "/tmp/backpack_canvas.ply"]


def test_environment_viewer_includes_window_frame_in_structural_room_mesh_for_scene_spec() -> None:
    scene = build_living_room_scene_spec()
    components = build_scene_mesh_components(scene)
    expected_room_vertices = sum(
        int(component.vertices_xyz.shape[0])
        for component in components
        if component.semantic_label in {"floor", "wall", "ceiling", "window_frame"}
    )
    expected_object_vertices = sum(
        int(component.vertices_xyz.shape[0])
        for component in components
        if component.semantic_label not in {"floor", "wall", "ceiling", "window_frame", "glass"}
    )

    viewer = Open3DEnvironmentViewer(o3d_module=_FakeO3D())
    viewer.update(
        OperatorSceneState(
            scene_spec=scene,
            robot_pose=scene.start_pose,
            phase=EpisodePhase.SELF_CALIBRATING,
        )
    )

    assert viewer._room_mesh.vertices.data.shape[0] >= expected_room_vertices
    assert viewer._object_mesh.vertices.data.shape[0] == expected_object_vertices


def test_environment_viewer_excludes_glass_from_solid_mesh_for_scene_spec() -> None:
    scene = build_living_room_scene_spec()
    components = build_scene_mesh_components(scene)
    glass_vertices = sum(
        int(component.vertices_xyz.shape[0])
        for component in components
        if component.semantic_label == "glass"
    )
    solid_object_vertices = sum(
        int(component.vertices_xyz.shape[0])
        for component in components
        if component.semantic_label not in {"floor", "wall", "ceiling", "window_frame", "glass"}
    )

    viewer = Open3DEnvironmentViewer(o3d_module=_FakeO3D())
    viewer.update(
        OperatorSceneState(
            scene_spec=scene,
            robot_pose=scene.start_pose,
            phase=EpisodePhase.SELF_CALIBRATING,
        )
    )

    assert glass_vertices > 0
    assert viewer._object_mesh.vertices.data.shape[0] == solid_object_vertices


def test_environment_viewer_adds_natural_backdrop_geometry_for_scene_spec() -> None:
    scene = build_living_room_scene_spec()
    components = build_scene_mesh_components(scene)
    structural_room_vertices = sum(
        int(component.vertices_xyz.shape[0])
        for component in components
        if component.semantic_label in {"floor", "wall", "ceiling", "window_frame"}
    )

    viewer = Open3DEnvironmentViewer(o3d_module=_FakeO3D())
    viewer.update(
        OperatorSceneState(
            scene_spec=scene,
            robot_pose=scene.start_pose,
            phase=EpisodePhase.SELF_CALIBRATING,
        )
    )

    assert viewer._room_mesh.vertices.data.shape[0] > structural_room_vertices


def test_environment_viewer_backdrop_uses_sky_grass_and_tree_palette_for_scene_spec() -> None:
    scene = build_living_room_scene_spec()
    viewer = Open3DEnvironmentViewer(o3d_module=_FakeO3D())
    viewer.update(
        OperatorSceneState(
            scene_spec=scene,
            robot_pose=scene.start_pose,
            phase=EpisodePhase.SELF_CALIBRATING,
        )
    )

    colors = np.asarray(viewer._room_mesh.vertex_colors.data, dtype=np.float64).reshape(-1, 3)

    def has_color(target_rgb: tuple[float, float, float]) -> bool:
        target = np.asarray(target_rgb, dtype=np.float64).reshape(1, 3)
        return bool(np.any(np.all(np.isclose(colors, target, atol=1e-3), axis=1)))

    assert has_color((0.83, 0.92, 0.99))
    assert has_color((0.32, 0.58, 0.28))
    assert has_color((0.22, 0.42, 0.18))


def test_environment_viewer_skips_synthetic_backdrop_for_authored_scene_spec() -> None:
    fake_o3d = _FakeO3D()
    viewer = Open3DEnvironmentViewer(o3d_module=fake_o3d)
    manifest = BlendSceneManifest(
        blend_file_path="/Users/chasoik/Downloads/InteriorTest.blend",
        room_size_xyz=(5.0, 3.0, 8.0),
        objects=(
            BlendSceneObject(
                object_id="Floor",
                object_type="MESH",
                semantic_label="floor",
                center_xyz=(0.0, 0.0, 0.0),
                size_xyz=(5.0, 0.01, 8.0),
                yaw_deg=0.0,
                vertices_xyz=np.asarray(
                    [[-2.5, 0.0, -4.0], [2.5, 0.0, -4.0], [2.5, 0.0, 4.0], [-2.5, 0.0, 4.0]],
                    dtype=np.float32,
                ),
                triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                collider=False,
            ),
            BlendSceneObject(
                object_id="Wall.001",
                object_type="MESH",
                semantic_label="wall",
                center_xyz=(0.0, 1.25, 3.95),
                size_xyz=(5.0, 2.5, 0.1),
                yaw_deg=0.0,
                vertices_xyz=np.asarray(
                    [[-2.5, 0.0, 3.9], [2.5, 0.0, 3.9], [2.5, 2.5, 4.0], [-2.5, 2.5, 4.0]],
                    dtype=np.float32,
                ),
                triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                collider=True,
            ),
        ),
    )
    scene = build_interior_test_tv_scene_spec(manifest)

    viewer.update(
        OperatorSceneState(
            scene_spec=scene,
            robot_pose=scene.start_pose,
            phase=EpisodePhase.SELF_CALIBRATING,
        )
    )

    assert viewer._room_mesh.vertices.data.shape[0] == 8


def test_environment_viewer_keeps_floor_below_ceiling_in_display_coordinates() -> None:
    scene = build_living_room_scene_spec()
    components = build_scene_mesh_components(scene)
    floor = next(component for component in components if component.component_id == "floor")
    ceiling = next(component for component in components if component.component_id == "ceiling")

    floor_display = _display_points_for_environment_view(floor.vertices_xyz)
    ceiling_display = _display_points_for_environment_view(ceiling.vertices_xyz)

    assert float(floor_display[:, 1].mean()) < float(ceiling_display[:, 1].mean())


def test_draw_detections_renders_perception_diagnostics_status_lines() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        draw_detections(
            np.zeros((16, 16, 3), dtype=np.uint8),
            [],
            10.0,
            slam_tracking_state="TRACKING",
            keyframe_id=1,
            perception_diagnostics=PerceptionDiagnostics(
                perception_mode="assisted",
                detection_source="runtime+fallback",
                depth_source="ground_truth",
                pose_source="ground_truth",
                gt_target_visible=True,
                benchmark_valid=False,
            ),
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert "Perception assisted" in fake_cv2.text_calls
    assert "Detect runtime+fallback" in fake_cv2.text_calls
    assert "Depth/Pose ground_truth/ground_truth" in fake_cv2.text_calls
    assert "Benchmark invalid GT-visible yes" in fake_cv2.text_calls


def test_draw_detections_renders_temporal_stereo_status_lines() -> None:
    class _FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def rectangle(self, canvas, pt1, pt2, color, thickness):
            return None

        def putText(self, canvas, text, org, font, scale, color, thickness, line_type):
            self.text_calls.append(text)
            return None

    import sys

    fake_cv2 = _FakeCV2()
    previous_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        draw_detections(
            np.zeros((16, 16, 3), dtype=np.uint8),
            [],
            10.0,
            slam_tracking_state="TRACKING",
            keyframe_id=1,
            temporal_stereo_diagnostics=TemporalStereoDiagnostics(
                enabled=True,
                applied=False,
                reference_keyframe_id=7,
                coverage_ratio=0.125,
                median_disparity_px=2.5,
                fit_sample_count=320,
                fit_rmse=0.28,
                fallback_reason="low_stereo_coverage",
            ),
        )
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2

    assert "Stereo on" in fake_cv2.text_calls
    assert "Stereo ref 7" in fake_cv2.text_calls
    assert "Stereo cov 12.5%" in fake_cv2.text_calls
    assert "Stereo fallback low_stereo_coverage" in fake_cv2.text_calls


def test_display_points_for_view_flips_y_and_z_axes() -> None:
    points = np.array(
        [
            [1.0, 2.0, 3.0],
            [-4.0, -5.0, 6.0],
        ],
        dtype=np.float32,
    )

    transformed = _display_points_for_view(points)

    assert np.allclose(
        transformed,
        np.array(
            [
                [1.0, -2.0, -3.0],
                [-4.0, 5.0, -6.0],
            ],
            dtype=np.float32,
        ),
    )


def test_display_points_for_environment_view_preserves_height_axis() -> None:
    points = np.array(
        [
            [1.0, 2.0, 3.0],
            [-4.0, -5.0, 6.0],
        ],
        dtype=np.float32,
    )

    transformed = _display_points_for_environment_view(points)

    assert np.allclose(
        transformed,
        np.array(
            [
                [1.0, 2.0, -3.0],
                [-4.0, -5.0, -6.0],
            ],
            dtype=np.float32,
        ),
    )


def test_open3d_viewer_applies_display_axis_transform_to_mesh_vertices() -> None:
    viewer = Open3DMeshViewer(o3d_module=_FakeO3D())
    vertices = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, 4.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    viewer.update(vertices, triangles, colors)

    assert np.allclose(
        viewer._mesh.vertices.data,
        np.array([[1.0, -2.0, -3.0], [-1.0, 2.0, -4.0], [0.0, -1.0, -2.0]], dtype=np.float32),
    )


def test_open3d_viewer_updates_graph_geometries_with_snapshot() -> None:
    viewer = Open3DMeshViewer(o3d_module=_FakeO3D())
    vertices = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    snapshot = SceneGraphSnapshot(
        frame_index=3,
        camera_pose_world=np.eye(4, dtype=np.float32),
        nodes=(
            GraphNode(
                id="ego",
                type="ego",
                label="camera",
                state="visible",
                confidence=1.0,
                world_centroid=np.zeros(3, dtype=np.float32),
                last_seen_frame=3,
                last_seen_direction="front",
                source_track_id=None,
            ),
            GraphNode(
                id="obj_table_1",
                type="object",
                label="table",
                state="visible",
                confidence=0.91,
                world_centroid=np.array([0.0, 0.0, 2.0], dtype=np.float32),
                last_seen_frame=3,
                last_seen_direction="front",
                source_track_id=1,
            ),
            GraphNode(
                id="seg_floor_1",
                type="segment",
                label="floor",
                state="visible",
                confidence=1.0,
                world_centroid=np.array([0.0, 1.0, 2.0], dtype=np.float32),
                last_seen_frame=3,
                last_seen_direction="front",
                source_track_id=1001,
            ),
        ),
        edges=(
            GraphEdge(
                source="ego",
                target="obj_table_1",
                relation="front",
                confidence=0.9,
                last_updated_frame=3,
                distance_bucket="mid",
                source_kind="detection",
            ),
            GraphEdge(
                source="obj_table_1",
                target="seg_floor_1",
                relation="on",
                confidence=0.8,
                last_updated_frame=3,
                distance_bucket=None,
                source_kind="segment",
            ),
        ),
        visible_node_ids=("ego", "obj_table_1", "seg_floor_1"),
        visible_edge_keys=(
            ("ego", "obj_table_1", "front"),
            ("obj_table_1", "seg_floor_1", "on"),
        ),
    )

    viewer.update(vertices, triangles, colors, snapshot)

    assert len(viewer._vis.geometry) == 5
    assert viewer._graph_nodes.points is not None
    assert viewer._graph_edges.lines is not None
    assert viewer._ego_lines.lines is not None


def test_open3d_viewer_color_only_update_skips_normal_recompute() -> None:
    viewer = Open3DMeshViewer(o3d_module=_FakeO3D())
    vertices = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.2, 1.0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[1.0, 0.0, 0.0]] * 3, dtype=np.float32)
    recolored = np.array([[0.0, 1.0, 0.0]] * 3, dtype=np.float32)

    viewer.update(vertices, triangles, colors, None, 1, 1)
    viewer._mesh.normals_computed = False

    viewer.update(vertices, triangles, recolored, None, 1, 2)

    assert viewer._mesh.normals_computed is False
    assert np.allclose(viewer._mesh.vertex_colors.data, recolored)


def test_interactive_view_state_holds_lod_mode_for_timeout_window() -> None:
    state = _InteractiveViewState(timeout_sec=0.3)

    assert state.update((1.0,), now=0.0) is False
    assert state.update((2.0,), now=0.1) is True
    assert state.update((2.0,), now=0.35) is True
    assert state.update((2.0,), now=0.41) is False
