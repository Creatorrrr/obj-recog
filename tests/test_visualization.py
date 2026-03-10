from __future__ import annotations

import numpy as np

from obj_recog.scene_graph import GraphEdge, GraphNode, SceneGraphSnapshot
from obj_recog.types import Detection, PanopticSegment
from obj_recog.visualization import (
    Open3DMeshViewer,
    _display_points_for_view,
    draw_detections,
    highlight_detected_points,
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


class _FakeVisualizer:
    def __init__(self) -> None:
        self.geometry = []
        self.reset_calls: list[bool] = []
        self.poll_count = 0
        self.renderer_updates = 0

    def create_window(self, window_name: str, width: int, height: int) -> bool:
        return True

    def add_geometry(self, geometry) -> None:
        self.geometry.append(geometry)

    def get_render_option(self) -> _FakeRenderOption:
        return _FakeRenderOption()

    def update_geometry(self, geometry) -> None:
        return None

    def poll_events(self) -> bool:
        self.poll_count += 1
        return True

    def update_renderer(self) -> None:
        self.renderer_updates += 1

    def reset_view_point(self, reset_bounding_box: bool) -> None:
        self.reset_calls.append(reset_bounding_box)

    def destroy_window(self) -> None:
        return None


class _FakeO3D:
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
    ]


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
        "front-right cup (near)",
    ]


def test_open3d_viewer_resets_view_on_first_non_empty_update_only() -> None:
    viewer = Open3DMeshViewer(o3d_module=_FakeO3D())
    vertices = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    viewer.update(vertices, triangles, colors)
    viewer.update(vertices, triangles, colors)

    assert viewer._vis.reset_calls == [True]
    assert len(viewer._vis.geometry) == 4


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

    assert len(viewer._vis.geometry) == 4
    assert viewer._graph_nodes.points is not None
    assert viewer._graph_edges.lines is not None
    assert viewer._ego_lines.lines is not None
