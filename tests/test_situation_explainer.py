from __future__ import annotations

import time

import numpy as np

from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.scene_graph import GraphEdge, GraphNode, SceneGraphSnapshot
from obj_recog.situation_explainer import (
    ExplanationResult,
    ExplanationStatus,
    _response_text,
    OpenAISituationExplainer,
    SituationExplanationWorker,
    build_explanation_snapshot,
)
from obj_recog.types import Detection, FrameArtifacts, PanopticSegment


def _make_artifacts() -> FrameArtifacts:
    frame = np.full((24, 32, 3), 120, dtype=np.uint8)
    intrinsics = CameraIntrinsics(fx=20.0, fy=20.0, cx=16.0, cy=12.0)
    detections = [
        Detection(
            xyxy=(1, 2, 11, 14),
            class_id=0,
            label=f"object_{index}",
            confidence=0.95 - (index * 0.01),
            color=(255, 0, 0),
        )
        for index in range(15)
    ]
    segments = [
        PanopticSegment(
            segment_id=index,
            label_id=index,
            label=f"segment_{index}",
            color_rgb=(255, 100, 50),
            mask=np.ones((24, 32), dtype=bool),
            area_pixels=1000 - index,
        )
        for index in range(15)
    ]
    nodes = tuple(
        [
            GraphNode(
                id="ego",
                type="ego",
                label="camera",
                state="visible",
                confidence=1.0,
                world_centroid=np.zeros(3, dtype=np.float32),
                last_seen_frame=10,
                last_seen_direction="front",
                source_track_id=None,
            )
        ]
        + [
            GraphNode(
                id=f"obj_{index}",
                type="object",
                label=f"object_{index}",
                state="visible",
                confidence=0.9 - (index * 0.01),
                world_centroid=np.array([0.1 * index, 0.0, 1.0 + (0.1 * index)], dtype=np.float32),
                last_seen_frame=10 - index,
                last_seen_direction="front",
                source_track_id=index,
            )
            for index in range(25)
        ]
    )
    edges = tuple(
        GraphEdge(
            source="ego",
            target=f"obj_{index}",
            relation="front-right" if index % 2 else "front",
            confidence=0.8 - (index * 0.01),
            last_updated_frame=10,
            distance_bucket="mid",
            source_kind="detection",
        )
        for index in range(25)
    )
    snapshot = SceneGraphSnapshot(
        frame_index=10,
        camera_pose_world=np.eye(4, dtype=np.float32),
        nodes=nodes,
        edges=edges,
        visible_node_ids=tuple(node.id for node in nodes),
        visible_edge_keys=tuple((edge.source, edge.target, edge.relation) for edge in edges),
    )
    return FrameArtifacts(
        frame_bgr=frame,
        intrinsics=intrinsics,
        detections=detections,
        depth_map=np.ones((24, 32), dtype=np.float32),
        points_xyz=np.empty((0, 3), dtype=np.float32),
        points_rgb=np.empty((0, 3), dtype=np.float32),
        dense_map_points_xyz=np.empty((0, 3), dtype=np.float32),
        dense_map_points_rgb=np.empty((0, 3), dtype=np.float32),
        mesh_vertices_xyz=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        mesh_triangles=np.array([[0, 0, 0]], dtype=np.int32),
        mesh_vertex_colors=np.array([[0.2, 0.3, 0.4]], dtype=np.float32),
        camera_pose_world=np.eye(4, dtype=np.float32),
        tracking_ok=True,
        is_keyframe=True,
        trajectory_xyz=np.empty((0, 3), dtype=np.float32),
        segment_id=1,
        slam_tracking_state="TRACKING",
        keyframe_id=7,
        sparse_map_points_xyz=np.empty((0, 3), dtype=np.float32),
        loop_closure_applied=False,
        segmentation_overlay_bgr=np.zeros_like(frame),
        segments=segments,
        scene_graph_snapshot=snapshot,
        visible_graph_nodes=list(snapshot.visible_nodes),
        visible_graph_edges=list(snapshot.visible_edges),
    )


def test_build_explanation_snapshot_includes_expected_sections_and_limits() -> None:
    artifacts = _make_artifacts()

    snapshot = build_explanation_snapshot(
        artifacts,
        snapshot_id=3,
        max_detections=12,
        max_graph_nodes=20,
        max_graph_edges=20,
    )

    assert snapshot.snapshot_id == 3
    assert snapshot.frame_bgr.shape == (24, 32, 3)
    assert snapshot.structured_context.index("Visible objects") < snapshot.structured_context.index(
        "Visible structural segments"
    )
    assert snapshot.structured_context.index("Visible structural segments") < snapshot.structured_context.index(
        "Ego-relative graph summary"
    )
    assert snapshot.structured_context.index("Ego-relative graph summary") < snapshot.structured_context.index(
        "SLAM/runtime state"
    )
    objects_section = snapshot.structured_context.split("Visible structural segments", maxsplit=1)[0]
    assert "object_11" in objects_section
    assert "object_12" not in objects_section
    assert "segment_11" in snapshot.structured_context
    assert "segment_12" not in snapshot.structured_context
    assert snapshot.structured_context.count("ego ->") <= 20
    assert "image_size=32x24" in snapshot.structured_context
    assert "bbox=" in snapshot.structured_context
    assert "coverage=" in snapshot.structured_context


def test_build_explanation_snapshot_degrades_without_graph_or_segments() -> None:
    artifacts = _make_artifacts()
    artifacts.segments = []
    artifacts.scene_graph_snapshot = None
    artifacts.visible_graph_nodes = []
    artifacts.visible_graph_edges = []
    artifacts.slam_tracking_state = "LOST"

    snapshot = build_explanation_snapshot(
        artifacts,
        snapshot_id=9,
        max_detections=4,
        max_graph_nodes=5,
        max_graph_edges=5,
    )

    assert "Visible objects" in snapshot.structured_context
    assert "Visible structural segments" not in snapshot.structured_context
    assert "Ego-relative graph summary" not in snapshot.structured_context
    assert "SLAM/runtime state" in snapshot.structured_context
    assert "LOST" in snapshot.structured_context


class _FakeExplainer:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def explain(self, snapshot) -> ExplanationResult:
        self.calls.append(snapshot)
        return ExplanationResult(
            text="현재 장면 설명\n핵심 객체: 의자\n공간 관계: 의자는 오른쪽\n불확실성: 낮음",
            status=ExplanationStatus.READY,
            latency_ms=12.5,
            model="fake-model",
            error_message=None,
        )


class _FailingExplainer:
    def explain(self, snapshot) -> ExplanationResult:
        raise TimeoutError("timed out")


def test_situation_explanation_worker_submit_and_poll_preserve_snapshot_id() -> None:
    worker = SituationExplanationWorker(explainer=_FakeExplainer())
    snapshot = build_explanation_snapshot(_make_artifacts(), snapshot_id=5)
    try:
        worker.submit(snapshot.snapshot_id, snapshot)
        for _ in range(50):
            result = worker.poll()
            if result is not None:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("worker did not produce a result")
    finally:
        worker.close()

    result_snapshot_id, explanation = result
    assert result_snapshot_id == 5
    assert explanation.status is ExplanationStatus.READY
    assert explanation.model == "fake-model"


def test_situation_explanation_worker_returns_error_result_on_exception() -> None:
    worker = SituationExplanationWorker(explainer=_FailingExplainer())
    snapshot = build_explanation_snapshot(_make_artifacts(), snapshot_id=7)
    try:
        worker.submit(snapshot.snapshot_id, snapshot)
        for _ in range(50):
            result = worker.poll()
            if result is not None:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("worker did not produce a result")
    finally:
        worker.close()

    result_snapshot_id, explanation = result
    assert result_snapshot_id == 7
    assert explanation.status is ExplanationStatus.ERROR
    assert "timed out" in str(explanation.error_message)


def test_response_text_falls_back_to_nested_text_values() -> None:
    class _NestedText:
        def __init__(self, value: str) -> None:
            self.value = value

    class _Content:
        def __init__(self, text) -> None:
            self.text = text

    class _Message:
        def __init__(self, content) -> None:
            self.content = content

    class _Response:
        output_text = ""
        output = [_Message([_Content(_NestedText("현재 장면 설명"))])]

    assert _response_text(_Response()) == "현재 장면 설명"


def test_openai_situation_explainer_adds_minimal_reasoning_for_gpt5_models() -> None:
    class _FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)

            class _Response:
                output_text = "현재 장면 설명"
                output = []

            return _Response()

    class _FakeClient:
        def __init__(self) -> None:
            self.responses = _FakeResponses()

    explainer = OpenAISituationExplainer(
        model="gpt-5-mini",
        timeout_sec=8.0,
        api_key="test-key",
        client=_FakeClient(),
    )
    snapshot = build_explanation_snapshot(_make_artifacts(), snapshot_id=1)

    result = explainer.explain(snapshot)

    assert result.status is ExplanationStatus.READY
    call = explainer._client.responses.calls[0]
    assert call["reasoning"] == {"effort": "minimal"}


def test_openai_situation_explainer_omits_reasoning_hint_for_non_gpt5_models() -> None:
    class _FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)

            class _Response:
                output_text = "현재 장면 설명"
                output = []

            return _Response()

    class _FakeClient:
        def __init__(self) -> None:
            self.responses = _FakeResponses()

    explainer = OpenAISituationExplainer(
        model="gpt-4.1-mini",
        timeout_sec=8.0,
        api_key="test-key",
        client=_FakeClient(),
    )
    snapshot = build_explanation_snapshot(_make_artifacts(), snapshot_id=2)

    result = explainer.explain(snapshot)

    assert result.status is ExplanationStatus.READY
    call = explainer._client.responses.calls[0]
    assert "reasoning" not in call
