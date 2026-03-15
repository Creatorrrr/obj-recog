from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.main import run
from obj_recog.scene_graph import GraphNode, SceneGraphSnapshot
from obj_recog.slam_bridge import SlamFrameResult
from obj_recog.types import Detection, PanopticSegment, SegmentationResult


class _FakeCV2:
    WND_PROP_VISIBLE = 1
    INTER_AREA = cv2.INTER_AREA
    COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY

    def __init__(self) -> None:
        self.imshow_calls: list[str] = []
        self.destroyed = False

    def imshow(self, window_name: str, frame: np.ndarray) -> None:
        self.imshow_calls.append(window_name)

    def waitKey(self, delay: int) -> int:
        _ = delay
        return -1

    def getWindowProperty(self, window_name: str, prop: int) -> float:
        _ = (window_name, prop)
        return 1.0

    def resize(self, image: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
        return cv2.resize(image, size, interpolation=interpolation)

    def cvtColor(self, image: np.ndarray, code: int) -> np.ndarray:
        return cv2.cvtColor(image, code)

    def destroyAllWindows(self) -> None:
        self.destroyed = True


class _CountingCV2(_FakeCV2):
    def __init__(self) -> None:
        super().__init__()
        self.waitkey_calls = 0

    def waitKey(self, delay: int) -> int:
        self.waitkey_calls += 1
        return super().waitKey(delay)


class _FakeViewer:
    def __init__(self) -> None:
        self.closed = False

    def update(self, *_args, **_kwargs) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


class _CountingViewer(_FakeViewer):
    def __init__(self) -> None:
        super().__init__()
        self.update_calls = 0

    def update(self, *_args, **_kwargs) -> bool:
        self.update_calls += 1
        return True


class _FakeEnvironmentViewer:
    def __init__(self) -> None:
        self.closed = False
        self.states: list[object] = []

    def update(self, scenario_state, *_args, **_kwargs) -> bool:
        self.states.append(scenario_state)
        return True

    def close(self) -> None:
        self.closed = True


class _FakeFrameSource:
    def __init__(self, packets: list[FramePacket]) -> None:
        self._packets = list(packets)
        self.closed = False
        self.runtime_observations: list[tuple[FramePacket, object]] = []

    def next_frame(self, *, timeout_sec: float | None = 1.0) -> FramePacket | None:
        _ = timeout_sec
        if not self._packets:
            return None
        return self._packets.pop(0)

    def record_runtime_observation(self, *, frame_packet: FramePacket, artifacts) -> None:
        self.runtime_observations.append((frame_packet, artifacts))

    def close(self) -> None:
        self.closed = True


class _PendingFrameSource(_FakeFrameSource):
    def __init__(self, packets: list[FramePacket]) -> None:
        super().__init__([])
        self._packets = list(packets)
        self._waiting = False
        self._stage = 0

    def next_frame(self, *, timeout_sec: float | None = 1.0) -> FramePacket | None:
        _ = timeout_sec
        if self._stage == 0:
            self._stage = 1
            return self._packets.pop(0)
        if self._stage == 1:
            self._waiting = True
            self._stage = 2
            return None
        if self._stage == 2:
            self._waiting = False
            self._stage = 3
            return self._packets.pop(0)
        return None

    def is_waiting_for_frame(self) -> bool:
        return self._waiting


class _CountingDetector:
    def __init__(self, **_kwargs) -> None:
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray):
        self.calls += 1
        return [
            Detection(
                xyxy=(1, 1, frame_bgr.shape[1] - 2, frame_bgr.shape[0] - 2),
                class_id=1,
                label="dining_table",
                confidence=0.93,
                color=(0, 255, 0),
            )
        ]


class _CountingDepthEstimator:
    def __init__(self, **_kwargs) -> None:
        self.calls = 0

    def estimate(self, frame_bgr: np.ndarray):
        self.calls += 1
        return np.full(frame_bgr.shape[:2], 2.0, dtype=np.float32)


class _FakeSlamBridge:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.calls = 0

    def track(self, frame_gray: np.ndarray, timestamp: float) -> SlamFrameResult:
        self.calls += 1
        _ = (frame_gray, timestamp)
        return SlamFrameResult(
            tracking_state="TRACKING",
            pose_world=np.eye(4, dtype=np.float32),
            keyframe_inserted=False,
            keyframe_id=None,
            optimized_keyframe_poses={},
            sparse_map_points_xyz=np.zeros((0, 3), dtype=np.float32),
            loop_closure_applied=False,
        )

    def close(self) -> None:
        return None


class _FakeSlamBridgeFactory:
    def __init__(self) -> None:
        self.instances: list[_FakeSlamBridge] = []

    def __call__(self, **kwargs):
        bridge = _FakeSlamBridge(**kwargs)
        self.instances.append(bridge)
        return bridge


class _FakeMapBuilder:
    requires_point_cloud = True

    def update(self, slam_result, frame_points_xyz: np.ndarray, frame_points_rgb: np.ndarray):
        _ = (slam_result, frame_points_xyz, frame_points_rgb)
        return type(
            "MapUpdate",
            (),
            {
                "dense_map_points_xyz": np.zeros((1, 3), dtype=np.float32),
                "dense_map_points_rgb": np.ones((1, 3), dtype=np.float32),
                "mesh_vertices_xyz": np.zeros((1, 3), dtype=np.float32),
                "mesh_triangles": np.empty((0, 3), dtype=np.int32),
                "mesh_vertex_colors": np.ones((1, 3), dtype=np.float32),
                "trajectory_xyz": np.zeros((1, 3), dtype=np.float32),
                "is_keyframe": True,
                "keyframe_id": 1,
                "sparse_map_points_xyz": np.zeros((0, 3), dtype=np.float32),
                "loop_closure_applied": False,
                "segment_id": 1,
            },
        )()

    def reset(self) -> None:
        return None


class _FakeExplainer:
    pass


class _FakeExplanationWorker:
    def __init__(self) -> None:
        self.submissions: list[tuple[int, object]] = []
        self._queued_result = None
        self.closed = False

    def submit(self, snapshot_id: int, snapshot) -> None:
        from obj_recog.situation_explainer import ExplanationResult, ExplanationStatus

        self.submissions.append((snapshot_id, snapshot))
        self._queued_result = (
            snapshot_id,
            ExplanationResult(
                text="TV is visible ahead.",
                status=ExplanationStatus.READY,
                latency_ms=42.0,
                model="fake-explainer",
                error_message=None,
            ),
        )

    def poll(self):
        queued = self._queued_result
        self._queued_result = None
        return queued

    def is_idle(self) -> bool:
        return self._queued_result is None

    def close(self) -> None:
        self.closed = True


class _ImmediateSegmentationWorker:
    def __init__(self, **_kwargs) -> None:
        self._submitted_frame_index = None
        self._returned = False
        self.closed = False

    def is_idle(self) -> bool:
        return True

    def submit(self, frame_index: int, _frame_bgr: np.ndarray) -> None:
        self._submitted_frame_index = frame_index
        self._returned = False

    def poll(self):
        if self._submitted_frame_index is None or self._returned:
            return None
        mask = np.ones((8, 8), dtype=bool)
        self._returned = True
        return (
            self._submitted_frame_index,
            SegmentationResult(
                overlay_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
                segment_id_map=np.ones((8, 8), dtype=np.int32),
                segments=[
                    PanopticSegment(
                        segment_id=1,
                        label_id=1,
                        label="floor",
                        color_rgb=(0, 255, 0),
                        mask=mask,
                        area_pixels=int(mask.sum()),
                    )
                ],
            ),
        )

    def close(self) -> None:
        self.closed = True


class _FakeSceneGraphMemory:
    def __init__(self, **_kwargs) -> None:
        self.update_calls: list[tuple[list[object], list[PanopticSegment]]] = []

    def update(
        self,
        *,
        frame_index: int,
        detections: list[object],
        segments: list[PanopticSegment],
        depth_map: np.ndarray,
        intrinsics,
        camera_pose_world: np.ndarray,
        slam_tracking_state: str,
    ) -> SceneGraphSnapshot:
        _ = (depth_map, intrinsics, camera_pose_world, slam_tracking_state)
        self.update_calls.append((list(detections), list(segments)))
        nodes = (
            GraphNode(
                id="ego",
                type="ego",
                label="camera",
                state="visible",
                confidence=1.0,
                world_centroid=np.zeros(3, dtype=np.float32),
                last_seen_frame=frame_index,
                last_seen_direction="front",
                source_track_id=None,
            ),
            GraphNode(
                id="seg-floor-1",
                type="segment",
                label="floor",
                state="visible",
                confidence=1.0,
                world_centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                last_seen_frame=frame_index,
                last_seen_direction="front",
                source_track_id=1,
            ),
        )
        return SceneGraphSnapshot(
            frame_index=frame_index,
            camera_pose_world=np.eye(4, dtype=np.float32),
            nodes=nodes,
            edges=(),
            visible_node_ids=("ego", "seg-floor-1"),
            visible_edge_keys=(),
        )


def _packet(timestamp_sec: float) -> FramePacket:
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    return FramePacket(
        frame_bgr=frame,
        timestamp_sec=timestamp_sec,
    )


def _write_runtime_files(tmp_path: Path) -> tuple[str, str]:
    vocabulary = tmp_path / "ORBvoc.txt"
    vocabulary.write_text("stub", encoding="utf-8")
    calibration = tmp_path / "camera.yaml"
    calibration.write_text(
        "\n".join(
            [
                "%YAML:1.0",
                "Camera.width: 8",
                "Camera.height: 8",
                "Camera.fx: 8.0",
                "Camera.fy: 8.0",
                "Camera.cx: 4.0",
                "Camera.cy: 4.0",
            ]
        ),
        encoding="utf-8",
    )
    return str(vocabulary), str(calibration)


def test_run_sim_mode_uses_runtime_depth_and_monocular_slam(tmp_path: Path) -> None:
    slam_vocabulary, calibration = _write_runtime_files(tmp_path)
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
        detection_interval=1,
        input_source="sim",
        segmentation_mode="off",
        graph_enabled=False,
        explanation_enabled=False,
        camera_calibration=calibration,
        slam_vocabulary=slam_vocabulary,
    )
    source = _FakeFrameSource([_packet(0.0), _packet(0.5)])
    fake_cv2 = _FakeCV2()
    detector = _CountingDetector()
    depth_estimator = _CountingDepthEstimator()
    slam_bridge_factory = _FakeSlamBridgeFactory()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_kwargs: detector,
        depth_estimator_factory=lambda **_kwargs: depth_estimator,
        tracker_factory=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("tracker should not be used")),
        slam_bridge_factory=slam_bridge_factory,
        map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
        viewer_factory=lambda: _FakeViewer(),
        environment_viewer_factory=lambda: _FakeEnvironmentViewer(),
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
    )

    assert detector.calls == 2
    assert depth_estimator.calls == 2
    assert len(slam_bridge_factory.instances) == 1
    assert slam_bridge_factory.instances[0].calls == 2
    assert "Situation Explanation" not in fake_cv2.imshow_calls
    assert len(source.runtime_observations) == 2
    assert all(
        observation[1].perception_diagnostics is not None
        and observation[1].perception_diagnostics.depth_source == "runtime"
        and observation[1].perception_diagnostics.pose_source == "runtime"
        for observation in source.runtime_observations
    )


def test_run_sim_mode_does_not_forward_environment_truth_to_open3d_view(tmp_path: Path) -> None:
    slam_vocabulary, calibration = _write_runtime_files(tmp_path)
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
        input_source="sim",
        segmentation_mode="off",
        graph_enabled=False,
        explanation_enabled=False,
        camera_calibration=calibration,
        slam_vocabulary=slam_vocabulary,
    )
    source = _FakeFrameSource([_packet(0.0)])
    environment_viewer = _FakeEnvironmentViewer()

    run(
        config,
        cv2_module=_FakeCV2(),
        detector_factory=lambda **_kwargs: _CountingDetector(),
        depth_estimator_factory=lambda **_kwargs: _CountingDepthEstimator(),
        tracker_factory=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("tracker should not be used")),
        slam_bridge_factory=_FakeSlamBridgeFactory(),
        map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
        viewer_factory=lambda: _FakeViewer(),
        environment_viewer_factory=lambda: environment_viewer,
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
    )

    assert source.closed is True
    assert environment_viewer.states == []
    assert environment_viewer.closed is False


def test_run_sim_mode_shows_explanation_panel_and_auto_refreshes_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slam_vocabulary, calibration = _write_runtime_files(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
        detection_interval=1,
        input_source="sim",
        segmentation_mode="off",
        graph_enabled=False,
        explanation_enabled=True,
        camera_calibration=calibration,
        slam_vocabulary=slam_vocabulary,
    )
    source = _FakeFrameSource([_packet(0.0), _packet(0.5), _packet(1.0)])
    fake_cv2 = _FakeCV2()
    detector = _CountingDetector()
    depth_estimator = _CountingDepthEstimator()
    slam_bridge_factory = _FakeSlamBridgeFactory()
    worker = _FakeExplanationWorker()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_kwargs: detector,
        depth_estimator_factory=lambda **_kwargs: depth_estimator,
        tracker_factory=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("tracker should not be used")),
        slam_bridge_factory=slam_bridge_factory,
        map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
        viewer_factory=lambda: _FakeViewer(),
        environment_viewer_factory=lambda: _FakeEnvironmentViewer(),
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
        situation_explainer_factory=lambda **_kwargs: _FakeExplainer(),
        explanation_worker_factory=lambda **_kwargs: worker,
        explanation_panel_renderer=lambda **_kwargs: np.zeros((80, 120, 3), dtype=np.uint8),
    )

    assert "Situation Explanation" in fake_cv2.imshow_calls
    assert len(worker.submissions) >= 1
    assert worker.closed is True


def test_run_sim_mode_records_runtime_observation_after_segments_and_scene_graph(tmp_path: Path) -> None:
    slam_vocabulary, calibration = _write_runtime_files(tmp_path)
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
        detection_interval=1,
        input_source="sim",
        segmentation_mode="panoptic",
        graph_enabled=True,
        explanation_enabled=False,
        camera_calibration=calibration,
        slam_vocabulary=slam_vocabulary,
    )
    source = _FakeFrameSource([_packet(0.0)])
    scene_graph_memory = _FakeSceneGraphMemory()

    run(
        config,
        cv2_module=_FakeCV2(),
        detector_factory=lambda **_kwargs: _CountingDetector(),
        depth_estimator_factory=lambda **_kwargs: _CountingDepthEstimator(),
        tracker_factory=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("tracker should not be used")),
        slam_bridge_factory=_FakeSlamBridgeFactory(),
        map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
        viewer_factory=lambda: _FakeViewer(),
        environment_viewer_factory=lambda: _FakeEnvironmentViewer(),
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
        segmentation_worker_factory=lambda **_kwargs: _ImmediateSegmentationWorker(),
        scene_graph_memory_factory=lambda **_kwargs: scene_graph_memory,
    )

    assert len(source.runtime_observations) == 1
    recorded_artifacts = source.runtime_observations[0][1]
    assert recorded_artifacts.segments
    assert recorded_artifacts.segments[0].label == "floor"
    assert recorded_artifacts.scene_graph_snapshot is not None
    assert scene_graph_memory.update_calls[0][1][0].label == "floor"


def test_run_sim_mode_waits_for_pending_frame_without_treating_it_as_eof(tmp_path: Path) -> None:
    slam_vocabulary, calibration = _write_runtime_files(tmp_path)
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
        detection_interval=1,
        input_source="sim",
        segmentation_mode="off",
        graph_enabled=False,
        explanation_enabled=False,
        camera_calibration=calibration,
        slam_vocabulary=slam_vocabulary,
    )
    source = _PendingFrameSource([_packet(0.0), _packet(0.5)])
    fake_cv2 = _CountingCV2()
    viewer = _CountingViewer()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_kwargs: _CountingDetector(),
        depth_estimator_factory=lambda **_kwargs: _CountingDepthEstimator(),
        tracker_factory=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("tracker should not be used")),
        slam_bridge_factory=_FakeSlamBridgeFactory(),
        map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
        viewer_factory=lambda: viewer,
        environment_viewer_factory=lambda: _FakeEnvironmentViewer(),
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
    )

    assert len(source.runtime_observations) == 2
    assert viewer.update_calls >= 3
    assert fake_cv2.waitkey_calls >= 3
    assert source.closed is True
