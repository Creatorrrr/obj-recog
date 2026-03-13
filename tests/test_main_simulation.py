from __future__ import annotations

import numpy as np

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.main import run
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.sim_protocol import EpisodePhase, OperatorSceneState
from obj_recog.sim_scene import build_living_room_scene_spec
from obj_recog.types import Detection


class _FakeCV2:
    WND_PROP_VISIBLE = 1

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

    def destroyAllWindows(self) -> None:
        self.destroyed = True


class _FakeViewer:
    def __init__(self) -> None:
        self.closed = False

    def update(self, *_args, **_kwargs) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


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


class _CountingTracker:
    def __init__(self, **_kwargs) -> None:
        self.calls = 0

    def update(self, **_kwargs):
        self.calls += 1
        return type(
            "TrackingResult",
            (),
            {
                "camera_pose_world": np.eye(4, dtype=np.float32),
                "tracking_ok": True,
                "did_reset": False,
            },
        )()

    def reset(self) -> None:
        return None


class _TrackerFactoryRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _CountingTracker()


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


def _packet(timestamp_sec: float) -> FramePacket:
    scene = build_living_room_scene_spec()
    operator_state = OperatorSceneState(
        scene_spec=scene,
        robot_pose=scene.start_pose,
        phase=EpisodePhase.SELF_CALIBRATING,
    )
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    return FramePacket(
        frame_bgr=frame,
        timestamp_sec=timestamp_sec,
        depth_map=np.full((8, 8), 2.0, dtype=np.float32),
        pose_world_gt=np.eye(4, dtype=np.float32),
        intrinsics_gt=CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0),
        scenario_state=operator_state,
    )


def test_run_uses_frame_source_for_sim_and_updates_open3d_environment_view() -> None:
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
    )
    source = _FakeFrameSource([_packet(0.0)])
    fake_cv2 = _FakeCV2()
    environment_viewer = _FakeEnvironmentViewer()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_kwargs: _CountingDetector(),
        depth_estimator_factory=lambda **_kwargs: _CountingDepthEstimator(),
        tracker_factory=lambda **_kwargs: _CountingTracker(),
        map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
        viewer_factory=lambda: _FakeViewer(),
        environment_viewer_factory=lambda: environment_viewer,
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
    )

    assert source.closed is True
    assert environment_viewer.closed is True
    assert len(environment_viewer.states) == 1
    assert "Environment Model" not in fake_cv2.imshow_calls
    assert "Object Recognition" in fake_cv2.imshow_calls


def test_run_sim_mode_keeps_runtime_models_active_and_suppresses_explanation_window() -> None:
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
    )
    source = _FakeFrameSource([_packet(0.0), _packet(0.5)])
    fake_cv2 = _FakeCV2()
    detector = _CountingDetector()
    depth_estimator = _CountingDepthEstimator()
    tracker = _CountingTracker()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_kwargs: detector,
        depth_estimator_factory=lambda **_kwargs: depth_estimator,
        tracker_factory=lambda **_kwargs: tracker,
        map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
        viewer_factory=lambda: _FakeViewer(),
        environment_viewer_factory=lambda: _FakeEnvironmentViewer(),
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
        explanation_worker_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sim mode should not create explanation worker")),
    )

    assert detector.calls == 2
    assert depth_estimator.calls == 0
    assert tracker.calls == 2
    assert "Situation Explanation" not in fake_cv2.imshow_calls
    assert len(source.runtime_observations) == 2
    assert all(
        observation[1].perception_diagnostics is not None
        and observation[1].perception_diagnostics.depth_source == "sensor"
        and observation[1].perception_diagnostics.pose_source == "runtime"
        for observation in source.runtime_observations
    )


def test_run_sim_mode_uses_relaxed_tracker_thresholds_for_sensor_rendering() -> None:
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
    )
    source = _FakeFrameSource([_packet(0.0)])
    tracker_factory = _TrackerFactoryRecorder()

    run(
        config,
        cv2_module=_FakeCV2(),
        detector_factory=lambda **_kwargs: _CountingDetector(),
        depth_estimator_factory=lambda **_kwargs: _CountingDepthEstimator(),
        tracker_factory=tracker_factory,
        map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
        viewer_factory=lambda: _FakeViewer(),
        environment_viewer_factory=lambda: _FakeEnvironmentViewer(),
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
    )

    assert tracker_factory.calls
    assert tracker_factory.calls[0]["min_correspondences"] == 4
    assert tracker_factory.calls[0]["min_inliers"] == 4
