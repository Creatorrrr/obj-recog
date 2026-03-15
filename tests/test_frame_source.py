from __future__ import annotations

import numpy as np

from obj_recog.camera import CameraSession
from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket, LiveCameraFrameSource
from obj_recog.main import _legacy_tracking_to_slam_result, process_frame
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.types import Detection


class _FakeCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame
        self.released = False

    def read(self, timeout_sec: float | None = None) -> tuple[bool, np.ndarray | None]:
        _ = timeout_sec
        if self._frame is None:
            return False, None
        frame, self._frame = self._frame, None
        return True, frame

    def release(self) -> None:
        self.released = True


class _StrictDetector:
    def detect(self, frame_bgr: np.ndarray):
        raise AssertionError(f"detector should not be called for GT packet: {frame_bgr.shape}")


class _StrictDepthEstimator:
    def estimate(self, frame_bgr: np.ndarray):
        raise AssertionError(f"depth estimator should not be called for GT packet: {frame_bgr.shape}")


class _FakeMapUpdate:
    def __init__(self, frame_points_xyz: np.ndarray, frame_points_rgb: np.ndarray, *, keyframe_id: int | None) -> None:
        self.dense_map_points_xyz = frame_points_xyz
        self.dense_map_points_rgb = frame_points_rgb
        self.mesh_vertices_xyz = frame_points_xyz
        self.mesh_triangles = np.empty((0, 3), dtype=np.int32)
        self.mesh_vertex_colors = frame_points_rgb
        self.trajectory_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        self.is_keyframe = True
        self.keyframe_id = keyframe_id
        self.sparse_map_points_xyz = np.empty((0, 3), dtype=np.float32)
        self.loop_closure_applied = False
        self.segment_id = 1


class _FakeMapBuilder:
    requires_point_cloud = True

    def __init__(self) -> None:
        self.last_frame_points_xyz: np.ndarray | None = None
        self.last_frame_points_rgb: np.ndarray | None = None

    def update(self, slam_result, frame_points_xyz: np.ndarray, frame_points_rgb: np.ndarray) -> _FakeMapUpdate:
        self.last_frame_points_xyz = np.asarray(frame_points_xyz, dtype=np.float32)
        self.last_frame_points_rgb = np.asarray(frame_points_rgb, dtype=np.float32)
        return _FakeMapUpdate(
            self.last_frame_points_xyz,
            self.last_frame_points_rgb,
            keyframe_id=slam_result.keyframe_id,
        )


class _CountingDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray):
        self.calls += 1
        return [
            Detection(
                xyxy=(0, 0, frame_bgr.shape[1] - 1, frame_bgr.shape[0] - 1),
                class_id=1,
                label="runtime-target",
                confidence=0.9,
                color=(255, 0, 0),
            )
        ]


class _EmptyDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray):
        self.calls += 1
        return []


class _RecordingDetector:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def detect(self, frame_bgr: np.ndarray):
        self.frames.append(np.asarray(frame_bgr, dtype=np.uint8).copy())
        return [
            Detection(
                xyxy=(0, 0, frame_bgr.shape[1] - 1, frame_bgr.shape[0] - 1),
                class_id=3,
                label="chair",
                confidence=0.88,
                color=(0, 255, 255),
            )
        ]


class _CountingDepthEstimator:
    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, frame_bgr: np.ndarray):
        self.calls += 1
        return np.full(frame_bgr.shape[:2], 1.5, dtype=np.float32)


class _CountingTracker:
    def __init__(self) -> None:
        self.calls = 0

    def update(self, *, frame_bgr: np.ndarray, depth_map: np.ndarray, intrinsics):
        _ = (frame_bgr, depth_map, intrinsics)
        self.calls += 1
        return type(
            "TrackingResult",
            (),
            {
                "camera_pose_world": np.eye(4, dtype=np.float32),
                "tracking_ok": True,
                "did_reset": True,
            },
        )()


def test_live_camera_frame_source_wraps_camera_session() -> None:
    frame = np.full((4, 6, 3), 127, dtype=np.uint8)
    capture = _FakeCapture(frame.copy())
    session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="Fake Camera",
        requested_name=None,
        used_fallback=False,
    )
    source = LiveCameraFrameSource(
        camera_session=session,
        time_source=lambda: 12.5,
    )

    packet = source.next_frame(timeout_sec=0.25)

    assert packet is not None
    assert packet.timestamp_sec == 12.5
    assert np.array_equal(packet.frame_bgr, frame)
    source.close()
    assert capture.released is True


def test_process_frame_uses_ground_truth_packet_without_runtime_models() -> None:
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
    )
    frame = np.full((8, 8, 3), 90, dtype=np.uint8)
    depth_map = np.full((8, 8), 2.0, dtype=np.float32)
    pose_world = np.eye(4, dtype=np.float32)
    pose_world[0, 3] = 1.25
    pose_world[2, 3] = 0.4
    packet = FramePacket(
        frame_bgr=frame,
        timestamp_sec=1.0,
        depth_map=depth_map,
        pose_world_gt=pose_world,
        intrinsics_gt=CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0),
        detections=[
            Detection(
                xyxy=(1, 1, 6, 6),
                class_id=7,
                label="target",
                confidence=0.99,
                color=(0, 255, 0),
            )
        ],
        tracking_state="TRACKING",
        keyframe_inserted=True,
        keyframe_id=7,
    )
    map_builder = _FakeMapBuilder()

    artifacts, cached = process_frame(
        frame_bgr=frame,
        detector=_StrictDetector(),
        depth_estimator=_StrictDepthEstimator(),
        map_builder=map_builder,
        config=config,
        frame_index=0,
        timestamp_sec=1.0,
        cached_detections=[],
        frame_packet=packet,
        prefer_frame_packet_ground_truth=True,
        cv2_module=object(),
    )

    assert len(cached) == 1
    assert cached[0].label == "target"
    assert artifacts.depth_map.shape == (8, 8)
    assert np.allclose(artifacts.camera_pose_world, pose_world)
    assert artifacts.keyframe_id == 7
    assert artifacts.is_keyframe is True
    assert map_builder.last_frame_points_xyz is not None
    assert map_builder.last_frame_points_xyz.shape[1] == 3


def test_process_frame_keeps_runtime_stack_active_by_default_even_with_packet_ground_truth() -> None:
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
    )
    frame = np.full((8, 8, 3), 90, dtype=np.uint8)
    packet = FramePacket(
        frame_bgr=frame,
        timestamp_sec=1.0,
        depth_map=np.full((8, 8), 9.0, dtype=np.float32),
        pose_world_gt=np.eye(4, dtype=np.float32),
        intrinsics_gt=CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0),
        detections=[
            Detection(
                xyxy=(1, 1, 6, 6),
                class_id=7,
                label="gt-target",
                confidence=0.99,
                color=(0, 255, 0),
            )
        ],
    )
    detector = _CountingDetector()
    depth_estimator = _CountingDepthEstimator()
    tracker = _CountingTracker()

    artifacts, cached = process_frame(
        frame_bgr=frame,
        detector=detector,
        depth_estimator=depth_estimator,
        map_builder=_FakeMapBuilder(),
        config=config,
        frame_index=0,
        timestamp_sec=1.0,
        cached_detections=[],
        tracker=tracker,
        frame_packet=packet,
        cv2_module=object(),
    )

    assert detector.calls == 1
    assert depth_estimator.calls == 1
    assert tracker.calls == 1
    assert cached[0].label == "runtime-target"
    assert artifacts.detections[0].label == "runtime-target"


def test_process_frame_assisted_mode_falls_back_to_packet_detections_and_depth() -> None:
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
        input_source="sim",
        sim_perception_mode="assisted",
    )
    frame = np.full((8, 8, 3), 90, dtype=np.uint8)
    packet = FramePacket(
        frame_bgr=frame,
        timestamp_sec=1.0,
        depth_map=np.full((8, 8), 3.25, dtype=np.float32),
        pose_world_gt=np.eye(4, dtype=np.float32),
        intrinsics_gt=CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0),
        detections=[
            Detection(
                xyxy=(1, 1, 6, 6),
                class_id=7,
                label="gt-target",
                confidence=0.99,
                color=(0, 255, 0),
            )
        ],
        scenario_state=type("ScenarioState", (), {"semantic_target_class": "gt-target"})(),
    )
    detector = _EmptyDetector()
    tracker = _CountingTracker()

    artifacts, cached = process_frame(
        frame_bgr=frame,
        detector=detector,
        depth_estimator=_StrictDepthEstimator(),
        map_builder=_FakeMapBuilder(),
        config=config,
        frame_index=0,
        timestamp_sec=1.0,
        cached_detections=[],
        tracker=tracker,
        frame_packet=packet,
        assist_frame_packet_ground_truth=True,
        cv2_module=object(),
    )

    assert detector.calls == 1
    assert tracker.calls == 1
    assert np.allclose(artifacts.depth_map, packet.depth_map)
    assert cached[0].label == "gt-target"
    assert artifacts.detections[0].label == "gt-target"
    assert artifacts.perception_diagnostics is not None
    assert artifacts.perception_diagnostics.detection_source == "runtime+fallback"
    assert artifacts.perception_diagnostics.depth_source == "ground_truth"
    assert artifacts.perception_diagnostics.pose_source == "ground_truth"
    assert artifacts.perception_diagnostics.gt_target_visible is True
    assert artifacts.perception_diagnostics.benchmark_valid is False


def test_process_frame_runtime_sim_mode_marks_benchmark_valid_runtime_sources() -> None:
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
        input_source="sim",
        sim_perception_mode="runtime",
    )
    frame = np.full((8, 8, 3), 90, dtype=np.uint8)
    detector = _CountingDetector()
    depth_estimator = _CountingDepthEstimator()
    tracker = _CountingTracker()

    artifacts, cached = process_frame(
        frame_bgr=frame,
        detector=detector,
        depth_estimator=depth_estimator,
        map_builder=_FakeMapBuilder(),
        config=config,
        frame_index=0,
        timestamp_sec=1.0,
        cached_detections=[],
        tracker=tracker,
    )

    assert cached[0].label == "runtime-target"
    assert artifacts.perception_diagnostics is not None
    assert artifacts.perception_diagnostics.perception_mode == "runtime"
    assert artifacts.perception_diagnostics.detection_source == "runtime"
    assert artifacts.perception_diagnostics.depth_source == "runtime_midas"
    assert artifacts.perception_diagnostics.pose_source == "runtime"
    assert artifacts.perception_diagnostics.gt_target_visible is False
    assert artifacts.perception_diagnostics.benchmark_valid is True


def test_process_frame_can_use_sensor_depth_without_ground_truth_pose() -> None:
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
        input_source="sim",
        sim_perception_mode="runtime",
    )
    frame = np.full((8, 8, 3), 90, dtype=np.uint8)
    packet = FramePacket(
        frame_bgr=frame,
        timestamp_sec=1.0,
        depth_map=np.full((8, 8), 2.75, dtype=np.float32),
        pose_world_gt=np.eye(4, dtype=np.float32),
        intrinsics_gt=CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0),
    )
    detector = _CountingDetector()
    depth_estimator = _CountingDepthEstimator()
    tracker = _CountingTracker()

    artifacts, cached = process_frame(
        frame_bgr=frame,
        detector=detector,
        depth_estimator=depth_estimator,
        map_builder=_FakeMapBuilder(),
        config=config,
        frame_index=0,
        timestamp_sec=1.0,
        cached_detections=[],
        tracker=tracker,
        frame_packet=packet,
        prefer_frame_packet_depth_sensor=True,
        cv2_module=object(),
    )

    assert cached[0].label == "runtime-target"
    assert depth_estimator.calls == 0
    assert tracker.calls == 1
    assert np.allclose(artifacts.depth_map, packet.depth_map)
    assert artifacts.perception_diagnostics is not None
    assert artifacts.perception_diagnostics.depth_source == "sensor"
    assert artifacts.perception_diagnostics.pose_source == "runtime"


def test_legacy_tracking_to_slam_result_assigns_frame_keyframe_id_on_reset() -> None:
    tracking_result = type(
        "TrackingResult",
        (),
        {
            "camera_pose_world": np.eye(4, dtype=np.float32),
            "tracking_ok": True,
            "did_reset": True,
        },
    )()

    slam_result = _legacy_tracking_to_slam_result(tracking_result, frame_index=7)

    assert slam_result.tracking_state == "TRACKING"
    assert slam_result.keyframe_inserted is True
    assert slam_result.keyframe_id == 7


def test_process_frame_passes_resized_raw_camera_frame_to_detector() -> None:
    config = AppConfig(
        camera_index=0,
        width=12,
        height=8,
        inference_width=6,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
    )
    frame = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    detector = _RecordingDetector()
    depth_estimator = _CountingDepthEstimator()
    tracker = _CountingTracker()

    artifacts, cached = process_frame(
        frame_bgr=frame,
        detector=detector,
        depth_estimator=depth_estimator,
        map_builder=_FakeMapBuilder(),
        config=config,
        frame_index=0,
        timestamp_sec=1.0,
        cached_detections=[],
        tracker=tracker,
    )

    assert len(detector.frames) == 1
    expected_height = int(round(frame.shape[0] * (config.inference_width / float(frame.shape[1]))))
    assert detector.frames[0].shape == (expected_height, int(config.inference_width), 3)
    assert not np.shares_memory(detector.frames[0], artifacts.frame_bgr)
    assert cached[0].label == "chair"
    assert depth_estimator.calls == 1
    assert tracker.calls == 1
