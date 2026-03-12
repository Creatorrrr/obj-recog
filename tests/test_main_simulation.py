from __future__ import annotations

import numpy as np
import pytest

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.main import process_frame, run
from obj_recog.reconstruct import CameraIntrinsics
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


class _StrictDetector:
    def __init__(self, **_kwargs) -> None:
        pass

    def detect(self, frame_bgr: np.ndarray):
        raise AssertionError(f"detector should not run for GT packets: {frame_bgr.shape}")


class _CountingDetector:
    def __init__(self, **_kwargs) -> None:
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray):
        self.calls += 1
        return [
            Detection(
                xyxy=(1, 1, 6, 6),
                class_id=1,
                label="runtime-target",
                confidence=0.9,
                color=(0, 255, 0),
            )
        ]


class _EmptyDetector:
    def __init__(self, **_kwargs) -> None:
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray):
        self.calls += 1
        return []


class _CountingDepthEstimator:
    def __init__(self, **_kwargs) -> None:
        self.calls = 0

    def estimate(self, frame_bgr: np.ndarray):
        self.calls += 1
        return np.full(frame_bgr.shape[:2], 2.0, dtype=np.float32)


class _StrictDepthEstimator:
    def __init__(self, **_kwargs) -> None:
        pass

    def estimate(self, frame_bgr: np.ndarray):
        raise AssertionError(f"depth estimator should not run for GT packets: {frame_bgr.shape}")


class _CountingTracker:
    def __init__(self, **_kwargs) -> None:
        self.calls = 0
        self.reset_calls = 0

    def update(self, **_kwargs):
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

    def reset(self) -> None:
        self.reset_calls += 1


class _WrongPoseTracker:
    def __init__(self, **_kwargs) -> None:
        self.calls = 0

    def update(self, **_kwargs):
        self.calls += 1
        bad_pose = np.eye(4, dtype=np.float32)
        bad_pose[0, 3] = 1.4
        return type(
            "TrackingResult",
            (),
            {
                "camera_pose_world": bad_pose,
                "tracking_ok": True,
                "did_reset": False,
            },
        )()

    def reset(self) -> None:
        return None


class _StrictTracker:
    def __init__(self, **_kwargs) -> None:
        pass

    def update(self, **_kwargs):
        raise AssertionError("tracker should not run for GT packets")

    def reset(self) -> None:
        raise AssertionError("tracker reset should not run for GT packets")


class _FakeMapUpdate:
    def __init__(self, frame_points_xyz: np.ndarray, frame_points_rgb: np.ndarray) -> None:
        self.dense_map_points_xyz = frame_points_xyz
        self.dense_map_points_rgb = frame_points_rgb
        self.mesh_vertices_xyz = frame_points_xyz
        self.mesh_triangles = np.empty((0, 3), dtype=np.int32)
        self.mesh_vertex_colors = frame_points_rgb
        self.trajectory_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        self.is_keyframe = True
        self.keyframe_id = 1
        self.sparse_map_points_xyz = np.empty((0, 3), dtype=np.float32)
        self.loop_closure_applied = False
        self.segment_id = 1


class _FakeMapBuilder:
    requires_point_cloud = True

    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0

    def update(self, slam_result, frame_points_xyz: np.ndarray, frame_points_rgb: np.ndarray) -> _FakeMapUpdate:
        _ = slam_result
        self.calls += 1
        return _FakeMapUpdate(np.asarray(frame_points_xyz, dtype=np.float32), np.asarray(frame_points_rgb, dtype=np.float32))

    def reset(self) -> None:
        self.reset_calls += 1


class _TsdfStyleMapBuilder:
    requires_point_cloud = False

    def __init__(
        self,
        *,
        window_keyframes: int,
        voxel_size: float,
        max_mesh_triangles: int,
        depth_trunc: float = 6.0,
        depth_sampling_stride: int = 6,
    ) -> None:
        self.window_keyframes = window_keyframes
        self.voxel_size = voxel_size
        self.max_mesh_triangles = max_mesh_triangles
        self.depth_trunc = depth_trunc
        self.depth_sampling_stride = depth_sampling_stride
        self.calls = 0

    def update(self, *, slam_result, frame_bgr: np.ndarray, depth_map: np.ndarray, intrinsics) -> _FakeMapUpdate:
        _ = (slam_result, intrinsics)
        self.calls += 1
        mesh_vertices_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        mesh_vertex_colors = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        update = _FakeMapUpdate(mesh_vertices_xyz, mesh_vertex_colors)
        update.mesh_triangles = np.empty((0, 3), dtype=np.int32)
        update.mesh_vertices_xyz = mesh_vertices_xyz
        update.mesh_vertex_colors = mesh_vertex_colors
        update.dense_map_points_xyz = mesh_vertices_xyz
        update.dense_map_points_rgb = mesh_vertex_colors
        return update

    def reset(self) -> None:
        return None


class _RecordingTsdfMapBuilder:
    requires_point_cloud = False

    def __init__(self) -> None:
        self.pose_world_history: list[np.ndarray] = []

    def update(self, *, slam_result, frame_bgr: np.ndarray, depth_map: np.ndarray, intrinsics) -> _FakeMapUpdate:
        _ = (frame_bgr, depth_map, intrinsics)
        self.pose_world_history.append(np.asarray(slam_result.pose_world, dtype=np.float32).copy())
        mesh_vertices_xyz = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        mesh_vertex_colors = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        update = _FakeMapUpdate(mesh_vertices_xyz, mesh_vertex_colors)
        update.mesh_vertices_xyz = mesh_vertices_xyz
        update.mesh_vertex_colors = mesh_vertex_colors
        update.dense_map_points_xyz = mesh_vertices_xyz
        update.dense_map_points_rgb = mesh_vertex_colors
        return update

    def reset(self) -> None:
        return None


class _FakeFrameSource:
    def __init__(self, packets: list[FramePacket]) -> None:
        self._packets = list(packets)
        self.closed = False

    def next_frame(self, *, timeout_sec: float | None = 1.0) -> FramePacket | None:
        _ = timeout_sec
        if not self._packets:
            return None
        return self._packets.pop(0)

    def close(self) -> None:
        self.closed = True


def _packet(timestamp_sec: float) -> FramePacket:
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    return FramePacket(
        frame_bgr=frame,
        timestamp_sec=timestamp_sec,
        depth_map=np.full((8, 8), 2.0, dtype=np.float32),
        pose_world_gt=np.eye(4, dtype=np.float32),
        intrinsics_gt=CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0),
        detections=[
            Detection(
                xyxy=(1, 1, 6, 6),
                class_id=1,
                label="target",
                confidence=0.99,
                color=(0, 255, 0),
            )
        ],
        tracking_state="TRACKING",
        keyframe_inserted=True,
        keyframe_id=1,
    )


def test_run_uses_frame_source_for_sim_input_without_opening_camera() -> None:
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
        segmentation_mode="off",
        graph_enabled=False,
        explanation_enabled=False,
    )
    source = _FakeFrameSource([_packet(0.0), _packet(0.25)])
    captured_factory_calls: list[AppConfig] = []
    viewer = _FakeViewer()
    detector = _CountingDetector()
    depth_estimator = _CountingDepthEstimator()
    tracker = _CountingTracker()

    def frame_source_factory(current_config: AppConfig, **_kwargs) -> _FakeFrameSource:
        captured_factory_calls.append(current_config)
        return source

    def fail_open_camera(*_args, **_kwargs):
        raise AssertionError("open_camera should not be used for sim input")

    run(
        config,
        cv2_module=_FakeCV2(),
        detector_factory=lambda **_kwargs: detector,
        depth_estimator_factory=lambda **_kwargs: depth_estimator,
        tracker_factory=lambda **_kwargs: tracker,
        map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
        viewer_factory=lambda: viewer,
        open_camera_fn=fail_open_camera,
        frame_source_factory=frame_source_factory,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
    )

    assert captured_factory_calls == [config]
    assert source.closed is True
    assert viewer.closed is True
    assert detector.calls == 1
    assert depth_estimator.calls == 2
    assert tracker.calls == 2


def test_run_rejects_external_sim_profile_without_custom_frame_source() -> None:
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
        input_source="sim",
        sim_profile="external",
        sim_perception_mode="runtime",
        segmentation_mode="off",
        graph_enabled=False,
        explanation_enabled=False,
    )

    with pytest.raises(RuntimeError, match="sim_profile=external requires --sim-external-manifest"):
        run(
            config,
            cv2_module=_FakeCV2(),
            detector_factory=_StrictDetector,
            depth_estimator_factory=_CountingDepthEstimator,
            tracker_factory=_CountingTracker,
            map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
            viewer_factory=lambda: _FakeViewer(),
            open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
            overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
        )


def test_run_builds_tsdf_style_map_builder_for_sim_tracker_path() -> None:
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
        segmentation_mode="off",
        graph_enabled=False,
        explanation_enabled=False,
    )
    source = _FakeFrameSource([_packet(0.0)])
    viewer = _FakeViewer()

    run(
        config,
        cv2_module=_FakeCV2(),
        detector_factory=lambda **_kwargs: _CountingDetector(),
        depth_estimator_factory=lambda **_kwargs: _CountingDepthEstimator(),
        tracker_factory=lambda **_kwargs: _CountingTracker(),
        map_builder_factory=_TsdfStyleMapBuilder,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
    )

    assert source.closed is True
    assert viewer.closed is True


def test_run_uses_ground_truth_packets_for_sim_when_enabled() -> None:
    config = AppConfig(
        camera_index=0,
        width=8,
        height=8,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=64,
        input_source="sim",
        sim_perception_mode="ground_truth",
        segmentation_mode="off",
        graph_enabled=False,
        explanation_enabled=False,
    )
    source = _FakeFrameSource([_packet(0.0), _packet(0.25)])
    viewer = _FakeViewer()

    run(
        config,
        cv2_module=_FakeCV2(),
        detector_factory=_StrictDetector,
        depth_estimator_factory=_StrictDepthEstimator,
        tracker_factory=_StrictTracker,
        map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
    )

    assert source.closed is True
    assert viewer.closed is True


def test_run_uses_assisted_packets_for_sim_when_enabled() -> None:
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
        segmentation_mode="off",
        graph_enabled=False,
        explanation_enabled=False,
    )
    source = _FakeFrameSource([_packet(0.0), _packet(0.25)])
    viewer = _FakeViewer()
    detector = _EmptyDetector()
    tracker = _CountingTracker()

    run(
        config,
        cv2_module=_FakeCV2(),
        detector_factory=lambda **_kwargs: detector,
        depth_estimator_factory=_StrictDepthEstimator,
        tracker_factory=lambda **_kwargs: tracker,
        map_builder_factory=lambda **_kwargs: _FakeMapBuilder(),
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, *_args, **_kwargs: frame_bgr,
    )

    assert source.closed is True
    assert viewer.closed is True
    assert detector.calls == 1
    assert tracker.calls == 2


def test_process_frame_uses_gt_pose_for_assisted_sim_mapping_when_available() -> None:
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
        segmentation_mode="off",
        graph_enabled=False,
        explanation_enabled=False,
    )
    frame_packet = _packet(0.0)
    tracker = _WrongPoseTracker()
    map_builder = _RecordingTsdfMapBuilder()

    artifacts, cached_detections = process_frame(
        frame_bgr=np.asarray(frame_packet.frame_bgr, dtype=np.uint8),
        detector=_EmptyDetector(),
        depth_estimator=_StrictDepthEstimator(),
        map_builder=map_builder,
        config=config,
        frame_index=0,
        timestamp_sec=frame_packet.timestamp_sec,
        cached_detections=[],
        tracker=tracker,
        frame_packet=frame_packet,
        assist_frame_packet_ground_truth=True,
    )

    assert tracker.calls == 1
    assert cached_detections == frame_packet.detections
    assert len(map_builder.pose_world_history) == 1
    np.testing.assert_allclose(map_builder.pose_world_history[0], frame_packet.pose_world_gt)
    np.testing.assert_allclose(artifacts.camera_pose_world, frame_packet.pose_world_gt)
