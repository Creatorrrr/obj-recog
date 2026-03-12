from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from obj_recog.camera import CameraDevice, CameraSession, open_camera
from obj_recog.config import AppConfig, parse_config
from obj_recog.frame_source import FramePacket
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.scene_graph import GraphEdge, GraphNode, SceneGraphSnapshot
from obj_recog.slam_bridge import SlamFrameResult
from obj_recog.main import _load_app_dotenv, main, process_frame, run
from obj_recog.types import Detection, PanopticSegment, SegmentationResult


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [
            Detection(
                xyxy=(0, 0, frame_bgr.shape[1] // 2, frame_bgr.shape[0] // 2),
                class_id=0,
                label="person",
                confidence=0.88,
                color=(255, 0, 0),
            )
        ]


class FakeDepthEstimator:
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        return np.full((h, w), 1.0, dtype=np.float32)


class FakeTrackingResult:
    def __init__(
        self,
        camera_pose_world: np.ndarray | None = None,
        tracking_ok: bool = True,
        did_reset: bool = False,
    ) -> None:
        self.camera_pose_world = (
            np.eye(4, dtype=np.float32) if camera_pose_world is None else camera_pose_world
        )
        self.tracking_ok = tracking_ok
        self.did_reset = did_reset


class FakeSlamBridge:
    def __init__(self, results: list[SlamFrameResult] | None = None) -> None:
        self.results = list(
            results
            or [
                SlamFrameResult(
                    tracking_state="TRACKING",
                    pose_world=np.eye(4, dtype=np.float32),
                    keyframe_inserted=True,
                    keyframe_id=1,
                    optimized_keyframe_poses={1: np.eye(4, dtype=np.float32)},
                    sparse_map_points_xyz=np.empty((0, 3), dtype=np.float32),
                    loop_closure_applied=False,
                )
            ]
        )
        self.frames: list[tuple[int, int, float]] = []
        self.closed = False

    def track(self, frame_gray: np.ndarray, timestamp: float) -> SlamFrameResult:
        self.frames.append((frame_gray.shape[1], frame_gray.shape[0], timestamp))
        if self.results:
            return self.results.pop(0)
        return SlamFrameResult(
            tracking_state="TRACKING",
            pose_world=np.eye(4, dtype=np.float32),
            keyframe_inserted=False,
            keyframe_id=None,
            optimized_keyframe_poses={},
            sparse_map_points_xyz=np.empty((0, 3), dtype=np.float32),
            loop_closure_applied=False,
        )

    def close(self) -> None:
        self.closed = True


class FakeTracker:
    def __init__(self, results: list[FakeTrackingResult] | None = None) -> None:
        self.results = list(results or [FakeTrackingResult(did_reset=True)])
        self.calls = 0
        self.reset_calls = 0

    def update(self, frame_bgr: np.ndarray, depth_map: np.ndarray, intrinsics) -> FakeTrackingResult:
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return FakeTrackingResult()

    def reset(self) -> None:
        self.reset_calls += 1


class FakeMapUpdate:
    def __init__(
        self,
        dense_map_points_xyz: np.ndarray,
        dense_map_points_rgb: np.ndarray,
        trajectory_xyz: np.ndarray,
        is_keyframe: bool,
        mesh_vertices_xyz: np.ndarray | None = None,
        mesh_triangles: np.ndarray | None = None,
        mesh_vertex_colors: np.ndarray | None = None,
        keyframe_id: int | None = None,
        sparse_map_points_xyz: np.ndarray | None = None,
        loop_closure_applied: bool = False,
        segment_id: int = 1,
    ) -> None:
        self.dense_map_points_xyz = dense_map_points_xyz
        self.dense_map_points_rgb = dense_map_points_rgb
        self.mesh_vertices_xyz = (
            dense_map_points_xyz if mesh_vertices_xyz is None else mesh_vertices_xyz
        )
        self.mesh_triangles = (
            np.empty((0, 3), dtype=np.int32) if mesh_triangles is None else mesh_triangles
        )
        self.mesh_vertex_colors = (
            dense_map_points_rgb if mesh_vertex_colors is None else mesh_vertex_colors
        )
        self.trajectory_xyz = trajectory_xyz
        self.is_keyframe = is_keyframe
        self.keyframe_id = keyframe_id
        self.sparse_map_points_xyz = (
            np.empty((0, 3), dtype=np.float32)
            if sparse_map_points_xyz is None
            else sparse_map_points_xyz
        )
        self.loop_closure_applied = loop_closure_applied
        self.segment_id = segment_id


class FakeMapBuilder:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0
        self.last_frame_points_xyz: np.ndarray | None = None
        self.last_frame_points_rgb: np.ndarray | None = None
        self.last_slam_result = None
        self.segmentation_observations: list[dict[str, object]] = []
        self._mesh_vertices_xyz = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        self._mesh_triangles = np.empty((0, 3), dtype=np.int32)
        self._mesh_vertex_colors = np.array([[0.2, 0.3, 0.4]], dtype=np.float32)

    def update(
        self,
        slam_result,
        frame_points_xyz: np.ndarray,
        frame_points_rgb: np.ndarray,
    ) -> FakeMapUpdate:
        self.calls += 1
        self.last_slam_result = slam_result
        self.last_frame_points_xyz = frame_points_xyz.copy()
        self.last_frame_points_rgb = frame_points_rgb.copy()
        size = 1 if slam_result.keyframe_inserted else min(frame_points_xyz.shape[0], 3)
        return FakeMapUpdate(
            dense_map_points_xyz=frame_points_xyz[:size],
            dense_map_points_rgb=frame_points_rgb[:size],
            trajectory_xyz=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            is_keyframe=slam_result.keyframe_inserted,
            keyframe_id=slam_result.keyframe_id,
            sparse_map_points_xyz=slam_result.sparse_map_points_xyz,
            loop_closure_applied=slam_result.loop_closure_applied,
            mesh_vertices_xyz=self._mesh_vertices_xyz,
            mesh_triangles=self._mesh_triangles,
            mesh_vertex_colors=self._mesh_vertex_colors,
            segment_id=2 if slam_result.loop_closure_applied and self.calls > 1 else 1,
        )

    def reset(self) -> None:
        self.reset_calls += 1

    def ingest_segmentation_observation(
        self,
        *,
        frame_index: int,
        camera_pose_world: np.ndarray,
        intrinsics,
        segment_id_map: np.ndarray,
        segments,
    ) -> None:
        self.segmentation_observations.append(
            {
                "frame_index": frame_index,
                "camera_pose_world": np.asarray(camera_pose_world, dtype=np.float32),
                "intrinsics": intrinsics,
                "segment_id_map": np.asarray(segment_id_map, dtype=np.int32),
                "segments": list(segments),
            }
        )
        self._mesh_vertex_colors = np.array([[0.8, 0.1, 0.2]], dtype=np.float32)

    def current_mesh_state(self):
        return (
            self._mesh_vertices_xyz.copy(),
            self._mesh_triangles.copy(),
            self._mesh_vertex_colors.copy(),
        )


class FakeCapture:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        opened: bool = True,
        frames: list[np.ndarray] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.opened = opened
        self.frames = list(frames or [np.zeros((height, width, 3), dtype=np.uint8)])
        self.released = False
        self.set_calls: list[tuple[int, float]] = []
        self.backend = None

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True

    def get(self, prop: int) -> float:
        if prop == FakeCV2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == FakeCV2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class TimeoutAwareCapture(FakeCapture):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.timeout_args: list[float | None] = []

    def read(self, timeout_sec: float | None = None) -> tuple[bool, np.ndarray | None]:
        self.timeout_args.append(timeout_sec)
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)


class FakeCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    WND_PROP_VISIBLE = 1
    CAP_AVFOUNDATION = 1200
    INTER_AREA = 1
    INTER_CUBIC = 2
    COLOR_BGR2GRAY = 6
    EVENT_LBUTTONDOWN = 1
    EVENT_MOUSEWHEEL = 10
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(
        self,
        captures: list[FakeCapture] | None = None,
        window_visible: float = 1.0,
        window_visibility_sequence: list[float] | None = None,
        key_sequence: list[int] | None = None,
        mouse_events: list[tuple[int, int, int]] | None = None,
    ) -> None:
        self._captures = list(captures or [])
        self.window_visible = window_visible
        self.window_visibility_sequence = list(window_visibility_sequence or [])
        self.key_sequence = list(key_sequence or [])
        self.mouse_events = list(mouse_events or [])
        self.destroyed = False
        self.destroy_window_calls: list[str] = []
        self.imshow_calls = 0
        self.imshow_windows: list[str] = []
        self.waitkey_calls = 0
        self.mouse_callbacks: dict[str, object] = {}

    def VideoCapture(self, camera_index: int, backend: int | None = None) -> FakeCapture:
        capture = self._captures.pop(0)
        capture.backend = backend
        return capture

    def resize(self, frame: np.ndarray, size: tuple[int, int], interpolation: int | None = None) -> np.ndarray:
        width, height = size
        channels = frame.shape[2] if frame.ndim == 3 else 1
        if channels == 1:
            return np.zeros((height, width), dtype=frame.dtype)
        return np.zeros((height, width, channels), dtype=frame.dtype)

    def cvtColor(self, frame: np.ndarray, code: int) -> np.ndarray:
        if code == self.COLOR_BGR2GRAY and frame.ndim == 3:
            return np.zeros(frame.shape[:2], dtype=frame.dtype)
        return frame

    def rectangle(self, frame, pt1, pt2, color, thickness):
        return frame

    def putText(self, frame, text, org, font, font_scale, color, thickness, line_type):
        return frame

    def imshow(self, window_name: str, frame: np.ndarray) -> None:
        self.imshow_calls += 1
        self.imshow_windows.append(window_name)

    def setMouseCallback(self, window_name: str, callback) -> None:
        self.mouse_callbacks[window_name] = callback

    def getMouseWheelDelta(self, flags: int) -> int:
        return int(flags)

    def waitKey(self, delay: int) -> int:
        self.waitkey_calls += 1
        if self.mouse_events:
            mouse_event = self.mouse_events.pop(0)
            if len(mouse_event) == 3:
                event, x, y = mouse_event
                window_name = "Object Recognition"
                flags = 0
            elif len(mouse_event) == 4 and isinstance(mouse_event[0], str):
                window_name, event, x, y = mouse_event
                flags = 0
            else:
                window_name, event, x, y, flags = mouse_event
            callback = self.mouse_callbacks.get(window_name)
            if callback is not None:
                callback(event, x, y, flags, None)
        if self.key_sequence:
            return self.key_sequence.pop(0)
        return -1

    def destroyAllWindows(self) -> None:
        self.destroyed = True

    def destroyWindow(self, window_name: str) -> None:
        self.destroy_window_calls.append(window_name)

    def getWindowProperty(self, window_name: str, prop: int) -> float:
        if self.window_visibility_sequence:
            return self.window_visibility_sequence.pop(0)
        return self.window_visible


class FakeViewer:
    def __init__(self) -> None:
        self.closed = False
        self.updates: list[int] = []
        self.last_triangles: np.ndarray | None = None
        self.last_colors: np.ndarray | None = None
        self.last_scene_graph_snapshot: SceneGraphSnapshot | None = None

    def update(
        self,
        mesh_vertices_xyz: np.ndarray,
        mesh_triangles: np.ndarray,
        mesh_vertex_colors: np.ndarray,
        *args,
    ) -> bool:
        self.updates.append(mesh_vertices_xyz.shape[0])
        self.last_triangles = mesh_triangles.copy()
        self.last_colors = mesh_vertex_colors.copy()
        if args:
            self.last_scene_graph_snapshot = args[0]
        return True

    def close(self) -> None:
        self.closed = True


class FakeSegmentationWorker:
    def __init__(self, results: list[tuple[int, SegmentationResult] | None] | None = None) -> None:
        self.results = list(results or [])
        self.submissions: list[tuple[int, np.ndarray]] = []
        self.closed = False

    def is_idle(self) -> bool:
        return True

    def submit(self, frame_index: int, frame_bgr: np.ndarray) -> None:
        self.submissions.append((frame_index, frame_bgr.copy()))

    def poll(self) -> tuple[int, SegmentationResult] | None:
        if self.results:
            return self.results.pop(0)
        return None

    def close(self) -> None:
        self.closed = True


class FakeExplanationWorker:
    def __init__(self, results: list[tuple[int, object] | None] | None = None) -> None:
        self.results = list(results or [])
        self.submissions: list[tuple[int, object]] = []
        self.closed = False

    def is_idle(self) -> bool:
        return True

    def submit(self, snapshot_id: int, payload) -> None:
        self.submissions.append((snapshot_id, payload))

    def poll(self):
        if self.results:
            return self.results.pop(0)
        return None

    def close(self) -> None:
        self.closed = True


class FakeExplainer:
    def __init__(self, text: str = "현재 장면 설명\n핵심 객체: 사람\n공간 관계: 사람은 앞쪽\n불확실성: 낮음") -> None:
        self.calls: list[object] = []
        self.text = text

    def explain(self, snapshot):
        from obj_recog.situation_explainer import ExplanationResult, ExplanationStatus

        self.calls.append(snapshot)
        return ExplanationResult(
            text=self.text,
            status=ExplanationStatus.READY,
            latency_ms=5.0,
            model="fake-model",
            error_message=None,
        )


class ProbeRecorder:
    def __init__(self) -> None:
        self.started_with: list[dict[str, object]] = []
        self.frames: list[dict[str, object]] = []
        self.finished = False

    def on_start(self, *, explanation_api_available: bool) -> None:
        self.started_with.append({"explanation_api_available": explanation_api_available})

    def record_frame(
        self,
        *,
        frame_index: int,
        frame_packet,
        artifacts,
        explanation_status,
        explanation_result,
        viewer_active: bool,
    ) -> None:
        self.frames.append(
            {
                "frame_index": frame_index,
                "frame_packet": frame_packet,
                "artifacts": artifacts,
                "explanation_status": explanation_status,
                "explanation_result": explanation_result,
                "viewer_active": viewer_active,
            }
        )

    def finish(self) -> None:
        self.finished = True


class FakeSceneGraphMemory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def update(
        self,
        *,
        frame_index: int,
        detections,
        segments,
        depth_map,
        intrinsics,
        camera_pose_world,
        slam_tracking_state: str,
    ) -> SceneGraphSnapshot:
        self.calls.append(
            {
                "frame_index": frame_index,
                "detections": list(detections),
                "segments": list(segments),
                "slam_tracking_state": slam_tracking_state,
            }
        )
        return SceneGraphSnapshot(
            frame_index=frame_index,
            camera_pose_world=np.asarray(camera_pose_world, dtype=np.float32),
            nodes=(
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
                    id="obj_person_1",
                    type="object",
                    label="person",
                    state="visible",
                    confidence=0.88,
                    world_centroid=np.array([0.0, 0.0, 2.0], dtype=np.float32),
                    last_seen_frame=frame_index,
                    last_seen_direction="front",
                    source_track_id=1,
                ),
            ),
            edges=(
                GraphEdge(
                    source="ego",
                    target="obj_person_1",
                    relation="front",
                    confidence=0.9,
                    last_updated_frame=frame_index,
                    distance_bucket="mid",
                    source_kind="detection",
                ),
            ),
            visible_node_ids=("ego", "obj_person_1"),
            visible_edge_keys=(("ego", "obj_person_1", "front"),),
        )


class FakeOpenCamera:
    def __init__(self, sessions: list[CameraSession]) -> None:
        self.sessions = list(sessions)
        self.calls: list[tuple[str | None, bool]] = []

    def __call__(
        self,
        config,
        cv2_module=None,
        preferred_name: str | None = None,
        force_default: bool = False,
    ) -> CameraSession:
        self.calls.append((preferred_name, force_default))
        return self.sessions.pop(0)


def test_process_frame_creates_frame_artifacts() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        camera_calibration="/tmp/camera.yaml",
        slam_vocabulary="/tmp/ORBvoc.txt",
        slam_width=640,
        slam_height=360,
    )
    frame = np.full((16, 16, 3), 127, dtype=np.uint8)
    detector = FakeDetector()
    depth_estimator = FakeDepthEstimator()
    slam_bridge = FakeSlamBridge()
    map_builder = FakeMapBuilder()

    artifacts, cached = process_frame(
        frame_bgr=frame,
        detector=detector,
        depth_estimator=depth_estimator,
        slam_bridge=slam_bridge,
        map_builder=map_builder,
        config=config,
        frame_index=0,
        timestamp_sec=12.25,
        cached_detections=[],
    )

    assert detector.calls == 1
    assert map_builder.calls == 1
    assert slam_bridge.frames == [(640, 360, 12.25)]
    assert len(artifacts.detections) == 1
    assert artifacts.depth_map.shape == (16, 16)
    assert artifacts.camera_pose_world.shape == (4, 4)
    assert artifacts.tracking_ok is True
    assert artifacts.is_keyframe is True
    assert artifacts.trajectory_xyz.shape == (1, 3)
    assert artifacts.dense_map_points_xyz.shape[1] == 3
    assert artifacts.dense_map_points_rgb.shape[1] == 3
    assert artifacts.mesh_vertices_xyz.shape[1] == 3
    assert artifacts.mesh_triangles.shape[1] == 3 or artifacts.mesh_triangles.shape[0] == 0
    assert artifacts.mesh_vertex_colors.shape[1] == 3
    assert artifacts.slam_tracking_state == "TRACKING"
    assert artifacts.keyframe_id == 1
    assert artifacts.depth_diagnostics is not None
    assert artifacts.depth_diagnostics.profile == "balanced"
    assert cached == artifacts.detections


def test_process_frame_reuses_cached_detections_between_detection_intervals() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        camera_calibration="/tmp/camera.yaml",
        slam_vocabulary="/tmp/ORBvoc.txt",
        slam_width=640,
        slam_height=360,
    )
    frame = np.full((16, 16, 3), 127, dtype=np.uint8)
    detector = FakeDetector()
    depth_estimator = FakeDepthEstimator()
    slam_bridge = FakeSlamBridge(
        results=[
            SlamFrameResult(
                tracking_state="TRACKING",
                pose_world=np.eye(4, dtype=np.float32),
                keyframe_inserted=False,
                keyframe_id=None,
                optimized_keyframe_poses={},
                sparse_map_points_xyz=np.empty((0, 3), dtype=np.float32),
                loop_closure_applied=False,
            )
        ]
    )
    map_builder = FakeMapBuilder()
    cached_detections = [
        Detection(
            xyxy=(1, 1, 4, 4),
            class_id=1,
            label="bottle",
            confidence=0.72,
            color=(0, 255, 0),
        )
    ]

    artifacts, cached = process_frame(
        frame_bgr=frame,
        detector=detector,
        depth_estimator=depth_estimator,
        slam_bridge=slam_bridge,
        map_builder=map_builder,
        config=config,
        frame_index=1,
        timestamp_sec=5.0,
        cached_detections=cached_detections,
    )

    assert detector.calls == 0
    assert artifacts.detections == cached_detections
    assert cached == cached_detections


def test_process_frame_uses_calibrated_intrinsics_for_dense_points() -> None:
    from obj_recog.calibration import CalibrationResult

    config = AppConfig(
        camera_index=0,
        width=2,
        height=2,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=8,
        detection_interval=2,
        inference_width=640,
        camera_calibration="/tmp/camera.yaml",
        slam_vocabulary="/tmp/ORBvoc.txt",
        slam_width=640,
        slam_height=360,
    )
    frame = np.full((2, 2, 3), 127, dtype=np.uint8)
    detector = FakeDetector()
    depth_estimator = FakeDepthEstimator()
    slam_bridge = FakeSlamBridge()
    map_builder = FakeMapBuilder()
    calibration = CalibrationResult(
        camera_matrix=np.array(
            [[8.0, 0.0, 1.0], [0.0, 6.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        distortion_coefficients=np.zeros((1, 5), dtype=np.float32),
        image_width=2,
        image_height=2,
        rms_error=0.1,
    )

    process_frame(
        frame_bgr=frame,
        detector=detector,
        depth_estimator=depth_estimator,
        slam_bridge=slam_bridge,
        map_builder=map_builder,
        config=config,
        frame_index=0,
        timestamp_sec=1.5,
        calibration=calibration,
        cached_detections=[],
    )

    assert map_builder.last_frame_points_xyz is not None
    assert map_builder.last_frame_points_xyz[0, 0] == pytest.approx(-0.125)
    assert map_builder.last_frame_points_xyz[0, 1] == pytest.approx(-1.0 / 6.0)


def test_run_requires_slam_settings_and_vocabulary_paths() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        camera_calibration=None,
        slam_vocabulary=None,
        slam_width=640,
        slam_height=360,
    )

    with pytest.raises(RuntimeError, match="SLAM vocabulary file not found|ORB-SLAM3 requires"):
        run(config, cv2_module=FakeCV2(), slam_bridge_factory=lambda **_: FakeSlamBridge())


def test_run_uses_promoted_bridge_from_runtime_calibration_resolver(tmp_path: Path) -> None:
    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    generated_path = tmp_path / "generated-camera.yaml"
    generated_path.write_text(
        "Camera.width: 640\nCamera.height: 360\nCamera.fx: 400.0\nCamera.fy: 400.0\nCamera.cx: 320.0\nCamera.cy: 180.0\nCamera.k1: 0.0\nCamera.k2: 0.0\nCamera.p1: 0.0\nCamera.p2: 0.0\nCamera.k3: 0.0\n",
        encoding="utf-8",
    )
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    fake_cv2 = FakeCV2(key_sequence=[ord("q")])
    capture = FakeCapture(width=16, height=16, frames=[np.full((16, 16, 3), 127, dtype=np.uint8)])
    viewer = FakeViewer()
    map_builder = FakeMapBuilder()
    map_builder._mesh_vertices_xyz = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    map_builder._mesh_triangles = np.array([[0, 0, 0]], dtype=np.int32)
    map_builder._mesh_vertex_colors = np.array([[0.2, 0.3, 0.4]], dtype=np.float32)
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    slam_bridge = FakeSlamBridge()
    resolver_calls: list[str] = []
    bridge_factory_calls: list[dict[str, object]] = []
    resolver_cv2: list[object] = []

    from obj_recog.calibration import load_orbslam3_settings

    def _resolver(config, camera_session, **kwargs):
        resolver_calls.append(camera_session.active_name)
        resolver_cv2.append(kwargs["cv2_module"])
        return type(
            "CalibrationBootstrap",
            (),
            {
                "source": "cache",
                "settings_path": str(generated_path),
                "calibration": load_orbslam3_settings(generated_path),
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": slam_bridge,
            },
        )()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: map_builder,
        slam_bridge_factory=lambda **kwargs: bridge_factory_calls.append(kwargs) or FakeSlamBridge(),
        viewer_factory=lambda: viewer,
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=_resolver,
    )

    assert resolver_calls == ["FaceTime HD Camera"]
    assert resolver_cv2 == [fake_cv2]
    assert bridge_factory_calls == []
    assert slam_bridge.frames == [(640, 360, pytest.approx(0.0, abs=1.0))]


def test_run_closes_calibration_window_before_runtime_model_init(tmp_path: Path) -> None:
    from obj_recog.auto_calibration import CALIBRATION_WINDOW_NAME
    from obj_recog.calibration import load_orbslam3_settings

    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    generated_path = tmp_path / "generated-camera.yaml"
    generated_path.write_text(
        "Camera.width: 640\nCamera.height: 360\nCamera.fx: 400.0\nCamera.fy: 400.0\nCamera.cx: 320.0\nCamera.cy: 180.0\nCamera.k1: 0.0\nCamera.k2: 0.0\nCamera.p1: 0.0\nCamera.p2: 0.0\nCamera.k3: 0.0\n",
        encoding="utf-8",
    )
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    fake_cv2 = FakeCV2(key_sequence=[ord("q")])
    capture = FakeCapture(width=16, height=16, frames=[np.full((16, 16, 3), 127, dtype=np.uint8)])
    viewer = FakeViewer()
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    slam_bridge = FakeSlamBridge()

    def _detector_factory(**_kwargs):
        assert fake_cv2.destroy_window_calls == [CALIBRATION_WINDOW_NAME]
        return FakeDetector()

    def _resolver(config, camera_session, **_kwargs):
        return type(
            "CalibrationBootstrap",
            (),
            {
                "source": "auto",
                "settings_path": str(generated_path),
                "calibration": load_orbslam3_settings(generated_path),
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": slam_bridge,
            },
        )()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=_detector_factory,
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: map_builder,
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: viewer,
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=_resolver,
    )


def test_run_emits_startup_logs_around_runtime_transition(tmp_path: Path) -> None:
    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    generated_path = tmp_path / "generated-camera.yaml"
    generated_path.write_text(
        "Camera.width: 640\nCamera.height: 360\nCamera.fx: 400.0\nCamera.fy: 400.0\nCamera.cx: 320.0\nCamera.cy: 180.0\nCamera.k1: 0.0\nCamera.k2: 0.0\nCamera.p1: 0.0\nCamera.p2: 0.0\nCamera.k3: 0.0\n",
        encoding="utf-8",
    )
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    fake_cv2 = FakeCV2(key_sequence=[ord("q")])
    capture = FakeCapture(width=16, height=16, frames=[np.full((16, 16, 3), 127, dtype=np.uint8)])
    viewer = FakeViewer()
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    slam_bridge = FakeSlamBridge()
    messages: list[str] = []

    from obj_recog.calibration import load_orbslam3_settings

    def _resolver(config, camera_session, **_kwargs):
        return type(
            "CalibrationBootstrap",
            (),
            {
                "source": "auto",
                "settings_path": str(generated_path),
                "calibration": load_orbslam3_settings(generated_path),
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": slam_bridge,
            },
        )()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: map_builder,
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: viewer,
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=_resolver,
        debug_log=messages.append,
    )

    assert any("runtime calibration ready" in message.lower() for message in messages)
    assert any("detector init start" in message.lower() for message in messages)
    assert any("detector init done" in message.lower() for message in messages)
    assert any("depth init start" in message.lower() for message in messages)
    assert any("depth init done" in message.lower() for message in messages)


def test_run_skips_calibration_loading_preview_when_self_calibration_is_disabled(tmp_path: Path) -> None:
    from obj_recog.calibration import load_orbslam3_settings

    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    generated_path = tmp_path / "generated-camera.yaml"
    generated_path.write_text(
        "Camera.width: 640\nCamera.height: 360\nCamera.fx: 400.0\nCamera.fy: 400.0\nCamera.cx: 320.0\nCamera.cy: 180.0\nCamera.k1: 0.0\nCamera.k2: 0.0\nCamera.p1: 0.0\nCamera.p2: 0.0\nCamera.k3: 0.0\n",
        encoding="utf-8",
    )
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    fake_cv2 = FakeCV2(key_sequence=[ord("q")])
    capture = FakeCapture(width=16, height=16, frames=[np.full((16, 16, 3), 127, dtype=np.uint8)])
    viewer = FakeViewer()
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    slam_bridge = FakeSlamBridge()

    def _detector_factory(**_kwargs):
        assert fake_cv2.destroy_window_calls == []
        assert fake_cv2.imshow_calls == 0
        return FakeDetector()

    def _resolver(config, camera_session, **_kwargs):
        return type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": str(generated_path),
                "calibration": load_orbslam3_settings(generated_path),
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": slam_bridge,
            },
        )()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=_detector_factory,
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: map_builder,
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: viewer,
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=_resolver,
    )


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        if not self._values:
            raise AssertionError("clock exhausted")
        return self._values.pop(0)


def test_run_uses_elapsed_seconds_for_slam_timestamps(tmp_path: Path) -> None:
    calibration_path = tmp_path / "camera.yaml"
    vocabulary_path = tmp_path / "ORBvoc.txt"
    calibration_path.write_text("Camera.width: 640\nCamera.height: 360\nCamera.fx: 400.0\nCamera.fy: 400.0\nCamera.cx: 320.0\nCamera.cy: 180.0\nCamera.k1: 0.0\nCamera.k2: 0.0\nCamera.p1: 0.0\nCamera.p2: 0.0\nCamera.k3: 0.0\n", encoding="utf-8")
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        camera_calibration=str(calibration_path),
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    fake_cv2 = FakeCV2(key_sequence=[ord("q")])
    capture = FakeCapture(width=16, height=16, frames=[np.full((16, 16, 3), 127, dtype=np.uint8)])
    viewer = FakeViewer()
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    slam_bridge = FakeSlamBridge()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: map_builder,
        slam_bridge_factory=lambda **_: slam_bridge,
        viewer_factory=lambda: viewer,
        open_camera_fn=FakeOpenCamera([camera_session]),
        time_source=FakeClock([100.0, 100.0, 100.25, 100.5]),
    )

    assert slam_bridge.frames == [(640, 360, 0.25)]
    assert viewer.last_triangles is not None


def test_open_camera_falls_back_when_requested_resolution_is_not_honored() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )
    first = FakeCapture(width=640, height=480)
    second = FakeCapture(width=640, height=480)
    fake_cv2 = FakeCV2(captures=[first, second])

    session = open_camera(config, cv2_module=fake_cv2, camera_lister=lambda: [])

    assert first.released is True
    assert session.capture is second
    assert session.active_index == 0
    assert session.used_fallback is False


def test_open_camera_prefers_named_device_and_falls_back_to_default_index() -> None:
    class _BrokenFFmpegCapture:
        def __init__(self, **kwargs) -> None:
            raise RuntimeError("ffmpeg could not open named camera")

    class ClosedCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            self.released = True

    config = AppConfig(
        camera_index=0,
        camera_name="iPhone",
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )
    fallback = FakeCapture(width=1280, height=720)
    fake_cv2 = FakeCV2(captures=[ClosedCapture(), fallback])
    devices = [
        CameraDevice(index=2, name="My iPhone Camera"),
        CameraDevice(index=0, name="FaceTime HD Camera"),
    ]

    session = open_camera(
        config,
        cv2_module=fake_cv2,
        camera_lister=lambda: devices,
        ffmpeg_capture_factory=_BrokenFFmpegCapture,
        platform_name="darwin",
    )

    assert session.capture is fallback
    assert session.active_index == 0
    assert session.active_name == "FaceTime HD Camera"
    assert session.requested_name == "iPhone"
    assert session.used_fallback is True


def test_run_exits_when_opencv_window_is_closed_and_cleans_up_resources() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )
    fake_cv2 = FakeCV2(window_visible=0.0)
    capture = FakeCapture(width=16, height=16, frames=[np.full((16, 16, 3), 127, dtype=np.uint8)])
    viewer = FakeViewer()
    tracker = FakeTracker()
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        tracker_factory=lambda **_: tracker,
        map_builder_factory=lambda **_: map_builder,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda cfg, cv2_module=None, preferred_name=None: camera_session,
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
    )

    assert viewer.closed is True
    assert capture.released is True
    assert fake_cv2.destroyed is True
    assert fake_cv2.imshow_calls == 2
    assert fake_cv2.imshow_windows == ["Object Recognition", "Situation Explanation"]


def test_run_shows_isometric_environment_window_for_sim_input() -> None:
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
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
    fake_cv2 = FakeCV2()
    viewer = FakeViewer()
    tracker = FakeTracker()
    map_builder = FakeMapBuilder()
    scenario_state = type(
        "ScenarioState",
        (),
        {
            "environment_objects": (
                {
                    "label": "backpack",
                    "center_world": (1.0, 0.4, 2.0),
                    "size_xyz": (0.4, 0.5, 0.3),
                    "color_bgr": (0, 255, 0),
                    "target_role": True,
                    "visible": True,
                },
            ),
            "rig_x": 0.0,
            "rig_z": 0.0,
            "yaw_deg": 0.0,
        },
    )()
    packet = FramePacket(
        frame_bgr=np.full((16, 16, 3), 127, dtype=np.uint8),
        timestamp_sec=0.0,
        depth_map=np.full((16, 16), 1.0, dtype=np.float32),
        pose_world_gt=np.eye(4, dtype=np.float32),
        intrinsics_gt=CameraIntrinsics(fx=8.0, fy=8.0, cx=8.0, cy=8.0),
        detections=[
            Detection(
                xyxy=(1, 1, 8, 8),
                class_id=0,
                label="backpack",
                confidence=0.9,
                color=(0, 255, 0),
            )
        ],
        scenario_state=scenario_state,
        tracking_state="TRACKING",
        keyframe_inserted=True,
        keyframe_id=1,
    )

    class _SinglePacketSource:
        def __init__(self) -> None:
            self._packets = [packet]
            self.closed = False

        def next_frame(self, *, timeout_sec: float | None = 1.0):
            _ = timeout_sec
            if not self._packets:
                return None
            return self._packets.pop(0)

        def close(self) -> None:
            self.closed = True

    source = _SinglePacketSource()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        tracker_factory=lambda **_: tracker,
        map_builder_factory=lambda **_: map_builder,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("open_camera should not be used")),
        frame_source_factory=lambda *_args, **_kwargs: source,
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
        environment_model_renderer=lambda state, **_kwargs: np.zeros((64, 64, 3), dtype=np.uint8),
    )

    assert source.closed is True
    assert "Environment Model" in fake_cv2.imshow_windows


def test_run_releases_camera_if_viewer_creation_fails() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )
    fake_cv2 = FakeCV2()
    capture = FakeCapture(width=16, height=16)
    tracker = FakeTracker()
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )

    with pytest.raises(RuntimeError, match="viewer boom"):
        run(
            config,
            cv2_module=fake_cv2,
            detector_factory=lambda **_: FakeDetector(),
            depth_estimator_factory=lambda **_: FakeDepthEstimator(),
            tracker_factory=lambda **_: tracker,
            map_builder_factory=lambda **_: map_builder,
            viewer_factory=lambda: (_ for _ in ()).throw(RuntimeError("viewer boom")),
            open_camera_fn=lambda cfg, cv2_module=None, preferred_name=None: camera_session,
            overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
        )

    assert capture.released is True
    assert fake_cv2.destroyed is True


def test_run_ignores_transient_window_visibility_before_first_waitkey() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )
    fake_cv2 = FakeCV2(window_visibility_sequence=[0.0, 1.0, 0.0])
    capture = FakeCapture(
        width=16,
        height=16,
        frames=[
            np.full((16, 16, 3), 127, dtype=np.uint8),
            np.full((16, 16, 3), 127, dtype=np.uint8),
        ],
    )
    viewer = FakeViewer()
    tracker = FakeTracker(
        results=[
            FakeTrackingResult(did_reset=True),
            FakeTrackingResult(did_reset=False),
        ]
    )
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        tracker_factory=lambda **_: tracker,
        map_builder_factory=lambda **_: map_builder,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda cfg, cv2_module=None, preferred_name=None: camera_session,
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
    )

    assert len(viewer.updates) == 2
    assert fake_cv2.waitkey_calls == 2


def test_run_resets_tracker_and_map_when_r_is_pressed() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        orb_features=1200,
        keyframe_translation=0.12,
        keyframe_rotation_deg=8.0,
        mapping_window_keyframes=20,
        map_voxel_size=0.03,
        max_map_points=150_000,
    )
    fake_cv2 = FakeCV2(key_sequence=[ord("r"), ord("q")])
    capture = FakeCapture(
        width=16,
        height=16,
        frames=[
            np.full((16, 16, 3), 127, dtype=np.uint8),
            np.full((16, 16, 3), 127, dtype=np.uint8),
        ],
    )
    viewer = FakeViewer()
    tracker = FakeTracker(
        results=[
            FakeTrackingResult(did_reset=True),
            FakeTrackingResult(did_reset=False),
        ]
    )
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        tracker_factory=lambda **_: tracker,
        map_builder_factory=lambda **_: map_builder,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda cfg, cv2_module=None, preferred_name=None: camera_session,
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
    )

    assert tracker.reset_calls == 1
    assert map_builder.reset_calls == 1


def test_run_resets_slam_bridge_and_map_when_r_is_pressed(tmp_path: Path) -> None:
    calibration_path = tmp_path / "camera.yaml"
    vocabulary_path = tmp_path / "ORBvoc.txt"
    calibration_path.write_text(
        "Camera.width: 640\nCamera.height: 360\nCamera.fx: 400.0\nCamera.fy: 400.0\nCamera.cx: 320.0\nCamera.cy: 180.0\nCamera.k1: 0.0\nCamera.k2: 0.0\nCamera.p1: 0.0\nCamera.p2: 0.0\nCamera.k3: 0.0\n",
        encoding="utf-8",
    )
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        camera_calibration=str(calibration_path),
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    fake_cv2 = FakeCV2(key_sequence=[ord("r"), ord("q")])
    capture = FakeCapture(
        width=16,
        height=16,
        frames=[
            np.full((16, 16, 3), 127, dtype=np.uint8),
            np.full((16, 16, 3), 127, dtype=np.uint8),
        ],
    )
    viewer = FakeViewer()
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    bridges = [FakeSlamBridge(), FakeSlamBridge()]
    created_bridges: list[FakeSlamBridge] = []

    def _slam_bridge_factory(**_kwargs):
        bridge = bridges.pop(0)
        created_bridges.append(bridge)
        return bridge

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: map_builder,
        slam_bridge_factory=_slam_bridge_factory,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda cfg, cv2_module=None, preferred_name=None: camera_session,
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
    )

    assert len(created_bridges) == 2
    assert created_bridges[0].closed is True
    assert created_bridges[1].closed is True
    assert map_builder.reset_calls == 1


def test_run_lists_cameras_and_exits_without_opening_camera(capsys: pytest.CaptureFixture[str]) -> None:
    config = AppConfig(
        camera_index=0,
        camera_name=None,
        list_cameras=True,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )
    fake_cv2 = FakeCV2()

    run(
        config,
        cv2_module=fake_cv2,
        camera_lister=lambda: [
            CameraDevice(index=0, name="FaceTime HD Camera"),
            CameraDevice(index=2, name="My iPhone Camera"),
        ],
    )

    captured = capsys.readouterr()
    assert "0: FaceTime HD Camera" in captured.out
    assert "2: My iPhone Camera" in captured.out


def test_run_falls_back_to_default_camera_after_named_camera_disconnects() -> None:
    config = AppConfig(
        camera_index=0,
        camera_name="iPhone",
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )
    fake_cv2 = FakeCV2(key_sequence=[-1, ord("q")])
    iphone_capture = FakeCapture(width=16, height=16, frames=[])
    fallback_capture = FakeCapture(
        width=16,
        height=16,
        frames=[np.full((16, 16, 3), 127, dtype=np.uint8)],
    )
    opener = FakeOpenCamera(
        sessions=[
            CameraSession(
                capture=iphone_capture,
                active_index=2,
                active_name="My iPhone Camera",
                requested_name="iPhone",
                used_fallback=False,
            ),
            CameraSession(
                capture=fallback_capture,
                active_index=0,
                active_name="FaceTime HD Camera",
                requested_name="iPhone",
                used_fallback=True,
            ),
        ]
    )
    viewer = FakeViewer()
    tracker = FakeTracker(
        results=[
            FakeTrackingResult(did_reset=True),
        ]
    )
    map_builder = FakeMapBuilder()
    overlay_calls: list[dict[str, object]] = []

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        tracker_factory=lambda **_: tracker,
        map_builder_factory=lambda **_: map_builder,
        viewer_factory=lambda: viewer,
        open_camera_fn=opener,
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: overlay_calls.append(kwargs) or frame_bgr,
    )

    assert opener.calls == [("iPhone", False), ("iPhone", True)]
    assert tracker.reset_calls == 1
    assert map_builder.reset_calls == 1
    assert overlay_calls[-1]["segmentation_alpha"] == pytest.approx(0.35)
    assert overlay_calls[-1]["segmentation_overlay_bgr"].shape == (16, 16, 3)
    assert overlay_calls[-1]["slam_tracking_state"] == "TRACKING"
    assert overlay_calls[-1]["mesh_triangle_count"] == 0
    assert overlay_calls[-1]["mesh_vertex_count"] >= 1


def test_run_passes_timeout_to_named_capture_reads() -> None:
    config = AppConfig(
        camera_index=0,
        camera_name="iPhone",
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )
    fake_cv2 = FakeCV2(key_sequence=[ord("q")])
    capture = TimeoutAwareCapture(
        width=16,
        height=16,
        frames=[np.full((16, 16, 3), 127, dtype=np.uint8)],
    )
    viewer = FakeViewer()
    tracker = FakeTracker(results=[FakeTrackingResult(did_reset=True)])
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=2,
        active_name="My iPhone Camera",
        requested_name="iPhone",
        used_fallback=False,
    )

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        tracker_factory=lambda **_: tracker,
        map_builder_factory=lambda **_: map_builder,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda cfg, cv2_module=None, preferred_name=None: camera_session,
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
    )

    assert capture.timeout_args == [1.0]


def test_run_submits_segmentation_frames_on_interval_and_reuses_last_result() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        segmentation_mode="panoptic",
        segmentation_alpha=0.35,
        segmentation_interval=2,
    )
    fake_cv2 = FakeCV2(key_sequence=[-1, -1, ord("q")])
    frames = [np.full((16, 16, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    capture = FakeCapture(width=16, height=16, frames=frames)
    viewer = FakeViewer()
    tracker = FakeTracker(
        results=[
            FakeTrackingResult(did_reset=True),
            FakeTrackingResult(did_reset=False),
            FakeTrackingResult(did_reset=False),
        ]
    )
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    worker = FakeSegmentationWorker(
        results=[
            (
                0,
                SegmentationResult(
                    overlay_bgr=np.full((16, 16, 3), 25, dtype=np.uint8),
                    segment_id_map=np.full((16, 16), 1, dtype=np.int32),
                    segments=[
                        PanopticSegment(
                            segment_id=1,
                            label_id=4,
                            label="chair",
                            color_rgb=(1, 2, 3),
                            mask=np.ones((16, 16), dtype=bool),
                            area_pixels=256,
                        )
                    ],
                ),
            ),
            None,
            None,
        ]
    )
    overlay_calls: list[dict[str, object]] = []

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        tracker_factory=lambda **_: tracker,
        map_builder_factory=lambda **_: map_builder,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda cfg, cv2_module=None, preferred_name=None: camera_session,
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: overlay_calls.append(kwargs) or frame_bgr,
        segmenter_factory=lambda **_: object(),
        segmentation_worker_factory=lambda **_: worker,
    )

    assert len(worker.submissions) == 2
    assert [submission[0] for submission in worker.submissions] == [0, 2]
    assert [submission[1][0, 0, 0] for submission in worker.submissions] == [10, 30]
    assert overlay_calls[0]["segmentation_overlay_bgr"].shape == (16, 16, 3)
    assert overlay_calls[1]["segmentation_overlay_bgr"].shape == (16, 16, 3)
    assert overlay_calls[2]["segmentation_overlay_bgr"].shape == (16, 16, 3)


def test_run_matches_segmentation_results_back_to_source_frame_and_refreshes_mesh_colors() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        segmentation_mode="panoptic",
        segmentation_interval=1,
    )
    fake_cv2 = FakeCV2(key_sequence=[-1, ord("q")])
    frames = [np.full((16, 16, 3), value, dtype=np.uint8) for value in (10, 20)]
    capture = FakeCapture(width=16, height=16, frames=frames)
    viewer = FakeViewer()
    tracker = FakeTracker(
        results=[
            FakeTrackingResult(did_reset=True),
            FakeTrackingResult(did_reset=False),
        ]
    )
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    worker = FakeSegmentationWorker(
        results=[
            (
                0,
                SegmentationResult(
                    overlay_bgr=np.full((16, 16, 3), 42, dtype=np.uint8),
                    segment_id_map=np.full((16, 16), 7, dtype=np.int32),
                    segments=[
                        PanopticSegment(
                            segment_id=7,
                            label_id=4,
                            label="chair",
                            color_rgb=(1, 2, 3),
                            mask=np.ones((16, 16), dtype=bool),
                            area_pixels=256,
                        )
                    ],
                ),
            ),
            None,
        ]
    )

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        tracker_factory=lambda **_: tracker,
        map_builder_factory=lambda **_: map_builder,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda cfg, cv2_module=None, preferred_name=None: camera_session,
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
        segmenter_factory=lambda **_: object(),
        segmentation_worker_factory=lambda **_: worker,
    )

    assert len(map_builder.segmentation_observations) == 1
    observation = map_builder.segmentation_observations[0]
    assert observation["frame_index"] == 0
    np.testing.assert_array_equal(
        observation["segment_id_map"],
        np.full((16, 16), 7, dtype=np.int32),
    )
    np.testing.assert_allclose(viewer.last_colors, np.array([[0.8, 0.1, 0.2]], dtype=np.float32))


def test_run_skips_segmentation_worker_when_mode_is_off() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        segmentation_mode="off",
    )
    fake_cv2 = FakeCV2(key_sequence=[ord("q")])
    capture = FakeCapture(width=16, height=16, frames=[np.full((16, 16, 3), 10, dtype=np.uint8)])
    viewer = FakeViewer()
    tracker = FakeTracker()
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        tracker_factory=lambda **_: tracker,
        map_builder_factory=lambda **_: map_builder,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda cfg, cv2_module=None, preferred_name=None: camera_session,
        segmentation_worker_factory=lambda **_: (_ for _ in ()).throw(AssertionError("segmentation worker should not start")),
    )


def test_run_updates_scene_graph_and_passes_snapshot_to_overlay_and_viewer() -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        segmentation_mode="off",
    )
    fake_cv2 = FakeCV2(key_sequence=[ord("q")])
    capture = FakeCapture(width=16, height=16, frames=[np.full((16, 16, 3), 20, dtype=np.uint8)])
    viewer = FakeViewer()
    tracker = FakeTracker()
    map_builder = FakeMapBuilder()
    scene_graph_memory = FakeSceneGraphMemory()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    overlay_calls: list[dict[str, object]] = []

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        tracker_factory=lambda **_: tracker,
        map_builder_factory=lambda **_: map_builder,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda cfg, cv2_module=None, preferred_name=None: camera_session,
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: overlay_calls.append(kwargs) or frame_bgr,
        scene_graph_memory_factory=lambda **_: scene_graph_memory,
    )

    assert len(scene_graph_memory.calls) == 1
    assert overlay_calls[-1]["scene_graph_snapshot"] is not None
    assert len(overlay_calls[-1]["visible_graph_nodes"]) == 2
    assert len(overlay_calls[-1]["visible_graph_edges"]) == 1
    assert viewer.last_scene_graph_snapshot is not None


def test_run_records_runtime_frames_into_validation_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        segmentation_mode="off",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("obj_recog.main._load_app_dotenv", lambda: None)
    fake_cv2 = FakeCV2(key_sequence=[ord("q")])
    capture = FakeCapture(width=16, height=16, frames=[np.full((16, 16, 3), 20, dtype=np.uint8)])
    viewer = FakeViewer()
    tracker = FakeTracker()
    map_builder = FakeMapBuilder()
    camera_session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    probe = ProbeRecorder()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        tracker_factory=lambda **_: tracker,
        map_builder_factory=lambda **_: map_builder,
        viewer_factory=lambda: viewer,
        open_camera_fn=lambda cfg, cv2_module=None, preferred_name=None: camera_session,
        validation_probe=probe,
    )

    assert probe.started_with == [{"explanation_api_available": False}]
    assert len(probe.frames) == 1
    assert probe.frames[0]["frame_index"] == 0
    assert probe.frames[0]["artifacts"].mesh_vertices_xyz.shape == (1, 3)
    assert probe.frames[0]["viewer_active"] is True
    assert probe.finished is True


def test_run_marks_explanation_disabled_without_api_key_and_keeps_loop_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("obj_recog.main._load_app_dotenv", lambda: None)
    fake_cv2 = FakeCV2(key_sequence=[ord("q")])
    camera_session = CameraSession(
        capture=FakeCapture(width=16, height=16, frames=[np.zeros((16, 16, 3), dtype=np.uint8)]),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    overlay_calls: list[dict[str, object]] = []

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: overlay_calls.append(kwargs) or frame_bgr,
    )

    assert overlay_calls[-1]["explanation_status"] == "disabled"
    assert "Situation Explanation" in fake_cv2.imshow_windows


def test_run_submits_explanation_snapshot_on_e_and_applies_latest_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from obj_recog.situation_explainer import ExplanationResult, ExplanationStatus

    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_cv2 = FakeCV2(key_sequence=[ord("e"), -1, ord("q")])
    camera_session = CameraSession(
        capture=FakeCapture(
            width=16,
            height=16,
            frames=[
                np.zeros((16, 16, 3), dtype=np.uint8),
                np.zeros((16, 16, 3), dtype=np.uint8),
                np.zeros((16, 16, 3), dtype=np.uint8),
            ],
        ),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    worker = FakeExplanationWorker(
        results=[
            None,
            (
                1,
                ExplanationResult(
                    text="현재 장면 설명\n핵심 객체: 사람\n공간 관계: 사람은 앞쪽\n불확실성: 낮음",
                    status=ExplanationStatus.READY,
                    latency_ms=123.0,
                    model="fake-model",
                    error_message=None,
                ),
            ),
        ]
    )
    overlay_calls: list[dict[str, object]] = []

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: overlay_calls.append(kwargs) or frame_bgr,
        explanation_worker_factory=lambda **_: worker,
    )

    assert len(worker.submissions) == 1
    snapshot_id, payload = worker.submissions[0]
    assert snapshot_id == 1
    assert "Visible objects" in payload.structured_context
    assert "Situation Explanation" in fake_cv2.imshow_windows
    assert overlay_calls[-1]["explanation_status"] in {"loading", "ready"}


def test_run_submits_explanation_snapshot_on_button_click(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_cv2 = FakeCV2(
        key_sequence=[-1, ord("q")],
        mouse_events=[(FakeCV2.EVENT_LBUTTONDOWN, 620, 460)],
    )
    camera_session = CameraSession(
        capture=FakeCapture(
            width=640,
            height=480,
            frames=[
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
            ],
        ),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    worker = FakeExplanationWorker()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        explanation_worker_factory=lambda **_: worker,
    )

    assert len(worker.submissions) == 1
    snapshot_id, payload = worker.submissions[0]
    assert snapshot_id == 1
    assert "Visible objects" in payload.structured_context


def test_run_auto_refreshes_explanation_while_toggle_is_on(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
        explanation_refresh_interval_sec=1.0,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_cv2 = FakeCV2(
        key_sequence=[-1, -1, ord("q")],
        mouse_events=[("Object Recognition", FakeCV2.EVENT_LBUTTONDOWN, 620, 460)],
    )
    camera_session = CameraSession(
        capture=FakeCapture(
            width=640,
            height=480,
            frames=[
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
            ],
        ),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    worker = FakeExplanationWorker()

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        explanation_worker_factory=lambda **_: worker,
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
        time_source=FakeClock([0.0, 0.0, 0.0, 0.5, 0.5, 1.6, 1.6, 2.0]),
    )

    assert [snapshot_id for snapshot_id, _payload in worker.submissions] == [1, 2]


def test_run_preserves_previous_explanation_body_while_auto_refresh_is_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from obj_recog.situation_explainer import ExplanationResult, ExplanationStatus

    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
        explanation_refresh_interval_sec=1.0,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_cv2 = FakeCV2(
        key_sequence=[-1, -1, -1, ord("q")],
        mouse_events=[("Object Recognition", FakeCV2.EVENT_LBUTTONDOWN, 620, 460)],
    )
    camera_session = CameraSession(
        capture=FakeCapture(
            width=640,
            height=480,
            frames=[
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
            ],
        ),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    worker = FakeExplanationWorker(
        results=[
            None,
            (
                1,
                ExplanationResult(
                    text="기존 분석 본문",
                    status=ExplanationStatus.READY,
                    latency_ms=12.0,
                    model="fake-model",
                    error_message=None,
                ),
            ),
            None,
            None,
        ]
    )
    panel_calls: list[dict[str, object]] = []

    def _fake_explanation_panel_renderer(
        *,
        status,
        text,
        model,
        latency_ms,
        timestamp_label,
        refresh_status="idle",
        scroll_offset=0,
        cv2_module=None,
        return_metadata=False,
        **_,
    ):
        panel_calls.append(
            {
                "status": status,
                "text": text,
                "refresh_status": refresh_status,
            }
        )
        panel = np.zeros((360, 960, 3), dtype=np.uint8)
        metadata = {
            "scroll_offset": int(scroll_offset),
            "max_scroll_offset": 0,
            "up_rect": None,
            "down_rect": None,
        }
        if return_metadata:
            return panel, metadata
        return panel

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
        explanation_worker_factory=lambda **_: worker,
        explanation_panel_renderer=_fake_explanation_panel_renderer,
        time_source=FakeClock([0.0, 0.0, 0.0, 0.5, 0.5, 0.8, 0.8, 1.6, 1.6, 1.8]),
    )

    assert panel_calls[-1]["text"] == "기존 분석 본문"
    assert panel_calls[-1]["refresh_status"] == "updating"


def test_run_preserves_previous_explanation_body_when_auto_refresh_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from obj_recog.situation_explainer import ExplanationResult, ExplanationStatus

    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
        explanation_refresh_interval_sec=1.0,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_cv2 = FakeCV2(
        key_sequence=[-1, -1, -1, ord("q")],
        mouse_events=[("Object Recognition", FakeCV2.EVENT_LBUTTONDOWN, 620, 460)],
    )
    camera_session = CameraSession(
        capture=FakeCapture(
            width=640,
            height=480,
            frames=[
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
            ],
        ),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    worker = FakeExplanationWorker(
        results=[
            None,
            (
                1,
                ExplanationResult(
                    text="기존 분석 본문",
                    status=ExplanationStatus.READY,
                    latency_ms=12.0,
                    model="fake-model",
                    error_message=None,
                ),
            ),
            None,
            (
                2,
                ExplanationResult(
                    text="",
                    status=ExplanationStatus.ERROR,
                    latency_ms=None,
                    model="fake-model",
                    error_message="refresh failed",
                ),
            ),
        ]
    )
    panel_calls: list[dict[str, object]] = []

    def _fake_explanation_panel_renderer(
        *,
        status,
        text,
        model,
        latency_ms,
        timestamp_label,
        refresh_status="idle",
        scroll_offset=0,
        cv2_module=None,
        return_metadata=False,
        **_,
    ):
        panel_calls.append(
            {
                "status": status,
                "text": text,
                "refresh_status": refresh_status,
            }
        )
        panel = np.zeros((360, 960, 3), dtype=np.uint8)
        metadata = {
            "scroll_offset": int(scroll_offset),
            "max_scroll_offset": 0,
            "up_rect": None,
            "down_rect": None,
        }
        if return_metadata:
            return panel, metadata
        return panel

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
        explanation_worker_factory=lambda **_: worker,
        explanation_panel_renderer=_fake_explanation_panel_renderer,
        time_source=FakeClock([0.0, 0.0, 0.0, 0.5, 0.5, 0.8, 0.8, 1.6, 1.6, 1.8]),
    )

    assert panel_calls[-1]["text"] == "기존 분석 본문"
    assert panel_calls[-1]["refresh_status"] == "failed"


def test_run_ignores_in_flight_refresh_result_after_toggle_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from obj_recog.situation_explainer import ExplanationResult, ExplanationStatus

    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
        explanation_refresh_interval_sec=1.0,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_cv2 = FakeCV2(
        key_sequence=[-1, -1, -1, ord("e"), ord("q")],
        mouse_events=[("Object Recognition", FakeCV2.EVENT_LBUTTONDOWN, 620, 460)],
    )
    camera_session = CameraSession(
        capture=FakeCapture(
            width=640,
            height=480,
            frames=[
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
            ],
        ),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    worker = FakeExplanationWorker(
        results=[
            None,
            (
                1,
                ExplanationResult(
                    text="기존 분석 본문",
                    status=ExplanationStatus.READY,
                    latency_ms=12.0,
                    model="fake-model",
                    error_message=None,
                ),
            ),
            None,
            None,
            (
                2,
                ExplanationResult(
                    text="늦게 도착한 새 분석",
                    status=ExplanationStatus.READY,
                    latency_ms=14.0,
                    model="fake-model",
                    error_message=None,
                ),
            ),
        ]
    )
    panel_calls: list[dict[str, object]] = []

    def _fake_explanation_panel_renderer(
        *,
        status,
        text,
        model,
        latency_ms,
        timestamp_label,
        refresh_status="idle",
        scroll_offset=0,
        cv2_module=None,
        return_metadata=False,
        **_,
    ):
        panel_calls.append(
            {
                "status": status,
                "text": text,
                "refresh_status": refresh_status,
            }
        )
        panel = np.zeros((360, 960, 3), dtype=np.uint8)
        metadata = {
            "scroll_offset": int(scroll_offset),
            "max_scroll_offset": 0,
            "up_rect": None,
            "down_rect": None,
        }
        if return_metadata:
            return panel, metadata
        return panel

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: frame_bgr,
        explanation_worker_factory=lambda **_: worker,
        explanation_panel_renderer=_fake_explanation_panel_renderer,
        time_source=FakeClock([0.0, 0.0, 0.0, 0.5, 0.5, 0.8, 0.8, 1.6, 1.6, 1.7, 1.7, 1.8]),
    )

    assert panel_calls[-1]["text"] == "기존 분석 본문"
    assert panel_calls[-1]["refresh_status"] == "idle"


def test_run_updates_explanation_scroll_offset_on_panel_button_click(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from obj_recog.situation_explainer import ExplanationResult, ExplanationStatus

    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_cv2 = FakeCV2(
        key_sequence=[-1, -1, ord("q")],
        mouse_events=[
            ("Object Recognition", FakeCV2.EVENT_LBUTTONDOWN, 620, 460),
            ("Situation Explanation", FakeCV2.EVENT_LBUTTONDOWN, 930, 320),
        ],
    )
    camera_session = CameraSession(
        capture=FakeCapture(
            width=640,
            height=480,
            frames=[
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
            ],
        ),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    worker = FakeExplanationWorker(
        results=[
            (
                1,
                ExplanationResult(
                    text="\n".join(f"line {index:02d}" for index in range(1, 21)),
                    status=ExplanationStatus.READY,
                    latency_ms=25.0,
                    model="fake-model",
                    error_message=None,
                ),
            ),
            None,
        ]
    )
    scroll_offsets: list[int] = []

    def _fake_explanation_panel_renderer(
        *,
        status,
        text,
        model,
        latency_ms,
        timestamp_label,
        scroll_offset=0,
        cv2_module=None,
        return_metadata=False,
        **_,
    ):
        scroll_offsets.append(int(scroll_offset))
        panel = np.zeros((360, 960, 3), dtype=np.uint8)
        metadata = {
            "scroll_offset": int(scroll_offset),
            "max_scroll_offset": 4,
            "up_rect": (900, 16, 944, 56),
            "down_rect": (900, 304, 944, 344),
        }
        if return_metadata:
            return panel, metadata
        return panel

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        explanation_worker_factory=lambda **_: worker,
        explanation_panel_renderer=_fake_explanation_panel_renderer,
    )

    assert scroll_offsets[0] == 0
    assert scroll_offsets[-1] == 1


def test_run_updates_explanation_scroll_offset_on_mouse_wheel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from obj_recog.situation_explainer import ExplanationResult, ExplanationStatus

    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_cv2 = FakeCV2(
        key_sequence=[-1, -1, ord("q")],
        mouse_events=[
            ("Object Recognition", FakeCV2.EVENT_LBUTTONDOWN, 620, 460),
            ("Situation Explanation", FakeCV2.EVENT_MOUSEWHEEL, 930, 320, -120),
        ],
    )
    camera_session = CameraSession(
        capture=FakeCapture(
            width=640,
            height=480,
            frames=[
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
            ],
        ),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    worker = FakeExplanationWorker(
        results=[
            (
                1,
                ExplanationResult(
                    text="\n".join(f"line {index:02d}" for index in range(1, 21)),
                    status=ExplanationStatus.READY,
                    latency_ms=25.0,
                    model="fake-model",
                    error_message=None,
                ),
            ),
            None,
        ]
    )
    scroll_offsets: list[int] = []

    def _fake_explanation_panel_renderer(
        *,
        status,
        text,
        model,
        latency_ms,
        timestamp_label,
        scroll_offset=0,
        cv2_module=None,
        return_metadata=False,
        **_,
    ):
        scroll_offsets.append(int(scroll_offset))
        panel = np.zeros((360, 960, 3), dtype=np.uint8)
        metadata = {
            "scroll_offset": int(scroll_offset),
            "max_scroll_offset": 4,
            "up_rect": (900, 16, 944, 56),
            "down_rect": (900, 304, 944, 344),
        }
        if return_metadata:
            return panel, metadata
        return panel

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        explanation_worker_factory=lambda **_: worker,
        explanation_panel_renderer=_fake_explanation_panel_renderer,
    )

    assert scroll_offsets[0] == 0
    assert scroll_offsets[-1] == 1


def test_run_toggles_depth_debug_level_with_d_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_cv2 = FakeCV2(key_sequence=[ord("d"), ord("q")])
    camera_session = CameraSession(
        capture=FakeCapture(
            width=16,
            height=16,
            frames=[
                np.zeros((16, 16, 3), dtype=np.uint8),
                np.zeros((16, 16, 3), dtype=np.uint8),
            ],
        ),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    overlay_calls: list[dict[str, object]] = []

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: overlay_calls.append(kwargs) or frame_bgr,
        explanation_worker_factory=lambda **_: FakeExplanationWorker(),
    )

    assert overlay_calls[0]["depth_debug_level"] == "basic"
    assert overlay_calls[-1]["depth_debug_level"] == "detailed"


def test_run_default_explanation_path_uses_default_explainer_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("obj_recog.main.OpenAISituationExplainer", lambda **_: FakeExplainer())
    fake_cv2 = FakeCV2(key_sequence=[ord("e"), -1, ord("q")])
    camera_session = CameraSession(
        capture=FakeCapture(
            width=16,
            height=16,
            frames=[
                np.zeros((16, 16, 3), dtype=np.uint8),
                np.zeros((16, 16, 3), dtype=np.uint8),
                np.zeros((16, 16, 3), dtype=np.uint8),
            ],
        ),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    overlay_calls: list[dict[str, object]] = []

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: overlay_calls.append(kwargs) or frame_bgr,
    )

    assert overlay_calls[-1]["explanation_status"] in {"loading", "ready"}


def test_run_ignores_stale_explanation_result_after_reset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from obj_recog.situation_explainer import ExplanationResult, ExplanationStatus

    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_cv2 = FakeCV2(key_sequence=[ord("e"), ord("r"), -1, ord("q")])
    camera_session = CameraSession(
        capture=FakeCapture(
            width=16,
            height=16,
            frames=[
                np.zeros((16, 16, 3), dtype=np.uint8),
                np.zeros((16, 16, 3), dtype=np.uint8),
                np.zeros((16, 16, 3), dtype=np.uint8),
                np.zeros((16, 16, 3), dtype=np.uint8),
            ],
        ),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    worker = FakeExplanationWorker(
        results=[
            None,
            None,
            (
                1,
                ExplanationResult(
                    text="현재 장면 설명\n핵심 객체: 사람\n공간 관계: 사람은 앞쪽\n불확실성: 낮음",
                    status=ExplanationStatus.READY,
                    latency_ms=123.0,
                    model="fake-model",
                    error_message=None,
                ),
            ),
            None,
        ]
    )
    overlay_calls: list[dict[str, object]] = []

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        overlay_renderer=lambda frame_bgr, detections, fps, **kwargs: overlay_calls.append(kwargs) or frame_bgr,
        explanation_worker_factory=lambda **_: worker,
    )

    assert all(call["explanation_status"] != "ready" for call in overlay_calls[1:])


def test_run_loads_openai_api_key_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vocabulary_path = tmp_path / "ORBvoc.txt"
    vocabulary_path.write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")
    config = AppConfig(
        camera_index=0,
        width=16,
        height=16,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
        disable_slam_calibration=True,
        slam_vocabulary=str(vocabulary_path),
        slam_width=640,
        slam_height=360,
    )
    fake_cv2 = FakeCV2(key_sequence=[ord("q")])
    camera_session = CameraSession(
        capture=FakeCapture(width=16, height=16, frames=[np.zeros((16, 16, 3), dtype=np.uint8)]),
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    captured: dict[str, object] = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    run(
        config,
        cv2_module=fake_cv2,
        detector_factory=lambda **_: FakeDetector(),
        depth_estimator_factory=lambda **_: FakeDepthEstimator(),
        map_builder_factory=lambda **_: FakeMapBuilder(),
        slam_bridge_factory=lambda **_: FakeSlamBridge(),
        viewer_factory=lambda: FakeViewer(),
        open_camera_fn=FakeOpenCamera([camera_session]),
        runtime_calibration_resolver=lambda *args, **kwargs: type(
            "CalibrationBootstrap",
            (),
            {
                "source": "disabled",
                "settings_path": "/tmp/generated.yaml",
                "calibration": None,
                "cache_entry": None,
                "warmup_restarted": False,
                "promoted_bridge": FakeSlamBridge(),
            },
        )(),
        situation_explainer_factory=lambda **kwargs: captured.setdefault("api_key", kwargs["api_key"]) or FakeExplainer(),
        explanation_worker_factory=lambda **_: FakeExplanationWorker(),
    )

    assert captured["api_key"] == "dotenv-key"


def test_load_app_dotenv_exposes_camera_calibration_for_parse_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "camera.yaml"
    calibration_path.write_text("Camera.width: 640\nCamera.height: 360\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        f"CAMERA_CALIBRATION={calibration_path}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CAMERA_CALIBRATION", raising=False)
    monkeypatch.chdir(tmp_path)

    _load_app_dotenv()
    config = parse_config([])

    assert config.camera_calibration == str(calibration_path)


def test_main_loads_dotenv_before_parse_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "camera.yaml"
    calibration_path.write_text("Camera.width: 640\nCamera.height: 360\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        f"CAMERA_CALIBRATION={calibration_path}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CAMERA_CALIBRATION", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["obj_recog.main"])

    captured: dict[str, AppConfig] = {}

    def fake_run(config: AppConfig, **_: object) -> None:
        captured["config"] = config

    monkeypatch.setattr("obj_recog.main.run", fake_run)

    main()

    assert captured["config"].camera_calibration == str(calibration_path)
