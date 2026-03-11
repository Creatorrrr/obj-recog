from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pytest

from obj_recog.auto_calibration import (
    ALGORITHM_VERSION,
    CalibrationValidationResult,
    SLAMBootstrapManager,
    SlamBootstrapMetrics,
    WarmupCalibrationResult,
    build_camera_fingerprint,
    calibration_requires_restart,
    create_approximate_calibration,
    default_calibration_cache_dir,
    ensure_runtime_calibration,
    load_cached_calibration_entry,
    refine_focal_lengths,
    store_calibration_cache,
)
from obj_recog.calibration import CalibrationResult, load_orbslam3_settings
from obj_recog.camera import CameraSession
from obj_recog.config import AppConfig
from obj_recog.slam_bridge import KeyframeObservation


class _Capture:
    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self._width = width
        self._height = height

    def get(self, prop: int) -> float:
        if prop == 3:
            return float(self._width)
        if prop == 4:
            return float(self._height)
        return 0.0


def _session(name: str = "FaceTime HD Camera", width: int = 1280, height: int = 720) -> CameraSession:
    return CameraSession(
        capture=_Capture(width=width, height=height),
        active_index=0,
        active_name=name,
        requested_name=None,
        used_fallback=False,
    )


def _config(**kwargs) -> AppConfig:
    values = dict(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=60_000,
        slam_vocabulary="/tmp/ORBvoc.txt",
        slam_width=640,
        slam_height=360,
    )
    values.update(kwargs)
    return AppConfig(**values)


def _metrics(**overrides) -> SlamBootstrapMetrics:
    values = dict(
        tracking_ok_ratio=0.92,
        valid_keyframes=7,
        unique_points=310,
        mean_track_length=3.6,
        median_reprojection_error=1.6,
    )
    values.update(overrides)
    return SlamBootstrapMetrics(**values)


def _warmup_result(
    calibration: CalibrationResult,
    *,
    bridge: object | None = None,
    warmup_restarted: bool = False,
    settings_path: str = "/tmp/generated.yaml",
    metrics: SlamBootstrapMetrics | None = None,
) -> WarmupCalibrationResult:
    return WarmupCalibrationResult(
        calibration=calibration,
        settings_path=settings_path,
        metrics=_metrics() if metrics is None else metrics,
        bridge=bridge,
        warmup_restarted=warmup_restarted,
    )


def test_default_calibration_cache_dir_uses_macos_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/tmp/test-home")

    cache_dir = default_calibration_cache_dir()

    assert str(cache_dir) == "/tmp/test-home/Library/Caches/obj-recog/calibration"


def test_build_camera_fingerprint_uses_camera_and_slam_shape() -> None:
    fingerprint = build_camera_fingerprint(_session(name="My Camera", width=1920, height=1080), _config())

    assert "My Camera" in fingerprint
    assert "1920x1080" in fingerprint
    assert "640x360" in fingerprint
    assert ALGORITHM_VERSION in fingerprint


def test_store_and_load_cached_calibration_entry_round_trips(tmp_path: Path) -> None:
    calibration = create_approximate_calibration(image_width=640, image_height=360)
    entry = store_calibration_cache(
        cache_dir=tmp_path,
        fingerprint="camera|640x360",
        calibration=calibration,
        active_name="FaceTime HD Camera",
        validation_metrics={"median_reprojection_error": 1.25},
        fps=30.0,
        created_at="2026-03-09T12:00:00Z",
    )

    loaded = load_cached_calibration_entry(tmp_path, "camera|640x360")

    assert loaded is not None
    assert Path(loaded.yaml_path).is_file()
    assert Path(loaded.metadata_path).is_file()
    metadata = json.loads(Path(loaded.metadata_path).read_text(encoding="utf-8"))
    assert metadata["active_name"] == "FaceTime HD Camera"
    assert metadata["validation_metrics"]["median_reprojection_error"] == 1.25
    assert loaded.camera_fingerprint == entry.camera_fingerprint
    assert loaded.stale is False


def test_refine_calibration_caps_observations_to_recent_keyframes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import obj_recog.auto_calibration as auto_calibration

    calibration = create_approximate_calibration(image_width=640, image_height=360)
    manager = SLAMBootstrapManager(calibration=calibration)
    manager._keyframe_poses = {
        keyframe_id: np.eye(4, dtype=np.float32)
        for keyframe_id in range(1, 13)
    }
    for keyframe_id in range(1, 13):
        for point_id in range(200):
            manager._observation_by_key[(keyframe_id, point_id)] = KeyframeObservation(
                keyframe_id=keyframe_id,
                point_id=(keyframe_id * 1_000) + point_id,
                u=float(point_id),
                v=float(point_id) * 0.5,
                x=float(point_id) * 0.01,
                y=float(point_id) * 0.02,
                z=1.0 + (point_id * 0.001),
            )

    captured: dict[str, object] = {}

    def _fake_refine(**kwargs):
        observations = kwargs["keyframe_observations"]
        captured["observations"] = observations
        captured["keyframes"] = sorted({observation.keyframe_id for observation in observations})
        return kwargs["initial_fx"], kwargs["initial_fy"], dict(kwargs["keyframe_poses"]), {}

    monkeypatch.setattr(auto_calibration, "refine_focal_lengths", _fake_refine)

    refined = manager.refine_calibration()

    assert refined is not None
    observations = captured["observations"]
    keyframes = captured["keyframes"]
    assert len(observations) <= (
        auto_calibration._MAX_REFINEMENT_KEYFRAMES * auto_calibration._MAX_REFINEMENT_OBSERVATIONS_PER_KEYFRAME
    )
    assert keyframes == list(
        range(13 - auto_calibration._MAX_REFINEMENT_KEYFRAMES, 13)
    )


def test_run_slam_self_calibration_emits_progress_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import obj_recog.auto_calibration as auto_calibration

    calibration = create_approximate_calibration(image_width=640, image_height=360)
    config = _config()
    session = _session()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    bridge = object()
    messages: list[str] = []

    class _Result:
        tracking_state = "TRACKING"
        tracked_feature_count = 128
        tracking_ok = True
        median_reprojection_error = 1.4
        optimized_keyframe_poses = {index: np.eye(4, dtype=np.float32) for index in range(1, 8)}
        keyframe_id = 7
        keyframe_observations = [
            KeyframeObservation(
                keyframe_id=keyframe_id,
                point_id=point_id,
                u=float(point_id),
                v=float(point_id) * 0.5,
                x=0.0,
                y=0.0,
                z=1.0,
            )
            for keyframe_id in range(1, 8)
            for point_id in range(260)
        ]

    result = _Result()

    monkeypatch.setattr(auto_calibration, "_resize_for_slam", lambda frame_bgr, config, cv2_module: np.zeros((360, 640), dtype=np.uint8))
    monkeypatch.setattr(
        auto_calibration,
        "read_camera_frame",
        lambda capture, timeout_sec=1.0: (True, frame.copy()),
    )

    def _factory(**kwargs):
        class _Bridge:
            def track(self, frame_gray, timestamp):
                return result

            def close(self):
                return None

        return _Bridge()

    def _fake_refine(self, *, cv2_module=None, debug_log=None):
        self._tracking_ok_frames = 10
        self._total_frames = 10
        self._reprojection_errors = [1.2]
        self._keyframe_poses = {index: np.eye(4, dtype=np.float32) for index in range(1, 8)}
        observations = {}
        for point_id in range(260):
            for keyframe_id in range(1, 8):
                observations[(keyframe_id, point_id)] = KeyframeObservation(
                    keyframe_id=keyframe_id,
                    point_id=point_id,
                    u=float(point_id),
                    v=float(point_id),
                    x=0.0,
                    y=0.0,
                    z=1.0,
                )
        self._observation_by_key = observations
        return calibration

    monkeypatch.setattr(auto_calibration.SLAMBootstrapManager, "refine_calibration", _fake_refine)

    auto_calibration.run_slam_self_calibration(
        camera_session=session,
        config=config,
        calibration=calibration,
        settings_path="/tmp/test.yaml",
        slam_bridge_factory=_factory,
        cv2_module=type(
            "_FakeCV2",
            (),
            {
                "INTER_AREA": 1,
                "COLOR_BGR2GRAY": 6,
                "FONT_HERSHEY_SIMPLEX": 0,
                "LINE_AA": 16,
                "resize": staticmethod(lambda frame, size, interpolation=None: np.zeros((size[1], size[0], 3), dtype=np.uint8)),
                "cvtColor": staticmethod(lambda frame, code: np.zeros(frame.shape[:2], dtype=np.uint8)),
                "putText": staticmethod(lambda *args, **kwargs: None),
                "imshow": staticmethod(lambda *args, **kwargs: None),
                "waitKey": staticmethod(lambda delay: -1),
                "destroyWindow": staticmethod(lambda name: None),
            },
        )(),
        frame_reader=lambda capture, timeout_sec=1.0: (True, frame.copy()),
        time_source=itertools.count(start=0.0, step=0.1).__next__,
        debug_log=messages.append,
    )

    assert any("acceptance reached" in message.lower() for message in messages)
    assert any("refinement started" in message.lower() for message in messages)
    assert any("warm-up complete" in message.lower() for message in messages)


def test_refine_calibration_logs_prepared_input(monkeypatch: pytest.MonkeyPatch) -> None:
    import obj_recog.auto_calibration as auto_calibration

    calibration = create_approximate_calibration(image_width=640, image_height=360)
    manager = SLAMBootstrapManager(calibration=calibration)
    manager._keyframe_poses = {
        keyframe_id: np.eye(4, dtype=np.float32)
        for keyframe_id in range(1, 10)
    }
    for keyframe_id in range(1, 10):
        for point_id in range(200):
            manager._observation_by_key[(keyframe_id, point_id)] = KeyframeObservation(
                keyframe_id=keyframe_id,
                point_id=point_id,
                u=float(point_id),
                v=float(point_id),
                x=0.1,
                y=0.2,
                z=1.0,
            )

    messages: list[str] = []

    def _fake_refine(**kwargs):
        return kwargs["initial_fx"], kwargs["initial_fy"], dict(kwargs["keyframe_poses"]), {}

    monkeypatch.setattr(auto_calibration, "refine_focal_lengths", _fake_refine)

    refined = manager.refine_calibration(debug_log=messages.append)

    assert refined is not None
    assert any("refinement input prepared" in message.lower() for message in messages)


def test_refine_focal_lengths_logs_solver_boundaries() -> None:
    points_world = np.array(
        [
            [-0.3, -0.2, 3.0],
            [-0.1, 0.2, 3.2],
            [0.2, -0.1, 2.8],
            [0.4, 0.3, 3.4],
        ],
        dtype=np.float32,
    )
    poses = {
        10: np.eye(4, dtype=np.float32),
        11: np.array(
            [
                [1.0, 0.0, 0.0, 0.15],
                [0.0, 1.0, 0.0, 0.01],
                [0.0, 0.0, 1.0, 0.02],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }
    observations: list[KeyframeObservation] = []
    for point_id, point_world in enumerate(points_world):
        for keyframe_id, pose in poses.items():
            camera_point = np.linalg.inv(pose)[:3, :] @ np.append(point_world, 1.0)
            observations.append(
                KeyframeObservation(
                    keyframe_id=keyframe_id,
                    point_id=point_id,
                    u=float(620.0 * (camera_point[0] / camera_point[2]) + 320.0),
                    v=float(610.0 * (camera_point[1] / camera_point[2]) + 180.0),
                    x=float(point_world[0]),
                    y=float(point_world[1]),
                    z=float(point_world[2]),
                )
            )
    messages: list[str] = []

    refine_focal_lengths(
        initial_fx=560.0,
        initial_fy=555.0,
        cx=320.0,
        cy=180.0,
        keyframe_poses=poses,
        keyframe_observations=observations,
        debug_log=messages.append,
    )

    assert any(
        "solver start" in message.lower()
        and ("variables=2," in message.lower() or "variables=2)" in message.lower())
        for message in messages
    )
    assert any("solver done" in message.lower() for message in messages)


def test_ensure_runtime_calibration_prefers_explicit_path(tmp_path: Path) -> None:
    calibration_path = tmp_path / "camera.yaml"
    calibration = create_approximate_calibration(image_width=640, image_height=360)
    calibration_path.write_text(
        "Camera.width: 640\nCamera.height: 360\nCamera.fx: 576.0\nCamera.fy: 576.0\nCamera.cx: 320.0\nCamera.cy: 180.0\nCamera.k1: 0.0\nCamera.k2: 0.0\nCamera.p1: 0.0\nCamera.p2: 0.0\nCamera.k3: 0.0\n",
        encoding="utf-8",
    )
    config = _config(camera_calibration=str(calibration_path))

    state = ensure_runtime_calibration(
        config,
        _session(),
        slam_bridge_factory=lambda **_: (_ for _ in ()).throw(AssertionError("bridge should not start")),
        validator_runner=lambda **_: (_ for _ in ()).throw(AssertionError("validator should not run")),
        warmup_runner=lambda **_: (_ for _ in ()).throw(AssertionError("warmup should not run")),
    )

    assert state.source == "explicit"
    assert state.settings_path == str(calibration_path)
    assert state.calibration.image_width == calibration.image_width
    assert state.warmup_restarted is False
    assert state.promoted_bridge is None


def test_ensure_runtime_calibration_generates_missing_explicit_path(tmp_path: Path) -> None:
    generated_path = tmp_path / "calibration" / "camera.yaml"
    generated = CalibrationResult(
        camera_matrix=np.array([[500.0, 0.0, 320.0], [0.0, 505.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        distortion_coefficients=np.zeros((1, 5), dtype=np.float32),
        image_width=640,
        image_height=360,
        rms_error=1.8,
    )
    config = _config(camera_calibration=str(generated_path))
    warmup_calls: list[int] = []

    state = ensure_runtime_calibration(
        config,
        _session(),
        slam_bridge_factory=lambda **_: object(),
        validator_runner=lambda **_: (_ for _ in ()).throw(AssertionError("validator should not run")),
        warmup_runner=lambda **_: warmup_calls.append(1) or _warmup_result(
            generated,
            bridge=object(),
            settings_path="/tmp/generated.yaml",
            metrics=_metrics(median_reprojection_error=1.8),
        ),
    )

    assert warmup_calls == [1]
    assert state.source == "auto"
    assert state.settings_path == str(generated_path)
    assert Path(state.settings_path).is_file()
    assert state.cache_entry is None
    assert state.promoted_bridge is not None
    loaded = load_orbslam3_settings(generated_path)
    assert loaded.image_width == 640
    assert loaded.image_height == 360
    assert float(loaded.camera_matrix[0, 0]) == pytest.approx(500.0)
    assert float(loaded.camera_matrix[1, 1]) == pytest.approx(505.0)


def test_ensure_runtime_calibration_writes_disabled_mode_to_missing_explicit_path(tmp_path: Path) -> None:
    generated_path = tmp_path / "calibration" / "camera.yaml"
    config = _config(
        camera_calibration=str(generated_path),
        disable_slam_calibration=True,
    )

    state = ensure_runtime_calibration(
        config,
        _session(),
        slam_bridge_factory=lambda **_: (_ for _ in ()).throw(AssertionError("bridge should not start")),
        validator_runner=lambda **_: (_ for _ in ()).throw(AssertionError("validator should not run")),
        warmup_runner=lambda **_: (_ for _ in ()).throw(AssertionError("warmup should not run")),
    )

    assert state.source == "disabled"
    assert state.settings_path == str(generated_path)
    assert Path(state.settings_path).is_file()
    assert state.cache_entry is None
    loaded = load_orbslam3_settings(generated_path)
    assert loaded.image_width == config.slam_width
    assert loaded.image_height == config.slam_height


def test_ensure_runtime_calibration_allows_disabling_self_calibration(tmp_path: Path) -> None:
    session = _session()
    config = _config(
        calibration_cache_dir=str(tmp_path),
        disable_slam_calibration=True,
    )

    state = ensure_runtime_calibration(
        config,
        session,
        slam_bridge_factory=lambda **_: (_ for _ in ()).throw(AssertionError("bridge should not start")),
        validator_runner=lambda **_: (_ for _ in ()).throw(AssertionError("validator should not run")),
        warmup_runner=lambda **_: (_ for _ in ()).throw(AssertionError("warmup should not run")),
    )

    assert state.source == "disabled"
    assert Path(state.settings_path).is_file()
    assert state.settings_path.endswith("-disabled.yaml")
    assert state.calibration.image_width == config.slam_width
    assert state.calibration.image_height == config.slam_height
    assert state.cache_entry is None
    assert state.warmup_restarted is False
    assert state.promoted_bridge is None


def test_ensure_runtime_calibration_reuses_validated_cache_with_promoted_bridge(tmp_path: Path) -> None:
    calibration = create_approximate_calibration(image_width=640, image_height=360)
    session = _session()
    config = _config(calibration_cache_dir=str(tmp_path))
    entry = store_calibration_cache(
        cache_dir=tmp_path,
        fingerprint=build_camera_fingerprint(session, config),
        calibration=calibration,
        active_name=session.active_name,
        validation_metrics={"median_reprojection_error": 1.0},
        fps=30.0,
        created_at="2026-03-09T12:00:00Z",
    )
    promoted_bridge = object()

    state = ensure_runtime_calibration(
        config,
        session,
        slam_bridge_factory=lambda **_: object(),
        validator_runner=lambda **_: CalibrationValidationResult(
            accepted=True,
            reason="ok",
            metrics=_metrics(),
            bridge=promoted_bridge,
        ),
        warmup_runner=lambda **_: (_ for _ in ()).throw(AssertionError("warmup should not run")),
    )

    assert state.source == "cache"
    assert state.settings_path == entry.yaml_path
    assert state.cache_entry is not None
    assert state.promoted_bridge is promoted_bridge
    assert state.warmup_restarted is False


def test_ensure_runtime_calibration_skips_stale_cache_and_runs_warmup(tmp_path: Path) -> None:
    calibration = create_approximate_calibration(image_width=640, image_height=360)
    session = _session()
    config = _config(calibration_cache_dir=str(tmp_path))
    entry = store_calibration_cache(
        cache_dir=tmp_path,
        fingerprint=build_camera_fingerprint(session, config),
        calibration=calibration,
        active_name=session.active_name,
        validation_metrics={"median_reprojection_error": 1.0},
        fps=30.0,
        created_at="2026-03-09T12:00:00Z",
        stale=True,
        stale_reason="bad tracking",
    )
    validator_calls: list[int] = []
    warmup_calls: list[int] = []

    state = ensure_runtime_calibration(
        config,
        session,
        slam_bridge_factory=lambda **_: object(),
        validator_runner=lambda **_: validator_calls.append(1) or CalibrationValidationResult(
            accepted=True,
            reason="ok",
            metrics=_metrics(),
        ),
        warmup_runner=lambda **_: warmup_calls.append(1) or _warmup_result(
            calibration,
            bridge=object(),
            settings_path=str(Path(entry.yaml_path)),
        ),
    )

    assert validator_calls == []
    assert warmup_calls == [1]
    assert state.source == "auto"


def test_ensure_runtime_calibration_marks_failed_cache_stale_and_overwrites_after_successful_warmup(
    tmp_path: Path,
) -> None:
    calibration = create_approximate_calibration(image_width=640, image_height=360)
    session = _session()
    config = _config(calibration_cache_dir=str(tmp_path))
    entry = store_calibration_cache(
        cache_dir=tmp_path,
        fingerprint=build_camera_fingerprint(session, config),
        calibration=calibration,
        active_name=session.active_name,
        validation_metrics={"median_reprojection_error": 1.0},
        fps=30.0,
        created_at="2026-03-09T12:00:00Z",
    )

    generated = CalibrationResult(
        camera_matrix=np.array([[500.0, 0.0, 320.0], [0.0, 505.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        distortion_coefficients=np.zeros((1, 5), dtype=np.float32),
        image_width=640,
        image_height=360,
        rms_error=1.8,
    )

    state = ensure_runtime_calibration(
        config,
        session,
        slam_bridge_factory=lambda **_: object(),
        validator_runner=lambda **_: CalibrationValidationResult(
            accepted=False,
            reason="bad reprojection",
            metrics=_metrics(median_reprojection_error=4.0),
        ),
        warmup_runner=lambda **_: _warmup_result(
            generated,
            bridge=object(),
            settings_path=str(Path(entry.yaml_path)),
            metrics=_metrics(median_reprojection_error=1.8),
        ),
    )

    assert state.source == "auto"
    assert Path(state.settings_path).is_file()
    metadata = json.loads(Path(entry.metadata_path).read_text(encoding="utf-8"))
    assert metadata["stale"] is False
    assert metadata["validation_metrics"]["median_reprojection_error"] == pytest.approx(1.8)


def test_ensure_runtime_calibration_respects_recalibrate_flag(tmp_path: Path) -> None:
    calibration = create_approximate_calibration(image_width=640, image_height=360)
    session = _session()
    config = _config(calibration_cache_dir=str(tmp_path), recalibrate=True)
    store_calibration_cache(
        cache_dir=tmp_path,
        fingerprint=build_camera_fingerprint(session, config),
        calibration=calibration,
        active_name=session.active_name,
        validation_metrics={"median_reprojection_error": 1.0},
        fps=30.0,
        created_at="2026-03-09T12:00:00Z",
    )

    state = ensure_runtime_calibration(
        config,
        session,
        slam_bridge_factory=lambda **_: object(),
        validator_runner=lambda **_: (_ for _ in ()).throw(AssertionError("validator should not run")),
        warmup_runner=lambda **_: _warmup_result(calibration, bridge=object()),
    )

    assert state.source == "auto"


def test_calibration_requires_restart_uses_one_percent_threshold() -> None:
    calibration = create_approximate_calibration(image_width=640, image_height=360)
    tiny_change = CalibrationResult(
        camera_matrix=np.array([[581.0, 0.0, 320.0], [0.0, 581.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        distortion_coefficients=np.zeros((1, 5), dtype=np.float32),
        image_width=640,
        image_height=360,
        rms_error=1.0,
    )
    large_change = CalibrationResult(
        camera_matrix=np.array([[590.0, 0.0, 320.0], [0.0, 590.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        distortion_coefficients=np.zeros((1, 5), dtype=np.float32),
        image_width=640,
        image_height=360,
        rms_error=1.0,
    )

    assert calibration_requires_restart(calibration, tiny_change) is False
    assert calibration_requires_restart(calibration, large_change) is True


def test_refine_focal_lengths_from_keyframe_observations_converges() -> None:
    points_world = np.array(
        [
            [-0.3, -0.2, 3.0],
            [-0.1, 0.2, 3.2],
            [0.2, -0.1, 2.8],
            [0.4, 0.3, 3.4],
            [0.0, 0.0, 2.9],
            [-0.25, 0.15, 3.1],
        ],
        dtype=np.float32,
    )
    fx_true = 620.0
    fy_true = 610.0
    cx = 320.0
    cy = 180.0
    poses = {
        10: np.eye(4, dtype=np.float32),
        11: np.array(
            [
                [1.0, 0.0, 0.0, 0.15],
                [0.0, 1.0, 0.0, 0.01],
                [0.0, 0.0, 1.0, 0.02],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        12: np.array(
            [
                [1.0, 0.0, 0.0, -0.12],
                [0.0, 1.0, 0.0, 0.02],
                [0.0, 0.0, 1.0, 0.04],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }
    observations: list[KeyframeObservation] = []
    for point_id, point_world in enumerate(points_world):
        for keyframe_id, pose in poses.items():
            camera_point = np.linalg.inv(pose)[:3, :] @ np.append(point_world, 1.0)
            observations.append(
                KeyframeObservation(
                    keyframe_id=keyframe_id,
                    point_id=point_id,
                    u=float(fx_true * (camera_point[0] / camera_point[2]) + cx),
                    v=float(fy_true * (camera_point[1] / camera_point[2]) + cy),
                    x=float(point_world[0]),
                    y=float(point_world[1]),
                    z=float(point_world[2]),
                )
            )

    refined_fx, refined_fy, refined_poses, refined_points = refine_focal_lengths(
        initial_fx=560.0,
        initial_fy=555.0,
        cx=cx,
        cy=cy,
        keyframe_poses=poses,
        keyframe_observations=observations,
    )

    assert refined_fx == pytest.approx(fx_true, rel=0.05)
    assert refined_fy == pytest.approx(fy_true, rel=0.05)
    assert sorted(refined_poses) == [10, 11, 12]
    assert sorted(refined_points) == list(range(points_world.shape[0]))
