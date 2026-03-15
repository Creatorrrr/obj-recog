from __future__ import annotations

import sys

import numpy as np
import pytest

from obj_recog.depth import DepthEstimator, RunningPercentileNormalizer
from obj_recog.metric_stabilization import MetricDepthCalibrator
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.types import PanopticSegment


def test_depth_estimator_reports_missing_timm_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHub:
        @staticmethod
        def load(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'timm'")

    class FakeTorch:
        hub = FakeHub()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    with pytest.raises(RuntimeError, match="timm"):
        DepthEstimator(device="cpu")


def test_running_percentile_normalizer_uses_ema_bounds() -> None:
    normalizer = RunningPercentileNormalizer(alpha=0.5)
    first = normalizer.normalize(
        np.array([[0.0, 5.0], [10.0, 15.0]], dtype=np.float32),
    )
    second = normalizer.normalize(
        np.array([[10.0, 15.0], [20.0, 25.0]], dtype=np.float32),
    )

    assert first.shape == (2, 2)
    assert second.shape == (2, 2)
    assert second.max() < first.max()


def test_running_percentile_normalizer_defaults_to_six_meter_range() -> None:
    normalizer = RunningPercentileNormalizer(alpha=1.0)

    depth_map = normalizer.normalize(
        np.array([[0.0, 5.0], [10.0, 15.0]], dtype=np.float32),
    )

    assert depth_map.min() == pytest.approx(0.3)
    assert depth_map.max() == pytest.approx(6.0)


def test_running_percentile_normalizer_depth_gamma_expands_mid_to_far_distances() -> None:
    raw_depth = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    fast = RunningPercentileNormalizer(alpha=1.0, gamma=1.0)
    depthy = RunningPercentileNormalizer(alpha=1.0, gamma=0.7)

    fast_depth = fast.normalize(raw_depth)
    depthy_depth = depthy.normalize(raw_depth)

    assert np.all(np.diff(fast_depth.reshape(-1)) <= 0.0)
    assert np.all(np.diff(depthy_depth.reshape(-1)) <= 0.0)
    assert depthy_depth[0, 1] > fast_depth[0, 1]
    assert depthy_depth[0, 2] > fast_depth[0, 2]


def test_depth_estimator_falls_back_to_eager_when_compiled_model_needs_triton() -> None:
    calls: list[str] = []

    class _CompiledModel:
        def __call__(self, _input_batch):
            calls.append("compiled")
            raise RuntimeError("Cannot find a working triton installation")

    class _EagerModel:
        def __call__(self, input_batch):
            calls.append("eager")
            return input_batch

    estimator = DepthEstimator.__new__(DepthEstimator)
    estimator._model = _CompiledModel()
    estimator._eager_model = _EagerModel()
    estimator._compiled_model_active = True
    debug_messages: list[str] = []
    estimator._debug_log = debug_messages.append

    result = estimator._run_model("payload")

    assert result == "payload"
    assert calls == ["compiled", "eager"]
    assert estimator._model is estimator._eager_model
    assert estimator._compiled_model_active is False
    assert any("fallback to eager" in message for message in debug_messages)


def test_metric_depth_calibrator_uses_floor_anchors_and_preserves_monotonicity() -> None:
    calibrator = MetricDepthCalibrator(min_depth=0.3, max_depth=6.0, ema_alpha=1.0)
    intrinsics = CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=2.0)
    floor_mask = np.zeros((8, 8), dtype=bool)
    floor_mask[4:, :] = True
    segment = PanopticSegment(
        segment_id=1,
        label_id=1,
        label="floor",
        color_rgb=(0, 255, 0),
        mask=floor_mask,
        area_pixels=int(floor_mask.sum()),
    )
    ys, _xs = np.nonzero(floor_mask)
    metric_floor_depth = 1.25 / np.maximum((ys.astype(np.float32) - intrinsics.cy) / intrinsics.fy, 1e-6)
    raw_depth = np.tile(np.linspace(3.0, 1.0, 8, dtype=np.float32), (8, 1))
    raw_depth[floor_mask] = metric_floor_depth / 0.5

    corrected = calibrator.correct(
        raw_depth,
        intrinsics=intrinsics,
        segments=[segment],
        camera_height_m=1.25,
        camera_pitch_deg=0.0,
    )

    corrected_floor = corrected.depth_map[floor_mask]
    assert corrected.correction_state == "live"
    assert corrected.scale_factor == pytest.approx(0.5, rel=0.1)
    assert np.median(corrected_floor) == pytest.approx(np.median(metric_floor_depth), rel=0.1)
    assert np.all(np.diff(corrected.depth_map[2]) <= 0.0)


def test_metric_depth_calibrator_freezes_previous_mapping_without_new_anchors() -> None:
    calibrator = MetricDepthCalibrator(min_depth=0.3, max_depth=6.0, ema_alpha=1.0)
    intrinsics = CameraIntrinsics(fx=6.0, fy=6.0, cx=3.0, cy=2.0)
    floor_mask = np.zeros((6, 6), dtype=bool)
    floor_mask[3:, :] = True
    segment = PanopticSegment(
        segment_id=1,
        label_id=1,
        label="floor",
        color_rgb=(0, 255, 0),
        mask=floor_mask,
        area_pixels=int(floor_mask.sum()),
    )
    ys, _xs = np.nonzero(floor_mask)
    metric_floor_depth = 1.25 / np.maximum((ys.astype(np.float32) - intrinsics.cy) / intrinsics.fy, 1e-6)
    first_raw = np.full((6, 6), 4.0, dtype=np.float32)
    first_raw[floor_mask] = metric_floor_depth / 0.5
    calibrator.correct(
        first_raw,
        intrinsics=intrinsics,
        segments=[segment],
        camera_height_m=1.25,
        camera_pitch_deg=0.0,
    )

    second_raw = np.full((6, 6), 5.0, dtype=np.float32)
    corrected = calibrator.correct(
        second_raw,
        intrinsics=intrinsics,
        segments=[],
        camera_height_m=1.25,
        camera_pitch_deg=0.0,
    )

    assert corrected.correction_state == "frozen"
    assert corrected.scale_factor == pytest.approx(0.5, rel=0.2)
    assert np.median(corrected.depth_map) == pytest.approx(2.5, rel=0.2)
