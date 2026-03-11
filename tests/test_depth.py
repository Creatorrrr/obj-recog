from __future__ import annotations

import sys

import numpy as np
import pytest

from obj_recog.depth import DepthEstimator, RunningPercentileNormalizer


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
