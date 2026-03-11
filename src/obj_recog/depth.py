from __future__ import annotations

import numpy as np

from obj_recog.types import DepthDiagnostics


class RunningPercentileNormalizer:
    def __init__(
        self,
        alpha: float = 0.6,
        low_percentile: float = 5.0,
        high_percentile: float = 95.0,
        min_depth: float = 0.3,
        max_depth: float = 6.0,
        gamma: float = 1.0,
    ) -> None:
        self._alpha = alpha
        self._low_percentile = low_percentile
        self._high_percentile = high_percentile
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._gamma = float(gamma)
        self._low: float | None = None
        self._high: float | None = None
        self._last_raw_percentiles: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def normalize(self, raw_depth: np.ndarray) -> np.ndarray:
        raw_depth = raw_depth.astype(np.float32, copy=False)
        self._last_raw_percentiles = (
            float(np.percentile(raw_depth, 5.0)),
            float(np.percentile(raw_depth, 50.0)),
            float(np.percentile(raw_depth, 95.0)),
        )
        current_low = float(np.percentile(raw_depth, self._low_percentile))
        current_high = float(np.percentile(raw_depth, self._high_percentile))

        if self._low is None or self._high is None:
            self._low = current_low
            self._high = current_high
        else:
            self._low = (self._alpha * current_low) + ((1.0 - self._alpha) * self._low)
            self._high = (self._alpha * current_high) + ((1.0 - self._alpha) * self._high)

        if self._high - self._low < 1e-6:
            normalized = np.zeros_like(raw_depth, dtype=np.float32)
        else:
            normalized = np.clip((raw_depth - self._low) / (self._high - self._low), 0.0, 1.0)

        distance_map = self._min_depth + ((1.0 - normalized) ** self._gamma) * (
            self._max_depth - self._min_depth
        )
        return np.clip(distance_map, self._min_depth, self._max_depth).astype(np.float32, copy=False)

    @property
    def low_high(self) -> tuple[float, float]:
        return float(self._low or 0.0), float(self._high or 0.0)

    @property
    def raw_percentiles(self) -> tuple[float, float, float]:
        return self._last_raw_percentiles


def normalize_inverse_depth(
    raw_depth: np.ndarray,
    min_depth: float = 0.3,
    max_depth: float = 6.0,
    gamma: float = 1.0,
) -> np.ndarray:
    return RunningPercentileNormalizer(
        alpha=1.0,
        min_depth=min_depth,
        max_depth=max_depth,
        gamma=gamma,
    ).normalize(raw_depth)


class DepthEstimator:
    def __init__(
        self,
        device: str,
        ema_alpha: float = 0.6,
        *,
        profile: str = "balanced",
        low_percentile: float = 5.0,
        high_percentile: float = 95.0,
        min_depth: float = 0.3,
        max_depth: float = 6.0,
        gamma: float = 1.0,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on local install.
            raise RuntimeError("torch is required for depth estimation") from exc

        self._torch = torch
        self._device = device
        self._ema_alpha = ema_alpha
        self._profile = str(profile)
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._previous_depth: np.ndarray | None = None
        self._last_diagnostics: DepthDiagnostics | None = None
        self._normalizer = RunningPercentileNormalizer(
            alpha=ema_alpha,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            min_depth=min_depth,
            max_depth=max_depth,
            gamma=gamma,
        )
        try:
            self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        except ModuleNotFoundError as exc:
            if exc.name == "timm" or "timm" in str(exc):
                raise RuntimeError(
                    "MiDaS requires the 'timm' package. Reinstall project dependencies and retry."
                ) from exc
            raise
        self._transform = transforms.small_transform
        self._model.to(device)
        self._model.eval()

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        with self._torch.no_grad():
            prediction = self._model(input_batch)
            prediction = self._torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        raw_depth = prediction.detach().cpu().numpy()
        depth_map = self._normalizer.normalize(raw_depth)

        if self._previous_depth is None:
            smoothed = depth_map
        else:
            smoothed = (self._ema_alpha * depth_map) + ((1.0 - self._ema_alpha) * self._previous_depth)

        self._previous_depth = smoothed.astype(np.float32, copy=False)
        valid = np.isfinite(self._previous_depth) & (self._previous_depth > 0.05)
        if np.any(valid):
            values = self._previous_depth[valid]
            normalized_percentiles = (
                float(np.percentile(values, 10.0)),
                float(np.percentile(values, 50.0)),
                float(np.percentile(values, 90.0)),
            )
            valid_ratio = float(valid.mean())
        else:
            normalized_percentiles = (self._min_depth, self._min_depth, self._min_depth)
            valid_ratio = 0.0
        self._last_diagnostics = DepthDiagnostics(
            calibration_source="unknown",
            profile=self._profile,
            raw_percentiles=self._normalizer.raw_percentiles,
            normalizer_low_high=self._normalizer.low_high,
            normalized_distance_percentiles=normalized_percentiles,
            valid_depth_ratio=valid_ratio,
            dense_z_span=0.0,
            mesh_z_span=0.0,
            intrinsics_summary=(0.0, 0.0, 0.0, 0.0),
            hint="monocular pseudo-depth scale limited",
        )
        return self._previous_depth

    def last_diagnostics(self) -> DepthDiagnostics | None:
        return self._last_diagnostics
