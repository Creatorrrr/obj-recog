from __future__ import annotations

import numpy as np


class RunningPercentileNormalizer:
    def __init__(
        self,
        alpha: float = 0.6,
        low_percentile: float = 5.0,
        high_percentile: float = 95.0,
        min_depth: float = 0.3,
        max_depth: float = 3.0,
    ) -> None:
        self._alpha = alpha
        self._low_percentile = low_percentile
        self._high_percentile = high_percentile
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._low: float | None = None
        self._high: float | None = None

    def normalize(self, raw_depth: np.ndarray) -> np.ndarray:
        raw_depth = raw_depth.astype(np.float32, copy=False)
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

        distance_map = self._min_depth + (1.0 - normalized) * (self._max_depth - self._min_depth)
        return np.clip(distance_map, self._min_depth, self._max_depth).astype(np.float32, copy=False)


def normalize_inverse_depth(
    raw_depth: np.ndarray,
    min_depth: float = 0.3,
    max_depth: float = 3.0,
) -> np.ndarray:
    return RunningPercentileNormalizer(
        alpha=1.0,
        min_depth=min_depth,
        max_depth=max_depth,
    ).normalize(raw_depth)


class DepthEstimator:
    def __init__(self, device: str, ema_alpha: float = 0.6) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on local install.
            raise RuntimeError("torch is required for depth estimation") from exc

        self._torch = torch
        self._device = device
        self._ema_alpha = ema_alpha
        self._previous_depth: np.ndarray | None = None
        self._normalizer = RunningPercentileNormalizer(alpha=ema_alpha)
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
        return self._previous_depth
