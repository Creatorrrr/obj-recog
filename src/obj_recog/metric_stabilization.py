from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from obj_recog.reconstruct import CameraIntrinsics


_FLOOR_LIKE_LABELS = {"floor", "rug", "walkable_floor", "walkable-floor"}


@dataclass(frozen=True, slots=True)
class MetricCorrectionResult:
    depth_map: np.ndarray
    scale_factor: float
    confidence: float
    anchor_count: int
    correction_state: str


class MetricDepthCalibrator:
    def __init__(
        self,
        *,
        min_depth: float = 0.3,
        max_depth: float = 6.0,
        ema_alpha: float = 0.35,
    ) -> None:
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._ema_alpha = float(np.clip(ema_alpha, 0.05, 1.0))
        self._scale_factor = 1.0
        self._confidence = 0.0
        self._anchor_count = 0
        self._correction_state = "raw"
        self._raw_knots: np.ndarray | None = None
        self._metric_knots: np.ndarray | None = None

    def correct(
        self,
        raw_depth_map: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None = None,
        segments: list[object] | tuple[object, ...] = (),
        camera_height_m: float = 1.25,
        camera_pitch_deg: float = 0.0,
    ) -> MetricCorrectionResult:
        raw_depth = np.asarray(raw_depth_map, dtype=np.float32)
        floor_scales = self._floor_scale_samples(
            raw_depth,
            intrinsics=intrinsics,
            segments=segments,
            camera_height_m=camera_height_m,
            camera_pitch_deg=camera_pitch_deg,
        )
        if floor_scales.size > 0:
            scale_sample = self._validated_scale_sample(float(np.median(floor_scales)))
            if scale_sample is not None:
                blended_scale = self._blend_scale(scale_sample)
                confidence = min(0.9, 0.35 + (0.02 * math.sqrt(float(floor_scales.size))))
                self._update_mapping(
                    raw_depth,
                    scale_factor=blended_scale,
                    confidence=confidence,
                    anchor_count=int(floor_scales.size),
                    correction_state="live",
                )
        elif self._raw_knots is None or self._metric_knots is None:
            self._initialize_identity_mapping(raw_depth)
            self._correction_state = "raw"
        else:
            self._anchor_count = 0
            self._correction_state = "frozen"
        return self.apply(raw_depth)

    def apply(self, raw_depth_map: np.ndarray) -> MetricCorrectionResult:
        raw_depth = np.asarray(raw_depth_map, dtype=np.float32)
        if self._raw_knots is None or self._metric_knots is None:
            self._initialize_identity_mapping(raw_depth)
        corrected = self._apply_mapping(raw_depth)
        return MetricCorrectionResult(
            depth_map=corrected,
            scale_factor=float(self._scale_factor),
            confidence=float(self._confidence),
            anchor_count=int(self._anchor_count),
            correction_state=str(self._correction_state),
        )

    def update_motion_anchor(
        self,
        previous_raw_depth_map: np.ndarray,
        current_raw_depth_map: np.ndarray,
        *,
        direction: str,
        commanded_delta_m: float,
    ) -> bool:
        commanded = float(commanded_delta_m)
        if commanded <= 0.0:
            return False
        sample = self._motion_scale_sample(
            previous_raw_depth_map=np.asarray(previous_raw_depth_map, dtype=np.float32),
            current_raw_depth_map=np.asarray(current_raw_depth_map, dtype=np.float32),
            direction=str(direction or ""),
            commanded_delta_m=commanded,
        )
        if sample is None:
            return False
        validated = self._validated_scale_sample(sample)
        if validated is None:
            return False
        blended_scale = self._blend_scale(validated)
        self._update_mapping(
            np.asarray(current_raw_depth_map, dtype=np.float32),
            scale_factor=blended_scale,
            confidence=max(self._confidence, 0.58),
            anchor_count=1,
            correction_state="live",
        )
        return True

    def _initialize_identity_mapping(self, raw_depth_map: np.ndarray) -> None:
        valid = np.asarray(raw_depth_map, dtype=np.float32)
        valid = valid[np.isfinite(valid) & (valid > 0.05)]
        if valid.size == 0:
            self._raw_knots = np.asarray([self._min_depth, (self._min_depth + self._max_depth) * 0.5, self._max_depth])
            self._metric_knots = self._raw_knots.copy()
            self._scale_factor = 1.0
            self._confidence = 0.0
            self._anchor_count = 0
            return
        knots = np.percentile(valid, (5.0, 50.0, 95.0)).astype(np.float32)
        knots = self._strictly_increasing(knots)
        self._raw_knots = knots
        self._metric_knots = np.clip(knots, self._min_depth, self._max_depth).astype(np.float32)
        self._scale_factor = 1.0
        self._confidence = 0.0
        self._anchor_count = 0

    def _update_mapping(
        self,
        raw_depth_map: np.ndarray,
        *,
        scale_factor: float,
        confidence: float,
        anchor_count: int,
        correction_state: str,
    ) -> None:
        valid = np.asarray(raw_depth_map, dtype=np.float32)
        valid = valid[np.isfinite(valid) & (valid > 0.05)]
        if valid.size == 0:
            return
        raw_knots = np.percentile(valid, (5.0, 50.0, 95.0)).astype(np.float32)
        raw_knots = self._strictly_increasing(raw_knots)
        metric_knots = np.clip(raw_knots * float(scale_factor), self._min_depth, self._max_depth).astype(np.float32)
        metric_knots = self._strictly_increasing(metric_knots)
        if self._raw_knots is None or self._metric_knots is None:
            self._raw_knots = raw_knots
            self._metric_knots = metric_knots
        else:
            alpha = self._ema_alpha
            self._raw_knots = ((1.0 - alpha) * self._raw_knots) + (alpha * raw_knots)
            self._metric_knots = ((1.0 - alpha) * self._metric_knots) + (alpha * metric_knots)
            self._raw_knots = self._strictly_increasing(self._raw_knots.astype(np.float32))
            self._metric_knots = self._strictly_increasing(self._metric_knots.astype(np.float32))
        self._scale_factor = float(scale_factor)
        self._confidence = float(np.clip(confidence, 0.0, 1.0))
        self._anchor_count = max(0, int(anchor_count))
        self._correction_state = str(correction_state)

    def _apply_mapping(self, raw_depth_map: np.ndarray) -> np.ndarray:
        raw_depth = np.asarray(raw_depth_map, dtype=np.float32)
        if self._raw_knots is None or self._metric_knots is None:
            return np.clip(raw_depth, self._min_depth, self._max_depth).astype(np.float32, copy=False)
        flat = raw_depth.reshape(-1)
        corrected_flat = np.interp(
            flat.astype(np.float64, copy=False),
            self._raw_knots.astype(np.float64, copy=False),
            self._metric_knots.astype(np.float64, copy=False),
        ).astype(np.float32, copy=False)
        corrected = corrected_flat.reshape(raw_depth.shape)
        corrected[~np.isfinite(raw_depth)] = np.nan
        return np.clip(corrected, self._min_depth, self._max_depth).astype(np.float32, copy=False)

    def _validated_scale_sample(self, scale_sample: float) -> float | None:
        sample = float(scale_sample)
        if not np.isfinite(sample) or sample <= 0.0:
            return None
        if self._confidence > 0.0:
            lower = self._scale_factor * 0.5
            upper = self._scale_factor * 2.0
            if sample < lower or sample > upper:
                return None
        return sample

    def _blend_scale(self, sample: float) -> float:
        if self._confidence <= 0.0:
            return float(sample)
        alpha = self._ema_alpha
        return float(((1.0 - alpha) * self._scale_factor) + (alpha * sample))

    def _floor_scale_samples(
        self,
        raw_depth_map: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None,
        segments: list[object] | tuple[object, ...],
        camera_height_m: float,
        camera_pitch_deg: float,
    ) -> np.ndarray:
        if intrinsics is None:
            return np.empty((0,), dtype=np.float32)
        height, width = raw_depth_map.shape[:2]
        if height <= 0 or width <= 0:
            return np.empty((0,), dtype=np.float32)
        floor_mask = np.zeros((height, width), dtype=bool)
        for segment in tuple(segments or ()):
            label = str(getattr(segment, "label", "")).strip().lower().replace(" ", "_")
            if label not in _FLOOR_LIKE_LABELS:
                continue
            mask = np.asarray(getattr(segment, "mask", np.zeros((height, width), dtype=bool)), dtype=bool)
            if mask.shape != floor_mask.shape:
                continue
            floor_mask |= mask
        if not np.any(floor_mask):
            return np.empty((0,), dtype=np.float32)
        lower_band = np.zeros_like(floor_mask)
        lower_band[int(round(height * 0.55)) :, :] = True
        sample_mask = floor_mask & lower_band
        ys, xs = np.nonzero(sample_mask)
        if ys.size == 0:
            return np.empty((0,), dtype=np.float32)
        if ys.size > 512:
            keep = np.linspace(0, ys.size - 1, 512, dtype=np.int32)
            ys = ys[keep]
            xs = xs[keep]
        y_norm = (ys.astype(np.float32) - float(intrinsics.cy)) / max(float(intrinsics.fy), 1e-6)
        pitch_rad = math.radians(float(camera_pitch_deg))
        denom = (y_norm * math.cos(pitch_rad)) - math.sin(pitch_rad)
        metric_depth = float(camera_height_m) / np.maximum(denom, 1e-6)
        raw_depth = raw_depth_map[ys, xs]
        valid = (
            np.isfinite(raw_depth)
            & (raw_depth > 0.05)
            & np.isfinite(metric_depth)
            & (metric_depth >= self._min_depth)
            & (metric_depth <= self._max_depth * 1.2)
            & (denom > 1e-3)
        )
        if not np.any(valid):
            return np.empty((0,), dtype=np.float32)
        return (metric_depth[valid] / raw_depth[valid]).astype(np.float32, copy=False)

    def _motion_scale_sample(
        self,
        *,
        previous_raw_depth_map: np.ndarray,
        current_raw_depth_map: np.ndarray,
        direction: str,
        commanded_delta_m: float,
    ) -> float | None:
        previous_roi, current_roi = self._motion_anchor_roi(
            previous_raw_depth_map,
            current_raw_depth_map,
            direction=direction,
        )
        if previous_roi.size == 0 or current_roi.size == 0:
            return None
        valid = (
            np.isfinite(previous_roi)
            & np.isfinite(current_roi)
            & (previous_roi > 0.05)
            & (current_roi > 0.05)
        )
        if not np.any(valid):
            return None
        delta = previous_roi[valid] - current_roi[valid]
        delta = delta[np.isfinite(delta)]
        if delta.size == 0:
            return None
        positive_delta = delta[delta > 0.02]
        if positive_delta.size == 0:
            return None
        observed_delta = float(np.median(positive_delta))
        if observed_delta <= 1e-6:
            return None
        return float(commanded_delta_m) / observed_delta

    @staticmethod
    def _motion_anchor_roi(
        previous_raw_depth_map: np.ndarray,
        current_raw_depth_map: np.ndarray,
        *,
        direction: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = previous_raw_depth_map.shape[:2]
        if current_raw_depth_map.shape[:2] != (height, width):
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        x_bounds = (0.35, 0.65)
        if direction == "left":
            x_bounds = (0.10, 0.40)
        elif direction == "right":
            x_bounds = (0.60, 0.90)
        y_bounds = (0.58, 0.92)
        x1 = int(max(0, min(width - 1, round(width * x_bounds[0]))))
        x2 = int(max(x1 + 1, min(width, round(width * x_bounds[1]))))
        y1 = int(max(0, min(height - 1, round(height * y_bounds[0]))))
        y2 = int(max(y1 + 1, min(height, round(height * y_bounds[1]))))
        return previous_raw_depth_map[y1:y2, x1:x2], current_raw_depth_map[y1:y2, x1:x2]

    @staticmethod
    def _strictly_increasing(values: np.ndarray) -> np.ndarray:
        adjusted = np.asarray(values, dtype=np.float32).copy()
        for index in range(1, adjusted.size):
            adjusted[index] = max(adjusted[index], adjusted[index - 1] + 1e-3)
        return adjusted
