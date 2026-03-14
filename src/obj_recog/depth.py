from __future__ import annotations

import numpy as np

from obj_recog.opencv_runtime import cvt_color
from obj_recog.runtime_accel import device_is_cuda, resolve_precision
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
        backend: str = "torch",
        precision: str = "auto",
        opencv_cuda: str = "off",
        debug_log=None,
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
        self._backend = str(backend)
        self._precision = resolve_precision(precision, device)
        self._use_cuda = device_is_cuda(device)
        self._opencv_cuda = str(opencv_cuda)
        self._debug_log = debug_log or (lambda _message: None)
        self._ema_alpha = ema_alpha
        self._profile = str(profile)
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._low_percentile = float(low_percentile) / 100.0
        self._high_percentile = float(high_percentile) / 100.0
        self._gamma = float(gamma)
        self._previous_depth: np.ndarray | None = None
        self._previous_depth_tensor = None
        self._last_depth_tensor = None
        self._low_tensor = None
        self._high_tensor = None
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
        if self._use_cuda:
            self._model = self._model.to(device=device, memory_format=torch.channels_last)
        else:
            self._model.to(device)
        self._model.eval()
        self._eager_model = self._model
        self._compiled_model_active = False
        self._autocast_enabled = self._use_cuda and self._precision == "fp16"
        compile_model = getattr(torch, "compile", None)
        if self._use_cuda and self._backend == "torch" and callable(compile_model):
            try:
                self._model = compile_model(self._model, mode="reduce-overhead")
                self._compiled_model_active = True
            except Exception as exc:
                self._debug_log(f"depth compile disabled ({exc})")
        self._warmup()

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cvt_color(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
            cv2_module=cv2,
            prefer_cuda=(self._use_cuda and self._opencv_cuda != "off"),
        )
        input_batch = self._transform(rgb)
        if hasattr(input_batch, "to"):
            input_batch = input_batch.to(self._device)
            if self._use_cuda:
                input_batch = input_batch.to(memory_format=self._torch.channels_last)

        with self._torch.inference_mode():
            with self._autocast():
                prediction = self._run_model(input_batch)
                prediction = self._torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

        raw_quantiles = self._quantiles(prediction, (0.05, 0.5, 0.95))
        self._normalizer._last_raw_percentiles = raw_quantiles
        depth_tensor = self._normalize_prediction_tensor(prediction)
        if self._previous_depth_tensor is None:
            smoothed = depth_tensor
        else:
            smoothed = (self._ema_alpha * depth_tensor) + (
                (1.0 - self._ema_alpha) * self._previous_depth_tensor
            )

        self._previous_depth_tensor = smoothed.detach()
        self._last_depth_tensor = self._previous_depth_tensor
        self._previous_depth = (
            self._previous_depth_tensor.detach()
            .to(device="cpu", dtype=self._torch.float32)
            .numpy()
            .astype(np.float32, copy=False)
        )
        valid = self._torch.isfinite(self._previous_depth_tensor) & (self._previous_depth_tensor > 0.05)
        if bool(valid.any().item()):
            values = self._previous_depth_tensor[valid]
            normalized_percentiles = self._quantiles(values, (0.1, 0.5, 0.9))
            valid_ratio = float(valid.float().mean().item())
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

    def last_depth_tensor(self):
        return self._last_depth_tensor

    def _warmup(self) -> None:
        if not self._use_cuda:
            return
        try:
            dummy = self._torch.zeros((1, 3, 256, 256), device=self._device, dtype=self._torch.float32)
            dummy = dummy.to(memory_format=self._torch.channels_last)
            with self._torch.inference_mode():
                with self._autocast():
                    warm = self._run_model(dummy)
                    _ = warm.shape
            synchronize = getattr(self._torch.cuda, "synchronize", None)
            if callable(synchronize):
                synchronize()
        except Exception as exc:
            self._debug_log(f"depth warmup skipped ({exc})")

    def _run_model(self, input_batch):
        try:
            return self._model(input_batch)
        except Exception as exc:
            if self._compiled_model_active and self._should_disable_compiled_model(exc):
                self._debug_log(f"depth compile fallback to eager ({exc})")
                self._model = self._eager_model
                self._compiled_model_active = False
                return self._model(input_batch)
            raise

    @staticmethod
    def _should_disable_compiled_model(exc: Exception) -> bool:
        message = str(exc).lower()
        if exc.__class__.__name__ == "TritonMissing":
            return True
        return (
            "triton" in message
            or "inductor" in message
            or "backend compiler failed" in message
        )

    def _autocast(self):
        if self._autocast_enabled:
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
        return self._torch.autocast(device_type="cpu", enabled=False)

    def _normalize_prediction_tensor(self, prediction):
        prediction = prediction.to(dtype=self._torch.float32)
        current_low = self._torch.quantile(prediction.reshape(-1), self._low_percentile)
        current_high = self._torch.quantile(prediction.reshape(-1), self._high_percentile)
        if self._low_tensor is None or self._high_tensor is None:
            self._low_tensor = current_low.detach()
            self._high_tensor = current_high.detach()
        else:
            self._low_tensor = (self._ema_alpha * current_low) + ((1.0 - self._ema_alpha) * self._low_tensor)
            self._high_tensor = (self._ema_alpha * current_high) + ((1.0 - self._ema_alpha) * self._high_tensor)
        self._normalizer._low = float(self._low_tensor.detach().cpu().item())
        self._normalizer._high = float(self._high_tensor.detach().cpu().item())

        denominator = self._high_tensor - self._low_tensor
        if float(denominator.abs().item()) < 1e-6:
            normalized = self._torch.zeros_like(prediction, dtype=self._torch.float32)
        else:
            normalized = self._torch.clamp((prediction - self._low_tensor) / denominator, 0.0, 1.0)
        distance_map = self._min_depth + ((1.0 - normalized) ** self._gamma) * (
            self._max_depth - self._min_depth
        )
        return self._torch.clamp(distance_map, self._min_depth, self._max_depth).to(
            dtype=self._torch.float32
        )

    def _quantiles(self, tensor, values: tuple[float, float, float]) -> tuple[float, float, float]:
        quantile_values = self._torch.tensor(values, device=tensor.device, dtype=self._torch.float32)
        result = self._torch.quantile(tensor.reshape(-1).to(dtype=self._torch.float32), quantile_values)
        result_cpu = result.detach().cpu().tolist()
        return tuple(float(item) for item in result_cpu)
