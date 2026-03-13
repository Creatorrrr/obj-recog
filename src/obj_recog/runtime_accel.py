from __future__ import annotations

from dataclasses import dataclass
import importlib.util


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    torch_available: bool
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    torch_cuda_version: str | None
    mps_available: bool
    tensorrt_available: bool
    opencv_cuda_available: bool


def _safe_bool(value) -> bool:
    try:
        return bool(value)
    except Exception:
        return False


def device_is_cuda(device: str | None) -> bool:
    return str(device or "").strip().lower().startswith("cuda")


def resolve_precision(requested_precision: str, device: str) -> str:
    requested = str(requested_precision or "auto").strip().lower()
    if requested not in {"auto", "fp16", "fp32"}:
        raise ValueError(f"unsupported precision: {requested_precision}")
    if requested == "auto":
        return "fp16" if device_is_cuda(device) else "fp32"
    if requested == "fp16" and not device_is_cuda(device):
        return "fp32"
    return requested


def detect_runtime_capabilities(*, torch_module=None, cv2_module=None) -> RuntimeCapabilities:
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            torch_module = None

    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError:
            cv2_module = None

    torch_available = torch_module is not None
    cuda_available = False
    cuda_device_count = 0
    cuda_device_name = None
    torch_cuda_version = None
    mps_available = False
    if torch_available:
        cuda_runtime = getattr(torch_module, "cuda", None)
        is_available = getattr(cuda_runtime, "is_available", None)
        device_count = getattr(cuda_runtime, "device_count", None)
        get_device_name = getattr(cuda_runtime, "get_device_name", None)
        cuda_available = callable(is_available) and _safe_bool(is_available())
        if callable(device_count):
            try:
                cuda_device_count = int(device_count())
            except Exception:
                cuda_device_count = 0
        version = getattr(getattr(torch_module, "version", None), "cuda", None)
        torch_cuda_version = None if version is None else str(version)
        if cuda_available and cuda_device_count > 0 and callable(get_device_name):
            try:
                cuda_device_name = str(get_device_name(0))
            except Exception:
                cuda_device_name = None
        backends = getattr(torch_module, "backends", None)
        mps_backend = getattr(backends, "mps", None)
        mps_is_available = getattr(mps_backend, "is_available", None)
        mps_available = callable(mps_is_available) and _safe_bool(mps_is_available())

    opencv_cuda_available = False
    if cv2_module is not None:
        cuda_namespace = getattr(cv2_module, "cuda", None)
        get_cuda_device_count = getattr(cuda_namespace, "getCudaEnabledDeviceCount", None)
        if callable(get_cuda_device_count):
            try:
                opencv_cuda_available = int(get_cuda_device_count()) > 0
            except Exception:
                opencv_cuda_available = False

    tensorrt_available = importlib.util.find_spec("tensorrt") is not None
    return RuntimeCapabilities(
        torch_available=torch_available,
        cuda_available=bool(cuda_available),
        cuda_device_count=int(cuda_device_count),
        cuda_device_name=cuda_device_name,
        torch_cuda_version=torch_cuda_version,
        mps_available=bool(mps_available),
        tensorrt_available=bool(tensorrt_available),
        opencv_cuda_available=bool(opencv_cuda_available),
    )
