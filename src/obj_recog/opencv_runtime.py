from __future__ import annotations

import os

import numpy as np


def opencv_cuda_available(cv2_module) -> bool:
    cuda_namespace = getattr(cv2_module, "cuda", None)
    get_count = getattr(cuda_namespace, "getCudaEnabledDeviceCount", None)
    if not callable(get_count):
        return False
    try:
        return int(get_count()) > 0
    except Exception:
        return False


def _opencv_cuda_requested(cuda_mode: str, cv2_module) -> bool:
    requested = str(cuda_mode or "off").strip().lower()
    if requested == "on":
        return True
    if requested == "auto":
        return opencv_cuda_available(cv2_module)
    return False


def configure_opencv_runtime(cv2_module, *, cuda_mode: str = "off"):
    if not _opencv_cuda_requested(cuda_mode, cv2_module):
        ocl = getattr(cv2_module, "ocl", None)
        set_use_opencl = getattr(ocl, "setUseOpenCL", None)
        if callable(set_use_opencl):
            try:
                set_use_opencl(False)
            except Exception:
                pass
    return cv2_module


def load_cv2(cv2_module=None, *, cuda_mode: str = "off"):
    if str(cuda_mode or "off").strip().lower() == "off":
        os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

    if cv2_module is not None:
        return configure_opencv_runtime(cv2_module, cuda_mode=cuda_mode)

    import cv2

    return configure_opencv_runtime(cv2, cuda_mode=cuda_mode)


def _gpu_mat_from_array(cv2_module, array: np.ndarray):
    gpu_mat = cv2_module.cuda_GpuMat()
    gpu_mat.upload(np.ascontiguousarray(array))
    return gpu_mat


def resize_image(
    image: np.ndarray,
    size: tuple[int, int],
    *,
    interpolation: int,
    cv2_module,
    prefer_cuda: bool = False,
) -> np.ndarray:
    cv2 = load_cv2(cv2_module)
    if prefer_cuda and opencv_cuda_available(cv2):
        try:
            gpu_image = _gpu_mat_from_array(cv2, image)
            resized = cv2.cuda.resize(gpu_image, size, interpolation=interpolation)
            return resized.download()
        except Exception:
            pass
    return cv2.resize(image, size, interpolation=interpolation)


def cvt_color(
    image: np.ndarray,
    code: int,
    *,
    cv2_module,
    prefer_cuda: bool = False,
) -> np.ndarray:
    cv2 = load_cv2(cv2_module)
    if prefer_cuda and opencv_cuda_available(cv2):
        try:
            gpu_image = _gpu_mat_from_array(cv2, image)
            converted = cv2.cuda.cvtColor(gpu_image, code)
            return converted.download()
        except Exception:
            pass
    return cv2.cvtColor(image, code)
