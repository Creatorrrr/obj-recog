from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np


@dataclass(frozen=True, slots=True)
class OpenCvRuntimeStatus:
    cuda_mode: str
    cv2_path: str | None
    cuda_api_present: bool
    cuda_device_count: int
    build_has_cuda: bool
    gpu_mat_ok: bool
    resize_ok: bool
    cvt_color_ok: bool
    errors: tuple[str, ...] = ()

    @property
    def cuda_available(self) -> bool:
        return self.cuda_api_present and self.cuda_device_count > 0

    @property
    def strict_ready(self) -> bool:
        return self.cuda_available and self.gpu_mat_ok and self.resize_ok and self.cvt_color_ok


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


def _opencv_build_has_cuda(cv2_module) -> bool:
    get_build_information = getattr(cv2_module, "getBuildInformation", None)
    if not callable(get_build_information):
        return False
    try:
        build_info = str(get_build_information() or "")
    except Exception:
        return False
    lowered = build_info.lower()
    if "cudaarithm" in lowered or "cudaimgproc" in lowered or "cudawarping" in lowered:
        return True
    for line in build_info.splitlines():
        lowered_line = line.strip().lower()
        if lowered_line.startswith("nvidia cuda:"):
            return "yes" in lowered_line or "true" in lowered_line or "on" in lowered_line
        if lowered_line.startswith("use cudnn:"):
            return "yes" in lowered_line or "true" in lowered_line or "on" in lowered_line
        if lowered_line.startswith("cuda:"):
            return "yes" in lowered_line or "true" in lowered_line or "on" in lowered_line
    return False


def probe_opencv_runtime(cv2_module, *, cuda_mode: str = "off") -> OpenCvRuntimeStatus:
    requested_mode = str(cuda_mode or "off").strip().lower()
    cuda_namespace = getattr(cv2_module, "cuda", None)
    cuda_api_present = bool(cuda_namespace is not None)
    get_count = getattr(cuda_namespace, "getCudaEnabledDeviceCount", None)
    errors: list[str] = []
    cuda_device_count = 0
    if callable(get_count):
        try:
            cuda_device_count = int(get_count())
        except Exception as exc:
            errors.append(f"cuda device count failed ({exc})")
            cuda_device_count = 0
    elif requested_mode != "off":
        errors.append("cv2.cuda namespace is unavailable")

    gpu_mat_ok = False
    resize_ok = False
    cvt_color_ok = False
    build_has_cuda = _opencv_build_has_cuda(cv2_module)
    if requested_mode != "off" and cuda_api_present and cuda_device_count > 0:
        try:
            sample = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            gpu_sample = _gpu_mat_from_array(cv2_module, sample)
            gpu_mat_ok = True
            resize_result = cv2_module.cuda.resize(
                gpu_sample,
                (2, 2),
                interpolation=getattr(cv2_module, "INTER_AREA", 3),
            )
            resized_array = resize_result.download()
            resize_ok = tuple(int(value) for value in resized_array.shape[:2]) == (2, 2)
            converted = cv2_module.cuda.cvtColor(
                gpu_sample,
                getattr(cv2_module, "COLOR_BGR2GRAY", 6),
            )
            converted_array = converted.download()
            cvt_color_ok = tuple(int(value) for value in converted_array.shape[:2]) == (3, 4)
            if not resize_ok:
                errors.append("cv2.cuda.resize smoke test returned an unexpected shape")
            if not cvt_color_ok:
                errors.append("cv2.cuda.cvtColor smoke test returned an unexpected shape")
        except Exception as exc:
            errors.append(f"opencv cuda smoke test failed ({exc})")
    elif requested_mode == "on":
        if not build_has_cuda:
            errors.append("OpenCV build does not report CUDA support")
        if cuda_device_count <= 0:
            errors.append("cv2.cuda reports no enabled CUDA devices")

    return OpenCvRuntimeStatus(
        cuda_mode=requested_mode,
        cv2_path=str(getattr(cv2_module, "__file__", None) or ""),
        cuda_api_present=cuda_api_present,
        cuda_device_count=int(cuda_device_count),
        build_has_cuda=bool(build_has_cuda),
        gpu_mat_ok=bool(gpu_mat_ok),
        resize_ok=bool(resize_ok),
        cvt_color_ok=bool(cvt_color_ok),
        errors=tuple(errors),
    )


def ensure_opencv_runtime(cv2_module, *, cuda_mode: str = "off") -> OpenCvRuntimeStatus:
    status = probe_opencv_runtime(cv2_module, cuda_mode=cuda_mode)
    if str(cuda_mode or "off").strip().lower() == "on" and not status.strict_ready:
        detail_text = "; ".join(status.errors) if status.errors else "strict CUDA verification failed"
        raise RuntimeError(f"--opencv-cuda on requested but CUDA OpenCV is not ready: {detail_text}")
    return status


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
