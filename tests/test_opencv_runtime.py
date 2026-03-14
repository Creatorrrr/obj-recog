from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from obj_recog.opencv_runtime import configure_opencv_runtime, cvt_color, load_cv2, resize_image


def test_configure_opencv_runtime_disables_opencl_when_available() -> None:
    calls: list[bool] = []

    fake_cv2 = SimpleNamespace(
        ocl=SimpleNamespace(
            setUseOpenCL=lambda enabled: calls.append(enabled),
        )
    )

    configured = configure_opencv_runtime(fake_cv2)

    assert configured is fake_cv2
    assert calls == [False]


def test_load_cv2_sets_opencl_runtime_env_and_configures_import(monkeypatch) -> None:
    calls: list[bool] = []
    fake_cv2 = SimpleNamespace(
        ocl=SimpleNamespace(
            setUseOpenCL=lambda enabled: calls.append(enabled),
        )
    )

    monkeypatch.delenv("OPENCV_OPENCL_RUNTIME", raising=False)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    loaded = load_cv2()

    assert loaded is fake_cv2
    assert os.environ["OPENCV_OPENCL_RUNTIME"] == "disabled"
    assert calls == [False]


def test_configure_opencv_runtime_preserves_opencl_when_cuda_is_requested() -> None:
    calls: list[bool] = []
    fake_cv2 = SimpleNamespace(
        ocl=SimpleNamespace(
            setUseOpenCL=lambda enabled: calls.append(enabled),
        ),
        cuda=SimpleNamespace(
            getCudaEnabledDeviceCount=lambda: 1,
        ),
    )

    configured = configure_opencv_runtime(fake_cv2, cuda_mode="auto")

    assert configured is fake_cv2
    assert calls == []


class _FakeGpuMat:
    def __init__(self) -> None:
        self._array = None

    def upload(self, array) -> None:
        self._array = np.asarray(array).copy()

    def download(self):
        return np.asarray(self._array).copy()


class _FakeCudaNamespace:
    def __init__(self, *, device_count: int) -> None:
        self._device_count = int(device_count)

    def getCudaEnabledDeviceCount(self) -> int:
        return self._device_count

    def resize(self, gpu_mat: _FakeGpuMat, size: tuple[int, int], interpolation: int = 0) -> _FakeGpuMat:
        _ = interpolation
        width, height = size
        source = gpu_mat.download()
        channels = source.shape[2] if source.ndim == 3 else 1
        resized = _FakeGpuMat()
        if channels == 1:
            resized.upload(np.zeros((height, width), dtype=source.dtype))
        else:
            resized.upload(np.zeros((height, width, channels), dtype=source.dtype))
        return resized

    def cvtColor(self, gpu_mat: _FakeGpuMat, code: int) -> _FakeGpuMat:
        _ = code
        source = gpu_mat.download()
        converted = _FakeGpuMat()
        converted.upload(np.zeros(source.shape[:2], dtype=source.dtype))
        return converted


def _make_fake_cuda_cv2(*, device_count: int, build_info: str | None = None):
    cuda_namespace = _FakeCudaNamespace(device_count=device_count)
    return SimpleNamespace(
        __file__="C:/fake/site-packages/cv2/__init__.py",
        INTER_AREA=3,
        COLOR_BGR2GRAY=6,
        cuda=cuda_namespace,
        cuda_GpuMat=lambda: _FakeGpuMat(),
        getBuildInformation=lambda: (
            "General configuration for OpenCV 4.13.0\nNVIDIA CUDA: YES\n"
            if build_info is None
            else build_info
        ),
    )


def test_probe_opencv_runtime_reports_cuda_smoke_success() -> None:
    from obj_recog.opencv_runtime import probe_opencv_runtime

    fake_cv2 = _make_fake_cuda_cv2(device_count=1)

    status = probe_opencv_runtime(fake_cv2, cuda_mode="on")

    assert status.cuda_mode == "on"
    assert status.cv2_path == "C:/fake/site-packages/cv2/__init__.py"
    assert status.build_has_cuda is True
    assert status.cuda_available is True
    assert status.gpu_mat_ok is True
    assert status.resize_ok is True
    assert status.cvt_color_ok is True
    assert status.strict_ready is True
    assert status.errors == ()


def test_ensure_opencv_runtime_raises_for_strict_mode_without_cuda_device() -> None:
    from obj_recog.opencv_runtime import ensure_opencv_runtime

    fake_cv2 = _make_fake_cuda_cv2(
        device_count=0,
        build_info="General configuration for OpenCV 4.13.0\nNVIDIA CUDA: NO\n",
    )

    with pytest.raises(RuntimeError, match="--opencv-cuda on requested"):
        ensure_opencv_runtime(fake_cv2, cuda_mode="on")


def test_probe_opencv_runtime_auto_mode_allows_cpu_fallback() -> None:
    from obj_recog.opencv_runtime import ensure_opencv_runtime

    fake_cv2 = _make_fake_cuda_cv2(
        device_count=0,
        build_info="General configuration for OpenCV 4.13.0\nNVIDIA CUDA: NO\n",
    )

    status = ensure_opencv_runtime(fake_cv2, cuda_mode="auto")

    assert status.cuda_mode == "auto"
    assert status.cuda_available is False
    assert status.strict_ready is False


def test_resize_image_prefers_cuda_when_available() -> None:
    fake_cv2 = _make_fake_cuda_cv2(device_count=1)
    image = np.zeros((6, 8, 3), dtype=np.uint8)

    resized = resize_image(
        image,
        (4, 3),
        interpolation=fake_cv2.INTER_AREA,
        cv2_module=fake_cv2,
        prefer_cuda=True,
    )

    assert resized.shape == (3, 4, 3)


def test_cvt_color_prefers_cuda_when_available() -> None:
    fake_cv2 = _make_fake_cuda_cv2(device_count=1)
    image = np.zeros((6, 8, 3), dtype=np.uint8)

    converted = cvt_color(
        image,
        fake_cv2.COLOR_BGR2GRAY,
        cv2_module=fake_cv2,
        prefer_cuda=True,
    )

    assert converted.shape == (6, 8)
