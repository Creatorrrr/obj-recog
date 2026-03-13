from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from obj_recog.opencv_runtime import configure_opencv_runtime, load_cv2


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
