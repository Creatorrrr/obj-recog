from __future__ import annotations

import os


def configure_opencv_runtime(cv2_module):
    ocl = getattr(cv2_module, "ocl", None)
    set_use_opencl = getattr(ocl, "setUseOpenCL", None)
    if callable(set_use_opencl):
        try:
            set_use_opencl(False)
        except Exception:
            pass
    return cv2_module


def load_cv2(cv2_module=None):
    os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

    if cv2_module is not None:
        return configure_opencv_runtime(cv2_module)

    import cv2

    return configure_opencv_runtime(cv2)
