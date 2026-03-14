from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    import cv2

    from obj_recog.opencv_runtime import ensure_opencv_runtime

    status = ensure_opencv_runtime(cv2, cuda_mode="on")
    print(f"cv2_version={getattr(cv2, '__version__', 'unknown')}")
    print(f"cv2_path={getattr(cv2, '__file__', 'unknown')}")
    print(f"cuda_device_count={status.cuda_device_count}")
    print(f"build_has_cuda={status.build_has_cuda}")
    print(f"gpu_mat_ok={status.gpu_mat_ok}")
    print(f"resize_ok={status.resize_ok}")
    print(f"cvt_color_ok={status.cvt_color_ok}")
    print("opencv_cuda_ready=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
