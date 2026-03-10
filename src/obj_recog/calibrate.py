from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from obj_recog.camera import open_camera, read_camera_frame
from obj_recog.calibration import CalibrationResult, render_orbslam3_settings_yaml, scale_calibration
from obj_recog.config import AppConfig
from obj_recog.opencv_runtime import load_cv2


_WINDOW_NAME = "Calibration Capture"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a chessboard sequence and write ORB-SLAM3 camera settings")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-name", type=str, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--slam-width", type=int, default=640)
    parser.add_argument("--slam-height", type=int, default=360)
    parser.add_argument("--board-cols", type=int, required=True)
    parser.add_argument("--board-rows", type=int, required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser


def _capture_calibration(
    board_cols: int,
    board_rows: int,
    samples: int,
    camera_config: AppConfig,
    *,
    cv2_module=None,
    camera_opener=open_camera,
    frame_reader=read_camera_frame,
    time_fn=time.monotonic,
    platform_name: str | None = None,
) -> CalibrationResult:
    cv2 = load_cv2(cv2_module)

    session = camera_opener(
        camera_config,
        cv2_module=cv2,
        preferred_name=camera_config.camera_name,
        allow_fallback=False,
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    object_template = np.zeros((board_rows * board_cols, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    last_capture_time = 0.0

    try:
        print(f"Calibration camera: {session.active_name}")
        while len(image_points) < samples:
            ok, frame = frame_reader(session.capture, timeout_sec=1.0)
            if not ok:
                raise RuntimeError("failed to read frame during calibration")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = _find_chessboard_corners(
                gray,
                board_cols=board_cols,
                board_rows=board_rows,
                criteria=criteria,
                cv2_module=cv2,
            )
            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, (board_cols, board_rows), corners, found)
                now = time_fn()
                if now - last_capture_time >= 0.75:
                    object_points.append(object_template.copy())
                    image_points.append(corners)
                    last_capture_time = now

            cv2.putText(
                preview,
                f"Samples {len(image_points)}/{samples} | q to cancel",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(_WINDOW_NAME, preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise RuntimeError("calibration canceled")

        rms_error, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
            object_points,
            image_points,
            (camera_config.width, camera_config.height),
            None,
            None,
        )
        return CalibrationResult(
            camera_matrix=np.asarray(camera_matrix, dtype=np.float32),
            distortion_coefficients=np.asarray(distortion, dtype=np.float32),
            image_width=int(camera_config.width),
            image_height=int(camera_config.height),
            rms_error=float(rms_error),
        )
    finally:
        session.capture.release()
        _close_preview_window(cv2, _WINDOW_NAME, platform_name=platform_name)


def _find_chessboard_corners(
    gray_frame: np.ndarray,
    *,
    board_cols: int,
    board_rows: int,
    criteria,
    cv2_module,
) -> tuple[bool, np.ndarray | None]:
    flags = 0
    flags |= getattr(cv2_module, "CALIB_CB_ADAPTIVE_THRESH", 0)
    flags |= getattr(cv2_module, "CALIB_CB_NORMALIZE_IMAGE", 0)
    found, corners = cv2_module.findChessboardCorners(gray_frame, (board_cols, board_rows), flags)
    if found and corners is not None:
        refined = cv2_module.cornerSubPix(
            gray_frame,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )
        return True, refined

    find_sb = getattr(cv2_module, "findChessboardCornersSB", None)
    if callable(find_sb):
        found_sb, corners_sb = find_sb(gray_frame, (board_cols, board_rows))
        if found_sb and corners_sb is not None:
            return True, corners_sb

    return False, None


def _close_preview_window(cv2_module, window_name: str, *, platform_name: str | None = None) -> None:
    current_platform = sys.platform if platform_name is None else platform_name
    if current_platform == "darwin":
        return

    destroy_window = getattr(cv2_module, "destroyWindow", None)
    if callable(destroy_window):
        destroy_window(window_name)
        return

    destroy_all = getattr(cv2_module, "destroyAllWindows", None)
    if callable(destroy_all):
        destroy_all()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    camera_config = AppConfig(
        camera_index=args.camera_index,
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=60_000,
        camera_calibration=None,
        slam_vocabulary=None,
        slam_width=args.slam_width,
        slam_height=args.slam_height,
    )
    calibration = _capture_calibration(
        args.board_cols,
        args.board_rows,
        args.samples,
        camera_config,
    )
    scaled = scale_calibration(
        calibration,
        target_width=args.slam_width,
        target_height=args.slam_height,
    )
    rendered = render_orbslam3_settings_yaml(scaled, fps=args.fps)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote ORB-SLAM3 calibration settings to {output_path}")


if __name__ == "__main__":
    main()
