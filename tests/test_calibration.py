from __future__ import annotations

import numpy as np
import pytest

from obj_recog.camera import CameraSession
from obj_recog.calibrate import _capture_calibration, _close_preview_window, _find_chessboard_corners
from obj_recog.calibration import (
    CalibrationResult,
    intrinsics_from_calibration,
    load_orbslam3_settings,
    render_orbslam3_settings_yaml,
    scale_calibration,
)
from obj_recog.config import AppConfig


def test_scale_calibration_updates_intrinsics_for_slam_resolution() -> None:
    calibration = CalibrationResult(
        camera_matrix=np.array(
            [[800.0, 0.0, 640.0], [0.0, 810.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        distortion_coefficients=np.array([[0.1, -0.02, 0.0, 0.0, 0.01]], dtype=np.float32),
        image_width=1280,
        image_height=720,
        rms_error=0.22,
    )

    scaled = scale_calibration(calibration, target_width=640, target_height=360)

    assert scaled.image_width == 640
    assert scaled.image_height == 360
    assert scaled.camera_matrix[0, 0] == 400.0
    assert scaled.camera_matrix[1, 1] == 405.0
    assert scaled.camera_matrix[0, 2] == 320.0
    assert scaled.camera_matrix[1, 2] == 180.0
    assert np.allclose(scaled.distortion_coefficients, calibration.distortion_coefficients)


def test_render_orbslam3_settings_yaml_contains_pinhole_fields() -> None:
    calibration = CalibrationResult(
        camera_matrix=np.array(
            [[400.0, 0.0, 320.0], [0.0, 405.0, 180.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        distortion_coefficients=np.array([[0.1, -0.02, 0.001, -0.002, 0.01]], dtype=np.float32),
        image_width=640,
        image_height=360,
        rms_error=0.18,
    )

    rendered = render_orbslam3_settings_yaml(calibration, fps=30.0)

    assert "Camera.type: PinHole" in rendered
    assert "Camera.fx: 400.0" in rendered
    assert "Camera.fy: 405.0" in rendered
    assert "Camera.cx: 320.0" in rendered
    assert "Camera.cy: 180.0" in rendered
    assert "Camera.k1: 0.1" in rendered
    assert "Camera.k2: -0.02" in rendered
    assert "Camera.p1: 0.001" in rendered
    assert "Camera.p2: -0.002" in rendered
    assert "Camera.k3: 0.01" in rendered
    assert "Camera.width: 640" in rendered
    assert "Camera.height: 360" in rendered
    assert "Camera.fps: 30.0" in rendered


def test_load_orbslam3_settings_parses_generated_yaml(tmp_path) -> None:
    calibration = CalibrationResult(
        camera_matrix=np.array(
            [[402.5, 0.0, 319.5], [0.0, 404.5, 181.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        distortion_coefficients=np.array([[0.1, -0.02, 0.001, -0.002, 0.01]], dtype=np.float32),
        image_width=640,
        image_height=360,
        rms_error=0.18,
    )
    settings_path = tmp_path / "camera.yaml"
    settings_path.write_text(render_orbslam3_settings_yaml(calibration, fps=30.0), encoding="utf-8")

    loaded = load_orbslam3_settings(settings_path)

    assert loaded.image_width == 640
    assert loaded.image_height == 360
    assert loaded.camera_matrix[0, 0] == 402.5
    assert loaded.camera_matrix[1, 1] == 404.5
    assert loaded.camera_matrix[0, 2] == 319.5
    assert loaded.camera_matrix[1, 2] == 181.5


def test_intrinsics_from_calibration_scales_to_runtime_resolution() -> None:
    calibration = CalibrationResult(
        camera_matrix=np.array(
            [[400.0, 0.0, 320.0], [0.0, 405.0, 180.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        distortion_coefficients=np.zeros((1, 5), dtype=np.float32),
        image_width=640,
        image_height=360,
        rms_error=0.18,
    )

    intrinsics = intrinsics_from_calibration(calibration, target_width=1280, target_height=720)

    assert intrinsics.fx == 800.0
    assert intrinsics.fy == 810.0
    assert intrinsics.cx == 640.0
    assert intrinsics.cy == 360.0


class _FakeCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame
        self.released = False

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self._frame.copy()

    def release(self) -> None:
        self.released = True


class _FakeCalibrationCv2:
    TERM_CRITERIA_EPS = 1
    TERM_CRITERIA_MAX_ITER = 2
    COLOR_BGR2GRAY = 6
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16
    CALIB_CB_ADAPTIVE_THRESH = 1 << 0
    CALIB_CB_NORMALIZE_IMAGE = 1 << 1

    def __init__(self) -> None:
        self.find_classic_calls = 0
        self.find_sb_calls = 0
        self.destroy_window_calls: list[str] = []
        self.destroy_all_calls = 0
        self.imshow_calls = 0

    def cvtColor(self, frame: np.ndarray, code: int) -> np.ndarray:
        assert code == self.COLOR_BGR2GRAY
        return frame[..., 0]

    def findChessboardCorners(self, gray: np.ndarray, board_shape: tuple[int, int], flags: int):
        self.find_classic_calls += 1
        assert board_shape == (9, 6)
        assert flags & self.CALIB_CB_ADAPTIVE_THRESH
        assert flags & self.CALIB_CB_NORMALIZE_IMAGE
        corners = np.zeros((board_shape[0] * board_shape[1], 1, 2), dtype=np.float32)
        return True, corners

    def findChessboardCornersSB(self, gray: np.ndarray, board_shape: tuple[int, int]):
        self.find_sb_calls += 1
        return False, None

    def cornerSubPix(self, gray, corners, win_size, zero_zone, criteria):
        return corners + 1.0

    def drawChessboardCorners(self, preview, board_shape, corners, found) -> None:
        pass

    def putText(self, *args, **kwargs) -> None:
        pass

    def imshow(self, window_name: str, frame: np.ndarray) -> None:
        self.imshow_calls += 1

    def waitKey(self, delay: int) -> int:
        return -1

    def calibrateCamera(self, object_points, image_points, image_size, camera_matrix, distortion):
        return (
            0.12,
            np.array([[800.0, 0.0, 640.0], [0.0, 810.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            np.zeros((1, 5), dtype=np.float32),
            [],
            [],
        )

    def destroyWindow(self, window_name: str) -> None:
        self.destroy_window_calls.append(window_name)

    def destroyAllWindows(self) -> None:
        self.destroy_all_calls += 1


def test_find_chessboard_corners_uses_classic_detector() -> None:
    fake_cv2 = _FakeCalibrationCv2()
    gray = np.zeros((10, 10), dtype=np.uint8)

    found, corners = _find_chessboard_corners(
        gray,
        board_cols=9,
        board_rows=6,
        criteria=(3, 30, 0.001),
        cv2_module=fake_cv2,
    )

    assert found is True
    assert corners is not None
    assert fake_cv2.find_classic_calls == 1
    assert fake_cv2.find_sb_calls == 0


def test_find_chessboard_corners_falls_back_to_sb_when_classic_misses() -> None:
    class _FallbackCv2(_FakeCalibrationCv2):
        def findChessboardCorners(self, gray: np.ndarray, board_shape: tuple[int, int], flags: int):
            self.find_classic_calls += 1
            return False, None

        def findChessboardCornersSB(self, gray: np.ndarray, board_shape: tuple[int, int]):
            self.find_sb_calls += 1
            corners = np.full((board_shape[0] * board_shape[1], 1, 2), 2.0, dtype=np.float32)
            return True, corners

    fake_cv2 = _FallbackCv2()
    gray = np.zeros((10, 10), dtype=np.uint8)

    found, corners = _find_chessboard_corners(
        gray,
        board_cols=9,
        board_rows=6,
        criteria=(3, 30, 0.001),
        cv2_module=fake_cv2,
    )

    assert found is True
    assert corners is not None
    assert float(corners[0, 0, 0]) == 2.0
    assert fake_cv2.find_classic_calls == 1
    assert fake_cv2.find_sb_calls == 1


def test_capture_calibration_uses_named_window_cleanup_on_non_macos() -> None:
    fake_cv2 = _FakeCalibrationCv2()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    capture = _FakeCapture(frame)
    session = CameraSession(
        capture=capture,
        active_index=0,
        active_name="FaceTime HD Camera",
        requested_name=None,
        used_fallback=False,
    )
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=60_000,
        camera_calibration=None,
        slam_vocabulary=None,
        slam_width=640,
        slam_height=360,
    )

    calibration = _capture_calibration(
        9,
        6,
        1,
        config,
        cv2_module=fake_cv2,
        camera_opener=lambda *args, **kwargs: session,
        time_fn=lambda: 1.0,
        platform_name="linux",
    )

    assert capture.released is True
    assert fake_cv2.destroy_window_calls == ["Calibration Capture"]
    assert fake_cv2.destroy_all_calls == 0
    assert calibration.image_width == 1280
    assert calibration.image_height == 720


def test_close_preview_window_skips_cleanup_on_macos() -> None:
    fake_cv2 = _FakeCalibrationCv2()

    _close_preview_window(fake_cv2, "Calibration Capture", platform_name="darwin")

    assert fake_cv2.destroy_window_calls == []
    assert fake_cv2.destroy_all_calls == 0


def test_capture_calibration_rejects_fallback_when_named_camera_requested() -> None:
    fake_cv2 = _FakeCalibrationCv2()
    config = AppConfig(
        camera_index=0,
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=60_000,
        camera_name="iPhone",
        camera_calibration=None,
        slam_vocabulary=None,
        slam_width=640,
        slam_height=360,
    )

    def _camera_opener(*args, **kwargs):
        assert kwargs["allow_fallback"] is False
        raise RuntimeError("requested camera 'iPhone' is unavailable")

    with pytest.raises(RuntimeError, match="requested camera 'iPhone' is unavailable"):
        _capture_calibration(
            9,
            6,
            1,
            config,
            cv2_module=fake_cv2,
            camera_opener=_camera_opener,
            time_fn=lambda: 1.0,
            platform_name="linux",
        )
