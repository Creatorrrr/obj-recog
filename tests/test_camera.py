from __future__ import annotations

import numpy as np
import pytest

from obj_recog.camera import (
    CameraDevice,
    _read_exact_bytes,
    choose_capture_mode,
    find_camera_device,
    open_camera,
    parse_avfoundation_devices,
    parse_avfoundation_video_modes,
    read_camera_frame,
)


def test_parse_avfoundation_devices_keeps_only_video_cameras() -> None:
    output = """
[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] FaceTime HD Camera
[AVFoundation indev @ 0x1] [1] My iPhone Desk View Camera
[AVFoundation indev @ 0x1] [2] My iPhone Camera
[AVFoundation indev @ 0x1] [3] Capture screen 0
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] My iPhone Microphone
""".strip()

    devices = parse_avfoundation_devices(output)

    assert devices == [
        CameraDevice(index=0, name="FaceTime HD Camera"),
        CameraDevice(index=1, name="My iPhone Desk View Camera"),
        CameraDevice(index=2, name="My iPhone Camera"),
    ]


def test_find_camera_device_prefers_exact_match_before_partial_match() -> None:
    devices = [
        CameraDevice(index=1, name="My iPhone Desk View Camera"),
        CameraDevice(index=2, name="My iPhone Camera"),
    ]

    exact = find_camera_device(devices, "My iPhone Camera")
    partial = find_camera_device(devices, "iPhone")

    assert exact == CameraDevice(index=2, name="My iPhone Camera")
    assert partial == CameraDevice(index=2, name="My iPhone Camera")


def test_find_camera_device_returns_none_for_unknown_name() -> None:
    devices = [CameraDevice(index=0, name="FaceTime HD Camera")]

    assert find_camera_device(devices, "External") is None


def test_parse_avfoundation_video_modes_extracts_supported_sizes() -> None:
    output = """
[AVFoundation indev @ 0x1] Selected video size (1280x720) is not supported by the device.
[AVFoundation indev @ 0x1] Supported modes:
[AVFoundation indev @ 0x1]   1920x1440@[1.000000 30.000000]fps
[AVFoundation indev @ 0x1]   1280x720@[1.000000 60.000000]fps
""".strip()

    modes = parse_avfoundation_video_modes(output)

    assert modes == [(1920, 1440), (1280, 720)]


def test_choose_capture_mode_prefers_smallest_sufficient_mode() -> None:
    chosen = choose_capture_mode(
        [(1920, 1440), (2560, 1440), (3840, 2160)],
        desired_width=1280,
        desired_height=720,
    )

    assert chosen == (1920, 1440)


def test_open_camera_force_default_preserves_requested_name_for_overlay_status() -> None:
    from obj_recog.config import AppConfig

    class _FakeCapture:
        def __init__(self) -> None:
            self.opened = True
            self.width = 1280
            self.height = 720
            self.frames = [np.zeros((720, 1280, 3), dtype=np.uint8)]
            self.released = False

        def isOpened(self) -> bool:
            return self.opened

        def set(self, prop: int, value: float) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == _FakeCV2.CAP_PROP_FRAME_WIDTH:
                return float(self.width)
            if prop == _FakeCV2.CAP_PROP_FRAME_HEIGHT:
                return float(self.height)
            return 0.0

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self) -> None:
            self.released = True

    class _FakeCV2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_AVFOUNDATION = 1200

        def __init__(self) -> None:
            self.capture = _FakeCapture()

        def VideoCapture(self, index: int, backend: int | None = None) -> _FakeCapture:
            return self.capture

    config = AppConfig(
        camera_index=0,
        camera_name="iPhone",
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )
    fake_cv2 = _FakeCV2()

    session = open_camera(
        config,
        cv2_module=fake_cv2,
        camera_lister=lambda: [
            CameraDevice(index=0, name="FaceTime HD Camera"),
            CameraDevice(index=2, name="My iPhone Camera"),
        ],
        preferred_name="iPhone",
        force_default=True,
    )

    assert session.active_name == "FaceTime HD Camera"
    assert session.requested_name == "iPhone"
    assert session.used_fallback is True


def test_open_camera_prefers_opencv_capture_for_named_device_when_available() -> None:
    from obj_recog.config import AppConfig

    class _FakeOpenCVCapture:
        def __init__(self) -> None:
            self.opened = True
            self.width = 1280
            self.height = 720
            self.frames = [np.zeros((720, 1280, 3), dtype=np.uint8)]
            self.released = False

        def isOpened(self) -> bool:
            return self.opened

        def set(self, prop: int, value: float) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == _FakeCV2.CAP_PROP_FRAME_WIDTH:
                return float(self.width)
            if prop == _FakeCV2.CAP_PROP_FRAME_HEIGHT:
                return float(self.height)
            return 0.0

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self) -> None:
            self.released = True

    class _FakeCV2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_AVFOUNDATION = 1200

        def __init__(self) -> None:
            self.calls: list[int] = []
            self.capture = _FakeOpenCVCapture()

        def VideoCapture(self, index: int, backend: int | None = None) -> _FakeOpenCVCapture:
            self.calls.append(index)
            return self.capture

    class _FailIfFFmpegUsed:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("FFmpeg capture should not be used when OpenCV can stream the named camera")

    config = AppConfig(
        camera_index=0,
        camera_name="iPhone",
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )
    fake_cv2 = _FakeCV2()

    session = open_camera(
        config,
        cv2_module=fake_cv2,
        camera_lister=lambda: [
            CameraDevice(index=0, name="FaceTime HD Camera"),
            CameraDevice(index=1, name="My iPhone Camera"),
        ],
        ffmpeg_capture_factory=_FailIfFFmpegUsed,
        platform_name="darwin",
    )

    assert fake_cv2.calls == [1]
    assert session.active_index == 1
    assert session.active_name == "My iPhone Camera"
    assert session.used_fallback is False
    assert session.capture is fake_cv2.capture


def test_open_camera_uses_ffmpeg_capture_for_named_device_when_opencv_open_fails() -> None:
    from obj_recog.config import AppConfig

    class _ClosedCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            self.released = True

    class _FakeCV2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_AVFOUNDATION = 1200

        def __init__(self) -> None:
            self.calls: list[int] = []

        def VideoCapture(self, index: int, backend: int | None = None) -> _ClosedCapture:
            self.calls.append(index)
            return _ClosedCapture()

    class _FakeFFmpegCapture:
        def __init__(
            self,
            *,
            device_index: int,
            width: int,
            height: int,
            input_width: int | None = None,
            input_height: int | None = None,
            ffmpeg_bin: str = "ffmpeg",
            pixel_format: str | None = None,
        ) -> None:
            self.device_index = device_index
            self.width = width
            self.height = height
            self.input_width = input_width
            self.input_height = input_height
            self.ffmpeg_bin = ffmpeg_bin
            self.pixel_format = pixel_format

        def read(self):
            return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

    config = AppConfig(
        camera_index=0,
        camera_name="iPhone",
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )
    fake_cv2 = _FakeCV2()

    session = open_camera(
        config,
        cv2_module=fake_cv2,
        camera_lister=lambda: [
            CameraDevice(index=0, name="FaceTime HD Camera"),
            CameraDevice(index=2, name="My iPhone Camera"),
        ],
        mode_lister=lambda index: [(1920, 1440)],
        ffmpeg_capture_factory=_FakeFFmpegCapture,
        platform_name="darwin",
    )

    assert fake_cv2.calls == [2]
    assert session.active_index == 2
    assert session.active_name == "My iPhone Camera"
    assert session.used_fallback is False
    assert isinstance(session.capture, _FakeFFmpegCapture)
    assert session.capture.input_width == 1920
    assert session.capture.input_height == 1440
    assert session.capture.pixel_format == "bgr0"


def test_open_camera_retries_next_pixel_format_when_first_frame_is_uniform_white() -> None:
    from obj_recog.config import AppConfig

    class _FakeFFmpegCapture:
        def __init__(
            self,
            *,
            device_index: int,
            width: int,
            height: int,
            input_width: int | None = None,
            input_height: int | None = None,
            ffmpeg_bin: str = "ffmpeg",
            pixel_format: str | None = None,
        ) -> None:
            self.pixel_format = pixel_format
            self.height = height
            self.width = width
            self.released = False

        def read(self):
            if self.pixel_format == "bgr0":
                return True, np.full((self.height, self.width, 3), 255, dtype=np.uint8)
            return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    class _FailIfOpenCVUsed:
        def VideoCapture(self, *args, **kwargs):
            raise AssertionError("OpenCV fallback should not be used for the named iPhone camera path")

    config = AppConfig(
        camera_index=0,
        camera_name="iPhone",
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )

    session = open_camera(
        config,
        cv2_module=_FailIfOpenCVUsed(),
        camera_lister=lambda: [
            CameraDevice(index=0, name="FaceTime HD Camera"),
            CameraDevice(index=2, name="My iPhone Camera"),
        ],
        mode_lister=lambda index: [(1920, 1440)],
        ffmpeg_capture_factory=_FakeFFmpegCapture,
        platform_name="darwin",
    )

    assert session.active_index == 2
    assert isinstance(session.capture, _FakeFFmpegCapture)
    assert session.capture.pixel_format == "0rgb"


def test_open_camera_falls_back_to_default_index_when_ffmpeg_named_capture_fails() -> None:
    from obj_recog.config import AppConfig

    class _ClosedCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            self.released = True

    class _FakeOpenCVCapture:
        def __init__(self) -> None:
            self.opened = True
            self.width = 1280
            self.height = 720
            self.frames = [np.zeros((720, 1280, 3), dtype=np.uint8)]
            self.released = False

        def isOpened(self) -> bool:
            return self.opened

        def set(self, prop: int, value: float) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == _FakeCV2.CAP_PROP_FRAME_WIDTH:
                return float(self.width)
            if prop == _FakeCV2.CAP_PROP_FRAME_HEIGHT:
                return float(self.height)
            return 0.0

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self) -> None:
            self.released = True

    class _FakeCV2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_AVFOUNDATION = 1200

        def __init__(self) -> None:
            self.named_capture = _ClosedCapture()
            self.default_capture = _FakeOpenCVCapture()
            self.calls: list[int] = []

        def VideoCapture(self, index: int, backend: int | None = None) -> _FakeOpenCVCapture:
            self.calls.append(index)
            if index == 2:
                return self.named_capture
            return self.default_capture

    class _BrokenFFmpegCapture:
        def __init__(self, **kwargs) -> None:
            raise RuntimeError("ffmpeg could not open named camera")

    config = AppConfig(
        camera_index=0,
        camera_name="iPhone",
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )

    fake_cv2 = _FakeCV2()

    session = open_camera(
        config,
        cv2_module=fake_cv2,
        camera_lister=lambda: [
            CameraDevice(index=0, name="FaceTime HD Camera"),
            CameraDevice(index=2, name="My iPhone Camera"),
        ],
        mode_lister=lambda index: [(1920, 1440)],
        ffmpeg_capture_factory=_BrokenFFmpegCapture,
        platform_name="darwin",
    )

    assert fake_cv2.calls == [2, 0]
    assert session.active_index == 0
    assert session.active_name == "FaceTime HD Camera"
    assert session.requested_name == "iPhone"
    assert session.used_fallback is True


def test_open_camera_raises_for_named_camera_when_fallback_disabled() -> None:
    from obj_recog.config import AppConfig

    class _ClosedCapture:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    class _FakeCV2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_AVFOUNDATION = 1200

        def __init__(self) -> None:
            self.calls: list[int] = []

        def VideoCapture(self, index: int, backend: int | None = None) -> _ClosedCapture:
            self.calls.append(index)
            return _ClosedCapture()

    class _BrokenFFmpegCapture:
        def __init__(self, **kwargs) -> None:
            raise RuntimeError("ffmpeg could not open named camera")

    config = AppConfig(
        camera_index=0,
        camera_name="iPhone",
        width=1280,
        height=720,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        detection_interval=2,
        inference_width=640,
    )

    fake_cv2 = _FakeCV2()

    with pytest.raises(RuntimeError, match="requested camera 'iPhone' is unavailable"):
        open_camera(
            config,
            cv2_module=fake_cv2,
            camera_lister=lambda: [
                CameraDevice(index=0, name="FaceTime HD Camera"),
                CameraDevice(index=2, name="My iPhone Camera"),
            ],
            mode_lister=lambda index: [(1920, 1440)],
            ffmpeg_capture_factory=_BrokenFFmpegCapture,
            platform_name="darwin",
            allow_fallback=False,
        )

    assert fake_cv2.calls == [2]


def test_read_camera_frame_passes_timeout_to_timeout_capable_capture() -> None:
    class _TimeoutAwareCapture:
        def __init__(self) -> None:
            self.timeout_args: list[float | None] = []

        def read(self, timeout_sec: float | None = None):
            self.timeout_args.append(timeout_sec)
            return True, np.zeros((1, 1, 3), dtype=np.uint8)

    capture = _TimeoutAwareCapture()

    ok, frame = read_camera_frame(capture, timeout_sec=0.75)

    assert ok is True
    assert frame is not None
    assert capture.timeout_args == [0.75]


def test_read_camera_frame_retries_without_timeout_for_opencv_keyword_error() -> None:
    class _FakeCV2Error(Exception):
        pass

    class _OpenCVCapture:
        def __init__(self) -> None:
            self.calls = 0

        def read(self, timeout_sec: float | None = None):
            self.calls += 1
            if self.calls == 1:
                raise _FakeCV2Error("'timeout_sec' is an invalid keyword argument for VideoCapture.read()")
            return True, np.zeros((1, 1, 3), dtype=np.uint8)

    capture = _OpenCVCapture()

    ok, frame = read_camera_frame(capture, timeout_sec=0.5)

    assert ok is True
    assert frame is not None
    assert capture.calls == 2


def test_read_camera_frame_does_not_swallow_unrelated_capture_errors() -> None:
    class _FakeCV2Error(Exception):
        pass

    class _BrokenCapture:
        def read(self, timeout_sec: float | None = None):
            raise _FakeCV2Error("camera disconnected")

    with pytest.raises(_FakeCV2Error, match="camera disconnected"):
        read_camera_frame(_BrokenCapture(), timeout_sec=0.5)


def test_read_exact_bytes_assembles_partial_chunks_before_returning() -> None:
    class _Pipe:
        def fileno(self) -> int:
            return 7

    chunks = [b"ab", b"cd", b"ef"]

    def _selector(readers, _writers, _errors, timeout):
        return readers, [], []

    def _reader(_fd: int, _remaining: int) -> bytes:
        return chunks.pop(0)

    payload = _read_exact_bytes(
        _Pipe(),
        6,
        timeout_sec=0.5,
        selector=_selector,
        reader=_reader,
    )

    assert payload == b"abcdef"
