from __future__ import annotations

import os
import re
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from obj_recog.config import AppConfig
from obj_recog.opencv_runtime import load_cv2


_UNSET = object()
_DEVICE_RE = re.compile(r"\[(\d+)\]\s+(.+)$")
_MODE_RE = re.compile(r"(\d+)x(\d+)@\[[^\]]+\]fps")
_PREFERRED_PIXEL_FORMATS: tuple[str | None, ...] = ("bgr0", "0rgb", "uyvy422", "yuyv422", "nv12", None)


@dataclass(frozen=True, slots=True)
class CameraDevice:
    index: int
    name: str


@dataclass(slots=True)
class CameraSession:
    capture: Any
    active_index: int
    active_name: str
    requested_name: str | None
    used_fallback: bool


class FFmpegVideoCapture:
    def __init__(
        self,
        *,
        device_index: int,
        width: int,
        height: int,
        input_width: int | None = None,
        input_height: int | None = None,
        ffmpeg_bin: str = "ffmpeg",
        framerate: int = 30,
        pixel_format: str | None = "bgr0",
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        self._input_width = int(input_width or width)
        self._input_height = int(input_height or height)
        self._frame_bytes = self._width * self._height * 3
        command = [
            ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-framerate",
            str(framerate),
            "-video_size",
            f"{self._input_width}x{self._input_height}",
        ]
        if pixel_format is not None:
            command.extend(["-pixel_format", pixel_format])
        command.extend(["-i", f"{device_index}:none"])
        if self._input_width != self._width or self._input_height != self._height:
            command.extend(["-vf", f"scale={self._width}:{self._height}"])
        command.extend(
            [
                "-pix_fmt",
                "bgr24",
                "-vcodec",
                "rawvideo",
                "-an",
                "-sn",
                "-dn",
                "-f",
                "rawvideo",
                "-",
            ]
        )
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes * 2,
        )

    def read(self, timeout_sec: float | None = None) -> tuple[bool, np.ndarray | None]:
        if self._process.poll() is not None or self._process.stdout is None:
            return False, None

        raw_frame = _read_exact_bytes(
            self._process.stdout,
            self._frame_bytes,
            timeout_sec=timeout_sec,
        )
        if raw_frame is None:
            return False, None

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self._height, self._width, 3)).copy()
        return True, frame

    def release(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)

    def get(self, prop: int) -> float:
        if prop == 3:
            return float(self._width)
        if prop == 4:
            return float(self._height)
        return 0.0


def _read_exact_bytes(
    pipe,
    byte_count: int,
    *,
    timeout_sec: float | None = None,
    selector=select.select,
    reader=None,
) -> bytes | None:
    if reader is None:
        reader = os.read

    remaining = byte_count
    chunks: list[bytes] = []
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec

    while remaining > 0:
        wait_timeout = None
        if deadline is not None:
            wait_timeout = max(0.0, deadline - time.monotonic())
            readable, _, _ = selector([pipe], [], [], wait_timeout)
            if not readable:
                return None

        chunk = reader(pipe.fileno(), remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def read_camera_frame(capture, *, timeout_sec: float | None = 1.0) -> tuple[bool, np.ndarray | None]:
    try:
        return capture.read(timeout_sec=timeout_sec)
    except TypeError:
        return capture.read()
    except Exception as exc:
        message = str(exc)
        if "invalid keyword argument" in message and "VideoCapture.read" in message:
            return capture.read()
        raise


def parse_avfoundation_devices(output: str) -> list[CameraDevice]:
    devices: list[CameraDevice] = []
    in_video_section = False

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if "AVFoundation video devices:" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if not in_video_section:
            continue

        match = _DEVICE_RE.search(line)
        if match is None:
            continue

        index = int(match.group(1))
        name = match.group(2).strip()
        if name.startswith("Capture screen "):
            continue
        devices.append(CameraDevice(index=index, name=name))

    return devices


def list_available_cameras() -> list[CameraDevice]:
    if sys.platform != "darwin":
        return []

    try:
        completed = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return parse_avfoundation_devices(output)


def parse_avfoundation_video_modes(output: str) -> list[tuple[int, int]]:
    modes: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_line in output.splitlines():
        match = _MODE_RE.search(raw_line)
        if match is None:
            continue
        mode = (int(match.group(1)), int(match.group(2)))
        if mode in seen:
            continue
        seen.add(mode)
        modes.append(mode)
    return modes


def list_camera_modes(device_index: int, ffmpeg_bin: str = "ffmpeg") -> list[tuple[int, int]]:
    if sys.platform != "darwin":
        return []

    try:
        completed = subprocess.run(
            [
                ffmpeg_bin,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "avfoundation",
                "-pixel_format",
                "nv12",
                "-video_size",
                "2x2",
                "-i",
                f"{device_index}:none",
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return parse_avfoundation_video_modes(output)


def choose_capture_mode(
    modes: list[tuple[int, int]],
    *,
    desired_width: int,
    desired_height: int,
) -> tuple[int, int] | None:
    if not modes:
        return None

    exact = (desired_width, desired_height)
    if exact in modes:
        return exact

    sufficient = [mode for mode in modes if mode[0] >= desired_width and mode[1] >= desired_height]
    if sufficient:
        return min(sufficient, key=lambda mode: (mode[0] * mode[1], mode[0], mode[1]))

    return max(modes, key=lambda mode: (mode[0] * mode[1], mode[0], mode[1]))


def find_camera_device(devices: list[CameraDevice], requested_name: str) -> CameraDevice | None:
    requested_key = requested_name.casefold()
    for device in devices:
        if device.name.casefold() == requested_key:
            return device

    partial_matches = [device for device in devices if requested_key in device.name.casefold()]
    if not partial_matches:
        return None

    def _partial_match_rank(device: CameraDevice) -> tuple[int, int]:
        normalized_name = device.name.casefold()
        desk_view_penalty = int("desk view" in normalized_name or "데스크뷰" in normalized_name)
        return (desk_view_penalty, len(normalized_name))

    return min(partial_matches, key=_partial_match_rank)


def _open_video_capture(cv2_module, index: int):
    backend = getattr(cv2_module, "CAP_AVFOUNDATION", None)
    if backend is None:
        return cv2_module.VideoCapture(index)
    return cv2_module.VideoCapture(index, backend)


def _requested_resolution_satisfied(capture, config: AppConfig, cv2_module) -> bool:
    width = capture.get(cv2_module.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT)
    return width >= config.width and height >= config.height


def _open_capture_index(index: int, config: AppConfig, cv2_module):
    capture = _open_video_capture(cv2_module, index)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open camera index {index}")

    capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, float(config.width))
    capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, float(config.height))

    ok, _ = capture.read()
    if ok and _requested_resolution_satisfied(capture, config, cv2_module):
        return capture

    capture.release()
    fallback = _open_video_capture(cv2_module, index)
    if not fallback.isOpened():
        raise RuntimeError(f"failed to open camera index {index}")

    fallback.set(cv2_module.CAP_PROP_FRAME_WIDTH, 640.0)
    fallback.set(cv2_module.CAP_PROP_FRAME_HEIGHT, 480.0)
    ok, _ = fallback.read()
    if not ok:
        fallback.release()
        raise RuntimeError(f"failed to read from camera index {index}")
    return fallback


def _open_named_capture(device_index: int, config: AppConfig, capture_factory, mode_lister) -> Any:
    supported_modes = mode_lister(device_index)
    attempted_sizes = [(config.width, config.height), (640, 480)]
    tried: set[tuple[int, int]] = set()

    def _frame_is_usable(frame: np.ndarray | None) -> bool:
        if frame is None or frame.size == 0:
            return False
        return not (float(frame.mean()) >= 254.5 and float(np.ptp(frame)) <= 1.0)

    for width, height in attempted_sizes:
        if (width, height) in tried:
            continue
        tried.add((width, height))
        input_mode = choose_capture_mode(
            supported_modes,
            desired_width=width,
            desired_height=height,
        )
        input_width = width if input_mode is None else input_mode[0]
        input_height = height if input_mode is None else input_mode[1]
        for pixel_format in _PREFERRED_PIXEL_FORMATS:
            try:
                capture = capture_factory(
                    device_index=device_index,
                    width=width,
                    height=height,
                    input_width=input_width,
                    input_height=input_height,
                    pixel_format=pixel_format,
                )
            except TypeError:
                capture = capture_factory(
                    device_index=device_index,
                    width=width,
                    height=height,
                    input_width=input_width,
                    input_height=input_height,
                )
                pixel_format = None
            except (OSError, RuntimeError, ValueError):
                continue
            if isinstance(capture, FFmpegVideoCapture):
                ok, frame = capture.read(timeout_sec=3.0)
            else:
                ok, frame = capture.read()
            if ok and _frame_is_usable(frame):
                return capture
            capture.release()

    raise RuntimeError(f"failed to read from named camera index {device_index}")


def open_camera(
    config: AppConfig,
    cv2_module=None,
    camera_lister=list_available_cameras,
    preferred_name: str | None | object = _UNSET,
    force_default: bool = False,
    allow_fallback: bool = True,
    ffmpeg_capture_factory=FFmpegVideoCapture,
    mode_lister=list_camera_modes,
    platform_name: str | None = None,
) -> CameraSession:
    cv2 = load_cv2(cv2_module)
    devices = camera_lister()
    current_platform = sys.platform if platform_name is None else platform_name

    if preferred_name is _UNSET:
        requested_name = config.camera_name
    else:
        requested_name = preferred_name

    last_named_camera_error: Exception | None = None

    if requested_name and not force_default and current_platform == "darwin":
        selected_device = find_camera_device(devices, requested_name)
        if selected_device is not None:
            try:
                capture = _open_capture_index(selected_device.index, config, cv2)
                return CameraSession(
                    capture=capture,
                    active_index=selected_device.index,
                    active_name=selected_device.name,
                    requested_name=requested_name,
                    used_fallback=False,
                )
            except Exception as exc:
                last_named_camera_error = exc

            try:
                capture = _open_named_capture(
                    selected_device.index,
                    config,
                    ffmpeg_capture_factory,
                    mode_lister,
                )
                return CameraSession(
                    capture=capture,
                    active_index=selected_device.index,
                    active_name=selected_device.name,
                    requested_name=requested_name,
                    used_fallback=False,
                )
            except (RuntimeError, OSError, ValueError) as exc:
                last_named_camera_error = exc

        if not allow_fallback:
            if selected_device is None:
                raise RuntimeError(f"requested camera '{requested_name}' is unavailable")
            raise RuntimeError(f"requested camera '{requested_name}' is unavailable") from last_named_camera_error

    capture = _open_capture_index(config.camera_index, config, cv2)
    active_name = next(
        (device.name for device in devices if device.index == config.camera_index),
        f"Camera {config.camera_index}",
    )
    return CameraSession(
        capture=capture,
        active_index=config.camera_index,
        active_name=active_name,
        requested_name=requested_name,
        used_fallback=bool(requested_name),
    )
