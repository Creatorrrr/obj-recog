from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import plistlib
import socket
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

from obj_recog.opencv_runtime import load_cv2
from obj_recog.sim_protocol import RigCapabilities, UnityRigDeltaCommand


@dataclass(frozen=True, slots=True)
class UnityRgbFrame:
    frame_bgr: np.ndarray
    timestamp_sec: float
    metadata: dict[str, object] | None = None


class UnityRgbClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_sec: float = 10.0,
        unity_player_path: str | None = None,
        player_args: tuple[str, ...] = (),
        cv2_module=None,
    ) -> None:
        self._host = str(host)
        self._port = int(port)
        self._timeout_sec = float(timeout_sec)
        self._unity_player_path = None if unity_player_path is None else str(unity_player_path)
        self._player_args = tuple(str(item) for item in player_args)
        self._cv2_module = cv2_module
        self._socket: socket.socket | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        if self._unity_player_path:
            self._launch_player()
        deadline = time.monotonic() + self._timeout_sec
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                connected = socket.create_connection((self._host, self._port), timeout=self._timeout_sec)
                connected.settimeout(self._timeout_sec)
                self._socket = connected
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.1)
        raise RuntimeError(
            f"Unity RGB client could not connect to {self._host}:{self._port}"
            + ("" if last_error is None else f" ({last_error})")
        )

    def reset_episode(self, *, scenario_id: str) -> UnityRgbFrame:
        return self._request_frame({"kind": "reset_episode", "scenario_id": str(scenario_id)})

    def apply_action(self, command: UnityRigDeltaCommand) -> UnityRgbFrame:
        return self._request_frame(
            {
                "kind": "rig_delta",
                "translate_forward_m": float(command.translate_forward_m),
                "translate_right_m": float(command.translate_right_m),
                "body_yaw_deg": float(command.body_yaw_deg),
                "camera_yaw_delta_deg": float(command.camera_yaw_delta_deg),
                "camera_pitch_delta_deg": float(command.camera_pitch_delta_deg),
                "pause_sec": float(command.pause_sec),
            }
        )

    def shutdown(self) -> None:
        sock = self._socket
        if sock is None:
            return
        try:
            _send_message(sock, {"kind": "shutdown"})
        except OSError:
            pass

    def close(self) -> None:
        try:
            self.shutdown()
        finally:
            if self._socket is not None:
                try:
                    self._socket.close()
                finally:
                    self._socket = None
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2.0)
            self._process = None

    def _request_frame(self, payload: dict[str, object]) -> UnityRgbFrame:
        self.connect()
        sock = self._require_socket()
        _send_message(sock, payload)
        response = _read_message(sock)
        return _frame_from_payload(response, cv2_module=self._cv2_module)

    def _require_socket(self) -> socket.socket:
        if self._socket is None:
            raise RuntimeError("Unity RGB client is not connected")
        return self._socket

    def _launch_player(self) -> None:
        if self._process is not None:
            return
        player_path = _resolve_unity_player_launch_path(self._unity_player_path or "")
        launch_args = list(self._player_args)
        if not any(str(item).startswith("--obj-recog-mode=") for item in launch_args):
            launch_args.insert(0, "--obj-recog-mode=agent")
        command = [
            str(player_path),
            *launch_args,
            f"--obj-recog-host={self._host}",
            f"--obj-recog-port={self._port}",
        ]
        self._process = subprocess.Popen(command)


def _resolve_unity_player_launch_path(unity_player_path: str | Path) -> Path:
    player_path = Path(unity_player_path).expanduser()
    if player_path.suffix.lower() == ".app":
        if not player_path.exists():
            raise RuntimeError(f"Unity player not found: {player_path}")
        for executable_path in _unity_app_bundle_executable_candidates(player_path):
            if executable_path.is_file():
                return executable_path
        raise RuntimeError(f"Unity player app bundle is missing executable: {player_path}")
    if not player_path.is_file():
        raise RuntimeError(f"Unity player not found: {player_path}")
    return player_path


def _unity_app_bundle_executable_candidates(player_path: Path) -> tuple[Path, ...]:
    contents_path = player_path / "Contents"
    macos_path = contents_path / "MacOS"
    candidates: list[Path] = []
    info_plist_path = contents_path / "Info.plist"
    if info_plist_path.is_file():
        info_plist = plistlib.loads(info_plist_path.read_bytes())
        executable_name = info_plist.get("CFBundleExecutable")
        if isinstance(executable_name, str) and executable_name:
            candidates.append(macos_path / executable_name)
    candidates.append(macos_path / player_path.stem)
    if macos_path.is_dir():
        files = sorted(path for path in macos_path.iterdir() if path.is_file())
        if len(files) == 1:
            candidates.append(files[0])
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return tuple(unique_candidates)


def rig_capabilities_from_metadata(metadata: dict[str, object] | None) -> RigCapabilities | None:
    payload = dict(metadata or {})
    required_keys = (
        "move_speed_mps",
        "turn_speed_deg_per_sec",
        "camera_yaw_speed_deg_per_sec",
        "camera_pitch_speed_deg_per_sec",
        "camera_yaw_limit_deg",
        "camera_pitch_limit_deg",
    )
    if not all(key in payload for key in required_keys):
        return None
    return RigCapabilities(
        move_speed_mps=float(payload["move_speed_mps"]),
        turn_speed_deg_per_sec=float(payload["turn_speed_deg_per_sec"]),
        camera_yaw_speed_deg_per_sec=float(payload["camera_yaw_speed_deg_per_sec"]),
        camera_pitch_speed_deg_per_sec=float(payload["camera_pitch_speed_deg_per_sec"]),
        camera_yaw_limit_deg=float(payload["camera_yaw_limit_deg"]),
        camera_pitch_limit_deg=float(payload["camera_pitch_limit_deg"]),
    )


def _send_message(sock: socket.socket, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(encoded)))
    sock.sendall(encoded)


def _read_message(sock: socket.socket) -> dict[str, object]:
    header = _recv_exact(sock, 4)
    size = struct.unpack(">I", header)[0]
    payload = _recv_exact(sock, size)
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("Unity RGB response must be a JSON object")
    return decoded


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise RuntimeError("Unity RGB socket closed unexpectedly")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _frame_from_payload(payload: dict[str, object], *, cv2_module=None) -> UnityRgbFrame:
    if str(payload.get("kind", "rgb_frame")) != "rgb_frame":
        raise RuntimeError(f"unexpected Unity RGB payload kind: {payload.get('kind')}")
    timestamp_sec = float(payload.get("timestamp_sec", 0.0))
    encoding = str(payload.get("image_encoding", "png")).lower()
    image_bytes_b64 = payload.get("image_bytes_b64")
    if not isinstance(image_bytes_b64, str) or not image_bytes_b64:
        raise RuntimeError("Unity RGB response is missing image_bytes_b64")
    image_bytes = base64.b64decode(image_bytes_b64.encode("ascii"))
    cv2 = load_cv2(cv2_module)
    if encoding == "png":
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError("Unity RGB response contains an invalid PNG payload")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"kind", "timestamp_sec", "image_encoding", "image_bytes_b64"}
        }
        return UnityRgbFrame(
            frame_bgr=np.asarray(frame_bgr, dtype=np.uint8),
            timestamp_sec=timestamp_sec,
            metadata=metadata,
        )
    raise RuntimeError(f"unsupported Unity RGB image encoding: {encoding}")
