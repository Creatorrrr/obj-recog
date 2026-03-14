from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import socket
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

from obj_recog.opencv_runtime import load_cv2
from obj_recog.sim_protocol import ActionPrimitive, UnityActionCommand


@dataclass(frozen=True, slots=True)
class UnityRgbFrame:
    frame_bgr: np.ndarray
    timestamp_sec: float


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

    def apply_action(self, command: UnityActionCommand) -> UnityRgbFrame:
        return self._request_frame(
            {
                "kind": "action",
                "primitive": str(command.primitive.value),
                "value": float(command.value),
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
        player_path = Path(self._unity_player_path or "")
        if not player_path.is_file():
            raise RuntimeError(f"Unity player not found: {player_path}")
        command = [
            str(player_path),
            f"--obj-recog-host={self._host}",
            f"--obj-recog-port={self._port}",
            *self._player_args,
        ]
        self._process = subprocess.Popen(command)


def command_from_step(primitive: ActionPrimitive, value: float) -> UnityActionCommand:
    return UnityActionCommand(primitive=primitive, value=float(value))


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
        return UnityRgbFrame(frame_bgr=np.asarray(frame_bgr, dtype=np.uint8), timestamp_sec=timestamp_sec)
    raise RuntimeError(f"unsupported Unity RGB image encoding: {encoding}")
