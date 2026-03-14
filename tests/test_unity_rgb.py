from __future__ import annotations

import base64
import json
import socket
import struct
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from obj_recog.sim_protocol import ActionPrimitive
from obj_recog.unity_rgb import UnityRgbClient, command_from_step


def _send_message(sock: socket.socket, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    sock.sendall(struct.pack(">I", len(encoded)))
    sock.sendall(encoded)


def _read_message(sock: socket.socket) -> dict[str, object]:
    size = struct.unpack(">I", _recv_exact(sock, 4))[0]
    return json.loads(_recv_exact(sock, size).decode("utf-8"))


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def test_unity_rgb_client_resets_and_applies_actions() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()
    received: list[dict[str, object]] = []
    frame = np.full((6, 8, 3), 123, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", frame)
    assert ok is True
    payload = {
        "kind": "rgb_frame",
        "timestamp_sec": 1.25,
        "image_encoding": "png",
        "image_bytes_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }

    def _server() -> None:
        conn, _addr = listener.accept()
        with conn:
            for _ in range(2):
                received.append(_read_message(conn))
                _send_message(conn, payload)
        listener.close()

    thread = threading.Thread(target=_server, daemon=True)
    thread.start()

    client = UnityRgbClient(host=str(host), port=int(port), timeout_sec=2.0)
    reset_frame = client.reset_episode(scenario_id="living_room_navigation_v1")
    action_frame = client.apply_action(command_from_step(ActionPrimitive.MOVE_FORWARD, 0.12))
    client.close()
    thread.join(timeout=2.0)

    assert received[0] == {"kind": "reset_episode", "scenario_id": "living_room_navigation_v1"}
    assert received[1] == {"kind": "action", "primitive": "move_forward", "value": 0.12}
    assert reset_frame.frame_bgr.shape == (6, 8, 3)
    assert action_frame.timestamp_sec == 1.25


def test_unity_rgb_client_launches_player_in_agent_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    player_path = tmp_path / "obj-recog-unity.exe"
    player_path.write_bytes(b"stub")
    launched: dict[str, object] = {}

    class _FakeProcess:
        def poll(self):
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> None:
            _ = timeout
            return None

    def _fake_popen(command):
        launched["command"] = list(command)
        return _FakeProcess()

    monkeypatch.setattr("obj_recog.unity_rgb.subprocess.Popen", _fake_popen)

    client = UnityRgbClient(
        host="127.0.0.1",
        port=8765,
        unity_player_path=str(player_path),
        player_args=("--quality-level=2",),
    )
    client._launch_player()

    assert launched["command"] == [
        str(player_path),
        "--obj-recog-mode=agent",
        "--quality-level=2",
        "--obj-recog-host=127.0.0.1",
        "--obj-recog-port=8765",
    ]
