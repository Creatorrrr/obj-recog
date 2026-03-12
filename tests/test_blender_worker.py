from __future__ import annotations

import io
import json
import os
from pathlib import Path
import select

import numpy as np
import pytest

from obj_recog.blender_worker import (
    BlenderFrameRequest,
    BlenderFrameResponse,
    BlenderWorkerClient,
    build_blender_worker_command,
)


class _FakeReadable:
    def __init__(self, lines: list[bytes], *, fileno_value: int = 11) -> None:
        self._lines = list(lines)
        self._fileno_value = fileno_value

    def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)

    def fileno(self) -> int:
        return self._fileno_value


class _FakeWritable(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout: _FakeReadable | None = None,
        stderr: _FakeReadable | None = None,
        stdin: _FakeWritable | None = None,
        poll_result: int | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.stdin = stdin
        self._poll_result = poll_result
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return self._poll_result

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        self._poll_result = 0
        return 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._poll_result = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self._poll_result = 0


def test_build_blender_worker_command_uses_expected_invocation_shape() -> None:
    command = build_blender_worker_command(
        blender_exec="/Applications/Blender.app/Contents/MacOS/Blender",
        worker_script=Path("/tmp/realtime_worker.py"),
    )

    assert command == [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "--background",
        "--python",
        "/tmp/realtime_worker.py",
    ]


def test_blender_worker_client_starts_with_expected_command() -> None:
    captured_commands: list[list[str]] = []
    fake_process = _FakeProcess(
        stdin=_FakeWritable(),
        stdout=_FakeReadable([b'{"status":"ok"}\n']),
        stderr=_FakeReadable([]),
        poll_result=None,
    )

    def _popen(command: list[str], **_kwargs):
        captured_commands.append(list(command))
        return fake_process

    client = BlenderWorkerClient(
        command=["blender", "--background", "--python", "worker.py"],
        popen_factory=_popen,
    )
    client.start()

    assert captured_commands == [["blender", "--background", "--python", "worker.py"]]


def test_blender_worker_client_request_frame_round_trips_json() -> None:
    fake_stdin = _FakeWritable()
    response_payload = {
        "rgb_path": "/tmp/rgb.png",
        "depth_path": "/tmp/depth.npy",
        "semantic_mask_path": "/tmp/semantic.png",
        "instance_mask_path": "/tmp/instance.png",
        "pose_world_gt": np.eye(4, dtype=np.float32).tolist(),
        "intrinsics_gt": {"fx": 10.0, "fy": 11.0, "cx": 4.0, "cy": 3.0},
        "render_time_ms": 14.5,
        "worker_state": "ready",
    }
    fake_process = _FakeProcess(
        stdin=fake_stdin,
        stdout=_FakeReadable([json.dumps(response_payload).encode("utf-8") + b"\n"]),
        stderr=_FakeReadable([]),
        poll_result=None,
    )

    client = BlenderWorkerClient(
        command=["blender", "--background", "--python", "worker.py"],
        popen_factory=lambda *_args, **_kwargs: fake_process,
        selector=lambda reads, _writes, _errors, _timeout: (reads, [], []),
    )
    client.start()
    response = client.request_frame(
        BlenderFrameRequest(
            frame_index=3,
            timestamp_sec=0.25,
            scenario_id="studio_open_v1",
            camera_pose_world=np.eye(4, dtype=np.float32),
            intrinsics={"fx": 10.0, "fy": 10.0, "cx": 4.0, "cy": 3.0},
            dynamic_actor_transforms={"actor-1": np.eye(4, dtype=np.float32)},
            lighting_seed=7,
        ),
        timeout_sec=0.5,
    )

    assert isinstance(response, BlenderFrameResponse)
    assert response.rgb_path == "/tmp/rgb.png"
    assert response.depth_path == "/tmp/depth.npy"
    np.testing.assert_allclose(response.pose_world_gt, np.eye(4, dtype=np.float32))
    assert response.intrinsics_gt == {"fx": 10.0, "fy": 11.0, "cx": 4.0, "cy": 3.0}
    written = fake_stdin.getvalue().decode("utf-8").strip()
    payload = json.loads(written)
    assert payload["frame_index"] == 3
    assert payload["scenario_id"] == "studio_open_v1"
    assert fake_stdin.flush_calls == 1


def test_blender_worker_client_skips_non_json_stdout_logs() -> None:
    fake_process = _FakeProcess(
        stdin=_FakeWritable(),
        stdout=_FakeReadable(
            [
                b"Blender 4.2.0 startup banner\n",
                b'{"rgb_path":"/tmp/rgb.png","depth_path":"/tmp/depth.npy","pose_world_gt":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],"intrinsics_gt":{"fx":10.0,"fy":10.0,"cx":3.0,"cy":2.0}}\n',
            ]
        ),
        stderr=_FakeReadable([]),
        poll_result=None,
    )
    client = BlenderWorkerClient(
        command=["blender", "--background", "--python", "worker.py"],
        popen_factory=lambda *_args, **_kwargs: fake_process,
        selector=lambda reads, _writes, _errors, _timeout: (reads, [], []),
    )
    client.start()

    response = client.request_frame(
        BlenderFrameRequest(
            frame_index=1,
            timestamp_sec=0.1,
            scenario_id="studio_open_v1",
            camera_pose_world=np.eye(4, dtype=np.float32),
            intrinsics={"fx": 10.0, "fy": 10.0, "cx": 4.0, "cy": 3.0},
            dynamic_actor_transforms={},
            lighting_seed=1,
        ),
        timeout_sec=0.5,
    )

    assert response.rgb_path == "/tmp/rgb.png"


def test_blender_frame_response_requires_gt_pose_and_intrinsics() -> None:
    with pytest.raises(ValueError, match="pose_world_gt"):
        BlenderFrameResponse.from_payload(
            {
                "rgb_path": "/tmp/rgb.png",
                "depth_path": "/tmp/depth.npy",
                "intrinsics_gt": {"fx": 10.0, "fy": 10.0, "cx": 3.0, "cy": 2.0},
            }
        )

    with pytest.raises(ValueError, match="intrinsics_gt"):
        BlenderFrameResponse.from_payload(
            {
                "rgb_path": "/tmp/rgb.png",
                "depth_path": "/tmp/depth.npy",
                "pose_world_gt": np.eye(4, dtype=np.float32).tolist(),
            }
        )


def test_blender_worker_client_raises_clean_error_when_process_fails_to_start() -> None:
    fake_process = _FakeProcess(
        stdin=_FakeWritable(),
        stdout=_FakeReadable([]),
        stderr=_FakeReadable([b"boom\n"]),
        poll_result=2,
    )

    client = BlenderWorkerClient(
        command=["blender", "--background", "--python", "worker.py"],
        popen_factory=lambda *_args, **_kwargs: fake_process,
    )

    with pytest.raises(RuntimeError, match="failed to start"):
        client.start()


def test_blender_worker_client_raises_clean_timeout_on_response_wait() -> None:
    fake_process = _FakeProcess(
        stdin=_FakeWritable(),
        stdout=_FakeReadable([]),
        stderr=_FakeReadable([]),
        poll_result=None,
    )

    client = BlenderWorkerClient(
        command=["blender", "--background", "--python", "worker.py"],
        popen_factory=lambda *_args, **_kwargs: fake_process,
        selector=lambda _reads, _writes, _errors, _timeout: ([], [], []),
    )
    client.start()

    with pytest.raises(TimeoutError, match="timed out"):
        client.request_frame(
            BlenderFrameRequest(
                frame_index=0,
                timestamp_sec=0.0,
                scenario_id="studio_open_v1",
                camera_pose_world=np.eye(4, dtype=np.float32),
                intrinsics={"fx": 10.0, "fy": 10.0, "cx": 4.0, "cy": 3.0},
                dynamic_actor_transforms={},
                lighting_seed=3,
            ),
            timeout_sec=0.25,
        )


def test_blender_worker_client_times_out_on_partial_stdout_without_newline() -> None:
    read_stream = None
    write_stream = None
    read_fd, write_fd = os.pipe()
    try:
        read_stream = os.fdopen(read_fd, "rb", buffering=0)
        write_stream = os.fdopen(write_fd, "wb", buffering=0)
        write_stream.write(b'{"rgb_path":"/tmp/rgb.png"')
        write_stream.flush()
        fake_process = _FakeProcess(
            stdin=_FakeWritable(),
            stdout=read_stream,
            stderr=_FakeReadable([]),
            poll_result=None,
        )
        client = BlenderWorkerClient(
            command=["blender", "--background", "--python", "worker.py"],
            popen_factory=lambda *_args, **_kwargs: fake_process,
            selector=select.select,
        )
        client.start()

        with pytest.raises(TimeoutError, match="timed out"):
            client.request_frame(
                BlenderFrameRequest(
                    frame_index=2,
                    timestamp_sec=0.2,
                    scenario_id="studio_open_v1",
                    camera_pose_world=np.eye(4, dtype=np.float32),
                    intrinsics={"fx": 10.0, "fy": 10.0, "cx": 4.0, "cy": 3.0},
                    dynamic_actor_transforms={},
                    lighting_seed=5,
                ),
                timeout_sec=0.05,
            )
    finally:
        try:
            write_stream.close()
        except Exception:
            pass
        try:
            read_stream.close()
        except Exception:
            pass


def test_blender_worker_client_close_terminates_process() -> None:
    fake_process = _FakeProcess(
        stdin=_FakeWritable(),
        stdout=_FakeReadable([]),
        stderr=_FakeReadable([]),
        poll_result=None,
    )
    client = BlenderWorkerClient(
        command=["blender", "--background", "--python", "worker.py"],
        popen_factory=lambda *_args, **_kwargs: fake_process,
    )
    client.start()

    client.close()

    assert fake_process.terminate_calls == 1
    assert fake_process.wait_calls == [1.0]
