from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from obj_recog.blender_worker import (
    BlenderFrameRequest,
    BlenderFrameResponse,
    BlenderSceneBuildRequest,
    BlenderSceneBuildResponse,
    BlenderWorkerClient,
    build_realtime_blender_worker_command,
)
from obj_recog.sim_scene import build_living_room_scene_spec


class _FakeReadable:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)

    def fileno(self) -> int:
        raise OSError("no fd")


class _FakeWritable(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1


class _FakeProcess:
    def __init__(self, *, stdout=None, stderr=None, stdin=None, poll_result=None) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.stdin = stdin
        self._poll_result = poll_result
        self.terminate_calls = 0
        self.wait_calls = []

    def poll(self):
        return self._poll_result

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._poll_result = 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        self._poll_result = 0
        return 0


def test_build_realtime_blender_worker_command_points_at_realtime_worker_script() -> None:
    command = build_realtime_blender_worker_command(
        blender_exec="/Applications/Blender.app/Contents/MacOS/Blender",
        repo_root=Path("/workspace"),
        output_root=Path("/tmp/reports"),
        blend_file=Path("/workspace/scripts/blender/scene_template/base_scene.blend"),
    )

    assert command == [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/workspace/scripts/blender/scene_template/base_scene.blend",
        "--background",
        "--python",
        "/workspace/scripts/blender/realtime_worker.py",
        "--",
        "--output-root",
        "/tmp/reports",
    ]


def test_blender_worker_client_builds_scene_before_rendering_frames() -> None:
    fake_stdin = _FakeWritable()
    fake_process = _FakeProcess(
        stdin=fake_stdin,
        stdout=_FakeReadable(
            [
                b'{"status":"ready","scene_id":"living_room_navigation_v1"}\n',
                b'{"rgb_path":"/tmp/rgb.npy","depth_path":"/tmp/depth.npy","semantic_mask_path":"/tmp/semantic.npy","instance_mask_path":"/tmp/instance.npy","camera_pose_world":[[1,0,0,0],[0,1,0,1.25],[0,0,1,0],[0,0,0,1]],"intrinsics":{"fx":10.0,"fy":10.0,"cx":8.0,"cy":6.0},"render_time_ms":8.2,"worker_state":"ready"}\n',
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
    build_response = client.build_scene(
        BlenderSceneBuildRequest(
            scene_spec=build_living_room_scene_spec(),
            image_width=16,
            image_height=12,
            horizontal_fov_deg=72.0,
            near_plane_m=0.2,
            far_plane_m=8.0,
        ),
        timeout_sec=0.5,
    )
    frame_response = client.request_frame(
        BlenderFrameRequest(
            frame_index=3,
            timestamp_sec=1.5,
            robot_pose=build_living_room_scene_spec().start_pose,
            camera_pose_world=np.eye(4, dtype=np.float32),
        ),
        timeout_sec=0.5,
    )

    assert isinstance(build_response, BlenderSceneBuildResponse)
    assert build_response.scene_id == "living_room_navigation_v1"
    assert isinstance(frame_response, BlenderFrameResponse)
    assert frame_response.rgb_path == "/tmp/rgb.npy"
    written_packets = [json.loads(line) for line in fake_stdin.getvalue().decode("utf-8").splitlines()]
    assert written_packets[0]["kind"] == "build_scene"
    assert written_packets[1]["kind"] == "render_frame"
    assert fake_stdin.flush_calls == 2


def test_blender_worker_client_rejects_render_before_scene_build() -> None:
    client = BlenderWorkerClient(command=["blender"])

    with pytest.raises(RuntimeError, match="build_scene"):
        client.request_frame(
            BlenderFrameRequest(
                frame_index=0,
                timestamp_sec=0.0,
                robot_pose=build_living_room_scene_spec().start_pose,
                camera_pose_world=np.eye(4, dtype=np.float32),
            )
        )


def test_blender_worker_response_requires_pose_and_intrinsics() -> None:
    with pytest.raises(ValueError, match="camera_pose_world"):
        BlenderFrameResponse.from_payload(
            {
                "rgb_path": "/tmp/rgb.npy",
                "depth_path": "/tmp/depth.npy",
                "semantic_mask_path": "/tmp/semantic.npy",
                "instance_mask_path": "/tmp/instance.npy",
                "intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 6.0},
            }
        )

    with pytest.raises(ValueError, match="intrinsics"):
        BlenderFrameResponse.from_payload(
            {
                "rgb_path": "/tmp/rgb.npy",
                "depth_path": "/tmp/depth.npy",
                "semantic_mask_path": "/tmp/semantic.npy",
                "instance_mask_path": "/tmp/instance.npy",
                "camera_pose_world": np.eye(4, dtype=np.float32).tolist(),
            }
        )


def test_blender_worker_client_raises_timeout_when_no_json_response_arrives() -> None:
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
        client.build_scene(
            BlenderSceneBuildRequest(
                scene_spec=build_living_room_scene_spec(),
                image_width=16,
                image_height=12,
                horizontal_fov_deg=72.0,
                near_plane_m=0.2,
                far_plane_m=8.0,
            ),
            timeout_sec=0.25,
        )
