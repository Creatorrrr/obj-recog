from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from obj_recog.slam_bridge import (
    KeyframeObservation,
    SlamFrameResult,
    OrbSlam3Bridge,
    decode_slam_response,
    encode_frame_packet,
    orbslam3_bridge_runtime_library_dirs,
    resolve_orbslam3_bridge_binary_path,
)


def test_encode_frame_packet_prefixes_header_and_payload() -> None:
    frame_gray = np.arange(12, dtype=np.uint8).reshape(3, 4)

    packet = encode_frame_packet(frame_gray, timestamp=12.5)

    assert packet[:4] == b"SLAM"
    assert len(packet) == 4 + 8 + 4 + 4 + 12
    assert struct.unpack(">Q", packet[4:12])[0] == 12_500_000
    assert struct.unpack(">I", packet[12:16])[0] == 4
    assert struct.unpack(">I", packet[16:20])[0] == 3


def test_decode_slam_response_restores_pose_arrays_and_sparse_points() -> None:
    payload = json.dumps(
        {
            "tracking_state": "TRACKING",
            "pose_world": [
                1.0,
                0.0,
                0.0,
                0.1,
                0.0,
                1.0,
                0.0,
                0.2,
                0.0,
                0.0,
                1.0,
                0.3,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "keyframe_inserted": True,
            "keyframe_id": 7,
            "optimized_keyframe_poses": {
                "7": [
                    1.0,
                    0.0,
                    0.0,
                    0.1,
                    0.0,
                    1.0,
                    0.0,
                    0.2,
                    0.0,
                    0.0,
                    1.0,
                    0.3,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            },
            "sparse_map_points": [[0.0, 0.1, 1.0], [0.2, -0.1, 1.5]],
            "loop_closure_applied": True,
            "tracked_feature_count": 142,
            "median_reprojection_error": 1.35,
            "map_points_changed": True,
            "keyframe_observations": [
                {
                    "keyframe_id": 7,
                    "point_id": 101,
                    "u": 123.4,
                    "v": 98.7,
                    "x": 0.2,
                    "y": -0.1,
                    "z": 1.5,
                },
                {
                    "keyframe_id": 8,
                    "point_id": 101,
                    "u": 120.0,
                    "v": 97.5,
                    "x": 0.2,
                    "y": -0.1,
                    "z": 1.5,
                },
            ],
        }
    )

    result = decode_slam_response(payload)

    assert isinstance(result, SlamFrameResult)
    assert result.tracking_state == "TRACKING"
    assert result.pose_world.shape == (4, 4)
    assert result.keyframe_inserted is True
    assert result.keyframe_id == 7
    assert sorted(result.optimized_keyframe_poses) == [7]
    assert result.sparse_map_points_xyz.shape == (2, 3)
    assert result.loop_closure_applied is True
    assert result.tracked_feature_count == 142
    assert result.median_reprojection_error == pytest.approx(1.35)
    assert result.map_points_changed is True
    assert len(result.keyframe_observations) == 2
    assert all(isinstance(item, KeyframeObservation) for item in result.keyframe_observations)
    assert result.keyframe_observations[0].keyframe_id == 7
    assert result.keyframe_observations[0].point_id == 101
    assert result.keyframe_observations[0].x == pytest.approx(0.2)


def test_decode_slam_response_accepts_map_changed_alias_for_map_points_changed() -> None:
    payload = json.dumps(
        {
            "tracking_state": "TRACKING",
            "pose_world": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
            "keyframe_inserted": False,
            "keyframe_id": None,
            "optimized_keyframe_poses": {},
            "sparse_map_points": [],
            "tracked_feature_count": 9,
            "median_reprojection_error": None,
            "keyframe_observations": [],
            "map_changed": True,
        }
    )

    result = decode_slam_response(payload)

    assert result.map_points_changed is True
    assert result.loop_closure_applied is True


class _FakePipeWriter:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None


class _BrokenPipeWriter:
    def write(self, payload: bytes) -> int:
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self) -> None:
        return None


class _FakePipeReader:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)

    def read(self) -> bytes:
        data = b"".join(self._lines)
        self._lines.clear()
        return data


class _FakeProcess:
    def __init__(
        self,
        stdout_lines: list[bytes],
        stderr_lines: list[bytes] | None = None,
        *,
        stdin=None,
        returncode: int | None = None,
    ) -> None:
        self.stdin = _FakePipeWriter() if stdin is None else stdin
        self.stdout = _FakePipeReader(stdout_lines)
        self.stderr = _FakePipeReader(stderr_lines or [])
        self._returncode: int | None = returncode

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self._returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        self._returncode = 0
        return 0

    def kill(self) -> None:
        self._returncode = -9


def test_orbslam3_bridge_track_skips_non_json_stdout_lines(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    vocab = tmp_path / "ORBvoc.txt"
    settings = tmp_path / "camera.yaml"
    bridge_binary = tmp_path / "orbslam3_bridge"
    vocab.write_text("", encoding="utf-8")
    settings.write_text("", encoding="utf-8")
    bridge_binary.write_text("", encoding="utf-8")
    fake_process = _FakeProcess(
        stdout_lines=[
            b"Loading ORB-SLAM3 vocabulary\n",
            json.dumps(
                {
                    "tracking_state": "TRACKING",
                    "pose_world": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
                    "keyframe_inserted": False,
                    "keyframe_id": None,
                    "optimized_keyframe_poses": {},
                    "sparse_map_points": [],
                    "loop_closure_applied": False,
                }
            ).encode("utf-8")
            + b"\n",
        ]
    )

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake_process)

    bridge = OrbSlam3Bridge(
        vocabulary_path=str(vocab),
        settings_path=str(settings),
        frame_width=4,
        frame_height=3,
        binary_path=str(bridge_binary),
    )

    result = bridge.track(np.zeros((3, 4), dtype=np.uint8), timestamp=0.25)

    assert result.tracking_state == "TRACKING"
    assert len(fake_process.stdin.writes) == 1


def test_orbslam3_bridge_track_surfaces_native_stderr_when_pipe_breaks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocab = tmp_path / "ORBvoc.txt"
    settings = tmp_path / "camera.yaml"
    bridge_binary = tmp_path / "orbslam3_bridge.exe"
    vocab.write_text("", encoding="utf-8")
    settings.write_text("", encoding="utf-8")
    bridge_binary.write_text("", encoding="utf-8")
    fake_process = _FakeProcess(
        stdout_lines=[],
        stderr_lines=[b"truncated grayscale payload\n"],
        stdin=_BrokenPipeWriter(),
        returncode=1,
    )

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake_process)

    bridge = OrbSlam3Bridge(
        vocabulary_path=str(vocab),
        settings_path=str(settings),
        frame_width=4,
        frame_height=3,
        binary_path=str(bridge_binary),
    )

    with pytest.raises(RuntimeError, match="truncated grayscale payload"):
        bridge.track(np.zeros((3, 4), dtype=np.uint8), timestamp=0.25)


def test_resolve_orbslam3_bridge_binary_path_accepts_windows_executable_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_is_file = Path.is_file

    def fake_is_file(path: Path) -> bool:
        normalized = str(path).replace("\\", "/")
        if normalized.endswith("/native/orbslam3_bridge/build/orbslam3_bridge"):
            return False
        if normalized.endswith("/native/orbslam3_bridge/build/orbslam3_bridge.exe"):
            return True
        return real_is_file(path)

    monkeypatch.setattr(Path, "is_file", fake_is_file)

    resolved = resolve_orbslam3_bridge_binary_path()

    assert resolved is not None
    assert str(resolved).replace("\\", "/").endswith("/native/orbslam3_bridge/build/orbslam3_bridge.exe")


def test_orbslam3_bridge_prepends_windows_runtime_dirs_to_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocab = tmp_path / "ORBvoc.txt"
    settings = tmp_path / "camera.yaml"
    bridge_binary = tmp_path / "orbslam3_bridge.exe"
    runtime_a = tmp_path / "runtime-a"
    runtime_b = tmp_path / "runtime-b"
    vocab.write_text("", encoding="utf-8")
    settings.write_text("", encoding="utf-8")
    bridge_binary.write_text("", encoding="utf-8")
    runtime_a.mkdir()
    runtime_b.mkdir()
    fake_process = _FakeProcess(stdout_lines=[])
    captured: dict[str, object] = {}

    def fake_popen(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return fake_process

    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    monkeypatch.setattr("obj_recog.slam_bridge.os.name", "nt", raising=False)
    monkeypatch.setattr(
        "obj_recog.slam_bridge.orbslam3_bridge_runtime_library_dirs",
        lambda **_: (runtime_a, runtime_b),
    )
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    OrbSlam3Bridge(
        vocabulary_path=str(vocab),
        settings_path=str(settings),
        frame_width=4,
        frame_height=3,
        binary_path=str(bridge_binary),
    )

    assert captured["env"] is not None
    path_entries = str(captured["env"]["PATH"]).split(";")
    assert path_entries[:2] == [str(runtime_a), str(runtime_b)]


def test_orbslam3_bridge_runtime_library_dirs_include_windows_dependency_bins(tmp_path) -> None:
    bridge_build = tmp_path / "native" / "orbslam3_bridge" / "build"
    orbslam_lib = tmp_path / "third_party" / "ORB_SLAM3" / "lib"
    dbow2_lib = tmp_path / "third_party" / "ORB_SLAM3" / "Thirdparty" / "DBoW2" / "lib"
    g2o_lib = tmp_path / "third_party" / "ORB_SLAM3" / "Thirdparty" / "g2o" / "lib"
    opencv_bin = tmp_path / "build" / "opencv-cuda" / "install" / "x64" / "vc17" / "bin"
    vcpkg_bin = tmp_path / "build" / "vcpkg" / "installed" / "x64-windows" / "bin"
    for directory in (bridge_build, orbslam_lib, dbow2_lib, g2o_lib, opencv_bin, vcpkg_bin):
        directory.mkdir(parents=True)

    runtime_dirs = orbslam3_bridge_runtime_library_dirs(repo_root=tmp_path)

    assert bridge_build in runtime_dirs
    assert orbslam_lib in runtime_dirs
    assert dbow2_lib in runtime_dirs
    assert g2o_lib in runtime_dirs
    assert opencv_bin in runtime_dirs
    assert vcpkg_bin in runtime_dirs
