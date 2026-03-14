from __future__ import annotations

import json
import os
import struct
import subprocess
from json import JSONDecodeError
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


TRACKING_OK_STATES = {"TRACKING", "RELOCALIZED"}
ORBSLAM3_BRIDGE_BUILD_HINT_PATH = Path("native") / "orbslam3_bridge" / "build" / "orbslam3_bridge"
_ORBSLAM3_BRIDGE_RELATIVE_CANDIDATES = (
    ORBSLAM3_BRIDGE_BUILD_HINT_PATH,
    ORBSLAM3_BRIDGE_BUILD_HINT_PATH.with_suffix(".exe"),
    ORBSLAM3_BRIDGE_BUILD_HINT_PATH.parent / "Release" / ORBSLAM3_BRIDGE_BUILD_HINT_PATH.with_suffix(".exe").name,
    ORBSLAM3_BRIDGE_BUILD_HINT_PATH.parent / "Debug" / ORBSLAM3_BRIDGE_BUILD_HINT_PATH.with_suffix(".exe").name,
    ORBSLAM3_BRIDGE_BUILD_HINT_PATH.parent / "RelWithDebInfo" / ORBSLAM3_BRIDGE_BUILD_HINT_PATH.with_suffix(".exe").name,
    ORBSLAM3_BRIDGE_BUILD_HINT_PATH.parent / "MinSizeRel" / ORBSLAM3_BRIDGE_BUILD_HINT_PATH.with_suffix(".exe").name,
)


@dataclass(frozen=True, slots=True)
class KeyframeObservation:
    keyframe_id: int
    point_id: int
    u: float
    v: float
    x: float
    y: float
    z: float


@dataclass(slots=True)
class SlamFrameResult:
    tracking_state: str
    pose_world: np.ndarray
    keyframe_inserted: bool
    keyframe_id: int | None
    optimized_keyframe_poses: dict[int, np.ndarray]
    sparse_map_points_xyz: np.ndarray
    loop_closure_applied: bool
    tracked_feature_count: int = 0
    median_reprojection_error: float | None = None
    keyframe_observations: list[KeyframeObservation] = field(default_factory=list)
    map_points_changed: bool = False

    @property
    def tracking_ok(self) -> bool:
        return self.tracking_state in TRACKING_OK_STATES


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def orbslam3_bridge_binary_candidates(*, repo_root: Path | None = None) -> tuple[Path, ...]:
    root = _repo_root() if repo_root is None else Path(repo_root)
    return tuple(root / relative_path for relative_path in _ORBSLAM3_BRIDGE_RELATIVE_CANDIDATES)


def resolve_orbslam3_bridge_binary_path(binary_path: str | Path | None = None) -> Path | None:
    if binary_path is not None:
        candidate = Path(binary_path)
        return candidate if candidate.is_file() else None

    for candidate in orbslam3_bridge_binary_candidates():
        if candidate.is_file():
            return candidate
    return None


def orbslam3_bridge_runtime_library_dirs(*, repo_root: Path | None = None) -> tuple[Path, ...]:
    root = _repo_root() if repo_root is None else Path(repo_root)
    candidates = (
        root / "native" / "orbslam3_bridge" / "build",
        root / "native" / "orbslam3_bridge" / "build" / "Release",
        root / "native" / "orbslam3_bridge" / "build" / "Debug",
        root / "native" / "orbslam3_bridge" / "build" / "RelWithDebInfo",
        root / "native" / "orbslam3_bridge" / "build" / "MinSizeRel",
        root / "third_party" / "ORB_SLAM3" / "lib",
        root / "third_party" / "ORB_SLAM3" / "Thirdparty" / "DBoW2" / "lib",
        root / "third_party" / "ORB_SLAM3" / "Thirdparty" / "g2o" / "lib",
    )
    ordered_existing_dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        ordered_existing_dirs.append(candidate)
    return tuple(ordered_existing_dirs)


def encode_frame_packet(frame_gray: np.ndarray, timestamp: float) -> bytes:
    frame_gray = np.asarray(frame_gray, dtype=np.uint8)
    if frame_gray.ndim != 2:
        raise ValueError("SLAM frame must be grayscale")
    timestamp_micros = max(0, int(round(float(timestamp) * 1_000_000.0)))
    header = struct.pack(">4sQII", b"SLAM", timestamp_micros, frame_gray.shape[1], frame_gray.shape[0])
    return header + frame_gray.tobytes(order="C")


def decode_slam_response(payload: str) -> SlamFrameResult:
    data = json.loads(payload)
    pose = np.asarray(data.get("pose_world") or np.eye(4, dtype=np.float32).reshape(-1).tolist(), dtype=np.float32)
    pose_world = pose.reshape(4, 4)
    keyframe_poses = {
        int(key): np.asarray(value, dtype=np.float32).reshape(4, 4)
        for key, value in (data.get("optimized_keyframe_poses") or {}).items()
    }
    sparse_points = np.asarray(data.get("sparse_map_points") or [], dtype=np.float32).reshape(-1, 3)
    keyframe_observations = [
        KeyframeObservation(
            keyframe_id=int(item["keyframe_id"]),
            point_id=int(item["point_id"]),
            u=float(item["u"]),
            v=float(item["v"]),
            x=float(item["x"]),
            y=float(item["y"]),
            z=float(item["z"]),
        )
        for item in (data.get("keyframe_observations") or [])
    ]
    return SlamFrameResult(
        tracking_state=str(data.get("tracking_state", "LOST")),
        pose_world=pose_world,
        keyframe_inserted=bool(data.get("keyframe_inserted", False)),
        keyframe_id=(None if data.get("keyframe_id") is None else int(data["keyframe_id"])),
        optimized_keyframe_poses=keyframe_poses,
        sparse_map_points_xyz=sparse_points,
        loop_closure_applied=bool(data.get("map_changed", data.get("loop_closure_applied", False))),
        tracked_feature_count=int(data.get("tracked_feature_count", 0)),
        median_reprojection_error=(
            None if data.get("median_reprojection_error") is None else float(data["median_reprojection_error"])
        ),
        keyframe_observations=keyframe_observations,
        map_points_changed=bool(data.get("map_points_changed", data.get("map_changed", False))),
    )


def _read_protocol_response(process: subprocess.Popen[bytes]) -> SlamFrameResult:
    if process.stdout is None:
        raise RuntimeError("ORB-SLAM3 bridge stdout is unavailable")

    ignored_lines: list[str] = []
    while True:
        line = process.stdout.readline()
        if not line:
            stderr = ""
            if process.stderr is not None:
                try:
                    stderr = process.stderr.readline().decode("utf-8", errors="replace")
                except Exception:
                    stderr = ""
            ignored = " | ".join(entry for entry in ignored_lines if entry)
            details = stderr or ignored
            raise RuntimeError(f"ORB-SLAM3 bridge returned no result. {details}".strip())

        decoded = line.decode("utf-8", errors="replace").strip()
        if not decoded:
            continue
        try:
            return decode_slam_response(decoded)
        except JSONDecodeError:
            ignored_lines.append(decoded)
            continue


class OrbSlam3Bridge:
    def __init__(
        self,
        *,
        vocabulary_path: str,
        settings_path: str,
        frame_width: int,
        frame_height: int,
        binary_path: str | None = None,
    ) -> None:
        if not vocabulary_path or not Path(vocabulary_path).is_file():
            raise RuntimeError("ORB-SLAM3 requires a valid --slam-vocabulary path")
        if not settings_path or not Path(settings_path).is_file():
            raise RuntimeError("ORB-SLAM3 requires a valid --camera-calibration path")

        self._frame_width = int(frame_width)
        self._frame_height = int(frame_height)
        resolved_binary_path = resolve_orbslam3_bridge_binary_path(binary_path)
        if resolved_binary_path is None:
            searched_locations = [Path(binary_path)] if binary_path is not None else list(orbslam3_bridge_binary_candidates())
            searched = ", ".join(str(path) for path in searched_locations)
            raise RuntimeError(
                "ORB-SLAM3 bridge binary not found. "
                f"Searched: {searched}. "
                f"Build {ORBSLAM3_BRIDGE_BUILD_HINT_PATH.as_posix()} first."
            )
        self._binary_path = str(resolved_binary_path)

        self._command = [
            self._binary_path,
            "--vocabulary",
            vocabulary_path,
            "--settings",
            settings_path,
            "--width",
            str(self._frame_width),
            "--height",
            str(self._frame_height),
        ]
        process_env = None
        if os.name == "nt":
            process_env = os.environ.copy()
            runtime_dirs = [str(path) for path in orbslam3_bridge_runtime_library_dirs()]
            if runtime_dirs:
                path_entries = runtime_dirs
                existing_path = process_env.get("PATH")
                if existing_path:
                    path_entries.append(existing_path)
                process_env["PATH"] = os.pathsep.join(path_entries)
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=process_env,
        )

    def track(self, frame_gray: np.ndarray, timestamp: float) -> SlamFrameResult:
        frame_gray = np.asarray(frame_gray, dtype=np.uint8)
        if frame_gray.shape != (self._frame_height, self._frame_width):
            raise ValueError("SLAM frame shape does not match configured bridge resolution")
        if self._process.poll() is not None or self._process.stdin is None or self._process.stdout is None:
            stderr = ""
            if self._process.stderr is not None:
                try:
                    stderr = self._process.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    stderr = ""
            raise RuntimeError(f"ORB-SLAM3 bridge is not running. {stderr}".strip())

        packet = encode_frame_packet(frame_gray, timestamp)
        self._process.stdin.write(packet)
        self._process.stdin.flush()
        return _read_protocol_response(self._process)

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)
