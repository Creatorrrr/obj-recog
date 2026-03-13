from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import select
import subprocess
import time
from typing import Any, Callable

import numpy as np

from obj_recog.sim_protocol import LivingRoomSceneSpec, RobotPose


def build_realtime_blender_worker_command(
    *,
    blender_exec: str,
    repo_root: str | Path,
    output_root: str | Path,
    blend_file: str | Path | None = None,
) -> list[str]:
    root = Path(repo_root)
    worker_script = root / "scripts" / "blender" / "realtime_worker.py"
    command = [str(blender_exec)]
    if blend_file is not None:
        command.append(str(blend_file))
    command.extend(
        [
            "--background",
            "--python",
            str(worker_script),
            "--",
            "--output-root",
            str(output_root),
        ]
    )
    return command


@dataclass(frozen=True, slots=True)
class BlenderSceneBuildRequest:
    scene_spec: LivingRoomSceneSpec
    image_width: int
    image_height: int
    horizontal_fov_deg: float
    near_plane_m: float
    far_plane_m: float

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": "build_scene",
            "scene_spec": _json_ready(self.scene_spec),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "horizontal_fov_deg": float(self.horizontal_fov_deg),
            "near_plane_m": float(self.near_plane_m),
            "far_plane_m": float(self.far_plane_m),
        }


@dataclass(frozen=True, slots=True)
class BlenderSceneBuildResponse:
    status: str
    scene_id: str

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> BlenderSceneBuildResponse:
        return cls(
            status=str(payload.get("status", "")),
            scene_id=str(payload.get("scene_id", "")),
        )


@dataclass(frozen=True, slots=True)
class BlenderFrameRequest:
    frame_index: int
    timestamp_sec: float
    robot_pose: RobotPose
    camera_pose_world: np.ndarray

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": "render_frame",
            "frame_index": int(self.frame_index),
            "timestamp_sec": float(self.timestamp_sec),
            "robot_pose": _json_ready(self.robot_pose),
            "camera_pose_world": _json_ready(self.camera_pose_world),
        }


@dataclass(frozen=True, slots=True)
class BlenderFrameResponse:
    rgb_path: str
    depth_path: str
    semantic_mask_path: str
    instance_mask_path: str
    camera_pose_world: np.ndarray
    intrinsics: dict[str, float]
    render_time_ms: float | None
    worker_state: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> BlenderFrameResponse:
        camera_pose_world = _require_payload_field(payload, "camera_pose_world")
        intrinsics = _require_payload_field(payload, "intrinsics")
        return cls(
            rgb_path=str(payload["rgb_path"]),
            depth_path=str(payload["depth_path"]),
            semantic_mask_path=str(payload["semantic_mask_path"]),
            instance_mask_path=str(payload["instance_mask_path"]),
            camera_pose_world=np.asarray(camera_pose_world, dtype=np.float32).reshape(4, 4),
            intrinsics={str(key): float(value) for key, value in dict(intrinsics).items()},
            render_time_ms=None if payload.get("render_time_ms") is None else float(payload["render_time_ms"]),
            worker_state=None if payload.get("worker_state") is None else str(payload["worker_state"]),
        )


class BlenderWorkerClient:
    def __init__(
        self,
        *,
        command: list[str] | tuple[str, ...],
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        selector=select.select,
    ) -> None:
        self._command = [str(item) for item in command]
        self._popen_factory = popen_factory
        self._selector = selector
        self._process: subprocess.Popen[bytes] | None = None
        self._scene_built = False

    def start(self) -> None:
        process = self._popen_factory(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.poll() is not None or process.stdin is None or process.stdout is None:
            raise RuntimeError("Blender worker failed to start")
        self._process = process

    def build_scene(
        self,
        request: BlenderSceneBuildRequest,
        *,
        timeout_sec: float | None = None,
    ) -> BlenderSceneBuildResponse:
        payload = self._request(request.to_payload(), timeout_sec=timeout_sec)
        self._scene_built = True
        return BlenderSceneBuildResponse.from_payload(payload)

    def request_frame(
        self,
        request: BlenderFrameRequest,
        *,
        timeout_sec: float | None = None,
    ) -> BlenderFrameResponse:
        if not self._scene_built:
            raise RuntimeError("Blender worker build_scene must complete before request_frame")
        payload = self._request(request.to_payload(), timeout_sec=timeout_sec)
        return BlenderFrameResponse.from_payload(payload)

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        process.wait(timeout=1.0)

    def _request(self, payload: dict[str, object], *, timeout_sec: float | None) -> dict[str, object]:
        process = self._require_process()
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n")
        process.stdin.flush()
        deadline = None if timeout_sec is None else (time.monotonic() + float(timeout_sec))
        skipped_stdout: list[str] = []

        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            line = _read_stdout_line(process.stdout, timeout_sec=remaining, selector=self._selector)
            if line is None:
                stderr_excerpt = _stderr_excerpt(process)
                if stderr_excerpt:
                    raise RuntimeError(
                        "Blender worker did not produce a JSON response before timeout. "
                        f"stderr: {stderr_excerpt}"
                    )
                raise TimeoutError(f"Blender worker response timed out after {timeout_sec} seconds")

            decoded_line = line.decode("utf-8", errors="replace").strip()
            if not decoded_line:
                if process.poll() is not None:
                    stderr_excerpt = _stderr_excerpt(process)
                    if stderr_excerpt:
                        raise RuntimeError(
                            "Blender worker exited before returning JSON. "
                            f"stderr: {stderr_excerpt}"
                        )
                    raise RuntimeError("Blender worker exited before returning JSON")
                continue

            try:
                decoded = json.loads(decoded_line)
            except json.JSONDecodeError:
                skipped_stdout.append(decoded_line)
                if process.poll() is not None:
                    stderr_excerpt = _stderr_excerpt(process)
                    stdout_excerpt = " | ".join(skipped_stdout[-3:])
                    details = stdout_excerpt
                    if stderr_excerpt:
                        details = f"stdout: {stdout_excerpt}; stderr: {stderr_excerpt}"
                    raise RuntimeError(
                        "Blender worker exited before returning JSON. "
                        f"{details}"
                    )
                continue

            if not isinstance(decoded, dict):
                raise RuntimeError("Blender worker response must be a JSON object")
            return decoded

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError("Blender worker has not been started")
        if self._process.poll() is not None:
            raise RuntimeError("Blender worker is not running")
        return self._process


def _read_stdout_line(stdout, *, timeout_sec: float | None, selector) -> bytes | None:
    if timeout_sec is not None:
        readable, _, _ = selector([stdout], [], [], timeout_sec)
        if not readable:
            return None
    line = stdout.readline()
    return None if not line else line


def _stderr_excerpt(process: subprocess.Popen[bytes], *, max_chars: int = 800) -> str:
    stderr = getattr(process, "stderr", None)
    if stderr is None:
        return ""
    if process.poll() is None:
        return ""
    try:
        content = stderr.read()
    except Exception:
        return ""
    if not content:
        return ""
    return content.decode("utf-8", errors="replace").strip()[:max_chars]


def _require_payload_field(payload: dict[str, object], key: str) -> object:
    if key not in payload or payload[key] is None:
        raise ValueError(f"Blender worker response is missing required field '{key}'")
    return payload[key]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, RobotPose):
        return {
            "x": float(value.x),
            "y": float(value.y),
            "z": float(value.z),
            "yaw_deg": float(value.yaw_deg),
            "camera_pan_deg": float(value.camera_pan_deg),
        }
    if isinstance(value, LivingRoomSceneSpec):
        return {
            "scene_id": value.scene_id,
            "room_size_xyz": list(value.room_size_xyz),
            "wall_thickness_m": float(value.wall_thickness_m),
            "window_wall": value.window_wall,
            "start_pose": _json_ready(value.start_pose),
            "hidden_goal_pose_xyz": list(value.hidden_goal_pose_xyz),
            "blend_file_path": value.blend_file_path,
            "goal_description": value.goal_description,
            "semantic_target_class": value.semantic_target_class,
            "objects": [
                {
                    "object_id": item.object_id,
                    "semantic_label": item.semantic_label,
                    "center_xyz": list(item.center_xyz),
                    "size_xyz": list(item.size_xyz),
                    "yaw_deg": float(item.yaw_deg),
                    "material_key": item.material_key,
                    "collider": bool(item.collider),
                }
                for item in value.objects
            ],
            "lights": [
                {
                    "light_id": light.light_id,
                    "light_type": light.light_type,
                    "location_xyz": list(light.location_xyz),
                    "rotation_deg_xyz": list(light.rotation_deg_xyz),
                    "color_rgb": list(light.color_rgb),
                    "energy": float(light.energy),
                }
                for light in value.lights
            ],
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value
