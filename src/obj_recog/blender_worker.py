from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import select
import subprocess
from typing import Any, Callable

import numpy as np


def build_blender_worker_command(
    *,
    blender_exec: str,
    worker_script: str | Path,
    blend_file: str | Path | None = None,
    extra_args: list[str] | tuple[str, ...] = (),
) -> list[str]:
    command = [str(blender_exec)]
    if blend_file is not None:
        command.append(str(blend_file))
    command.extend(["--background", "--python", str(worker_script)])
    if extra_args:
        command.extend(["--", *[str(item) for item in extra_args]])
    return command


@dataclass(frozen=True, slots=True)
class BlenderFrameRequest:
    frame_index: int
    timestamp_sec: float
    scenario_id: str
    camera_pose_world: np.ndarray
    intrinsics: dict[str, float]
    dynamic_actor_transforms: dict[str, np.ndarray]
    lighting_seed: int

    def to_payload(self) -> dict[str, object]:
        return {
            "frame_index": int(self.frame_index),
            "timestamp_sec": float(self.timestamp_sec),
            "scenario_id": str(self.scenario_id),
            "camera_pose_world": _json_ready(self.camera_pose_world),
            "intrinsics": {str(key): float(value) for key, value in self.intrinsics.items()},
            "dynamic_actor_transforms": {
                str(key): _json_ready(value) for key, value in self.dynamic_actor_transforms.items()
            },
            "lighting_seed": int(self.lighting_seed),
        }


@dataclass(frozen=True, slots=True)
class BlenderFrameResponse:
    rgb_path: str
    depth_path: str
    semantic_mask_path: str | None
    instance_mask_path: str | None
    pose_world_gt: np.ndarray
    intrinsics_gt: dict[str, float]
    render_time_ms: float | None
    worker_state: str | None

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> BlenderFrameResponse:
        return cls(
            rgb_path=str(payload["rgb_path"]),
            depth_path=str(payload["depth_path"]),
            semantic_mask_path=(
                None if payload.get("semantic_mask_path") in (None, "") else str(payload["semantic_mask_path"])
            ),
            instance_mask_path=(
                None if payload.get("instance_mask_path") in (None, "") else str(payload["instance_mask_path"])
            ),
            pose_world_gt=np.asarray(payload.get("pose_world_gt") or np.eye(4, dtype=np.float32), dtype=np.float32).reshape(4, 4),
            intrinsics_gt={
                str(key): float(value)
                for key, value in dict(payload.get("intrinsics_gt") or {}).items()
            },
            render_time_ms=(
                None if payload.get("render_time_ms") is None else float(payload["render_time_ms"])
            ),
            worker_state=(
                None if payload.get("worker_state") in (None, "") else str(payload["worker_state"])
            ),
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

    @property
    def command(self) -> list[str]:
        return list(self._command)

    def start(self) -> None:
        process = self._popen_factory(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.poll() is not None or process.stdin is None or process.stdout is None:
            stderr = _read_stderr_line(process)
            raise RuntimeError(f"Blender worker failed to start. {stderr}".strip())
        self._process = process

    def request_frame(
        self,
        request: BlenderFrameRequest,
        *,
        timeout_sec: float | None = None,
    ) -> BlenderFrameResponse:
        process = self._require_process()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Blender worker streams are unavailable")
        packet = json.dumps(request.to_payload(), separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        process.stdin.write(packet)
        process.stdin.flush()
        response_line = _read_stdout_line(process.stdout, timeout_sec=timeout_sec, selector=self._selector)
        if response_line is None:
            raise TimeoutError(f"Blender worker response timed out after {timeout_sec} seconds")
        if not response_line.strip():
            raise RuntimeError("Blender worker returned an empty response")
        try:
            payload = json.loads(response_line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Blender worker returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Blender worker response must be a JSON object")
        return BlenderFrameResponse.from_payload(payload)

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError("Blender worker has not been started")
        if self._process.poll() is not None:
            stderr = _read_stderr_line(self._process)
            raise RuntimeError(f"Blender worker is not running. {stderr}".strip())
        return self._process


def _read_stdout_line(stdout, *, timeout_sec: float | None, selector) -> bytes | None:
    if timeout_sec is not None:
        readable, _, _ = selector([stdout], [], [], timeout_sec)
        if not readable:
            return None
    return stdout.readline()


def _read_stderr_line(process: subprocess.Popen[bytes]) -> str:
    if process.stderr is None:
        return ""
    try:
        return process.stderr.readline().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
