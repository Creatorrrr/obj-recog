from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    output_root: Path


class PythonWorkerRuntime:
    def __init__(self, *, output_root: Path) -> None:
        self._output_root = Path(output_root)
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._scene_id = "uninitialized"
        self._image_width = 64
        self._image_height = 48
        self._intrinsics = {"fx": 32.0, "fy": 32.0, "cx": 32.0, "cy": 24.0}

    def process_request(self, payload: dict[str, object]) -> dict[str, object]:
        kind = str(payload.get("kind", ""))
        if kind == "build_scene":
            self._scene_id = str(payload["scene_spec"]["scene_id"])
            self._image_width = int(payload["image_width"])
            self._image_height = int(payload["image_height"])
            self._intrinsics = {
                "fx": float(self._image_width * 0.65),
                "fy": float(self._image_width * 0.65),
                "cx": float(self._image_width * 0.5),
                "cy": float(self._image_height * 0.5),
            }
            return {"status": "ready", "scene_id": self._scene_id}
        if kind == "render_frame":
            started = time.perf_counter()
            frame_index = int(payload["frame_index"])
            rgb = np.full((self._image_height, self._image_width, 3), 60 + frame_index, dtype=np.uint8)
            depth = np.full((self._image_height, self._image_width), 2.5, dtype=np.float32)
            semantic = np.zeros((self._image_height, self._image_width), dtype=np.uint8)
            instance = np.zeros((self._image_height, self._image_width), dtype=np.uint8)

            rgb_path = self._output_root / f"frame-{frame_index:04d}.npy"
            depth_path = self._output_root / f"frame-{frame_index:04d}-depth.npy"
            semantic_path = self._output_root / f"frame-{frame_index:04d}-semantic.npy"
            instance_path = self._output_root / f"frame-{frame_index:04d}-instance.npy"
            np.save(rgb_path, rgb)
            np.save(depth_path, depth)
            np.save(semantic_path, semantic)
            np.save(instance_path, instance)
            return {
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "semantic_mask_path": str(semantic_path),
                "instance_mask_path": str(instance_path),
                "camera_pose_world": payload["camera_pose_world"],
                "intrinsics": dict(self._intrinsics),
                "render_time_ms": (time.perf_counter() - started) * 1000.0,
                "worker_state": "ready",
            }
        raise RuntimeError(f"unsupported worker request kind: {kind}")


def create_worker_runtime(*, output_root: str | Path, force_python_fallback: bool = False):
    _ = force_python_fallback
    return PythonWorkerRuntime(output_root=Path(output_root))


def parse_config(argv: list[str] | None = None) -> WorkerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(_argv_after_double_dash(argv or sys.argv))
    return WorkerConfig(output_root=Path(args.output_root))


def run_worker_loop(
    *,
    argv: list[str] | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
    force_python_fallback: bool = False,
) -> int:
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    config = parse_config(argv)
    runtime = create_worker_runtime(output_root=config.output_root, force_python_fallback=force_python_fallback)
    print(json.dumps({"worker_state": "bootstrapping"}), file=stderr, flush=True)
    for raw_line in stdin:
        stripped = str(raw_line).strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        response = runtime.process_request(payload)
        print(json.dumps(response), file=stdout, flush=True)
    return 0


def _argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return argv[1:]
    index = argv.index("--")
    return argv[index + 1 :]


if __name__ == "__main__":
    raise SystemExit(run_worker_loop())
