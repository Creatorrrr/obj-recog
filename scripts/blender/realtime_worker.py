from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    scene_manifest_path: Path
    render_root: Path
    asset_cache_dir: Path
    quality: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent Blender realtime render worker")
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--render-root", required=True)
    parser.add_argument("--asset-cache-dir", required=True)
    parser.add_argument("--quality", choices=("low", "high"), default="low")
    return parser


def parse_config(argv: list[str] | None = None) -> WorkerConfig:
    args = build_arg_parser().parse_args(argv)
    return WorkerConfig(
        scene_manifest_path=Path(args.scene_manifest).expanduser().resolve(),
        render_root=Path(args.render_root).expanduser().resolve(),
        asset_cache_dir=Path(args.asset_cache_dir).expanduser().resolve(),
        quality=str(args.quality),
    )


def emit_startup_banner(config: WorkerConfig) -> None:
    payload = {
        "worker_state": "bootstrapping",
        "scene_manifest_path": str(config.scene_manifest_path),
        "render_root": str(config.render_root),
        "asset_cache_dir": str(config.asset_cache_dir),
        "quality": config.quality,
    }
    sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    config.render_root.mkdir(parents=True, exist_ok=True)
    emit_startup_banner(config)
    raise SystemExit(
        "scripts/blender/realtime_worker.py is a bootstrap stub. "
        "The live Blender render loop will be added in a later task."
    )


if __name__ == "__main__":
    raise SystemExit(main())
