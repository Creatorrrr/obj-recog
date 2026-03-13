from __future__ import annotations

from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def _load_worker_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "blender"
        / "realtime_worker.py"
    )
    spec = importlib.util.spec_from_file_location("realtime_worker_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scene_manifest_payload() -> dict[str, object]:
    return {
        "scene_id": "studio_open_v1",
        "scenario_family": "studio",
        "difficulty_level": 1,
        "semantic_target_class": "backpack",
        "asset_manifest_id": "manifest-1234",
        "environment": {
            "room_width_m": 6.0,
            "room_depth_m": 8.0,
            "room_height_m": 3.0,
        },
        "camera_rig": {
            "image_width": 64,
            "image_height": 48,
            "fps": 4.0,
            "horizontal_fov_deg": 75.0,
            "near_plane_m": 0.1,
            "far_plane_m": 25.0,
        },
        "placements": [
            {
                "asset_id": "desk_basic",
                "semantic_class": "desk",
                "asset_family": "furniture",
                "target_role": False,
                "source_kind": "static",
                "source_key": "static-0",
                "center_world": [0.0, 0.5, 4.5],
                "size_xyz": [1.4, 0.8, 0.8],
                "yaw_deg": 0.0,
                "preview_sprite_path": "/tmp/desk.png",
                "preview_mesh_path": "/tmp/desk.ply",
                "asset_metadata_path": "/tmp/desk.json",
                "asset_provenance": "external",
                "blender_library_path": "/tmp/assets/libraries/furniture/desk_basic.blend",
                "blender_object_name": "DeskBasic",
                "recommended_scale": 1.0,
                "lod_hint": "low",
                "material_variant": "furniture-desk",
                "render_representation": "mesh",
            },
            {
                "asset_id": "person_walker",
                "semantic_class": "person",
                "asset_family": "character",
                "target_role": False,
                "source_kind": "dynamic",
                "source_key": "occluder-1",
                "center_world": [0.0, 0.9, 3.5],
                "size_xyz": [0.6, 1.8, 0.6],
                "yaw_deg": 0.0,
                "preview_sprite_path": "/tmp/person.png",
                "preview_mesh_path": "/tmp/person.ply",
                "asset_metadata_path": "/tmp/person.json",
                "asset_provenance": "external",
                "blender_library_path": "/tmp/assets/libraries/character/person_walker.blend",
                "blender_object_name": "PersonWalker",
                "recommended_scale": 1.0,
                "lod_hint": "low",
                "material_variant": "character-person",
                "render_representation": "mesh",
            },
            {
                "asset_id": "backpack_canvas",
                "semantic_class": "backpack",
                "asset_family": "prop",
                "target_role": True,
                "source_kind": "static",
                "source_key": "static-1",
                "center_world": [0.4, 0.35, 2.8],
                "size_xyz": [0.45, 0.55, 0.3],
                "yaw_deg": 0.0,
                "preview_sprite_path": "/tmp/backpack.png",
                "preview_mesh_path": "/tmp/backpack.ply",
                "asset_metadata_path": "/tmp/backpack.json",
                "asset_provenance": "external",
                "blender_library_path": "/tmp/assets/libraries/prop/backpack_canvas.blend",
                "blender_object_name": "BackpackCanvas",
                "recommended_scale": 1.0,
                "lod_hint": "low",
                "material_variant": "prop-backpack",
                "render_representation": "mesh",
            },
        ],
    }


def test_realtime_worker_script_processes_request_in_python_mode(tmp_path: Path) -> None:
    module = _load_worker_module()
    manifest_path = tmp_path / "scene.json"
    manifest_path.write_text(json.dumps(_scene_manifest_payload()), encoding="utf-8")
    config = module.WorkerConfig(
        scene_manifest_path=manifest_path,
        render_root=tmp_path / "render-root",
        asset_cache_dir=tmp_path / "assets",
        quality="low",
    )
    scene_manifest = module.load_scene_manifest(config.scene_manifest_path)
    runtime = module.create_worker_runtime(config, scene_manifest, force_python_fallback=True)

    response = runtime.process_request(
        {
            "frame_index": 3,
            "timestamp_sec": 0.75,
            "scenario_id": "studio_open_v1",
            "camera_pose_world": np.eye(4, dtype=np.float32).tolist(),
            "intrinsics": {"fx": 46.0, "fy": 46.0, "cx": 32.0, "cy": 24.0},
            "dynamic_actor_transforms": {
                "occluder-1": (
                    np.array(
                        [
                            [1.0, 0.0, 0.0, 0.2],
                            [0.0, 1.0, 0.0, 1.1],
                            [0.0, 0.0, 1.0, 3.1],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ).tolist()
                )
            },
            "lighting_seed": 7,
        }
    )

    assert response["worker_state"] == "ready"
    assert Path(str(response["rgb_path"])).is_file()
    assert Path(str(response["depth_path"])).is_file()
    assert Path(str(response["semantic_mask_path"])).is_file()
    assert Path(str(response["instance_mask_path"])).is_file()
    assert tuple(response["intrinsics_gt"].keys()) == ("fx", "fy", "cx", "cy")
    assert np.asarray(np.load(response["depth_path"]), dtype=np.float32).shape == (48, 64)
    assert np.asarray(np.load(response["semantic_mask_path"]), dtype=np.uint8).shape == (48, 64)
    assert np.asarray(np.load(response["instance_mask_path"]), dtype=np.uint8).shape == (48, 64)
    assert any(item["label"] == "backpack" for item in response["detections"])


def test_realtime_worker_script_loop_writes_json_response_line(tmp_path: Path) -> None:
    module = _load_worker_module()
    manifest_path = tmp_path / "scene.json"
    manifest_path.write_text(json.dumps(_scene_manifest_payload()), encoding="utf-8")
    config = module.WorkerConfig(
        scene_manifest_path=manifest_path,
        render_root=tmp_path / "render-root",
        asset_cache_dir=tmp_path / "assets",
        quality="low",
    )
    request_payload = {
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "scenario_id": "studio_open_v1",
        "camera_pose_world": np.eye(4, dtype=np.float32).tolist(),
        "intrinsics": {"fx": 46.0, "fy": 46.0, "cx": 32.0, "cy": 24.0},
        "dynamic_actor_transforms": {},
        "lighting_seed": 3,
    }
    stdin = io.StringIO(json.dumps(request_payload) + "\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = module.run_worker_loop(
        config=config,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        force_python_fallback=True,
    )

    assert exit_code == 0
    stderr_lines = [line for line in stderr.getvalue().splitlines() if line.strip()]
    assert stderr_lines
    startup_payload = json.loads(stderr_lines[0])
    assert startup_payload["worker_state"] == "bootstrapping"
    response_payload = json.loads(stdout.getvalue().strip())
    assert response_payload["worker_state"] == "ready"
    assert response_payload["render_time_ms"] >= 0.0
    assert Path(response_payload["rgb_path"]).is_file()


def test_realtime_worker_parse_config_uses_args_after_double_dash() -> None:
    module = _load_worker_module()
    config = module.parse_config(
        [
            "blender",
            "--background",
            "--python",
            "scripts/blender/realtime_worker.py",
            "--",
            "--scene-manifest",
            "/tmp/scene.json",
            "--render-root",
            "/tmp/render-root",
            "--asset-cache-dir",
            "/tmp/assets",
            "--quality",
            "high",
        ]
    )

    assert str(config.scene_manifest_path).endswith("/tmp/scene.json")
    assert str(config.render_root).endswith("/tmp/render-root")
    assert str(config.asset_cache_dir).endswith("/tmp/assets")
    assert config.quality == "high"


def test_realtime_worker_strict_external_assets_rejects_python_fallback(tmp_path: Path) -> None:
    module = _load_worker_module()
    payload = _scene_manifest_payload()
    payload["require_external_assets"] = True
    manifest_path = tmp_path / "scene.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    config = module.WorkerConfig(
        scene_manifest_path=manifest_path,
        render_root=tmp_path / "render-root",
        asset_cache_dir=tmp_path / "assets",
        quality="low",
    )
    scene_manifest = module.load_scene_manifest(config.scene_manifest_path)

    with pytest.raises(RuntimeError, match="Python fallback is disabled"):
        module.create_worker_runtime(config, scene_manifest, force_python_fallback=True)
