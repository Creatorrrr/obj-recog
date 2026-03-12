from __future__ import annotations

from pathlib import Path

from PIL import Image

from obj_recog.sim_assets import (
    build_scenario_asset_manifest,
    ensure_asset_payload,
    ensure_preview_sprite,
    load_asset_catalog,
    write_blender_scene_manifest,
)
from obj_recog.simulation import CameraRigSpec
from obj_recog.simulation import SCENARIO_SPECS


def test_load_asset_catalog_exposes_semantic_assets_with_download_metadata() -> None:
    catalog = load_asset_catalog()

    for asset_id in ("desk_basic", "chair_modern", "backpack_canvas", "person_walker", "suitcase_hardshell"):
        assert asset_id in catalog
        entry = catalog[asset_id]
        assert entry.semantic_class
        assert entry.license_name == "CC0"
        assert entry.download_url.startswith("https://")
        assert entry.asset_family
        assert entry.blender_library_relpath.endswith(".blend")
        assert entry.blender_object_name
        assert entry.recommended_scale > 0.0
        assert entry.lod_hint in {"low", "medium", "high"}
        assert entry.material_variant


def test_build_scenario_asset_manifest_is_deterministic(tmp_path: Path) -> None:
    scenario = SCENARIO_SPECS["studio_open_v1"]

    manifest_a = build_scenario_asset_manifest(
        scenario,
        seed=7,
        cache_dir=tmp_path,
        quality="low",
    )
    manifest_b = build_scenario_asset_manifest(
        scenario,
        seed=7,
        cache_dir=tmp_path,
        quality="low",
    )

    assert manifest_a.manifest_id == manifest_b.manifest_id
    assert tuple((item.asset_id, item.semantic_class, item.target_role) for item in manifest_a.placements) == tuple(
        (item.asset_id, item.semantic_class, item.target_role) for item in manifest_b.placements
    )
    assert manifest_a.semantic_target_class == "backpack"


def test_build_scenario_asset_manifest_uses_mesh_first_placements(tmp_path: Path) -> None:
    scenario = SCENARIO_SPECS["studio_open_v1"]
    manifest = build_scenario_asset_manifest(
        scenario,
        seed=7,
        cache_dir=tmp_path,
        quality="low",
    )

    assert manifest.placements
    for placement in manifest.placements:
        assert placement.render_representation == "mesh"
        assert placement.blender_library_path.endswith(".blend")
        assert placement.blender_object_name
        assert placement.recommended_scale > 0.0
        assert placement.lod_hint in {"low", "medium", "high"}
        assert placement.material_variant


def test_ensure_preview_sprite_writes_nonempty_rgba_sprite(tmp_path: Path) -> None:
    catalog = load_asset_catalog()

    sprite_path = ensure_preview_sprite(
        catalog["backpack_canvas"],
        cache_dir=tmp_path,
        quality="low",
    )

    assert sprite_path.is_file()
    image = Image.open(sprite_path)
    assert image.mode == "RGBA"
    alpha = image.getchannel("A")
    assert alpha.getbbox() is not None


def test_ensure_asset_payload_caches_downloaded_bytes(tmp_path: Path) -> None:
    catalog = load_asset_catalog()
    calls: list[str] = []

    def _downloader(url: str) -> bytes:
        calls.append(url)
        return b"fake-asset-binary"

    asset_path = ensure_asset_payload(
        catalog["chair_modern"],
        cache_dir=tmp_path,
        downloader=_downloader,
    )

    assert asset_path.is_file()
    assert asset_path.read_bytes() == b"fake-asset-binary"
    assert calls == [catalog["chair_modern"].download_url]


def test_write_blender_scene_manifest_serializes_scene_and_camera_bundle(tmp_path: Path) -> None:
    scenario = SCENARIO_SPECS["studio_open_v1"]
    asset_manifest = build_scenario_asset_manifest(
        scenario,
        seed=7,
        cache_dir=tmp_path,
        quality="low",
    )
    rig = CameraRigSpec(
        image_width=320,
        image_height=180,
        fps=6.0,
        horizontal_fov_deg=72.0,
        near_plane_m=0.2,
        far_plane_m=8.0,
        enable_distortion=False,
        depth_noise_std=0.01,
        motion_blur=0.1,
        yaw_rate_limit_deg=45.0,
        linear_velocity_limit_mps=0.5,
    )

    manifest_path = write_blender_scene_manifest(
        scenario=scenario,
        asset_manifest=asset_manifest,
        rig=rig,
        output_dir=tmp_path,
    )

    payload = manifest_path.read_text(encoding="utf-8")
    assert '"scene_id": "studio_open_v1"' in payload
    assert '"semantic_target_class": "backpack"' in payload
    assert '"camera_rig"' in payload
