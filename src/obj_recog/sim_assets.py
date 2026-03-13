from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from typing import Any

from PIL import Image, ImageDraw, ImageFilter


ASSET_CATALOG_VERSION = 1
_DEFAULT_BOOTSTRAP_ARCHIVE_TYPE = "zip"
_DEFAULT_BOOTSTRAP_IMPORT_FORMAT = "blend"


@dataclass(frozen=True, slots=True)
class AssetBootstrapSpec:
    archive_url: str
    archive_sha256: str
    archive_type: str
    extract_member_glob: str
    import_format: str
    source_object_name: str | None
    export_object_name: str
    up_axis: str = "Z"
    forward_axis: str = "-Y"
    uniform_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class AssetCacheMetadata:
    asset_id: str
    catalog_version: int
    provenance: str
    archive_url: str
    archive_sha256: str
    blender_library_path: str
    preview_mesh_path: str
    export_object_name: str


@dataclass(frozen=True, slots=True)
class AssetCatalogEntry:
    asset_id: str
    semantic_class: str
    asset_family: str
    license_name: str
    download_url: str
    recommended_scale: float
    footprint_xy: tuple[float, float]
    preview_sprite_name: str
    preview_style: str
    base_palette: tuple[tuple[int, int, int], ...]
    catalog_version: int = ASSET_CATALOG_VERSION

    @property
    def blender_library_relpath(self) -> str:
        return f"libraries/{self.asset_family}/{self.asset_id}.blend"

    @property
    def preview_mesh_relpath(self) -> str:
        return f"preview_meshes/{self.asset_family}/{self.asset_id}.ply"

    @property
    def metadata_relpath(self) -> str:
        return f"metadata/{self.asset_family}/{self.asset_id}.json"

    @property
    def blender_object_name(self) -> str:
        return "".join(part.capitalize() for part in self.asset_id.split("_"))

    @property
    def lod_hint(self) -> str:
        if self.asset_family in {"architecture", "warehouse"}:
            return "high"
        if self.asset_family in {"character", "electronics"}:
            return "medium"
        return "low"

    @property
    def material_variant(self) -> str:
        return f"{self.asset_family}-{self.preview_style}"

    @property
    def bootstrap(self) -> AssetBootstrapSpec:
        return _BOOTSTRAP_SPECS[self.asset_id]


@dataclass(frozen=True, slots=True)
class ScenarioAssetPlacement:
    asset_id: str
    semantic_class: str
    asset_family: str
    target_role: bool
    source_kind: str
    source_key: str
    center_world: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    yaw_deg: float
    color_bgr: tuple[int, int, int]
    preview_sprite_path: str
    preview_mesh_path: str
    asset_metadata_path: str
    asset_provenance: str
    blender_library_path: str
    blender_object_name: str
    recommended_scale: float
    lod_hint: str
    material_variant: str
    render_representation: str = "mesh"


@dataclass(frozen=True, slots=True)
class ScenarioAssetManifest:
    manifest_id: str
    scenario_id: str
    semantic_target_class: str
    placements: tuple[ScenarioAssetPlacement, ...]


def _bootstrap_spec_for_entry(entry: AssetCatalogEntry) -> AssetBootstrapSpec:
    archive_sha256 = hashlib.sha256(f"obj-recog:{entry.asset_id}".encode("utf-8")).hexdigest()
    return AssetBootstrapSpec(
        archive_url=entry.download_url,
        archive_sha256=archive_sha256,
        archive_type=_DEFAULT_BOOTSTRAP_ARCHIVE_TYPE,
        extract_member_glob="**/*",
        import_format=_DEFAULT_BOOTSTRAP_IMPORT_FORMAT,
        source_object_name=entry.blender_object_name,
        export_object_name=entry.blender_object_name,
    )


_TARGET_CLASS_BY_SCENARIO = {
    "studio_open_v1": "backpack",
    "office_clutter_v1": "backpack",
    "lab_corridor_v1": "suitcase",
    "showroom_occlusion_v1": "backpack",
    "office_crossflow_v1": "laptop",
    "warehouse_moving_target_v1": "suitcase",
}

_ASSET_BINDINGS = {
    "table": ("desk_basic", "desk"),
    "desk": ("desk_basic", "desk"),
    "chair": ("chair_modern", "chair"),
    "plant": ("plant_potted", "potted plant"),
    "cabinet": ("cabinet_metal", "cabinet"),
    "box": ("box_cardboard", "box"),
    "partition": ("partition_panel", "partition"),
    "cart": ("cart_utility", "cart"),
    "pillar": ("pillar_concrete", "pillar"),
    "display": ("display_showcase", "chair"),
    "pedestal": ("pedestal_gallery", "table"),
    "doorframe": ("doorframe_office", "door"),
    "shelf": ("shelf_warehouse", "shelf"),
    "crate": ("crate_wood", "box"),
    "occluder": ("person_walker", "person"),
    "distractor": ("person_walker", "person"),
}

_CATALOG_DATA: tuple[AssetCatalogEntry, ...] = (
    AssetCatalogEntry(
        asset_id="desk_basic",
        semantic_class="desk",
        asset_family="furniture",
        license_name="CC0",
        download_url="https://kenney.nl/assets/furniture-kit",
        recommended_scale=1.0,
        footprint_xy=(1.4, 0.8),
        preview_sprite_name="desk_basic.png",
        preview_style="desk",
        base_palette=((139, 110, 79), (171, 145, 104), (87, 64, 45)),
    ),
    AssetCatalogEntry(
        asset_id="chair_modern",
        semantic_class="chair",
        asset_family="furniture",
        license_name="CC0",
        download_url="https://kenney.nl/assets/furniture-kit",
        recommended_scale=1.0,
        footprint_xy=(0.75, 0.75),
        preview_sprite_name="chair_modern.png",
        preview_style="chair",
        base_palette=((88, 149, 190), (55, 104, 150), (29, 58, 102)),
    ),
    AssetCatalogEntry(
        asset_id="plant_potted",
        semantic_class="potted plant",
        asset_family="decor",
        license_name="CC0",
        download_url="https://polyhaven.com/textures",
        recommended_scale=1.0,
        footprint_xy=(0.6, 0.6),
        preview_sprite_name="plant_potted.png",
        preview_style="plant",
        base_palette=((60, 121, 77), (90, 165, 103), (78, 91, 48)),
    ),
    AssetCatalogEntry(
        asset_id="backpack_canvas",
        semantic_class="backpack",
        asset_family="prop",
        license_name="CC0",
        download_url="https://quaternius.com/packs/ultimatemodularkit.html",
        recommended_scale=1.0,
        footprint_xy=(0.45, 0.3),
        preview_sprite_name="backpack_canvas.png",
        preview_style="backpack",
        base_palette=((74, 90, 128), (105, 130, 178), (35, 45, 72)),
    ),
    AssetCatalogEntry(
        asset_id="suitcase_hardshell",
        semantic_class="suitcase",
        asset_family="prop",
        license_name="CC0",
        download_url="https://quaternius.com/packs/ultimatemodularkit.html",
        recommended_scale=1.0,
        footprint_xy=(0.65, 0.35),
        preview_sprite_name="suitcase_hardshell.png",
        preview_style="suitcase",
        base_palette=((64, 94, 124), (108, 139, 168), (38, 51, 72)),
    ),
    AssetCatalogEntry(
        asset_id="laptop_open",
        semantic_class="laptop",
        asset_family="electronics",
        license_name="CC0",
        download_url="https://kenney.nl/assets/furniture-kit",
        recommended_scale=1.0,
        footprint_xy=(0.35, 0.25),
        preview_sprite_name="laptop_open.png",
        preview_style="laptop",
        base_palette=((72, 72, 76), (134, 147, 164), (28, 28, 33)),
    ),
    AssetCatalogEntry(
        asset_id="monitor_flat",
        semantic_class="monitor",
        asset_family="electronics",
        license_name="CC0",
        download_url="https://kenney.nl/assets/furniture-kit",
        recommended_scale=1.0,
        footprint_xy=(0.5, 0.2),
        preview_sprite_name="monitor_flat.png",
        preview_style="monitor",
        base_palette=((58, 61, 69), (115, 134, 152), (18, 20, 24)),
    ),
    AssetCatalogEntry(
        asset_id="book_stack",
        semantic_class="book",
        asset_family="prop",
        license_name="CC0",
        download_url="https://kenney.nl/assets/furniture-kit",
        recommended_scale=1.0,
        footprint_xy=(0.35, 0.2),
        preview_sprite_name="book_stack.png",
        preview_style="books",
        base_palette=((104, 57, 66), (152, 108, 72), (77, 101, 136)),
    ),
    AssetCatalogEntry(
        asset_id="box_cardboard",
        semantic_class="box",
        asset_family="storage",
        license_name="CC0",
        download_url="https://quaternius.com/packs/ultimatemodularkit.html",
        recommended_scale=1.0,
        footprint_xy=(0.8, 0.8),
        preview_sprite_name="box_cardboard.png",
        preview_style="box",
        base_palette=((142, 111, 78), (188, 156, 112), (97, 74, 48)),
    ),
    AssetCatalogEntry(
        asset_id="cabinet_metal",
        semantic_class="cabinet",
        asset_family="storage",
        license_name="CC0",
        download_url="https://kenney.nl/assets/furniture-kit",
        recommended_scale=1.0,
        footprint_xy=(0.8, 0.6),
        preview_sprite_name="cabinet_metal.png",
        preview_style="cabinet",
        base_palette=((118, 125, 137), (164, 173, 183), (81, 88, 98)),
    ),
    AssetCatalogEntry(
        asset_id="partition_panel",
        semantic_class="partition",
        asset_family="architecture",
        license_name="CC0",
        download_url="https://quaternius.com/packs/ultimatemodularkit.html",
        recommended_scale=1.0,
        footprint_xy=(0.45, 2.0),
        preview_sprite_name="partition_panel.png",
        preview_style="panel",
        base_palette=((124, 128, 141), (173, 177, 186), (87, 91, 99)),
    ),
    AssetCatalogEntry(
        asset_id="cart_utility",
        semantic_class="cart",
        asset_family="prop",
        license_name="CC0",
        download_url="https://quaternius.com/packs/ultimatemodularkit.html",
        recommended_scale=1.0,
        footprint_xy=(0.8, 0.6),
        preview_sprite_name="cart_utility.png",
        preview_style="cart",
        base_palette=((79, 117, 137), (121, 171, 190), (54, 76, 90)),
    ),
    AssetCatalogEntry(
        asset_id="pillar_concrete",
        semantic_class="pillar",
        asset_family="architecture",
        license_name="CC0",
        download_url="https://polyhaven.com/textures",
        recommended_scale=1.0,
        footprint_xy=(0.45, 0.45),
        preview_sprite_name="pillar_concrete.png",
        preview_style="pillar",
        base_palette=((124, 126, 132), (167, 170, 178), (90, 92, 96)),
    ),
    AssetCatalogEntry(
        asset_id="display_showcase",
        semantic_class="chair",
        asset_family="showroom",
        license_name="CC0",
        download_url="https://kenney.nl/assets/furniture-kit",
        recommended_scale=1.0,
        footprint_xy=(0.8, 0.8),
        preview_sprite_name="display_showcase.png",
        preview_style="chair",
        base_palette=((124, 94, 145), (182, 144, 204), (78, 58, 101)),
    ),
    AssetCatalogEntry(
        asset_id="pedestal_gallery",
        semantic_class="table",
        asset_family="showroom",
        license_name="CC0",
        download_url="https://kenney.nl/assets/furniture-kit",
        recommended_scale=1.0,
        footprint_xy=(0.7, 0.7),
        preview_sprite_name="pedestal_gallery.png",
        preview_style="pedestal",
        base_palette=((210, 202, 185), (182, 173, 154), (150, 140, 122)),
    ),
    AssetCatalogEntry(
        asset_id="doorframe_office",
        semantic_class="door",
        asset_family="architecture",
        license_name="CC0",
        download_url="https://quaternius.com/packs/ultimatemodularkit.html",
        recommended_scale=1.0,
        footprint_xy=(0.9, 0.2),
        preview_sprite_name="doorframe_office.png",
        preview_style="doorframe",
        base_palette=((136, 119, 88), (185, 159, 114), (96, 80, 57)),
    ),
    AssetCatalogEntry(
        asset_id="shelf_warehouse",
        semantic_class="shelf",
        asset_family="warehouse",
        license_name="CC0",
        download_url="https://quaternius.com/packs/ultimatemodularkit.html",
        recommended_scale=1.0,
        footprint_xy=(0.9, 2.8),
        preview_sprite_name="shelf_warehouse.png",
        preview_style="shelf",
        base_palette=((108, 133, 159), (152, 183, 210), (78, 95, 114)),
    ),
    AssetCatalogEntry(
        asset_id="crate_wood",
        semantic_class="box",
        asset_family="warehouse",
        license_name="CC0",
        download_url="https://quaternius.com/packs/ultimatemodularkit.html",
        recommended_scale=1.0,
        footprint_xy=(0.9, 0.9),
        preview_sprite_name="crate_wood.png",
        preview_style="crate",
        base_palette=((118, 88, 63), (164, 126, 83), (80, 55, 37)),
    ),
    AssetCatalogEntry(
        asset_id="person_walker",
        semantic_class="person",
        asset_family="character",
        license_name="CC0",
        download_url="https://quaternius.com/packs/ultimatemodularkit.html",
        recommended_scale=1.0,
        footprint_xy=(0.55, 0.4),
        preview_sprite_name="person_walker.png",
        preview_style="person",
        base_palette=((82, 99, 135), (214, 176, 140), (53, 60, 73)),
    ),
)

_BOOTSTRAP_SPECS: dict[str, AssetBootstrapSpec] = {
    entry.asset_id: _bootstrap_spec_for_entry(entry)
    for entry in _CATALOG_DATA
}


def load_asset_catalog() -> dict[str, AssetCatalogEntry]:
    return {entry.asset_id: entry for entry in _CATALOG_DATA}


def preview_mesh_path(entry: AssetCatalogEntry, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / entry.preview_mesh_relpath


def asset_metadata_path(entry: AssetCatalogEntry, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / entry.metadata_relpath


def load_asset_cache_metadata(path: str | Path) -> AssetCacheMetadata | None:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return AssetCacheMetadata(
        asset_id=str(payload["asset_id"]),
        catalog_version=int(payload["catalog_version"]),
        provenance=str(payload["provenance"]),
        archive_url=str(payload["archive_url"]),
        archive_sha256=str(payload["archive_sha256"]),
        blender_library_path=str(payload["blender_library_path"]),
        preview_mesh_path=str(payload["preview_mesh_path"]),
        export_object_name=str(payload["export_object_name"]),
    )


def write_asset_cache_metadata(metadata: AssetCacheMetadata, path: str | Path) -> Path:
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "asset_id": metadata.asset_id,
                "catalog_version": int(metadata.catalog_version),
                "provenance": metadata.provenance,
                "archive_url": metadata.archive_url,
                "archive_sha256": metadata.archive_sha256,
                "blender_library_path": metadata.blender_library_path,
                "preview_mesh_path": metadata.preview_mesh_path,
                "export_object_name": metadata.export_object_name,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return metadata_path


def resolve_asset_cache_metadata(entry: AssetCatalogEntry, *, cache_dir: str | Path) -> AssetCacheMetadata | None:
    return load_asset_cache_metadata(asset_metadata_path(entry, cache_dir))


def asset_cache_metadata_is_current(
    metadata: AssetCacheMetadata | None,
    *,
    entry: AssetCatalogEntry,
) -> bool:
    if metadata is None:
        return False
    return (
        metadata.asset_id == entry.asset_id
        and int(metadata.catalog_version) == int(entry.catalog_version)
        and str(metadata.archive_sha256) == str(entry.bootstrap.archive_sha256)
        and str(metadata.export_object_name) == str(entry.bootstrap.export_object_name)
    )


def photoreal_asset_preflight_issues(
    asset_manifest: ScenarioAssetManifest,
    *,
    cache_dir: str | Path,
) -> list[str]:
    catalog = load_asset_catalog()
    issues: list[str] = []
    checked_asset_ids: set[str] = set()
    for placement in asset_manifest.placements:
        asset_id = str(placement.asset_id)
        if asset_id in checked_asset_ids:
            continue
        checked_asset_ids.add(asset_id)
        entry = catalog[asset_id]
        metadata = resolve_asset_cache_metadata(entry, cache_dir=cache_dir)
        if not asset_cache_metadata_is_current(metadata, entry=entry):
            issues.append(f"{asset_id}: metadata missing or stale")
            continue
        if str(metadata.provenance) != "external":
            issues.append(f"{asset_id}: provenance={metadata.provenance}")
            continue
        if not Path(metadata.blender_library_path).is_file():
            issues.append(f"{asset_id}: missing blend {metadata.blender_library_path}")
            continue
        if not Path(metadata.preview_mesh_path).is_file():
            issues.append(f"{asset_id}: missing preview mesh {metadata.preview_mesh_path}")
            continue
    return issues


def ensure_preview_sprite(
    entry: AssetCatalogEntry,
    *,
    cache_dir: str | Path,
    quality: str,
) -> Path:
    root = Path(cache_dir) / "preview_sprites" / str(quality)
    root.mkdir(parents=True, exist_ok=True)
    sprite_path = root / entry.preview_sprite_name
    if sprite_path.is_file():
        return sprite_path
    image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    _draw_sprite(draw, entry.preview_style, entry.base_palette)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.25))
    image.save(sprite_path)
    return sprite_path


def ensure_asset_payload(
    entry: AssetCatalogEntry,
    *,
    cache_dir: str | Path,
    downloader=None,
) -> Path:
    root = Path(cache_dir) / "downloads" / entry.asset_id
    root.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(entry.download_url)
    filename = Path(parsed.path).name or f"{entry.asset_id}.bin"
    payload_path = root / filename
    if payload_path.is_file():
        return payload_path
    fetch = downloader or _download_url_bytes
    payload = fetch(entry.download_url)
    payload_path.write_bytes(payload)
    return payload_path


def build_scenario_asset_manifest(
    scenario: Any,
    *,
    seed: int,
    cache_dir: str | Path,
    quality: str,
) -> ScenarioAssetManifest:
    catalog = load_asset_catalog()
    scenario_id = str(getattr(scenario, "scene_id"))
    semantic_target_class = _TARGET_CLASS_BY_SCENARIO.get(scenario_id, "backpack")
    manifest_key = f"{scenario_id}:{seed}:{quality}:{semantic_target_class}".encode("utf-8")
    placements: list[ScenarioAssetPlacement] = []
    for index, item in enumerate(tuple(getattr(scenario, "static_objects", ()))):
        label = str(getattr(item, "label"))
        placement = _placement_from_spec(
            item=item,
            label=label,
            source_kind="static",
            source_key=f"static-{index}",
            semantic_target_class=semantic_target_class,
            seed=seed,
            cache_dir=cache_dir,
            quality=quality,
            catalog=catalog,
        )
        placements.append(placement)
        manifest_key += (
            f"|{placement.asset_id}:{placement.semantic_class}:{placement.target_role}:{placement.source_kind}:{placement.source_key}"
        ).encode("utf-8")
    for index, actor in enumerate(tuple(getattr(scenario, "dynamic_actors", ()))):
        label = str(getattr(actor, "label"))
        placement = _placement_from_spec(
            item=actor,
            label=label,
            source_kind="dynamic",
            source_key=str(getattr(actor, "actor_id", f"dynamic-{index}")),
            semantic_target_class=semantic_target_class,
            seed=seed + 97,
            cache_dir=cache_dir,
            quality=quality,
            catalog=catalog,
        )
        placements.append(placement)
        manifest_key += (
            f"|{placement.asset_id}:{placement.semantic_class}:{placement.target_role}:{placement.source_kind}:{placement.source_key}"
        ).encode("utf-8")
    manifest_id = hashlib.sha1(manifest_key).hexdigest()[:12]
    return ScenarioAssetManifest(
        manifest_id=manifest_id,
        scenario_id=scenario_id,
        semantic_target_class=semantic_target_class,
        placements=tuple(placements),
    )


def write_blender_scene_manifest(
    *,
    scenario: Any,
    asset_manifest: ScenarioAssetManifest,
    rig: Any,
    output_dir: str | Path,
    require_external_assets: bool = False,
) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{asset_manifest.scenario_id}-{asset_manifest.manifest_id}-scene.json"
    payload = {
        "scene_id": str(getattr(scenario, "scene_id")),
        "scenario_family": str(getattr(scenario, "scenario_family")),
        "difficulty_level": int(getattr(scenario, "difficulty_level")),
        "semantic_target_class": asset_manifest.semantic_target_class,
        "asset_manifest_id": asset_manifest.manifest_id,
        "require_external_assets": bool(require_external_assets),
        "environment": {
            "room_width_m": float(getattr(getattr(scenario, "environment"), "room_width_m")),
            "room_depth_m": float(getattr(getattr(scenario, "environment"), "room_depth_m")),
            "room_height_m": float(getattr(getattr(scenario, "environment"), "room_height_m")),
        },
        "camera_rig": {
            "image_width": int(getattr(rig, "image_width")),
            "image_height": int(getattr(rig, "image_height")),
            "fps": float(getattr(rig, "fps")),
            "horizontal_fov_deg": float(getattr(rig, "horizontal_fov_deg")),
            "near_plane_m": float(getattr(rig, "near_plane_m")),
            "far_plane_m": float(getattr(rig, "far_plane_m")),
        },
        "placements": [
            {
                "asset_id": placement.asset_id,
                "semantic_class": placement.semantic_class,
                "asset_family": placement.asset_family,
                "target_role": placement.target_role,
                "source_kind": placement.source_kind,
                "source_key": placement.source_key,
                "center_world": list(placement.center_world),
                "size_xyz": list(placement.size_xyz),
                "yaw_deg": placement.yaw_deg,
                "preview_sprite_path": placement.preview_sprite_path,
                "preview_mesh_path": placement.preview_mesh_path,
                "asset_metadata_path": placement.asset_metadata_path,
                "asset_provenance": placement.asset_provenance,
                "blender_library_path": placement.blender_library_path,
                "blender_object_name": placement.blender_object_name,
                "recommended_scale": placement.recommended_scale,
                "lod_hint": placement.lod_hint,
                "material_variant": placement.material_variant,
                "render_representation": placement.render_representation,
            }
            for placement in asset_manifest.placements
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _asset_id_for_target_class(semantic_target_class: str) -> str:
    by_class = {
        "backpack": "backpack_canvas",
        "suitcase": "suitcase_hardshell",
        "laptop": "laptop_open",
    }
    return by_class.get(semantic_target_class, "backpack_canvas")


def _placement_from_spec(
    *,
    item: Any,
    label: str,
    source_kind: str,
    source_key: str,
    semantic_target_class: str,
    seed: int,
    cache_dir: str | Path,
    quality: str,
    catalog: dict[str, AssetCatalogEntry],
) -> ScenarioAssetPlacement:
    if label == "target":
        asset_id = _asset_id_for_target_class(semantic_target_class)
        semantic_class = semantic_target_class
        target_role = True
    else:
        asset_id, semantic_class = _ASSET_BINDINGS.get(label, ("box_cardboard", "box"))
        target_role = False
    entry = catalog[asset_id]
    sprite_path = ensure_preview_sprite(entry, cache_dir=cache_dir, quality=quality)
    mesh_path = preview_mesh_path(entry, cache_dir)
    metadata_path = asset_metadata_path(entry, cache_dir)
    metadata = load_asset_cache_metadata(metadata_path)
    asset_provenance = "missing" if metadata is None else str(metadata.provenance)
    center_world = getattr(item, "center_world", None)
    if center_world is None:
        center_world = getattr(item, "base_center_world")
    return ScenarioAssetPlacement(
        asset_id=asset_id,
        semantic_class=semantic_class,
        asset_family=entry.asset_family,
        target_role=target_role,
        source_kind=source_kind,
        source_key=source_key,
        center_world=tuple(float(v) for v in center_world),
        size_xyz=tuple(float(v) for v in getattr(item, "size_xyz")),
        yaw_deg=float((abs(hash((source_key, seed))) % 36000) / 100.0),
        color_bgr=tuple(int(v) for v in getattr(item, "color_bgr")),
        preview_sprite_path=str(sprite_path),
        preview_mesh_path=str(mesh_path),
        asset_metadata_path=str(metadata_path),
        asset_provenance=asset_provenance,
        blender_library_path=str(Path(cache_dir) / entry.blender_library_relpath),
        blender_object_name=entry.blender_object_name,
        recommended_scale=float(entry.recommended_scale),
        lod_hint=entry.lod_hint,
        material_variant=entry.material_variant,
    )


def _draw_sprite(draw: ImageDraw.ImageDraw, style: str, palette: tuple[tuple[int, int, int], ...]) -> None:
    p0 = tuple(palette[0]) + (255,)
    p1 = tuple(palette[min(1, len(palette) - 1)]) + (255,)
    p2 = tuple(palette[min(2, len(palette) - 1)]) + (255,)
    shadow = (16, 18, 20, 92)
    draw.ellipse((52, 204, 204, 236), fill=shadow)
    if style == "desk":
        draw.rounded_rectangle((44, 108, 212, 142), radius=12, fill=p1, outline=p2, width=3)
        draw.rectangle((58, 142, 72, 214), fill=p2)
        draw.rectangle((184, 142, 198, 214), fill=p2)
        draw.rectangle((94, 142, 108, 208), fill=p2)
        draw.rectangle((148, 142, 162, 208), fill=p2)
    elif style == "chair":
        draw.rounded_rectangle((76, 132, 180, 168), radius=12, fill=p0, outline=p2, width=3)
        draw.rounded_rectangle((88, 74, 168, 132), radius=16, fill=p1, outline=p2, width=3)
        draw.rectangle((92, 168, 106, 218), fill=p2)
        draw.rectangle((150, 168, 164, 218), fill=p2)
        draw.rectangle((82, 102, 94, 178), fill=p2)
        draw.rectangle((162, 102, 174, 178), fill=p2)
    elif style == "plant":
        draw.rounded_rectangle((90, 152, 166, 212), radius=10, fill=p2, outline=p0, width=3)
        draw.ellipse((88, 84, 146, 160), fill=p0)
        draw.ellipse((116, 58, 176, 148), fill=p1)
        draw.ellipse((62, 62, 124, 150), fill=p1)
        draw.rectangle((122, 98, 130, 178), fill=(55, 95, 52, 255))
    elif style == "backpack":
        draw.rounded_rectangle((82, 82, 174, 194), radius=24, fill=p0, outline=p2, width=4)
        draw.rounded_rectangle((96, 110, 160, 154), radius=14, fill=p1, outline=p2, width=3)
        draw.arc((60, 88, 100, 170), start=250, end=70, fill=p2, width=10)
        draw.arc((156, 88, 196, 170), start=110, end=290, fill=p2, width=10)
        draw.rectangle((120, 68, 136, 88), fill=p2)
    elif style == "suitcase":
        draw.rounded_rectangle((66, 94, 190, 192), radius=18, fill=p0, outline=p2, width=4)
        draw.rounded_rectangle((110, 60, 146, 96), radius=10, outline=p2, width=8)
        draw.line((92, 122, 164, 122), fill=p1, width=6)
        draw.line((92, 150, 164, 150), fill=p1, width=6)
        draw.rectangle((86, 192, 102, 218), fill=p2)
        draw.rectangle((154, 192, 170, 218), fill=p2)
    elif style == "laptop":
        draw.rounded_rectangle((72, 74, 184, 148), radius=10, fill=p2, outline=p1, width=4)
        draw.rectangle((84, 86, 172, 136), fill=(66, 134, 178, 255))
        draw.polygon(((56, 148), (200, 148), (176, 192), (80, 192)), fill=p0, outline=p2)
    elif style == "monitor":
        draw.rounded_rectangle((62, 62, 194, 142), radius=10, fill=p2, outline=p1, width=4)
        draw.rectangle((74, 74, 182, 130), fill=(72, 132, 170, 255))
        draw.rectangle((118, 142, 138, 176), fill=p1)
        draw.rounded_rectangle((88, 176, 168, 194), radius=8, fill=p0)
    elif style == "books":
        draw.rounded_rectangle((70, 162, 190, 192), radius=6, fill=p0)
        draw.rounded_rectangle((84, 132, 180, 160), radius=6, fill=p1)
        draw.rounded_rectangle((98, 104, 170, 130), radius=6, fill=p2)
    elif style == "box":
        draw.polygon(((78, 116), (148, 92), (200, 126), (130, 152)), fill=p1, outline=p2)
        draw.polygon(((78, 116), (130, 152), (130, 218), (78, 184)), fill=p0, outline=p2)
        draw.polygon(((130, 152), (200, 126), (200, 194), (130, 218)), fill=p2, outline=p2)
        draw.line((110, 104, 168, 140), fill=(240, 225, 190, 255), width=4)
    elif style == "cabinet":
        draw.rounded_rectangle((82, 58, 174, 214), radius=12, fill=p0, outline=p2, width=4)
        draw.line((128, 70, 128, 202), fill=p2, width=4)
        draw.ellipse((116, 118, 124, 126), fill=p2)
        draw.ellipse((132, 118, 140, 126), fill=p2)
    elif style == "panel":
        draw.rounded_rectangle((94, 46, 162, 224), radius=8, fill=p0, outline=p2, width=4)
        draw.line((106, 70, 150, 70), fill=p1, width=5)
        draw.line((106, 108, 150, 108), fill=p1, width=5)
    elif style == "cart":
        draw.rounded_rectangle((74, 94, 182, 150), radius=10, fill=p0, outline=p2, width=4)
        draw.line((88, 94, 62, 58), fill=p2, width=8)
        draw.ellipse((78, 174, 108, 204), fill=p2)
        draw.ellipse((150, 174, 180, 204), fill=p2)
        draw.rectangle((92, 150, 100, 176), fill=p2)
        draw.rectangle((156, 150, 164, 176), fill=p2)
    elif style == "pillar":
        draw.rounded_rectangle((102, 36, 154, 222), radius=12, fill=p0, outline=p2, width=4)
        draw.line((114, 60, 114, 198), fill=p1, width=6)
        draw.line((142, 60, 142, 198), fill=p1, width=6)
    elif style == "pedestal":
        draw.rounded_rectangle((88, 78, 168, 206), radius=10, fill=p0, outline=p2, width=4)
        draw.rounded_rectangle((74, 56, 182, 86), radius=10, fill=p1, outline=p2, width=3)
    elif style == "doorframe":
        draw.rectangle((86, 34, 112, 220), fill=p2)
        draw.rectangle((144, 34, 170, 220), fill=p2)
        draw.rectangle((86, 34, 170, 60), fill=p1)
    elif style == "shelf":
        draw.rounded_rectangle((74, 44, 182, 218), radius=6, outline=p2, width=5)
        for y in (86, 132, 176):
            draw.line((82, y, 174, y), fill=p1, width=6)
        draw.rectangle((96, 94, 130, 120), fill=(172, 132, 88, 255))
        draw.rectangle((138, 140, 164, 168), fill=(146, 106, 72, 255))
    elif style == "crate":
        draw.rounded_rectangle((78, 96, 186, 202), radius=10, fill=p0, outline=p2, width=4)
        for offset in (0, 24, 48):
            draw.line((90 + offset, 102, 90 + offset, 196), fill=p1, width=5)
        draw.line((88, 122, 176, 122), fill=p1, width=5)
        draw.line((88, 154, 176, 154), fill=p1, width=5)
    elif style == "person":
        draw.ellipse((102, 42, 154, 92), fill=p1)
        draw.rounded_rectangle((88, 92, 168, 168), radius=18, fill=p0)
        draw.line((110, 168, 94, 224), fill=p2, width=14)
        draw.line((144, 168, 160, 224), fill=p2, width=14)
        draw.line((88, 110, 54, 156), fill=p2, width=12)
        draw.line((168, 110, 202, 156), fill=p2, width=12)
    else:
        draw.rounded_rectangle((72, 72, 184, 184), radius=20, fill=p0, outline=p2, width=4)


def _download_url_bytes(url: str) -> bytes:
    with urlopen(url, timeout=20) as response:  # nosec: asset downloads are opt-in and catalog-controlled
        return response.read()
