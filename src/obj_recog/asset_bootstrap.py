from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, Iterable
import zipfile

from obj_recog.config import DEFAULT_ASSET_CACHE_DIR, DEFAULT_ASSET_QUALITY, SIM_SCENARIO_CHOICES
from obj_recog.sim_assets import (
    AssetCacheMetadata,
    AssetCatalogEntry,
    _ASSET_BINDINGS,
    _TARGET_CLASS_BY_SCENARIO,
    _asset_id_for_target_class,
    asset_cache_metadata_is_current,
    asset_metadata_path,
    ensure_asset_payload,
    load_asset_catalog,
    load_asset_cache_metadata,
    preview_mesh_path,
    write_asset_cache_metadata,
)


def required_bootstrap_entries(*, scenarios: Iterable[str]) -> tuple[AssetCatalogEntry, ...]:
    from obj_recog.simulation import SCENARIO_SPECS

    catalog = load_asset_catalog()
    asset_ids: set[str] = set()
    for scenario_id in tuple(str(item) for item in scenarios):
        scenario = SCENARIO_SPECS[scenario_id]
        target_class = _TARGET_CLASS_BY_SCENARIO.get(scenario_id, "backpack")
        for item in tuple(getattr(scenario, "static_objects", ())):
            label = str(getattr(item, "label"))
            asset_id = _asset_id_for_target_class(target_class) if label == "target" else _ASSET_BINDINGS.get(label, ("box_cardboard", "box"))[0]
            asset_ids.add(str(asset_id))
        for item in tuple(getattr(scenario, "dynamic_actors", ())):
            label = str(getattr(item, "label"))
            asset_id = _asset_id_for_target_class(target_class) if label == "target" else _ASSET_BINDINGS.get(label, ("box_cardboard", "box"))[0]
            asset_ids.add(str(asset_id))
    return tuple(sorted((catalog[asset_id] for asset_id in asset_ids), key=lambda entry: entry.asset_id))


def bootstrap_asset_catalog_entry(
    entry: AssetCatalogEntry,
    *,
    cache_dir: str | Path = DEFAULT_ASSET_CACHE_DIR,
    blender_exec: str,
    force_rebuild: bool = False,
    downloader=None,
    asset_builder: Callable[..., None] | None = None,
) -> AssetCacheMetadata:
    cache_root = Path(cache_dir)
    metadata_path = asset_metadata_path(entry, cache_root)
    output_blend_path = cache_root / entry.blender_library_relpath
    output_preview_mesh_path = preview_mesh_path(entry, cache_root)
    existing = load_asset_cache_metadata(metadata_path)
    if (
        not force_rebuild
        and asset_cache_metadata_is_current(existing, entry=entry)
        and output_blend_path.is_file()
        and output_preview_mesh_path.is_file()
    ):
        assert existing is not None
        return existing

    archive_path = ensure_asset_payload(entry, cache_dir=cache_root, downloader=downloader)
    _validate_archive_payload(entry, archive_path)
    with tempfile.TemporaryDirectory(prefix=f"obj-recog-{entry.asset_id}-") as temp_dir:
        extracted_root = _extract_archive(
            archive_path,
            archive_type=entry.bootstrap.archive_type,
            destination=Path(temp_dir),
        )
        builder = _build_normalized_asset_artifacts if asset_builder is None else asset_builder
        builder(
            entry=entry,
            extracted_root=extracted_root,
            output_blend_path=output_blend_path,
            output_preview_mesh_path=output_preview_mesh_path,
            blender_exec=str(blender_exec),
        )

    metadata = AssetCacheMetadata(
        asset_id=entry.asset_id,
        catalog_version=entry.catalog_version,
        provenance="external",
        archive_url=entry.bootstrap.archive_url,
        archive_sha256=entry.bootstrap.archive_sha256,
        blender_library_path=str(output_blend_path),
        preview_mesh_path=str(output_preview_mesh_path),
        export_object_name=entry.bootstrap.export_object_name,
    )
    write_asset_cache_metadata(metadata, metadata_path)
    return metadata


def bootstrap_assets_for_scenarios(
    *,
    scenarios: Iterable[str],
    cache_dir: str | Path = DEFAULT_ASSET_CACHE_DIR,
    asset_quality: str = DEFAULT_ASSET_QUALITY,
    blender_exec: str,
    force_rebuild: bool = False,
) -> tuple[AssetCacheMetadata, ...]:
    _ = asset_quality
    return tuple(
        bootstrap_asset_catalog_entry(
            entry,
            cache_dir=cache_dir,
            blender_exec=blender_exec,
            force_rebuild=force_rebuild,
        )
        for entry in required_bootstrap_entries(scenarios=tuple(scenarios))
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap external photoreal assets into the local cache")
    parser.add_argument("--scenario", choices=SIM_SCENARIO_CHOICES, default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--asset-cache-dir", type=str, default=DEFAULT_ASSET_CACHE_DIR)
    parser.add_argument("--asset-quality", choices=("low", "high"), default=DEFAULT_ASSET_QUALITY)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--blender-exec", type=str, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if bool(args.all_scenarios) == bool(args.scenario):
        parser.error("choose exactly one of --scenario or --all-scenarios")
    scenarios = SIM_SCENARIO_CHOICES if args.all_scenarios else (str(args.scenario),)
    bootstrap_assets_for_scenarios(
        scenarios=scenarios,
        cache_dir=args.asset_cache_dir,
        asset_quality=args.asset_quality,
        blender_exec=str(args.blender_exec),
        force_rebuild=bool(args.force_rebuild),
    )
    return 0


def _extract_archive(archive_path: str | Path, *, archive_type: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = Path(archive_path)
    if archive_type == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return destination
    if archive_type == "tar.gz":
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination)
        return destination
    copied_path = destination / archive_path.name
    shutil.copy2(archive_path, copied_path)
    return destination


def _validate_archive_payload(entry: AssetCatalogEntry, archive_path: str | Path) -> None:
    archive_path = Path(archive_path)
    archive_type = str(entry.bootstrap.archive_type)
    if archive_type == "zip" and not zipfile.is_zipfile(archive_path):
        raise RuntimeError(
            f"{entry.asset_id} bootstrap download is not a ZIP archive. "
            f"Update asset catalog archive_url to a direct archive: {entry.bootstrap.archive_url}"
        )
    if archive_type == "tar.gz" and not tarfile.is_tarfile(archive_path):
        raise RuntimeError(
            f"{entry.asset_id} bootstrap download is not a tar archive. "
            f"Update asset catalog archive_url to a direct archive: {entry.bootstrap.archive_url}"
        )


def _build_normalized_asset_artifacts(
    *,
    entry: AssetCatalogEntry,
    extracted_root: Path,
    output_blend_path: Path,
    output_preview_mesh_path: Path,
    blender_exec: str,
) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "blender" / "asset_bootstrap_worker.py"
    output_blend_path.parent.mkdir(parents=True, exist_ok=True)
    output_preview_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(blender_exec),
        "--background",
        "--python",
        str(script_path),
        "--",
        "--input-root",
        str(extracted_root),
        "--archive-type",
        entry.bootstrap.archive_type,
        "--extract-member-glob",
        entry.bootstrap.extract_member_glob,
        "--import-format",
        entry.bootstrap.import_format,
        "--source-object-name",
        "" if entry.bootstrap.source_object_name is None else str(entry.bootstrap.source_object_name),
        "--export-object-name",
        str(entry.bootstrap.export_object_name),
        "--uniform-scale",
        str(entry.bootstrap.uniform_scale),
        "--up-axis",
        str(entry.bootstrap.up_axis),
        "--forward-axis",
        str(entry.bootstrap.forward_axis),
        "--output-blend",
        str(output_blend_path),
        "--output-preview-mesh",
        str(output_preview_mesh_path),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
