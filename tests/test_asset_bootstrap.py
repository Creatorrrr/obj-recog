from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pytest

from obj_recog.asset_bootstrap import bootstrap_asset_catalog_entry, required_bootstrap_entries
from obj_recog.sim_assets import asset_metadata_path, load_asset_cache_metadata, load_asset_catalog


def _zip_payload(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def test_required_bootstrap_entries_are_scenario_scoped() -> None:
    entries = required_bootstrap_entries(scenarios=("studio_open_v1",))

    asset_ids = {entry.asset_id for entry in entries}
    assert "chair_modern" in asset_ids
    assert "desk_basic" in asset_ids
    assert "backpack_canvas" in asset_ids
    assert "suitcase_hardshell" not in asset_ids


def test_bootstrap_asset_catalog_entry_writes_external_provenance_metadata(tmp_path: Path) -> None:
    entry = load_asset_catalog()["chair_modern"]
    download_calls: list[str] = []
    builder_calls: list[tuple[Path, Path]] = []

    def _downloader(url: str) -> bytes:
        download_calls.append(url)
        return _zip_payload({"models/chair.glb": b"glb-payload"})

    def _builder(
        *,
        entry,
        extracted_root: Path,
        output_blend_path: Path,
        output_preview_mesh_path: Path,
        blender_exec: str,
    ) -> None:
        assert blender_exec == "/Applications/Blender.app/Contents/MacOS/Blender"
        assert extracted_root.is_dir()
        output_blend_path.parent.mkdir(parents=True, exist_ok=True)
        output_preview_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        output_blend_path.write_bytes(b"blend")
        output_preview_mesh_path.write_text("ply", encoding="utf-8")
        builder_calls.append((output_blend_path, output_preview_mesh_path))

    metadata = bootstrap_asset_catalog_entry(
        entry,
        cache_dir=tmp_path,
        blender_exec="/Applications/Blender.app/Contents/MacOS/Blender",
        force_rebuild=True,
        downloader=_downloader,
        asset_builder=_builder,
    )

    assert metadata.provenance == "external"
    assert Path(metadata.blender_library_path).is_file()
    assert Path(metadata.preview_mesh_path).is_file()
    assert download_calls == [entry.bootstrap.archive_url]
    assert len(builder_calls) == 1
    stored = load_asset_cache_metadata(asset_metadata_path(entry, tmp_path))
    assert stored.provenance == "external"


def test_bootstrap_asset_catalog_entry_rejects_non_archive_payloads(tmp_path: Path) -> None:
    entry = load_asset_catalog()["chair_modern"]

    def _downloader(_url: str) -> bytes:
        return b"<html>not-an-archive</html>"

    with pytest.raises(RuntimeError, match="not a ZIP archive"):
        bootstrap_asset_catalog_entry(
            entry,
            cache_dir=tmp_path,
            blender_exec="/Applications/Blender.app/Contents/MacOS/Blender",
            force_rebuild=True,
            downloader=_downloader,
        )
