from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil

import pytest

from obj_recog.unity_vendor_check import UnityVendorCheckError, main, validate_unity_vendor_setup


def _write_vendor_fixture(tmp_path: Path) -> tuple[Path, Path]:
    unity_root = tmp_path / "unity"
    (unity_root / "ProjectSettings").mkdir(parents=True)
    (unity_root / "ProjectSettings" / "ProjectVersion.txt").write_text(
        "m_EditorVersion: 6000.3.11f1\n",
        encoding="utf-8",
    )

    demo_asset = unity_root / "Assets" / "Brick Project Studio" / "Apartment Kit" / "Scenes" / "DemoSettings.lighting"
    demo_asset.parent.mkdir(parents=True, exist_ok=True)
    demo_asset.write_text("stub", encoding="utf-8")
    demo_asset.with_name(f"{demo_asset.name}.meta").write_text(
        "\n".join(
            [
                "fileFormatVersion: 2",
                "guid: a2f6163588e6e074996614a2b15b04b7",
                "AssetOrigin:",
                "  serializedVersion: 1",
                "  productId: 124055",
                "  packageName: Apartment Kit",
                "  packageVersion: 4.2",
                "  assetPath: Assets/Brick Project Studio/Apartment Kit/Scenes/DemoSettings.lighting",
            ]
        ),
        encoding="utf-8",
    )

    guid_samples = [
        (
            "Assets/Brick Project Studio/Apartment Kit/_Prefabs/Apt Build Kit/Interiors/Flooring & Ceilings/Int_apt_01_Floor_01.prefab",
            "bee09117d2231334d90a559b1ff7f323",
        ),
        (
            "Assets/Brick Project Studio/Apartment Kit/_Prefabs/Apt Build Kit/Interiors/Flooring & Ceilings/Int_apt_02_Floor_01.prefab",
            "5aac716a11271124887a33ec1836f27b",
        ),
        (
            "Assets/Brick Project Studio/Apartment Kit/_Prefabs/Apt Build Kit/Interiors/Walls Frames & Doors/Apt_02/Int_Apt_02_Wall_01.prefab",
            "fb28b01c668360548b1caee1ee42ac7e",
        ),
        (
            "Assets/Brick Project Studio/_BPS Basic Assets/_Prefabs/Base Scene/BA_Concrete Plane.prefab",
            "0a39e6bc93e859145a2f868ff0277c31",
        ),
    ]
    for asset_path, guid in guid_samples:
        asset_file = unity_root.joinpath(*PurePosixPath(asset_path).parts)
        asset_file.parent.mkdir(parents=True, exist_ok=True)
        asset_file.write_text("stub", encoding="utf-8")
        asset_file.with_name(f"{asset_file.name}.meta").write_text(
            f"fileFormatVersion: 2\nguid: {guid}\n",
            encoding="utf-8",
        )

    manifest_path = unity_root / "vendor_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "unity_editor_version": "6000.3.11f1",
                "packages": [
                    {
                        "name": "Apartment Kit",
                        "version": "4.2",
                        "product_id": 124055,
                        "required_paths": [
                            "Assets/Brick Project Studio/Apartment Kit",
                            "Assets/Brick Project Studio/_BPS Basic Assets",
                        ],
                        "asset_origin_sample": {
                            "asset_path": "Assets/Brick Project Studio/Apartment Kit/Scenes/DemoSettings.lighting",
                            "package_name": "Apartment Kit",
                            "package_version": "4.2",
                            "product_id": 124055,
                        },
                        "guid_samples": [
                            {
                                "asset_path": "Assets/Brick Project Studio/Apartment Kit/Scenes/DemoSettings.lighting",
                                "guid": "a2f6163588e6e074996614a2b15b04b7",
                            },
                            {
                                "asset_path": "Assets/Brick Project Studio/Apartment Kit/_Prefabs/Apt Build Kit/Interiors/Flooring & Ceilings/Int_apt_01_Floor_01.prefab",
                                "guid": "bee09117d2231334d90a559b1ff7f323",
                            },
                            {
                                "asset_path": "Assets/Brick Project Studio/Apartment Kit/_Prefabs/Apt Build Kit/Interiors/Flooring & Ceilings/Int_apt_02_Floor_01.prefab",
                                "guid": "5aac716a11271124887a33ec1836f27b",
                            },
                            {
                                "asset_path": "Assets/Brick Project Studio/Apartment Kit/_Prefabs/Apt Build Kit/Interiors/Walls Frames & Doors/Apt_02/Int_Apt_02_Wall_01.prefab",
                                "guid": "fb28b01c668360548b1caee1ee42ac7e",
                            },
                            {
                                "asset_path": "Assets/Brick Project Studio/_BPS Basic Assets/_Prefabs/Base Scene/BA_Concrete Plane.prefab",
                                "guid": "0a39e6bc93e859145a2f868ff0277c31",
                            },
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return unity_root, manifest_path


def test_validate_unity_vendor_setup_accepts_expected_layout(tmp_path: Path) -> None:
    unity_root, manifest_path = _write_vendor_fixture(tmp_path)

    validate_unity_vendor_setup(unity_project_root=unity_root, manifest_path=manifest_path)


def test_validate_unity_vendor_setup_reports_missing_vendor_path(tmp_path: Path) -> None:
    unity_root, manifest_path = _write_vendor_fixture(tmp_path)
    shutil.rmtree(unity_root / "Assets" / "Brick Project Studio" / "_BPS Basic Assets")

    with pytest.raises(UnityVendorCheckError, match="missing required path Assets/Brick Project Studio/_BPS Basic Assets"):
        validate_unity_vendor_setup(unity_project_root=unity_root, manifest_path=manifest_path)


def test_validate_unity_vendor_setup_reports_guid_mismatch(tmp_path: Path) -> None:
    unity_root, manifest_path = _write_vendor_fixture(tmp_path)
    sample_meta = (
        unity_root
        / "Assets"
        / "Brick Project Studio"
        / "Apartment Kit"
        / "_Prefabs"
        / "Apt Build Kit"
        / "Interiors"
        / "Flooring & Ceilings"
        / "Int_apt_01_Floor_01.prefab.meta"
    )
    sample_meta.write_text("fileFormatVersion: 2\nguid: 11111111111111111111111111111111\n", encoding="utf-8")

    with pytest.raises(UnityVendorCheckError, match="guid mismatch"):
        validate_unity_vendor_setup(unity_project_root=unity_root, manifest_path=manifest_path)


def test_unity_vendor_check_main_returns_nonzero_on_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    unity_root, manifest_path = _write_vendor_fixture(tmp_path)
    (unity_root / "ProjectSettings" / "ProjectVersion.txt").write_text(
        "m_EditorVersion: 6000.2.0f1\n",
        encoding="utf-8",
    )

    exit_code = main(["--unity-project-root", str(unity_root), "--manifest", str(manifest_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Unity editor version mismatch" in captured.err
