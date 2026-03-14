from __future__ import annotations

from pathlib import Path


def test_unity_project_scaffold_exists() -> None:
    unity_root = Path(__file__).resolve().parents[1] / "unity"
    manifest_text = (unity_root / "Packages" / "manifest.json").read_text(encoding="utf-8")

    assert (unity_root / "Packages" / "manifest.json").is_file()
    assert (unity_root / "ProjectSettings" / "ProjectVersion.txt").is_file()
    assert (unity_root / "ProjectSettings" / "EditorBuildSettings.asset").is_file()
    assert (unity_root / "Assets" / "Scenes" / "LivingRoomMain.unity").is_file()
    assert (unity_root / "Assets" / "Scripts" / "Runtime" / "LivingRoomAppBootstrap.cs").is_file()
    assert (unity_root / "Assets" / "Scripts" / "Runtime" / "AgentTcpServer.cs").is_file()
    assert '"com.unity.modules.animation": "1.0.0"' in manifest_text


def test_unity_bootstrap_scene_references_runtime_bootstrap() -> None:
    unity_root = Path(__file__).resolve().parents[1] / "unity"
    scene_text = (unity_root / "Assets" / "Scenes" / "LivingRoomMain.unity").read_text(encoding="utf-8")
    meta_text = (
        unity_root / "Assets" / "Scripts" / "Runtime" / "LivingRoomAppBootstrap.cs.meta"
    ).read_text(encoding="utf-8")
    bootstrap_text = (
        unity_root / "Assets" / "Scripts" / "Runtime" / "LivingRoomAppBootstrap.cs"
    ).read_text(encoding="utf-8")

    assert "LivingRoomBootstrap" in scene_text
    assert "m_Name: Living Room" in scene_text
    assert "m_Name: Apartment_02" in scene_text
    assert "guid: 328aa4dbf0da4ce290d498722e6d4d67" in scene_text
    assert "guid: 328aa4dbf0da4ce290d498722e6d4d67" in meta_text
    assert "ObjRecogRuntime" in bootstrap_text
    assert "__GeneratedLivingRoom" not in bootstrap_text
