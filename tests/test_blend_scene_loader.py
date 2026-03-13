from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from obj_recog.blend_scene_loader import load_blend_scene_manifest, manifest_from_payload


def test_manifest_from_payload_classifies_structural_and_target_objects() -> None:
    manifest = manifest_from_payload(
        {
            "blend_file_path": "/tmp/InteriorTest.blend",
            "room_size_xyz": [5.0, 3.0, 8.0],
            "objects": [
                {
                    "object_id": "Floor",
                    "object_type": "MESH",
                    "center_xyz": [0.0, 0.0, 0.0],
                    "size_xyz": [5.0, 8.0, 0.1],
                    "yaw_deg": 0.0,
                    "vertices_xyz": [],
                    "triangles": [],
                },
                {
                    "object_id": "Wall.001",
                    "object_type": "MESH",
                    "center_xyz": [0.0, 4.0, 1.0],
                    "size_xyz": [5.0, 0.1, 2.5],
                    "yaw_deg": 0.0,
                    "vertices_xyz": [],
                    "triangles": [],
                },
                {
                    "object_id": "Window.001",
                    "object_type": "MESH",
                    "center_xyz": [0.0, 4.0, 1.2],
                    "size_xyz": [2.0, 0.05, 1.8],
                    "yaw_deg": 0.0,
                    "vertices_xyz": [],
                    "triangles": [],
                },
                {
                    "object_id": "TV",
                    "object_type": "MESH",
                    "center_xyz": [0.0, 3.9, 1.3],
                    "size_xyz": [1.5, 0.05, 0.8],
                    "yaw_deg": 0.0,
                    "vertices_xyz": [],
                    "triangles": [],
                },
                {
                    "object_id": "Meja",
                    "object_type": "MESH",
                    "center_xyz": [0.2, 1.2, 0.5],
                    "size_xyz": [1.0, 1.0, 0.8],
                    "yaw_deg": 0.0,
                    "vertices_xyz": [],
                    "triangles": [],
                },
            ],
        }
    )

    labels = {item.object_id: item.semantic_label for item in manifest.objects}
    colliders = {item.object_id: item.collider for item in manifest.objects}

    assert labels["Floor"] == "floor"
    assert labels["Wall.001"] == "wall"
    assert labels["Window.001"] == "glass"
    assert labels["TV"] == "tv"
    assert labels["Meja"] == "table"
    assert colliders["Floor"] is False
    assert colliders["Wall.001"] is True
    assert colliders["Window.001"] is True
    assert colliders["TV"] is False
    assert colliders["Meja"] is True


def test_load_blend_scene_manifest_reads_output_from_extraction_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def _fake_run(command, *, check, capture_output, text):
        assert check is False
        assert capture_output is True
        assert text is True
        captured["command"] = list(command)
        output_path = Path(command[command.index("--output-json") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "blend_file_path": "/Users/chasoik/Downloads/InteriorTest.blend",
                    "room_size_xyz": [5.0, 3.0, 8.0],
                    "objects": [
                        {
                            "object_id": "TV",
                            "object_type": "MESH",
                            "center_xyz": [0.0, 3.926, 1.322],
                            "size_xyz": [1.5, 0.05, 0.8],
                            "yaw_deg": 0.0,
                            "vertices_xyz": [],
                            "triangles": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout='{"status":"ok"}', stderr="")

    monkeypatch.setattr("obj_recog.blend_scene_loader.subprocess.run", _fake_run)

    manifest = load_blend_scene_manifest(
        blend_file_path=tmp_path / "InteriorTest.blend",
        blender_exec="/Applications/Blender.app/Contents/MacOS/Blender",
    )

    assert manifest.blend_file_path == "/Users/chasoik/Downloads/InteriorTest.blend"
    assert len(manifest.objects) == 1
    assert manifest.objects[0].semantic_label == "tv"
    assert "--python" in captured["command"]
    assert captured["command"][captured["command"].index("--python") + 1].endswith(
        "scripts/blender/extract_scene_manifest.py"
    )
