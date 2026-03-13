from __future__ import annotations

from obj_recog.procedural_assets import procedural_asset_blueprint


def test_procedural_asset_blueprint_builds_recognizable_chair_and_desk() -> None:
    chair = procedural_asset_blueprint("chair_modern", semantic_class="chair")
    desk = procedural_asset_blueprint("desk_basic", semantic_class="desk")

    assert len(chair.parts) >= 6
    assert len(desk.parts) >= 5
    assert any(part.kind == "leg" for part in chair.parts)
    assert any(part.kind == "backrest" for part in chair.parts)
    assert any(part.kind == "top" for part in desk.parts)
    assert any(part.kind == "leg" for part in desk.parts)


def test_procedural_asset_blueprint_supports_core_scene_assets() -> None:
    for asset_id, semantic_class in (
        ("plant_potted", "potted plant"),
        ("backpack_canvas", "backpack"),
        ("suitcase_hardshell", "suitcase"),
        ("laptop_open", "laptop"),
    ):
        blueprint = procedural_asset_blueprint(asset_id, semantic_class=semantic_class)
        assert blueprint.parts
        assert all(part.primitive_type for part in blueprint.parts)
        assert all(len(part.location_xyz) == 3 for part in blueprint.parts)
        assert all(len(part.dimensions_xyz) == 3 for part in blueprint.parts)
