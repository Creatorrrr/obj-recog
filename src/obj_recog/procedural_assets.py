from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProceduralAssetPart:
    primitive_type: str
    kind: str
    location_xyz: tuple[float, float, float]
    dimensions_xyz: tuple[float, float, float]
    rotation_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class ProceduralAssetBlueprint:
    asset_id: str
    semantic_class: str
    parts: tuple[ProceduralAssetPart, ...]


def procedural_asset_blueprint(asset_id: str, *, semantic_class: str) -> ProceduralAssetBlueprint:
    key = str(asset_id)
    semantic = str(semantic_class)
    if key == "desk_basic" or semantic == "desk":
        parts = (
            _cube("top", (0.0, 0.22, 0.0), (1.45, 0.10, 0.82)),
            _cube("leg", (-0.58, -0.16, -0.28), (0.09, 0.76, 0.09)),
            _cube("leg", (0.58, -0.16, -0.28), (0.09, 0.76, 0.09)),
            _cube("leg", (-0.58, -0.16, 0.28), (0.09, 0.76, 0.09)),
            _cube("leg", (0.58, -0.16, 0.28), (0.09, 0.76, 0.09)),
            _cube("panel", (0.0, 0.02, -0.28), (1.10, 0.36, 0.05)),
        )
    elif key == "chair_modern" or semantic == "chair":
        parts = (
            _cube("seat", (0.0, -0.08, 0.02), (0.62, 0.12, 0.62)),
            _cube("backrest", (0.0, 0.28, -0.22), (0.60, 0.54, 0.10)),
            _cube("leg", (-0.22, -0.36, -0.22), (0.08, 0.62, 0.08)),
            _cube("leg", (0.22, -0.36, -0.22), (0.08, 0.62, 0.08)),
            _cube("leg", (-0.22, -0.36, 0.22), (0.08, 0.62, 0.08)),
            _cube("leg", (0.22, -0.36, 0.22), (0.08, 0.62, 0.08)),
            _cube("brace", (0.0, -0.06, -0.22), (0.46, 0.08, 0.08)),
        )
    elif key == "plant_potted" or semantic == "potted plant":
        parts = (
            _cube("pot", (0.0, -0.34, 0.0), (0.38, 0.30, 0.38)),
            _cube("stem", (0.0, -0.04, 0.0), (0.05, 0.42, 0.05)),
            _sphere("leaf", (-0.16, 0.18, 0.0), (0.32, 0.24, 0.22)),
            _sphere("leaf", (0.16, 0.22, -0.04), (0.30, 0.24, 0.24)),
            _sphere("leaf", (0.0, 0.34, 0.12), (0.34, 0.26, 0.24)),
        )
    elif key == "backpack_canvas" or semantic == "backpack":
        parts = (
            _cube("body", (0.0, 0.0, 0.0), (0.54, 0.74, 0.30)),
            _cube("pocket", (0.0, -0.12, 0.17), (0.30, 0.22, 0.11)),
            _cube("strap", (-0.16, 0.04, -0.14), (0.06, 0.62, 0.05)),
            _cube("strap", (0.16, 0.04, -0.14), (0.06, 0.62, 0.05)),
            _cube("handle", (0.0, 0.42, -0.06), (0.14, 0.06, 0.05)),
        )
    elif key == "suitcase_hardshell" or semantic == "suitcase":
        parts = (
            _cube("body", (0.0, -0.02, 0.0), (0.56, 0.78, 0.32)),
            _cube("handle", (0.0, 0.44, -0.10), (0.16, 0.08, 0.04)),
            _cube("wheel", (-0.18, -0.44, -0.12), (0.08, 0.08, 0.08)),
            _cube("wheel", (0.18, -0.44, -0.12), (0.08, 0.08, 0.08)),
        )
    elif key == "laptop_open" or semantic == "laptop":
        parts = (
            _cube("base", (0.0, -0.12, 0.0), (0.62, 0.04, 0.42)),
            _cube("screen", (0.0, 0.10, -0.14), (0.58, 0.32, 0.04), rotation_xyz_deg=(-68.0, 0.0, 0.0)),
            _cube("hinge", (0.0, -0.08, -0.18), (0.50, 0.03, 0.04)),
        )
    else:
        parts = (_cube("body", (0.0, 0.0, 0.0), (0.75, 0.75, 0.75)),)
    return ProceduralAssetBlueprint(asset_id=key, semantic_class=semantic, parts=parts)


def _cube(
    kind: str,
    location_xyz: tuple[float, float, float],
    dimensions_xyz: tuple[float, float, float],
    *,
    rotation_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ProceduralAssetPart:
    return ProceduralAssetPart(
        primitive_type="CUBE",
        kind=str(kind),
        location_xyz=tuple(float(value) for value in location_xyz),
        dimensions_xyz=tuple(float(value) for value in dimensions_xyz),
        rotation_xyz_deg=tuple(float(value) for value in rotation_xyz_deg),
    )


def _sphere(
    kind: str,
    location_xyz: tuple[float, float, float],
    dimensions_xyz: tuple[float, float, float],
) -> ProceduralAssetPart:
    return ProceduralAssetPart(
        primitive_type="UV_SPHERE",
        kind=str(kind),
        location_xyz=tuple(float(value) for value in location_xyz),
        dimensions_xyz=tuple(float(value) for value in dimensions_xyz),
    )
