from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    key: str
    base_color_rgb: tuple[float, float, float]
    roughness: float
    metallic: float = 0.0
    alpha: float = 1.0


MATERIAL_SPECS: dict[str, MaterialSpec] = {
    "wood_floor": MaterialSpec("wood_floor", (0.55, 0.42, 0.30), roughness=0.65),
    "painted_wall": MaterialSpec("painted_wall", (0.90, 0.90, 0.88), roughness=0.92),
    "matte_ceiling": MaterialSpec("matte_ceiling", (0.96, 0.96, 0.94), roughness=0.98),
    "tinted_glass": MaterialSpec("tinted_glass", (0.70, 0.82, 0.88), roughness=0.05, alpha=0.35),
    "sofa_fabric": MaterialSpec("sofa_fabric", (0.52, 0.60, 0.66), roughness=0.88),
    "wood_table": MaterialSpec("wood_table", (0.46, 0.32, 0.20), roughness=0.58),
    "tv_panel": MaterialSpec("tv_panel", (0.08, 0.09, 0.11), roughness=0.22, metallic=0.05),
    "chair_wood": MaterialSpec("chair_wood", (0.42, 0.30, 0.22), roughness=0.72),
}


def material_color_rgb(material_key: str) -> tuple[float, float, float]:
    return MATERIAL_SPECS.get(material_key, MATERIAL_SPECS["painted_wall"]).base_color_rgb


def material_alpha(material_key: str) -> float:
    return float(MATERIAL_SPECS.get(material_key, MATERIAL_SPECS["painted_wall"]).alpha)
