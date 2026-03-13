from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


DEFAULT_INTERIOR_TEST_BLEND_FILE = "/Users/chasoik/Downloads/InteriorTest.blend"


@dataclass(frozen=True, slots=True)
class BlendSceneObject:
    object_id: str
    object_type: str
    semantic_label: str
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    yaw_deg: float
    vertices_xyz: np.ndarray
    triangles: np.ndarray
    collider: bool = True


@dataclass(frozen=True, slots=True)
class BlendSceneManifest:
    blend_file_path: str
    room_size_xyz: tuple[float, float, float]
    objects: tuple[BlendSceneObject, ...]


def normalize_blender_xyz_to_sim_xyz(blender_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in blender_xyz)
    return (x, z, y)


def tv_front_goal_from_manifest(
    manifest: BlendSceneManifest,
    *,
    tv_object_id: str = "TV",
    offset_m: float = 0.8,
    camera_height_m: float = 1.25,
) -> tuple[float, float, float]:
    tv_object = next(item for item in manifest.objects if item.object_id == tv_object_id)
    return (
        float(tv_object.center_xyz[0]),
        float(camera_height_m),
        float(tv_object.center_xyz[2]) - float(offset_m),
    )


def load_blend_scene_manifest(*, blend_file_path: str | Path, blender_exec: str) -> BlendSceneManifest:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "blender" / "extract_scene_manifest.py"
    if not script_path.is_file():
        raise RuntimeError(f"Blender manifest extraction script is missing: {script_path}")

    with tempfile.TemporaryDirectory(prefix="obj-recog-blend-manifest-") as temp_dir:
        output_json = Path(temp_dir) / "scene_manifest.json"
        command = [
            str(blender_exec),
            str(blend_file_path),
            "--background",
            "--python",
            str(script_path),
            "--",
            "--output-json",
            str(output_json),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr_excerpt = completed.stderr.strip()[:1200]
            raise RuntimeError(
                "Blender scene manifest extraction failed. "
                f"stderr: {stderr_excerpt}"
            )
        if not output_json.is_file():
            stdout_excerpt = completed.stdout.strip()[:1200]
            raise RuntimeError(
                "Blender scene manifest extraction did not produce an output JSON file. "
                f"stdout: {stdout_excerpt}"
            )
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    return manifest_from_payload(payload)


def manifest_from_payload(payload: dict[str, object]) -> BlendSceneManifest:
    objects = []
    for item in list(payload.get("objects") or []):
        center_xyz = normalize_blender_xyz_to_sim_xyz(tuple(float(value) for value in item["center_xyz"]))
        size_raw = tuple(float(value) for value in item["size_xyz"])
        size_xyz = (size_raw[0], size_raw[2], size_raw[1])
        vertices_raw = np.asarray(item.get("vertices_xyz") or [], dtype=np.float32).reshape((-1, 3))
        triangles = np.asarray(item.get("triangles") or [], dtype=np.int32).reshape((-1, 3))
        if vertices_raw.size > 0:
            vertices_xyz = np.asarray(
                [normalize_blender_xyz_to_sim_xyz(tuple(vertex.tolist())) for vertex in vertices_raw],
                dtype=np.float32,
            ).reshape((-1, 3))
        else:
            vertices_xyz = np.empty((0, 3), dtype=np.float32)
        object_id = str(item["object_id"])
        semantic_label = _semantic_label_for_object_id(object_id)
        objects.append(
            BlendSceneObject(
                object_id=object_id,
                object_type=str(item.get("object_type", "MESH")),
                semantic_label=semantic_label,
                center_xyz=center_xyz,
                size_xyz=size_xyz,
                yaw_deg=float(item.get("yaw_deg", 0.0)),
                vertices_xyz=vertices_xyz,
                triangles=triangles,
                collider=_collider_for_object(semantic_label),
            )
        )

    room_size_raw = tuple(float(value) for value in payload.get("room_size_xyz") or (5.0, 3.0, 8.0))
    room_size_xyz = (
        float(room_size_raw[0]),
        float(room_size_raw[1]),
        float(room_size_raw[2]),
    )
    return BlendSceneManifest(
        blend_file_path=str(payload.get("blend_file_path") or DEFAULT_INTERIOR_TEST_BLEND_FILE),
        room_size_xyz=room_size_xyz,
        objects=tuple(objects),
    )


def _semantic_label_for_object_id(object_id: str) -> str:
    normalized = str(object_id).strip().lower()
    if "tv" in normalized:
        return "tv"
    if "floor" in normalized:
        return "floor"
    if "plafon" in normalized or "ceiling" in normalized:
        return "ceiling"
    if "window" in normalized:
        return "glass"
    if "wall" in normalized or "sidetegel" in normalized:
        return "wall"
    if "sofa" in normalized or "bangku" in normalized:
        return "sofa"
    if "meja" in normalized:
        return "table"
    if "piano" in normalized:
        return "piano"
    if "lamp" in normalized:
        return "lamp"
    if "pot" in normalized or "tree" in normalized:
        return "plant"
    if "book" in normalized:
        return "book"
    if "carpet" in normalized:
        return "rug"
    return normalized.replace(".", "_")


def _collider_for_object(semantic_label: str) -> bool:
    return semantic_label not in {"tv", "floor", "ceiling", "rug", "book"}
