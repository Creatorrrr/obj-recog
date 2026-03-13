from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, TextIO

import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - Pillow is available in tests, optional in Blender runtime
    Image = None
    ImageDraw = None

try:  # pragma: no cover - exercised manually inside Blender
    import bpy  # type: ignore
    import mathutils  # type: ignore
except Exception:  # pragma: no cover - default test path uses the Python fallback
    bpy = None
    mathutils = None


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    scene_manifest_path: Path
    render_root: Path
    asset_cache_dir: Path
    quality: str


@dataclass(frozen=True, slots=True)
class ScenePlacement:
    asset_id: str
    semantic_class: str
    asset_family: str
    target_role: bool
    source_kind: str
    source_key: str
    center_world: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    yaw_deg: float
    preview_sprite_path: str
    blender_library_path: str
    blender_object_name: str
    recommended_scale: float
    lod_hint: str
    material_variant: str
    render_representation: str


@dataclass(frozen=True, slots=True)
class SceneManifest:
    scene_id: str
    scenario_family: str
    difficulty_level: int
    semantic_target_class: str
    asset_manifest_id: str
    environment: dict[str, float]
    camera_rig: dict[str, float]
    placements: tuple[ScenePlacement, ...]


class WorkerRuntime:
    def process_request(self, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent Blender realtime render worker")
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--render-root", required=True)
    parser.add_argument("--asset-cache-dir", required=True)
    parser.add_argument("--quality", choices=("low", "high"), default="low")
    return parser


def parse_config(argv: list[str] | None = None) -> WorkerConfig:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_args:
        raw_args = raw_args[raw_args.index("--") + 1 :]
    args = build_arg_parser().parse_args(raw_args)
    return WorkerConfig(
        scene_manifest_path=Path(args.scene_manifest).expanduser().resolve(),
        render_root=Path(args.render_root).expanduser().resolve(),
        asset_cache_dir=Path(args.asset_cache_dir).expanduser().resolve(),
        quality=str(args.quality),
    )


def emit_startup_banner(config: WorkerConfig, *, stderr: TextIO | None = None) -> None:
    stream = sys.stderr if stderr is None else stderr
    payload = {
        "worker_state": "bootstrapping",
        "scene_manifest_path": str(config.scene_manifest_path),
        "render_root": str(config.render_root),
        "asset_cache_dir": str(config.asset_cache_dir),
        "quality": config.quality,
    }
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


def load_scene_manifest(path: str | Path) -> SceneManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    placements = tuple(
        ScenePlacement(
            asset_id=str(item["asset_id"]),
            semantic_class=str(item["semantic_class"]),
            asset_family=str(item["asset_family"]),
            target_role=bool(item["target_role"]),
            source_kind=str(item["source_kind"]),
            source_key=str(item["source_key"]),
            center_world=_triple(item["center_world"]),
            size_xyz=_triple(item["size_xyz"]),
            yaw_deg=float(item["yaw_deg"]),
            preview_sprite_path=str(item["preview_sprite_path"]),
            blender_library_path=str(item["blender_library_path"]),
            blender_object_name=str(item["blender_object_name"]),
            recommended_scale=float(item["recommended_scale"]),
            lod_hint=str(item["lod_hint"]),
            material_variant=str(item["material_variant"]),
            render_representation=str(item.get("render_representation", "mesh")),
        )
        for item in payload["placements"]
    )
    return SceneManifest(
        scene_id=str(payload["scene_id"]),
        scenario_family=str(payload["scenario_family"]),
        difficulty_level=int(payload["difficulty_level"]),
        semantic_target_class=str(payload["semantic_target_class"]),
        asset_manifest_id=str(payload["asset_manifest_id"]),
        environment={str(key): float(value) for key, value in dict(payload["environment"]).items()},
        camera_rig={str(key): float(value) for key, value in dict(payload["camera_rig"]).items()},
        placements=placements,
    )


def create_worker_runtime(
    config: WorkerConfig,
    scene_manifest: SceneManifest,
    *,
    force_python_fallback: bool = False,
) -> WorkerRuntime:
    if not force_python_fallback and bpy is not None and mathutils is not None:
        return BlenderRealtimeWorkerRuntime(config=config, scene_manifest=scene_manifest)
    return PythonFallbackWorkerRuntime(config=config, scene_manifest=scene_manifest)


def run_worker_loop(
    *,
    config: WorkerConfig,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    force_python_fallback: bool = False,
) -> int:
    config.render_root.mkdir(parents=True, exist_ok=True)
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr
    emit_startup_banner(config, stderr=error_stream)
    scene_manifest = load_scene_manifest(config.scene_manifest_path)
    runtime = create_worker_runtime(
        config,
        scene_manifest,
        force_python_fallback=force_python_fallback,
    )
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("worker request must be a JSON object")
            response = runtime.process_request(payload)
        except Exception as exc:
            error_stream.write(
                json.dumps(
                    {
                        "worker_state": "error",
                        "error_message": str(exc),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            error_stream.flush()
            continue
        output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
        output_stream.flush()
    return 0


class PythonFallbackWorkerRuntime(WorkerRuntime):
    def __init__(self, *, config: WorkerConfig, scene_manifest: SceneManifest) -> None:
        self._config = config
        self._scene_manifest = scene_manifest

    def process_request(self, payload: dict[str, object]) -> dict[str, object]:
        return _process_request_common(
            config=self._config,
            scene_manifest=self._scene_manifest,
            payload=payload,
            rgb_renderer=self._render_rgb_bundle,
        )

    def _render_rgb_bundle(
        self,
        *,
        output_path: Path,
        width: int,
        height: int,
        camera_pose_world: np.ndarray,
        intrinsics: dict[str, float],
        active_placements: list[ScenePlacement],
        visible_items: list[dict[str, object]],
        target_class: str,
    ) -> None:
        del camera_pose_world, intrinsics, active_placements
        if Image is None or ImageDraw is None:
            raise RuntimeError("Pillow is required for the Python fallback worker runtime")
        image = Image.new("RGB", (width, height), (38, 41, 46))
        draw = ImageDraw.Draw(image, "RGBA")
        horizon = max(1, int(height * 0.45))
        for row in range(horizon):
            alpha = row / max(horizon - 1, 1)
            color = (
                int(56 + (34 * alpha)),
                int(66 + (28 * alpha)),
                int(78 + (22 * alpha)),
            )
            draw.line((0, row, width, row), fill=color, width=1)
        for row in range(horizon, height):
            alpha = (row - horizon) / max(height - horizon - 1, 1)
            color = (
                int(42 + (26 * alpha)),
                int(44 + (18 * alpha)),
                int(39 + (12 * alpha)),
            )
            draw.line((0, row, width, row), fill=color, width=1)
        grid = (78, 88, 102, 120)
        vanishing_x = width // 2
        for offset in range(-5, 6):
            start_x = int(((offset + 6) / 12.0) * width)
            draw.line((start_x, height - 1, vanishing_x, horizon), fill=grid, width=1)
        for row in range(1, 7):
            y = int(horizon + (1.0 - (0.78**row)) * (height - horizon))
            draw.line((0, y, width, y), fill=grid, width=1)
        for item in sorted(visible_items, key=lambda value: float(value["depth_m"]), reverse=True):
            x1, y1, x2, y2 = item["bbox"]
            color = tuple(int(v) for v in item["color_bgr"])
            rgb = (color[2], color[1], color[0])
            border = (min(rgb[0] + 24, 255), min(rgb[1] + 24, 255), min(rgb[2] + 24, 255), 255)
            fill = (rgb[0], rgb[1], rgb[2], 230)
            shadow = (14, 16, 18, 96)
            draw.ellipse((x1, y2 - 6, x2, y2 + 8), fill=shadow)
            draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=fill, outline=border, width=2)
            if str(item["label"]) == target_class:
                draw.rounded_rectangle((x1 - 2, y1 - 2, x2 + 2, y2 + 2), radius=10, outline=(240, 215, 96, 255), width=2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


class BlenderRealtimeWorkerRuntime(WorkerRuntime):  # pragma: no cover - exercised manually with Blender installed
    def __init__(self, *, config: WorkerConfig, scene_manifest: SceneManifest) -> None:
        assert bpy is not None
        assert mathutils is not None
        self._config = config
        self._scene_manifest = scene_manifest
        self._scene = bpy.context.scene
        self._objects: dict[str, object] = {}
        self._camera = None
        self._light = None
        self._configure_scene()
        self._ensure_environment()
        self._ensure_camera()
        self._ensure_light()
        self._ensure_scene_objects()

    def process_request(self, payload: dict[str, object]) -> dict[str, object]:
        return _process_request_common(
            config=self._config,
            scene_manifest=self._scene_manifest,
            payload=payload,
            rgb_renderer=self._render_rgb_bundle,
        )

    def _configure_scene(self) -> None:
        assert bpy is not None
        scene = self._scene
        scene.render.engine = (
            "BLENDER_EEVEE_NEXT"
            if "BLENDER_EEVEE_NEXT" in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
            else "BLENDER_EEVEE"
        )
        scene.render.resolution_x = int(self._scene_manifest.camera_rig["image_width"])
        scene.render.resolution_y = int(self._scene_manifest.camera_rig["image_height"])
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        if scene.world is None:
            scene.world = bpy.data.worlds.new(name="RealtimeWorkerWorld")
        scene.world.use_nodes = True
        background = scene.world.node_tree.nodes.get("Background")
        if background is not None:
            background.inputs[0].default_value = (0.045, 0.05, 0.06, 1.0)
            background.inputs[1].default_value = 0.85

    def _ensure_environment(self) -> None:
        assert bpy is not None
        self._clear_nonpersistent_objects()
        environment = self._scene_manifest.environment
        floor = self._new_primitive("worker-floor", "PLANE")
        floor.scale = (environment["room_width_m"] * 0.5, environment["room_depth_m"] * 0.5, 1.0)
        floor.location = (0.0, 0.0, environment["room_depth_m"] * 0.5)
        self._assign_material(floor, color_rgb=(0.28, 0.27, 0.24), roughness=0.95, metallic=0.0)
        wall_depth = 0.06
        wall_height = environment["room_height_m"]
        room_width = environment["room_width_m"]
        room_depth = environment["room_depth_m"]
        walls = (
            ("worker-wall-back", (0.0, wall_height * 0.5, room_depth), (room_width, wall_height, wall_depth)),
            ("worker-wall-left", (-room_width * 0.5, wall_height * 0.5, room_depth * 0.5), (wall_depth, wall_height, room_depth)),
            ("worker-wall-right", (room_width * 0.5, wall_height * 0.5, room_depth * 0.5), (wall_depth, wall_height, room_depth)),
        )
        for name, location, size in walls:
            wall = self._new_primitive(name, "CUBE")
            wall.location = location
            wall.dimensions = size
            self._assign_material(wall, color_rgb=(0.72, 0.73, 0.76), roughness=0.9, metallic=0.0)

    def _ensure_camera(self) -> None:
        assert bpy is not None
        camera_data = bpy.data.cameras.new(name="RealtimeWorkerCameraData")
        camera_object = bpy.data.objects.new("RealtimeWorkerCamera", camera_data)
        self._scene.collection.objects.link(camera_object)
        self._camera = camera_object
        self._scene.camera = camera_object

    def _ensure_light(self) -> None:
        assert bpy is not None
        light_data = bpy.data.lights.new(name="RealtimeWorkerSunData", type="SUN")
        light_object = bpy.data.objects.new("RealtimeWorkerSun", light_data)
        light_object.location = (2.0, 5.0, -1.0)
        light_object.rotation_euler = (math.radians(52.0), 0.0, math.radians(28.0))
        light_data.energy = 3.0
        self._scene.collection.objects.link(light_object)
        self._light = light_object

    def _ensure_scene_objects(self) -> None:
        for placement in self._scene_manifest.placements:
            obj = self._create_object_for_placement(placement)
            self._objects[placement.source_key] = obj
            self._apply_placement_transform(obj, placement)

    def _render_rgb_bundle(
        self,
        *,
        output_path: Path,
        width: int,
        height: int,
        camera_pose_world: np.ndarray,
        intrinsics: dict[str, float],
        active_placements: list[ScenePlacement],
        visible_items: list[dict[str, object]],
        target_class: str,
    ) -> None:
        del visible_items, target_class
        assert bpy is not None
        self._apply_request_state(camera_pose_world=camera_pose_world, intrinsics=intrinsics, active_placements=active_placements)
        self._scene.render.resolution_x = int(width)
        self._scene.render.resolution_y = int(height)
        self._scene.render.filepath = str(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.render.render(write_still=True)

    def _new_primitive(self, name: str, primitive_type: str):
        assert bpy is not None
        if primitive_type == "PLANE":
            bpy.ops.mesh.primitive_plane_add(size=1.0)
        elif primitive_type == "CYLINDER":
            bpy.ops.mesh.primitive_cylinder_add(vertices=18, radius=0.5, depth=1.0)
        elif primitive_type == "UV_SPHERE":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, segments=24, ring_count=12)
        else:
            bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        obj.name = name
        return obj

    def _create_object_for_placement(self, placement: ScenePlacement):
        obj = self._try_link_asset_object(placement)
        if obj is None:
            primitive = _primitive_type_for_placement(placement)
            obj = self._new_primitive(f"worker-{placement.source_key}", primitive)
        self._assign_material(
            obj,
            color_rgb=_color_rgb_for_placement(placement),
            roughness=_roughness_for_asset_family(placement.asset_family),
            metallic=0.0 if placement.asset_family != "electronics" else 0.2,
        )
        return obj

    def _try_link_asset_object(self, placement: ScenePlacement):
        assert bpy is not None
        library_path = Path(placement.blender_library_path)
        if not library_path.is_file():
            return None
        try:
            with bpy.data.libraries.load(str(library_path), link=False) as (data_from, data_to):
                if placement.blender_object_name not in data_from.objects:
                    return None
                data_to.objects = [placement.blender_object_name]
            obj = next((item for item in data_to.objects if item is not None), None)
            if obj is None:
                return None
            bpy.context.scene.collection.objects.link(obj)
            return obj
        except Exception:
            return None

    def _apply_placement_transform(self, obj, placement: ScenePlacement) -> None:
        obj.location = placement.center_world
        obj.rotation_euler = (0.0, math.radians(float(placement.yaw_deg)), 0.0)
        obj.dimensions = tuple(float(value) for value in placement.size_xyz)

    def _update_dynamic_objects(self, dynamic_actor_transforms: dict[str, np.ndarray]) -> None:
        for placement in self._scene_manifest.placements:
            if placement.source_kind != "dynamic":
                continue
            transform = dynamic_actor_transforms.get(placement.source_key)
            if transform is None:
                continue
            obj = self._objects.get(placement.source_key)
            if obj is None:
                continue
            obj.matrix_world = mathutils.Matrix(np.asarray(transform, dtype=np.float32).tolist())
            obj.dimensions = tuple(float(value) for value in placement.size_xyz)

    def _clear_nonpersistent_objects(self) -> None:
        assert bpy is not None
        for obj in list(bpy.data.objects):
            if obj.type in {"MESH", "CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)

    def _assign_material(self, obj, *, color_rgb: tuple[float, float, float], roughness: float, metallic: float) -> None:
        assert bpy is not None
        material_name = f"mat-{obj.name}"
        material = bpy.data.materials.get(material_name)
        if material is None:
            material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        principled = material.node_tree.nodes.get("Principled BSDF")
        if principled is not None:
            principled.inputs["Base Color"].default_value = (*color_rgb, 1.0)
            principled.inputs["Roughness"].default_value = float(roughness)
            principled.inputs["Metallic"].default_value = float(metallic)
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

    def _apply_request_state(
        self,
        *,
        camera_pose_world: np.ndarray,
        intrinsics: dict[str, float],
        active_placements: list[ScenePlacement],
    ) -> None:
        assert bpy is not None
        assert mathutils is not None
        if self._camera is None:
            raise RuntimeError("Blender worker camera is unavailable")
        blender_camera_world = _camera_pose_to_blender_matrix(camera_pose_world)
        self._camera.matrix_world = mathutils.Matrix(blender_camera_world.tolist())
        camera_data = self._camera.data
        camera_data.sensor_fit = "HORIZONTAL"
        image_width = max(int(self._scene_manifest.camera_rig["image_width"]), 1)
        angle_x = 2.0 * math.atan(image_width / max(2.0 * float(intrinsics["fx"]), 1e-6))
        camera_data.angle = float(angle_x)
        placements_by_key = {placement.source_key: placement for placement in active_placements}
        for source_key, obj in self._objects.items():
            placement = placements_by_key.get(source_key)
            if placement is None:
                continue
            self._apply_placement_transform(obj, placement)


def _process_request_common(
    *,
    config: WorkerConfig,
    scene_manifest: SceneManifest,
    payload: dict[str, object],
    rgb_renderer,
) -> dict[str, object]:
    frame_index = int(payload["frame_index"])
    scenario_id = str(payload["scenario_id"])
    if scenario_id != scene_manifest.scene_id:
        raise ValueError(f"request scenario_id {scenario_id!r} does not match loaded scene {scene_manifest.scene_id!r}")
    timestamp_sec = float(payload["timestamp_sec"])
    camera_pose_world = _coerce_matrix(payload["camera_pose_world"])
    intrinsics = {
        "fx": float(dict(payload["intrinsics"])["fx"]),
        "fy": float(dict(payload["intrinsics"])["fy"]),
        "cx": float(dict(payload["intrinsics"])["cx"]),
        "cy": float(dict(payload["intrinsics"])["cy"]),
    }
    dynamic_actor_transforms = {
        str(key): _coerce_matrix(value)
        for key, value in dict(payload.get("dynamic_actor_transforms") or {}).items()
    }
    width = int(scene_manifest.camera_rig["image_width"])
    height = int(scene_manifest.camera_rig["image_height"])
    far_plane_m = float(scene_manifest.camera_rig["far_plane_m"])
    active_placements = _active_placements(
        scene_manifest.placements,
        dynamic_actor_transforms=dynamic_actor_transforms,
    )
    start_time = time.perf_counter()
    visible_items = _visible_items_for_request(
        active_placements,
        camera_pose_world=camera_pose_world,
        intrinsics=intrinsics,
        image_width=width,
        image_height=height,
    )
    frame_stem = f"{scene_manifest.scene_id}-f{frame_index:06d}"
    rgb_path = config.render_root / f"{frame_stem}.png"
    depth_path = config.render_root / f"{frame_stem}-depth.npy"
    semantic_mask_path = config.render_root / f"{frame_stem}-semantic.npy"
    instance_mask_path = config.render_root / f"{frame_stem}-instance.npy"
    rgb_renderer(
        output_path=rgb_path,
        width=width,
        height=height,
        camera_pose_world=camera_pose_world,
        intrinsics=intrinsics,
        active_placements=active_placements,
        visible_items=visible_items,
        target_class=scene_manifest.semantic_target_class,
    )
    depth_map, semantic_mask, instance_mask, detections = _build_aux_passes(
        visible_items=visible_items,
        image_height=height,
        image_width=width,
        far_plane_m=far_plane_m,
    )
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(depth_path, depth_map.astype(np.float32))
    np.save(semantic_mask_path, semantic_mask.astype(np.uint8))
    np.save(instance_mask_path, instance_mask.astype(np.uint8))
    render_time_ms = max(0.0, (time.perf_counter() - start_time) * 1000.0)
    return {
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "semantic_mask_path": str(semantic_mask_path),
        "instance_mask_path": str(instance_mask_path),
        "pose_world_gt": camera_pose_world.astype(np.float32).tolist(),
        "intrinsics_gt": {
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
        },
        "render_time_ms": render_time_ms,
        "worker_state": "ready",
        "detections": detections,
    }


def _active_placements(
    placements: tuple[ScenePlacement, ...],
    *,
    dynamic_actor_transforms: dict[str, np.ndarray],
) -> list[ScenePlacement]:
    active: list[ScenePlacement] = []
    for placement in placements:
        center_world = placement.center_world
        yaw_deg = placement.yaw_deg
        transform = dynamic_actor_transforms.get(placement.source_key)
        if transform is not None:
            center_world = tuple(float(value) for value in np.asarray(transform, dtype=np.float32)[:3, 3])
            yaw_deg = _yaw_deg_from_matrix(transform)
        active.append(
            ScenePlacement(
                asset_id=placement.asset_id,
                semantic_class=placement.semantic_class,
                asset_family=placement.asset_family,
                target_role=placement.target_role,
                source_kind=placement.source_kind,
                source_key=placement.source_key,
                center_world=center_world,
                size_xyz=placement.size_xyz,
                yaw_deg=yaw_deg,
                preview_sprite_path=placement.preview_sprite_path,
                blender_library_path=placement.blender_library_path,
                blender_object_name=placement.blender_object_name,
                recommended_scale=placement.recommended_scale,
                lod_hint=placement.lod_hint,
                material_variant=placement.material_variant,
                render_representation=placement.render_representation,
            )
        )
    return active


def _visible_items_for_request(
    placements: list[ScenePlacement],
    *,
    camera_pose_world: np.ndarray,
    intrinsics: dict[str, float],
    image_width: int,
    image_height: int,
) -> list[dict[str, object]]:
    visible: list[dict[str, object]] = []
    semantic_ids = {name: index + 1 for index, name in enumerate(sorted({item.semantic_class for item in placements}))}
    for instance_id, placement in enumerate(placements, start=1):
        projection = _project_bbox(
            center_world=placement.center_world,
            size_xyz=placement.size_xyz,
            yaw_deg=placement.yaw_deg,
            camera_pose_world=camera_pose_world,
            intrinsics=intrinsics,
            image_width=image_width,
            image_height=image_height,
        )
        if projection is None:
            continue
        bbox, depth_m = projection
        visible.append(
            {
                "bbox": bbox,
                "depth_m": depth_m,
                "label": placement.semantic_class,
                "asset_id": placement.asset_id,
                "target_role": placement.target_role,
                "instance_id": instance_id,
                "semantic_id": semantic_ids[placement.semantic_class],
                "color_bgr": _color_bgr_for_placement(placement),
            }
        )
    visible.sort(key=lambda item: float(item["depth_m"]), reverse=True)
    return visible


def _build_aux_passes(
    *,
    visible_items: list[dict[str, object]],
    image_height: int,
    image_width: int,
    far_plane_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    depth_map = np.full((image_height, image_width), float(far_plane_m), dtype=np.float32)
    semantic_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    instance_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    detections: list[dict[str, object]] = []
    for item in visible_items:
        x1, y1, x2, y2 = (int(value) for value in item["bbox"])
        if x2 <= x1 or y2 <= y1:
            continue
        depth_map[y1:y2, x1:x2] = np.minimum(depth_map[y1:y2, x1:x2], float(item["depth_m"]))
        semantic_mask[y1:y2, x1:x2] = int(item["semantic_id"])
        instance_mask[y1:y2, x1:x2] = int(item["instance_id"])
        detections.append(
            {
                "xyxy": [x1, y1, x2, y2],
                "class_id": int(item["semantic_id"]),
                "label": str(item["label"]),
                "confidence": 0.99 if bool(item["target_role"]) else 0.95,
                "color": list(item["color_bgr"]),
            }
        )
    return depth_map, semantic_mask, instance_mask, detections


def _project_bbox(
    *,
    center_world: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    yaw_deg: float,
    camera_pose_world: np.ndarray,
    intrinsics: dict[str, float],
    image_width: int,
    image_height: int,
) -> tuple[tuple[int, int, int, int], float] | None:
    world_from_camera = np.asarray(camera_pose_world, dtype=np.float32).reshape(4, 4)
    camera_from_world = np.linalg.inv(world_from_camera)
    half = np.asarray(size_xyz, dtype=np.float32) * 0.5
    yaw_rad = math.radians(float(yaw_deg))
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    rotation = np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )
    corners_local = np.asarray(
        [
            [sx, sy, sz]
            for sx in (-half[0], half[0])
            for sy in (-half[1], half[1])
            for sz in (-half[2], half[2])
        ],
        dtype=np.float32,
    )
    corners_world = (corners_local @ rotation.T) + np.asarray(center_world, dtype=np.float32)
    homogeneous = np.concatenate(
        [corners_world, np.ones((corners_world.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    corners_camera = (camera_from_world @ homogeneous.T).T[:, :3]
    valid = corners_camera[:, 2] > 1e-3
    if not bool(np.any(valid)):
        return None
    projected = corners_camera[valid]
    xs = (float(intrinsics["fx"]) * projected[:, 0] / projected[:, 2]) + float(intrinsics["cx"])
    ys = float(intrinsics["cy"]) - (float(intrinsics["fy"]) * projected[:, 1] / projected[:, 2])
    x1 = max(0, int(np.floor(np.min(xs))))
    y1 = max(0, int(np.floor(np.min(ys))))
    x2 = min(int(image_width), int(np.ceil(np.max(xs))))
    y2 = min(int(image_height), int(np.ceil(np.max(ys))))
    if x2 <= x1 or y2 <= y1:
        return None
    depth_m = float(np.median(projected[:, 2]))
    return (x1, y1, x2, y2), depth_m


def _triple(values: Any) -> tuple[float, float, float]:
    items = tuple(values)
    if len(items) != 3:
        raise ValueError(f"expected 3 values, got {items!r}")
    return tuple(float(value) for value in items)


def _coerce_matrix(values: object) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix, got shape {matrix.shape}")
    return matrix


def _yaw_deg_from_matrix(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    return math.degrees(math.atan2(float(matrix[0, 2]), float(matrix[2, 2])))


def _camera_pose_to_blender_matrix(camera_pose_world: np.ndarray) -> np.ndarray:
    pose = np.asarray(camera_pose_world, dtype=np.float32).reshape(4, 4)
    axis_flip = np.eye(4, dtype=np.float32)
    axis_flip[2, 2] = -1.0
    return pose @ axis_flip


def _primitive_type_for_placement(placement: ScenePlacement) -> str:
    if placement.semantic_class in {"person", "pillar"}:
        return "CYLINDER"
    if placement.semantic_class in {"potted plant"}:
        return "UV_SPHERE"
    return "CUBE"


def _roughness_for_asset_family(asset_family: str) -> float:
    if asset_family == "electronics":
        return 0.35
    if asset_family == "character":
        return 0.7
    return 0.82


def _color_rgb_for_placement(placement: ScenePlacement) -> tuple[float, float, float]:
    bgr = _color_bgr_for_placement(placement)
    return (
        float(bgr[2]) / 255.0,
        float(bgr[1]) / 255.0,
        float(bgr[0]) / 255.0,
    )


def _color_bgr_for_placement(placement: ScenePlacement) -> tuple[int, int, int]:
    base_by_family = {
        "furniture": (104, 140, 188),
        "prop": (86, 118, 176),
        "electronics": (82, 86, 96),
        "storage": (124, 152, 188),
        "architecture": (144, 142, 136),
        "warehouse": (104, 132, 174),
        "character": (88, 136, 214),
        "decor": (86, 148, 106),
    }
    base = base_by_family.get(placement.asset_family, (120, 140, 170))
    tweak = (abs(hash((placement.asset_id, placement.source_key))) % 32) - 16
    return tuple(int(np.clip(channel + tweak, 24, 230)) for channel in base)


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    return run_worker_loop(config=config)


if __name__ == "__main__":
    raise SystemExit(main())
