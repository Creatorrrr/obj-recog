from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from obj_recog.sim_materials import material_alpha, material_color_rgb
from obj_recog.sim_protocol import LivingRoomLightSpec, LivingRoomObjectSpec, LivingRoomSceneSpec, RobotPose
from obj_recog.sim_scene import build_scene_mesh_components


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    output_root: Path


@dataclass(frozen=True, slots=True)
class _RenderLight:
    light_type: str
    location_xyz: np.ndarray
    color_rgb: np.ndarray
    energy: float


@dataclass(frozen=True, slots=True)
class _RenderPrimitive:
    primitive_id: str
    semantic_label: str
    semantic_id: int
    instance_id: int
    material_key: str
    center_xyz: np.ndarray
    half_size_xyz: np.ndarray
    rotation_world_to_local: np.ndarray
    rotation_local_to_world: np.ndarray


class SoftwareRendererRuntime:
    def __init__(self, *, output_root: Path) -> None:
        self._output_root = Path(output_root)
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._scene_id = "uninitialized"
        self._image_width = 64
        self._image_height = 48
        self._horizontal_fov_deg = 72.0
        self._near_plane_m = 0.2
        self._far_plane_m = 8.0
        self._intrinsics = {"fx": 32.0, "fy": 32.0, "cx": 32.0, "cy": 24.0}
        self._scene_center = np.zeros(3, dtype=np.float32)
        self._primitives: tuple[_RenderPrimitive, ...] = ()
        self._lights: tuple[_RenderLight, ...] = ()

    def process_request(self, payload: dict[str, object]) -> dict[str, object]:
        kind = str(payload.get("kind", ""))
        if kind == "build_scene":
            return self._build_scene(payload)
        if kind == "render_frame":
            return self._render_frame(payload)
        raise RuntimeError(f"unsupported worker request kind: {kind}")

    def _build_scene(self, payload: dict[str, object]) -> dict[str, object]:
        scene_spec = dict(payload["scene_spec"])
        living_room_scene = _coerce_scene_spec(scene_spec)
        self._scene_id = str(living_room_scene.scene_id)
        self._image_width = int(payload["image_width"])
        self._image_height = int(payload["image_height"])
        self._horizontal_fov_deg = float(payload["horizontal_fov_deg"])
        self._near_plane_m = float(payload["near_plane_m"])
        self._far_plane_m = float(payload["far_plane_m"])
        focal = 0.5 * float(self._image_width) / max(math.tan(math.radians(self._horizontal_fov_deg) * 0.5), 1e-6)
        self._intrinsics = {
            "fx": float(focal),
            "fy": float(focal),
            "cx": float(self._image_width) * 0.5,
            "cy": float(self._image_height) * 0.5,
        }
        room_width, room_height, _room_depth = (float(value) for value in living_room_scene.room_size_xyz)
        self._scene_center = np.array((0.0, room_height * 0.5, 0.0), dtype=np.float32)
        self._primitives = _build_primitives_from_components(living_room_scene)
        self._lights = tuple(
            _RenderLight(
                light_type=str(item["light_type"]),
                location_xyz=np.asarray(item["location_xyz"], dtype=np.float32).reshape(3),
                color_rgb=np.asarray(item["color_rgb"], dtype=np.float32).reshape(3),
                energy=float(item["energy"]),
            )
            for item in list(scene_spec.get("lights") or [])
        )
        return {"status": "ready", "scene_id": self._scene_id}

    def _render_frame(self, payload: dict[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        frame_index = int(payload["frame_index"])
        robot_pose = dict(payload["robot_pose"])
        origin_world = np.array(
            [
                float(robot_pose["x"]),
                float(robot_pose["y"]),
                float(robot_pose["z"]),
            ],
            dtype=np.float32,
        )
        total_yaw_deg = float(robot_pose["yaw_deg"]) + float(robot_pose.get("camera_pan_deg", 0.0))
        camera_pose_world = _camera_pose_world(origin_world, total_yaw_deg)
        dirs_camera, dirs_world = self._camera_rays(total_yaw_deg)
        rgb_bgr, depth_map, semantic_mask, instance_mask = self._trace_scene(
            origin_world=origin_world,
            dirs_camera=dirs_camera,
            dirs_world=dirs_world,
        )

        rgb_path = self._output_root / f"frame-{frame_index:04d}.npy"
        depth_path = self._output_root / f"frame-{frame_index:04d}-depth.npy"
        semantic_path = self._output_root / f"frame-{frame_index:04d}-semantic.npy"
        instance_path = self._output_root / f"frame-{frame_index:04d}-instance.npy"
        np.save(rgb_path, rgb_bgr)
        np.save(depth_path, depth_map)
        np.save(semantic_path, semantic_mask)
        np.save(instance_path, instance_mask)
        return {
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "semantic_mask_path": str(semantic_path),
            "instance_mask_path": str(instance_path),
            "camera_pose_world": camera_pose_world.tolist(),
            "intrinsics": dict(self._intrinsics),
            "render_time_ms": (time.perf_counter() - started) * 1000.0,
            "worker_state": "ready",
        }

    def _camera_rays(self, total_yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(self._image_width, dtype=np.float32)
        ys = np.arange(self._image_height, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        dir_x = (grid_x - float(self._intrinsics["cx"])) / float(self._intrinsics["fx"])
        dir_y = -(grid_y - float(self._intrinsics["cy"])) / float(self._intrinsics["fy"])
        dir_z = np.ones_like(dir_x, dtype=np.float32)
        dirs_camera = np.stack((dir_x, dir_y, dir_z), axis=-1).reshape(-1, 3)
        dirs_camera /= np.linalg.norm(dirs_camera, axis=1, keepdims=True).clip(min=1e-6)

        camera_rotation = _rotation_y_matrix(-total_yaw_deg)
        dirs_world = dirs_camera @ camera_rotation.T
        dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True).clip(min=1e-6)
        return dirs_camera, dirs_world

    def _trace_scene(
        self,
        *,
        origin_world: np.ndarray,
        dirs_camera: np.ndarray,
        dirs_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pixel_count = int(dirs_world.shape[0])
        best_t = np.full(pixel_count, np.inf, dtype=np.float32)
        hit_indices = np.full(pixel_count, -1, dtype=np.int32)
        hit_normals = np.zeros((pixel_count, 3), dtype=np.float32)
        best_opaque_t = np.full(pixel_count, np.inf, dtype=np.float32)
        opaque_hit_indices = np.full(pixel_count, -1, dtype=np.int32)
        opaque_hit_normals = np.zeros((pixel_count, 3), dtype=np.float32)

        for index, primitive in enumerate(self._primitives):
            hit_t, hit_mask, hit_normals_world = _intersect_box(
                origin_world=origin_world,
                dirs_world=dirs_world,
                primitive=primitive,
                near_plane_m=self._near_plane_m,
                far_plane_m=self._far_plane_m,
            )
            better = hit_mask & (hit_t < best_t)
            if np.any(better):
                best_t[better] = hit_t[better]
                hit_indices[better] = int(index)
                hit_normals[better] = hit_normals_world[better]

            if material_alpha(primitive.material_key) >= 0.999:
                better_opaque = hit_mask & (hit_t < best_opaque_t)
                if np.any(better_opaque):
                    best_opaque_t[better_opaque] = hit_t[better_opaque]
                    opaque_hit_indices[better_opaque] = int(index)
                    opaque_hit_normals[better_opaque] = hit_normals_world[better_opaque]

        rgb = np.zeros((pixel_count, 3), dtype=np.float32)
        depth = np.full(pixel_count, float(self._far_plane_m), dtype=np.float32)
        semantic = np.zeros(pixel_count, dtype=np.uint8)
        instance = np.zeros(pixel_count, dtype=np.uint8)

        miss_mask = hit_indices < 0
        if np.any(miss_mask):
            rgb[miss_mask] = _background_rgb(dirs_camera[miss_mask])

        hit_mask = ~miss_mask
        if np.any(hit_mask):
            hit_points_world = np.zeros((pixel_count, 3), dtype=np.float32)
            hit_points_camera = np.zeros((pixel_count, 3), dtype=np.float32)
            hit_points_world[hit_mask] = (
                origin_world.reshape(1, 3) + (dirs_world[hit_mask] * best_t[hit_mask, None])
            )
            hit_points_camera[hit_mask] = dirs_camera[hit_mask] * best_t[hit_mask, None]

            opaque_mask = opaque_hit_indices >= 0
            if np.any(opaque_mask):
                opaque_points_camera = dirs_camera[opaque_mask] * best_opaque_t[opaque_mask, None]
                depth[opaque_mask] = np.clip(
                    opaque_points_camera[:, 2],
                    self._near_plane_m,
                    self._far_plane_m,
                )

            for index, primitive in enumerate(self._primitives):
                primitive_mask = hit_indices == int(index)
                if not np.any(primitive_mask):
                    continue

                primitive_points = hit_points_world[primitive_mask]
                primitive_normals = hit_normals[primitive_mask]
                base_rgb = _textured_material_rgb(
                    primitive.material_key,
                    primitive.semantic_label,
                    primitive_points,
                    primitive_normals,
                )
                front_rgb = _apply_lighting(
                    base_rgb=base_rgb,
                    normals_world=primitive_normals,
                    hit_points_world=primitive_points,
                    lights=self._lights,
                    scene_center=self._scene_center,
                )

                alpha = material_alpha(primitive.material_key)
                if alpha >= 0.999:
                    rgb[primitive_mask] = front_rgb
                    semantic[primitive_mask] = np.uint8(primitive.semantic_id)
                    instance[primitive_mask] = np.uint8(primitive.instance_id)
                    continue

                behind_mask = primitive_mask & (opaque_hit_indices >= 0) & (best_opaque_t > (best_t + 1e-4))
                blended_rgb = front_rgb.copy()
                if np.any(behind_mask):
                    behind_indices = opaque_hit_indices[behind_mask]
                    behind_points_world = (
                        origin_world.reshape(1, 3) + (dirs_world[behind_mask] * best_opaque_t[behind_mask, None])
                    )
                    behind_normals = opaque_hit_normals[behind_mask]
                    behind_rgb = np.zeros((behind_points_world.shape[0], 3), dtype=np.float32)
                    behind_semantic = np.zeros(behind_points_world.shape[0], dtype=np.uint8)
                    behind_instance = np.zeros(behind_points_world.shape[0], dtype=np.uint8)
                    for behind_index, behind_primitive in enumerate(self._primitives):
                        sample_mask = behind_indices == int(behind_index)
                        if not np.any(sample_mask):
                            continue
                        sample_points = behind_points_world[sample_mask]
                        sample_normals = behind_normals[sample_mask]
                        sample_base_rgb = _textured_material_rgb(
                            behind_primitive.material_key,
                            behind_primitive.semantic_label,
                            sample_points,
                            sample_normals,
                        )
                        behind_rgb[sample_mask] = _apply_lighting(
                            base_rgb=sample_base_rgb,
                            normals_world=sample_normals,
                            hit_points_world=sample_points,
                            lights=self._lights,
                            scene_center=self._scene_center,
                        )
                        behind_semantic[sample_mask] = np.uint8(behind_primitive.semantic_id)
                        behind_instance[sample_mask] = np.uint8(behind_primitive.instance_id)

                    primitive_positions = np.flatnonzero(primitive_mask)
                    behind_positions = np.flatnonzero(behind_mask)
                    lookup = {pixel_index: position for position, pixel_index in enumerate(behind_positions.tolist())}
                    for local_index, pixel_index in enumerate(primitive_positions.tolist()):
                        behind_position = lookup.get(pixel_index)
                        if behind_position is None:
                            background_rgb = _background_rgb(dirs_camera[pixel_index : pixel_index + 1])[0]
                            blended_rgb[local_index] = (front_rgb[local_index] * alpha) + (
                                background_rgb * (1.0 - alpha)
                            )
                            continue
                        blended_rgb[local_index] = (front_rgb[local_index] * alpha) + (
                            behind_rgb[behind_position] * (1.0 - alpha)
                        )
                        semantic[pixel_index] = behind_semantic[behind_position]
                        instance[pixel_index] = behind_instance[behind_position]
                else:
                    background_rgb = _background_rgb(dirs_camera[primitive_mask])
                    blended_rgb = (front_rgb * alpha) + (background_rgb * (1.0 - alpha))

                rgb[primitive_mask] = blended_rgb

        rgb = np.clip(rgb, 0.0, 1.0)
        vignette = _vignette(
            width=self._image_width,
            height=self._image_height,
        ).reshape(-1, 1)
        rgb = np.clip(rgb * vignette, 0.0, 1.0)
        rgb_bgr = (rgb.reshape(self._image_height, self._image_width, 3)[:, :, ::-1] * 255.0).astype(np.uint8)
        depth_map = depth.reshape(self._image_height, self._image_width).astype(np.float32)
        semantic_mask = semantic.reshape(self._image_height, self._image_width)
        instance_mask = instance.reshape(self._image_height, self._image_width)
        return rgb_bgr, depth_map, semantic_mask, instance_mask


def create_worker_runtime(*, output_root: str | Path, force_python_fallback: bool = False):
    _ = force_python_fallback
    return SoftwareRendererRuntime(output_root=Path(output_root))


def parse_config(argv: list[str] | None = None) -> WorkerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(_argv_after_double_dash(argv or sys.argv))
    return WorkerConfig(output_root=Path(args.output_root))


def run_worker_loop(
    *,
    argv: list[str] | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
    force_python_fallback: bool = False,
) -> int:
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    config = parse_config(argv)
    runtime = create_worker_runtime(output_root=config.output_root, force_python_fallback=force_python_fallback)
    print(json.dumps({"worker_state": "bootstrapping"}), file=stderr, flush=True)
    for raw_line in stdin:
        stripped = str(raw_line).strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        response = runtime.process_request(payload)
        print(json.dumps(response), file=stdout, flush=True)
    return 0


def _argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return argv[1:]
    index = argv.index("--")
    return argv[index + 1 :]


def _coerce_scene_spec(scene_spec_obj: dict[str, object]) -> LivingRoomSceneSpec:
    room_size_xyz = tuple(float(value) for value in scene_spec_obj["room_size_xyz"])
    wall_thickness_m = float(scene_spec_obj["wall_thickness_m"])
    hidden_goal_pose_xyz = tuple(float(value) for value in scene_spec_obj["hidden_goal_pose_xyz"])
    start_pose_data = dict(scene_spec_obj["start_pose"])
    objects = tuple(
        LivingRoomObjectSpec(
            object_id=str(item["object_id"]),
            semantic_label=str(item["semantic_label"]),
            center_xyz=tuple(float(value) for value in item["center_xyz"]),
            size_xyz=tuple(float(value) for value in item["size_xyz"]),
            yaw_deg=float(item["yaw_deg"]),
            material_key=str(item["material_key"]),
            collider=bool(item.get("collider", True)),
        )
        for item in list(scene_spec_obj.get("objects") or [])
    )
    lights = tuple(
        LivingRoomLightSpec(
            light_id=str(item["light_id"]),
            light_type=str(item["light_type"]),
            location_xyz=tuple(float(value) for value in item["location_xyz"]),
            rotation_deg_xyz=tuple(float(value) for value in item["rotation_deg_xyz"]),
            color_rgb=tuple(float(value) for value in item["color_rgb"]),
            energy=float(item["energy"]),
        )
        for item in list(scene_spec_obj.get("lights") or [])
    )
    start_pose = RobotPose(
        x=float(start_pose_data["x"]),
        y=float(start_pose_data["y"]),
        z=float(start_pose_data["z"]),
        yaw_deg=float(start_pose_data["yaw_deg"]),
        camera_pan_deg=float(start_pose_data.get("camera_pan_deg", 0.0)),
    )
    return LivingRoomSceneSpec(
        scene_id=str(scene_spec_obj["scene_id"]),
        room_size_xyz=room_size_xyz,
        wall_thickness_m=wall_thickness_m,
        window_wall=str(scene_spec_obj.get("window_wall", "front")),
        start_pose=start_pose,
        hidden_goal_pose_xyz=hidden_goal_pose_xyz,
        objects=objects,
        lights=lights,
    )


def _extract_box_component(component) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract an OBB primitive from a shared mesh component.

    The software renderer is intentionally box-first: only components that are exact
    box/slab/panel meshes (8 corners + 12 triangles + consistent corner lattice)
    are accepted. Any component that cannot be represented as such fails loudly at
    build time to avoid silent geometry approximation.
    """

    component_id = str(getattr(component, "component_id", "unknown"))
    vertices = np.asarray(component.vertices_xyz, dtype=np.float32).reshape((-1, 3))
    triangles = np.asarray(component.triangles, dtype=np.int64)
    if vertices.shape != (8, 3):
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: expected 8 vertices")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: triangles must be Nx3")
    if triangles.size == 0:
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: no triangles provided")
    if triangles.min(initial=0) < 0 or triangles.max(initial=0) >= vertices.shape[0]:
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: triangle index out of bounds")
    tri_count = len(triangles)
    if tri_count != 12:
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: expected 12 triangles")

    tri_area = np.linalg.norm(
        np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]], vertices[triangles[:, 2]] - vertices[triangles[:, 0]]),
        axis=1,
    )
    if np.any(tri_area <= 1e-8):
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: degenerate triangle")

    triangle_edges = np.concatenate(
        [
            np.sort(triangles[:, [0, 1]], axis=1),
            np.sort(triangles[:, [1, 2]], axis=1),
            np.sort(triangles[:, [2, 0]], axis=1),
        ],
        axis=0,
    )
    unique_edge_count = len({tuple(edge) for edge in triangle_edges.tolist()})
    if unique_edge_count < 18:
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: invalid edge topology")

    triangle_topology = {tuple(sorted(map(int, tri))) for tri in triangles.tolist()}
    if len(triangle_topology) != 12:
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: expected box-like triangle topology")
    if np.isnan(vertices).any() or np.isinf(vertices).any():
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: invalid vertex values")

    center = vertices.mean(axis=0)
    centered = vertices - center
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    covariance = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(covariance)
    axes = np.asarray(eigvecs.T, dtype=np.float32)
    up_scores = np.abs(axes @ world_up)
    up_axis_index = int(np.argmax(up_scores))
    up_axis = np.asarray(axes[up_axis_index], dtype=np.float32)
    if up_scores[up_axis_index] < 0.90:
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: expected near-vertical local Y axis")

    local_axes = [axis for idx, axis in enumerate(axes) if idx != up_axis_index]
    local_spans = [float((centered @ axis).ptp()) for axis in local_axes]
    order = np.argsort(local_spans)
    local_x_axis = np.asarray(local_axes[int(order[1])], dtype=np.float32)
    local_z_axis = np.asarray(local_axes[int(order[0])], dtype=np.float32)

    rotation_world_to_local = np.stack([local_x_axis, up_axis, local_z_axis]).astype(np.float32)
    if np.linalg.det(rotation_world_to_local) < 0.0:
        local_z_axis = -local_z_axis
        rotation_world_to_local = np.stack([local_x_axis, up_axis, local_z_axis]).astype(np.float32)

    local_vertices = centered @ rotation_world_to_local.T
    local_min = local_vertices.min(axis=0)
    local_max = local_vertices.max(axis=0)
    span = local_max - local_min
    if np.any(span <= 1e-6):
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: degenerate extents")

    tol = max(1e-5, float(span.max()) * 1e-4)
    levels: list[list[float]] = []
    for axis_index in range(3):
        axis_values = np.sort(local_vertices[:, axis_index])
        axis_levels = [float(axis_values[0])]
        for value in axis_values[1:]:
            if abs(float(value) - axis_levels[-1]) > tol:
                axis_levels.append(float(value))
        if len(axis_levels) != 2:
            raise RuntimeError(
                f"component '{component_id}' does not satisfy box/slab/panel contract: non-box corner lattice on axis {axis_index}"
            )
        levels.append(axis_levels)

    expected = np.array([(x, y, z) for x in levels[0] for y in levels[1] for z in levels[2]], dtype=np.float32)
    residual = np.linalg.norm(local_vertices[:, None, :] - expected[None, :, :], axis=2).min(axis=1)
    if float(np.max(residual)) > (tol * 2.0):
        raise RuntimeError(f"component '{component_id}' does not satisfy box/slab/panel contract: invalid corners")

    return center, span * 0.5, rotation_world_to_local.astype(np.float32)


def _build_primitives_from_components(scene_spec: LivingRoomSceneSpec) -> tuple[_RenderPrimitive, ...]:
    semantic_ids: dict[str, int] = {}
    primitives: list[_RenderPrimitive] = []
    next_instance_id = 1

    def semantic_id(label: str) -> int:
        if label not in semantic_ids:
            semantic_ids[label] = len(semantic_ids) + 1
        return semantic_ids[label]

    def add_primitive(
        primitive_id: str,
        semantic_label: str,
        center_xyz,
        half_size_xyz,
        material_key: str,
        *,
        rotation_world_to_local: np.ndarray,
    ) -> None:
        nonlocal next_instance_id
        primitives.append(
            _RenderPrimitive(
                primitive_id=str(primitive_id),
                semantic_label=str(semantic_label),
                semantic_id=semantic_id(str(semantic_label)),
                instance_id=next_instance_id,
                material_key=str(material_key),
                center_xyz=np.asarray(center_xyz, dtype=np.float32).reshape(3),
                half_size_xyz=np.asarray(half_size_xyz, dtype=np.float32).reshape(3),
                rotation_world_to_local=np.asarray(rotation_world_to_local, dtype=np.float32),
                rotation_local_to_world=np.asarray(rotation_world_to_local, dtype=np.float32).T,
            )
        )
        next_instance_id += 1

    for component in build_scene_mesh_components(scene_spec):
        center_xyz, half_size_xyz, rotation_world_to_local = _extract_box_component(component)
        add_primitive(
            primitive_id=str(component.component_id),
            semantic_label=str(component.semantic_label),
            center_xyz=center_xyz,
            half_size_xyz=half_size_xyz,
            material_key=str(component.material_key),
            rotation_world_to_local=rotation_world_to_local,
        )

    return tuple(primitives)


def _rotation_y_matrix(yaw_deg: float) -> np.ndarray:
    yaw_rad = math.radians(float(yaw_deg))
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    return np.array(
        [
            [cos_yaw, 0.0, sin_yaw],
            [0.0, 1.0, 0.0],
            [-sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float32,
    )


def _camera_pose_world(origin_world: np.ndarray, total_yaw_deg: float) -> np.ndarray:
    yaw_rad = math.radians(-float(total_yaw_deg))
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 0] = cos_yaw
    pose[0, 2] = sin_yaw
    pose[2, 0] = -sin_yaw
    pose[2, 2] = cos_yaw
    pose[:3, 3] = np.asarray(origin_world, dtype=np.float32).reshape(3)
    return pose


def _intersect_box(
    *,
    origin_world: np.ndarray,
    dirs_world: np.ndarray,
    primitive: _RenderPrimitive,
    near_plane_m: float,
    far_plane_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_origin = (origin_world.reshape(1, 3) - primitive.center_xyz.reshape(1, 3)) @ primitive.rotation_world_to_local
    local_origin = local_origin.reshape(3)
    local_dirs = dirs_world @ primitive.rotation_world_to_local
    mins = -primitive.half_size_xyz
    maxs = primitive.half_size_xyz

    safe_dirs = np.where(np.abs(local_dirs) < 1e-6, np.where(local_dirs < 0.0, -1e-6, 1e-6), local_dirs)
    t1 = (mins.reshape(1, 3) - local_origin.reshape(1, 3)) / safe_dirs
    t2 = (maxs.reshape(1, 3) - local_origin.reshape(1, 3)) / safe_dirs
    t_near = np.maximum.reduce(np.minimum(t1, t2), axis=1)
    t_far = np.minimum.reduce(np.maximum(t1, t2), axis=1)
    hit_t = np.where(t_near >= near_plane_m, t_near, t_far)
    hit_mask = (t_far >= np.maximum(t_near, near_plane_m)) & (hit_t >= near_plane_m) & (hit_t <= far_plane_m)

    local_hit = local_origin.reshape(1, 3) + (local_dirs * hit_t[:, None])
    normalized = np.abs(local_hit / primitive.half_size_xyz.reshape(1, 3).clip(min=1e-6))
    face_axis = np.argmax(normalized, axis=1)
    face_sign = np.sign(local_hit[np.arange(local_hit.shape[0]), face_axis])
    face_sign = np.where(face_sign == 0.0, 1.0, face_sign)
    normals_local = np.zeros_like(local_hit, dtype=np.float32)
    normals_local[np.arange(local_hit.shape[0]), face_axis] = face_sign.astype(np.float32)
    normals_world = normals_local @ primitive.rotation_local_to_world
    normals_world /= np.linalg.norm(normals_world, axis=1, keepdims=True).clip(min=1e-6)
    return hit_t.astype(np.float32), hit_mask, normals_world.astype(np.float32)


def _background_rgb(dirs_camera: np.ndarray) -> np.ndarray:
    horizon = np.clip(0.5 + (dirs_camera[:, 1] * 0.5), 0.0, 1.0)
    sky = np.stack(
        (
            0.25 + (0.30 * horizon),
            0.34 + (0.34 * horizon),
            0.46 + (0.42 * horizon),
        ),
        axis=1,
    )
    return np.clip(sky, 0.0, 1.0).astype(np.float32)


def _textured_material_rgb(
    material_key: str,
    semantic_label: str,
    hit_points_world: np.ndarray,
    normals_world: np.ndarray,
) -> np.ndarray:
    base = np.tile(np.asarray(material_color_rgb(material_key), dtype=np.float32).reshape(1, 3), (hit_points_world.shape[0], 1))
    x = hit_points_world[:, 0]
    y = hit_points_world[:, 1]
    z = hit_points_world[:, 2]
    up_facing = np.clip(normals_world[:, 1], 0.0, 1.0)

    if material_key == "wood_floor":
        grain = 0.08 * np.sin((x * 3.2) + (z * 16.0))
        cross = 0.04 * np.cos(z * 8.0)
        base += (grain + cross)[:, None] * np.array([0.22, 0.13, 0.05], dtype=np.float32)
    elif material_key in {"wood_table", "chair_wood"}:
        grain = 0.07 * np.sin((x + z) * 14.0)
        base += grain[:, None] * np.array([0.18, 0.09, 0.04], dtype=np.float32)
    elif material_key == "painted_wall":
        plaster = 0.02 * np.sin((x * 2.6) + (y * 7.0)) + 0.01 * np.cos(z * 5.5)
        base += plaster[:, None]
    elif material_key == "matte_ceiling":
        stipple = 0.015 * np.cos((x * 11.0) + (z * 9.0))
        base += stipple[:, None]
    elif material_key == "sofa_fabric":
        weave = 0.04 * np.sin(x * 20.0) * np.cos(z * 14.0)
        base += weave[:, None] * np.array([0.5, 0.6, 0.7], dtype=np.float32)
    elif material_key == "tv_panel":
        scanline = 0.03 * np.sin((y * 18.0) + (x * 2.0))
        base += scanline[:, None] * np.array([0.35, 0.45, 0.70], dtype=np.float32)
    elif material_key == "tinted_glass":
        glaze = 0.10 * np.clip((y - 0.5) / 1.8, 0.0, 1.0)
        reflection = 0.08 * np.clip(1.0 - np.abs(normals_world[:, 2]), 0.0, 1.0)
        base = (base * 0.78) + glaze[:, None] * np.array([0.25, 0.32, 0.42], dtype=np.float32)
        base += reflection[:, None] * np.array([0.20, 0.24, 0.28], dtype=np.float32)

    if semantic_label in {"dining_table", "coffee_table"}:
        top_highlight = 0.06 * up_facing
        base += top_highlight[:, None] * np.array([0.20, 0.16, 0.10], dtype=np.float32)

    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _apply_lighting(
    *,
    base_rgb: np.ndarray,
    normals_world: np.ndarray,
    hit_points_world: np.ndarray,
    lights: tuple[_RenderLight, ...],
    scene_center: np.ndarray,
) -> np.ndarray:
    lit = base_rgb * np.array([0.22, 0.24, 0.28], dtype=np.float32).reshape(1, 3)

    for light in lights:
        if light.light_type.upper() == "SUN":
            light_dirs = np.tile(
                _normalized(light.location_xyz.reshape(3) - scene_center.reshape(3)).reshape(1, 3),
                (hit_points_world.shape[0], 1),
            )
            attenuation = np.full((hit_points_world.shape[0], 1), 0.95 + (0.03 * light.energy), dtype=np.float32)
        else:
            vectors = light.location_xyz.reshape(1, 3) - hit_points_world
            distances = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-3)
            light_dirs = vectors / distances
            attenuation = np.clip((light.energy / 1100.0) / (1.0 + (distances ** 2 * 0.18)), 0.0, 1.0)

        diffuse = np.clip(np.sum(normals_world * light_dirs, axis=1, keepdims=True), 0.0, 1.0)
        lit += base_rgb * diffuse * attenuation * light.color_rgb.reshape(1, 3)

    bounce = 0.08 * np.clip(normals_world[:, 1:2], 0.0, 1.0)
    lit += bounce * np.array([0.95, 0.88, 0.72], dtype=np.float32).reshape(1, 3)
    return np.clip(lit, 0.0, 1.0).astype(np.float32)


def _normalized(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-6:
        return np.array((0.0, 1.0, 0.0), dtype=np.float32)
    return array / norm


def _vignette(*, width: int, height: int) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    radius = np.sqrt((grid_x ** 2) + (grid_y ** 2))
    return np.clip(1.08 - (radius * 0.22), 0.82, 1.05).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(run_worker_loop())
