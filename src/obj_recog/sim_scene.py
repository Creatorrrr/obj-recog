from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from obj_recog.blend_scene_loader import DEFAULT_INTERIOR_TEST_BLEND_FILE, BlendSceneManifest, tv_front_goal_from_manifest
from obj_recog.sim_materials import material_color_rgb
from obj_recog.sim_protocol import LivingRoomLightSpec, LivingRoomObjectSpec, LivingRoomSceneSpec, RobotPose


@dataclass(frozen=True, slots=True)
class SceneMeshComponent:
    component_id: str
    semantic_label: str
    material_key: str
    vertices_xyz: np.ndarray
    triangles: np.ndarray
    vertex_colors: np.ndarray


def build_living_room_scene_spec() -> LivingRoomSceneSpec:
    room_width = 7.2
    room_height = 2.5
    room_depth = 5.4
    wall_thickness = 0.12
    dining_table_center = (1.75, 0.40, 1.65)
    dining_table_size = (1.60, 0.75, 0.90)
    near_edge_z = dining_table_center[2] - (dining_table_size[2] * 0.5)

    objects = (
        LivingRoomObjectSpec(
            object_id="sofa",
            semantic_label="sofa",
            center_xyz=(-1.85, 0.45, -0.15),
            size_xyz=(2.20, 0.90, 0.95),
            yaw_deg=0.0,
            material_key="sofa_fabric",
        ),
        LivingRoomObjectSpec(
            object_id="coffee_table",
            semantic_label="coffee_table",
            center_xyz=(-1.45, 0.22, 0.95),
            size_xyz=(1.00, 0.45, 0.60),
            yaw_deg=0.0,
            material_key="wood_table",
        ),
        LivingRoomObjectSpec(
            object_id="tv_console",
            semantic_label="tv_console",
            center_xyz=(2.15, 0.38, -1.80),
            size_xyz=(1.80, 0.55, 0.45),
            yaw_deg=0.0,
            material_key="wood_table",
        ),
        LivingRoomObjectSpec(
            object_id="tv_panel",
            semantic_label="tv_panel",
            center_xyz=(2.15, 1.15, -1.58),
            size_xyz=(1.40, 0.80, 0.08),
            yaw_deg=0.0,
            material_key="tv_panel",
            collider=False,
        ),
        LivingRoomObjectSpec(
            object_id="dining_table",
            semantic_label="dining_table",
            center_xyz=dining_table_center,
            size_xyz=dining_table_size,
            yaw_deg=0.0,
            material_key="wood_table",
        ),
        LivingRoomObjectSpec(
            object_id="dining_chair_front",
            semantic_label="dining_chair",
            center_xyz=(1.75, 0.45, 0.72),
            size_xyz=(0.52, 0.90, 0.52),
            yaw_deg=180.0,
            material_key="chair_wood",
        ),
        LivingRoomObjectSpec(
            object_id="dining_chair_back",
            semantic_label="dining_chair",
            center_xyz=(1.75, 0.45, 2.58),
            size_xyz=(0.52, 0.90, 0.52),
            yaw_deg=0.0,
            material_key="chair_wood",
        ),
        LivingRoomObjectSpec(
            object_id="dining_chair_left",
            semantic_label="dining_chair",
            center_xyz=(0.72, 0.45, 1.65),
            size_xyz=(0.52, 0.90, 0.52),
            yaw_deg=90.0,
            material_key="chair_wood",
        ),
        LivingRoomObjectSpec(
            object_id="dining_chair_right",
            semantic_label="dining_chair",
            center_xyz=(2.78, 0.45, 1.65),
            size_xyz=(0.52, 0.90, 0.52),
            yaw_deg=-90.0,
            material_key="chair_wood",
        ),
    )
    lights = (
        LivingRoomLightSpec(
            light_id="sun_main",
            light_type="SUN",
            location_xyz=(0.0, 4.5, 4.8),
            rotation_deg_xyz=(55.0, 0.0, 180.0),
            color_rgb=(1.0, 0.96, 0.90),
            energy=3.8,
        ),
        LivingRoomLightSpec(
            light_id="living_ceiling",
            light_type="AREA",
            location_xyz=(-1.2, 2.2, 0.3),
            rotation_deg_xyz=(90.0, 0.0, 0.0),
            color_rgb=(1.0, 0.97, 0.93),
            energy=950.0,
        ),
        LivingRoomLightSpec(
            light_id="dining_ceiling",
            light_type="AREA",
            location_xyz=(1.8, 2.15, 1.65),
            rotation_deg_xyz=(90.0, 0.0, 0.0),
            color_rgb=(1.0, 0.90, 0.82),
            energy=830.0,
        ),
    )
    return LivingRoomSceneSpec(
        scene_id="living_room_navigation_v1",
        room_size_xyz=(room_width, room_height, room_depth),
        wall_thickness_m=wall_thickness,
        window_wall="front",
        start_pose=RobotPose(x=-2.4, y=1.25, z=-1.85, yaw_deg=0.0, camera_pan_deg=0.0),
        hidden_goal_pose_xyz=(dining_table_center[0], 1.25, near_edge_z - 0.75),
        objects=objects,
        lights=lights,
        goal_description="Reach the front position of the dining table using only current visible evidence.",
        semantic_target_class="dining_table",
    )


def build_interior_test_tv_scene_spec(manifest: BlendSceneManifest | None = None) -> LivingRoomSceneSpec:
    resolved_manifest = manifest
    if resolved_manifest is None:
        resolved_manifest = BlendSceneManifest(
            blend_file_path=DEFAULT_INTERIOR_TEST_BLEND_FILE,
            room_size_xyz=(5.0, 3.0, 8.0),
            objects=(),
        )
    hidden_goal_pose_xyz = (
        tv_front_goal_from_manifest(resolved_manifest, tv_object_id="TV", offset_m=0.8)
        if any(item.object_id == "TV" for item in resolved_manifest.objects)
        else (0.0, 1.25, 3.1260)
    )
    room_width, room_height, room_depth = (float(value) for value in resolved_manifest.room_size_xyz)
    return LivingRoomSceneSpec(
        scene_id="interior_test_tv_navigation_v1",
        room_size_xyz=(room_width, room_height, room_depth),
        wall_thickness_m=0.05,
        window_wall="authored",
        start_pose=RobotPose(x=0.0, y=1.25, z=-2.8, yaw_deg=0.0, camera_pan_deg=0.0),
        hidden_goal_pose_xyz=hidden_goal_pose_xyz,
        objects=tuple(_manifest_object_to_scene_object(item) for item in resolved_manifest.objects),
        lights=(),
        blend_file_path=str(Path(resolved_manifest.blend_file_path)),
        goal_description="Reach the front position of the TV using only current visible evidence.",
        semantic_target_class="tv",
        scene_metadata={"blend_manifest": resolved_manifest},
    )


def build_scene_mesh_components(scene: LivingRoomSceneSpec) -> tuple[SceneMeshComponent, ...]:
    blend_manifest = scene.scene_metadata.get("blend_manifest")
    if isinstance(blend_manifest, BlendSceneManifest) and blend_manifest.objects:
        return _build_authored_scene_mesh_components(blend_manifest)
    if scene.blend_file_path:
        return tuple(
            _make_box_component(
                item.object_id,
                item.semantic_label,
                center_xyz=item.center_xyz,
                size_xyz=item.size_xyz,
                material_key=item.material_key,
                yaw_deg=item.yaw_deg,
            )
            for item in scene.objects
        )

    room_width, room_height, room_depth = scene.room_size_xyz
    wall_thickness = scene.wall_thickness_m
    half_w = room_width * 0.5
    half_d = room_depth * 0.5

    components = [
        _make_box_component(
            "floor",
            "floor",
            center_xyz=(0.0, -0.03, 0.0),
            size_xyz=(room_width, 0.06, room_depth),
            material_key="wood_floor",
        ),
        _make_box_component(
            "wall_left",
            "wall",
            center_xyz=(-half_w + (wall_thickness * 0.5), room_height * 0.5, 0.0),
            size_xyz=(wall_thickness, room_height, room_depth),
            material_key="painted_wall",
        ),
        _make_box_component(
            "wall_right",
            "wall",
            center_xyz=(half_w - (wall_thickness * 0.5), room_height * 0.5, 0.0),
            size_xyz=(wall_thickness, room_height, room_depth),
            material_key="painted_wall",
        ),
        _make_box_component(
            "wall_back",
            "wall",
            center_xyz=(0.0, room_height * 0.5, -half_d + (wall_thickness * 0.5)),
            size_xyz=(room_width, room_height, wall_thickness),
            material_key="painted_wall",
        ),
        _make_box_component(
            "front_glass",
            "glass",
            center_xyz=(0.0, 1.30, half_d - 0.02),
            size_xyz=(room_width - 0.30, 2.15, 0.04),
            material_key="tinted_glass",
        ),
        _make_box_component(
            "front_window_left_frame",
            "window_frame",
            center_xyz=(-(room_width - 0.30) * 0.5 + 0.07, 1.30, half_d - 0.02),
            size_xyz=(0.08, 2.15, 0.08),
            material_key="painted_wall",
        ),
        _make_box_component(
            "front_window_right_frame",
            "window_frame",
            center_xyz=((room_width - 0.30) * 0.5 - 0.07, 1.30, half_d - 0.02),
            size_xyz=(0.08, 2.15, 0.08),
            material_key="painted_wall",
        ),
        _make_box_component(
            "front_window_top_frame",
            "window_frame",
            center_xyz=(0.0, 2.30, half_d - 0.02),
            size_xyz=(room_width - 0.30, 0.18, 0.08),
            material_key="painted_wall",
        ),
        _make_box_component(
            "front_window_bottom_frame",
            "window_frame",
            center_xyz=(0.0, 0.21, half_d - 0.02),
            size_xyz=(room_width - 0.30, 0.16, 0.08),
            material_key="painted_wall",
        ),
        _make_box_component(
            "ceiling",
            "ceiling",
            center_xyz=(0.0, room_height + 0.03, 0.0),
            size_xyz=(room_width, 0.06, room_depth),
            material_key="matte_ceiling",
        ),
    ]

    for item in scene.objects:
        if item.object_id == "sofa":
            components.extend(_build_sofa_components(item))
        elif item.object_id.startswith("dining_chair"):
            components.extend(_build_dining_chair_components(item))
        elif item.object_id == "dining_table":
            components.extend(_build_dining_table_components(item))
        elif item.object_id == "tv_console":
            components.extend(_build_tv_console_components(item))
        else:
            components.append(
                _make_box_component(
                    item.object_id,
                    item.semantic_label,
                    center_xyz=item.center_xyz,
                    size_xyz=item.size_xyz,
                    material_key=item.material_key,
                    yaw_deg=item.yaw_deg,
                )
            )
    return tuple(components)


def _build_authored_scene_mesh_components(manifest: BlendSceneManifest) -> tuple[SceneMeshComponent, ...]:
    components: list[SceneMeshComponent] = []
    for item in manifest.objects:
        if item.vertices_xyz.size == 0 or item.triangles.size == 0:
            components.append(
                _make_box_component(
                    item.object_id,
                    item.semantic_label,
                    center_xyz=item.center_xyz,
                    size_xyz=item.size_xyz,
                    material_key=_material_key_for_semantic_label(item.semantic_label),
                    yaw_deg=item.yaw_deg,
                )
            )
            continue
        color_rgb = np.asarray(material_color_rgb(_material_key_for_semantic_label(item.semantic_label)), dtype=np.float32)
        vertex_colors = np.tile(color_rgb.reshape(1, 3), (item.vertices_xyz.shape[0], 1))
        components.append(
            SceneMeshComponent(
                component_id=item.object_id,
                semantic_label=item.semantic_label,
                material_key=_material_key_for_semantic_label(item.semantic_label),
                vertices_xyz=np.asarray(item.vertices_xyz, dtype=np.float32).reshape((-1, 3)),
                triangles=np.asarray(item.triangles, dtype=np.int32).reshape((-1, 3)),
                vertex_colors=vertex_colors,
            )
        )
    return tuple(components)


def pose_distance_to_goal(scene: LivingRoomSceneSpec, pose: RobotPose) -> float:
    goal = np.asarray(scene.hidden_goal_pose_xyz, dtype=np.float32)
    robot = np.asarray((pose.x, pose.y, pose.z), dtype=np.float32)
    return float(np.linalg.norm(goal - robot))


def _manifest_object_to_scene_object(item) -> LivingRoomObjectSpec:
    return LivingRoomObjectSpec(
        object_id=str(item.object_id),
        semantic_label=str(item.semantic_label),
        center_xyz=tuple(float(value) for value in item.center_xyz),
        size_xyz=tuple(max(float(value), 1e-3) for value in item.size_xyz),
        yaw_deg=float(item.yaw_deg),
        material_key=_material_key_for_semantic_label(str(item.semantic_label)),
        collider=bool(item.collider),
    )


def _material_key_for_semantic_label(semantic_label: str) -> str:
    normalized = str(semantic_label).lower()
    if normalized == "floor":
        return "wood_floor"
    if normalized in {"wall", "window_frame"}:
        return "painted_wall"
    if normalized in {"ceiling", "plafon"}:
        return "matte_ceiling"
    if normalized in {"tv", "tv_panel"}:
        return "tv_panel"
    if normalized in {"table", "meja", "coffee_table", "dining_table"}:
        return "wood_table"
    if normalized in {"chair", "bangku", "dining_chair"}:
        return "chair_wood"
    return "painted_wall"


def _build_sofa_components(item: LivingRoomObjectSpec) -> tuple[SceneMeshComponent, ...]:
    width, height, depth = item.size_xyz
    x, y, z = item.center_xyz

    base_y = y - (height * 0.5) + 0.16
    base = _make_box_component(
        f"{item.object_id}_base",
        item.semantic_label,
        center_xyz=(x, base_y, z),
        size_xyz=(width * 0.97, 0.32, depth * 0.97),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    seat = _make_box_component(
        f"{item.object_id}_seat",
        item.semantic_label,
        center_xyz=(x, y - 0.16, z),
        size_xyz=(width * 0.88, 0.16, depth * 0.58),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    backrest = _make_box_component(
        f"{item.object_id}_backrest",
        item.semantic_label,
        center_xyz=(x, y + 0.02, z - (depth * 0.22)),
        size_xyz=(width * 0.93, 0.45, depth * 0.20),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    left_arm = _make_box_component(
        f"{item.object_id}_left_arm",
        item.semantic_label,
        center_xyz=(x - width * 0.47, y + 0.05, z),
        size_xyz=(0.18, 0.50, depth * 0.78),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    right_arm = _make_box_component(
        f"{item.object_id}_right_arm",
        item.semantic_label,
        center_xyz=(x + width * 0.47, y + 0.05, z),
        size_xyz=(0.18, 0.50, depth * 0.78),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    seat_cushion = _make_box_component(
        f"{item.object_id}_seat_cushion_a",
        item.semantic_label,
        center_xyz=(x - 0.35, y - 0.11, z + 0.02),
        size_xyz=(width * 0.42, 0.10, depth * 0.54),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    seat_cushion_b = _make_box_component(
        f"{item.object_id}_seat_cushion_b",
        item.semantic_label,
        center_xyz=(x + 0.35, y - 0.11, z + 0.02),
        size_xyz=(width * 0.42, 0.10, depth * 0.54),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    back_cushion_a = _make_box_component(
        f"{item.object_id}_back_cushion_a",
        item.semantic_label,
        center_xyz=(x - 0.55, y + 0.13, z - (depth * 0.16)),
        size_xyz=(width * 0.18, 0.09, depth * 0.26),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    back_cushion_b = _make_box_component(
        f"{item.object_id}_back_cushion_b",
        item.semantic_label,
        center_xyz=(x + 0.55, y + 0.13, z - (depth * 0.16)),
        size_xyz=(width * 0.18, 0.09, depth * 0.26),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    return (
        base,
        seat,
        backrest,
        left_arm,
        right_arm,
        seat_cushion,
        seat_cushion_b,
        back_cushion_a,
        back_cushion_b,
    )


def _build_dining_table_components(item: LivingRoomObjectSpec) -> tuple[SceneMeshComponent, ...]:
    width, height, depth = item.size_xyz
    x, y, z = item.center_xyz
    # Keep the table geometry nontrivial while preserving compact top and support logic.
    top = _make_box_component(
        f"{item.object_id}_top",
        item.semantic_label,
        center_xyz=(x, y + 0.43, z),
        size_xyz=(width * 0.88, 0.08, depth * 0.80),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    apron = _make_box_component(
        f"{item.object_id}_apron",
        item.semantic_label,
        center_xyz=(x, y + 0.23, z),
        size_xyz=(width * 0.88, 0.10, depth * 0.88),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    leg_size = (0.08, 0.45, 0.08)
    leg_z_offset = (depth * 0.88) * 0.45
    leg_x_offset = (width * 0.88) * 0.45
    leg_y = y - (height * 0.5) + (leg_size[1] * 0.5)
    legs = (
        _make_box_component(
            f"{item.object_id}_leg_front_left",
            item.semantic_label,
            center_xyz=(x - leg_x_offset, leg_y, z - leg_z_offset),
            size_xyz=leg_size,
            material_key=item.material_key,
            yaw_deg=item.yaw_deg,
        ),
        _make_box_component(
            f"{item.object_id}_leg_front_right",
            item.semantic_label,
            center_xyz=(x + leg_x_offset, leg_y, z - leg_z_offset),
            size_xyz=leg_size,
            material_key=item.material_key,
            yaw_deg=item.yaw_deg,
        ),
        _make_box_component(
            f"{item.object_id}_leg_back_left",
            item.semantic_label,
            center_xyz=(x - leg_x_offset, leg_y, z + leg_z_offset),
            size_xyz=leg_size,
            material_key=item.material_key,
            yaw_deg=item.yaw_deg,
        ),
        _make_box_component(
            f"{item.object_id}_leg_back_right",
            item.semantic_label,
            center_xyz=(x + leg_x_offset, leg_y, z + leg_z_offset),
            size_xyz=leg_size,
            material_key=item.material_key,
            yaw_deg=item.yaw_deg,
        ),
    )
    return (top, apron, *legs)


def _build_dining_chair_components(item: LivingRoomObjectSpec) -> tuple[SceneMeshComponent, ...]:
    width, height, depth = item.size_xyz
    x, y, z = item.center_xyz
    seat_y = y - (height * 0.5) + 0.14
    seat = _make_box_component(
        f"{item.object_id}_seat",
        item.semantic_label,
        center_xyz=(x, seat_y, z),
        size_xyz=(width * 0.86, 0.09, depth * 0.86),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    backrest = _make_box_component(
        f"{item.object_id}_backrest",
        item.semantic_label,
        center_xyz=(x, y + 0.12, z + (depth * 0.28)),
        size_xyz=(width * 0.86, 0.60, 0.05),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    leg_y = y - (height * 0.5) + (0.36 * 0.5)
    leg_size = (0.06, 0.36, 0.06)
    x_off = width * 0.38
    z_off = depth * 0.38
    legs = (
        _make_box_component(
            f"{item.object_id}_leg_front_left",
            item.semantic_label,
            center_xyz=(x - x_off, leg_y, z - z_off),
            size_xyz=leg_size,
            material_key=item.material_key,
            yaw_deg=item.yaw_deg,
        ),
        _make_box_component(
            f"{item.object_id}_leg_front_right",
            item.semantic_label,
            center_xyz=(x + x_off, leg_y, z - z_off),
            size_xyz=leg_size,
            material_key=item.material_key,
            yaw_deg=item.yaw_deg,
        ),
        _make_box_component(
            f"{item.object_id}_leg_back_left",
            item.semantic_label,
            center_xyz=(x - x_off, leg_y, z + z_off),
            size_xyz=leg_size,
            material_key=item.material_key,
            yaw_deg=item.yaw_deg,
        ),
        _make_box_component(
            f"{item.object_id}_leg_back_right",
            item.semantic_label,
            center_xyz=(x + x_off, leg_y, z + z_off),
            size_xyz=leg_size,
            material_key=item.material_key,
            yaw_deg=item.yaw_deg,
        ),
    )
    return (seat, backrest, *legs)


def _build_tv_console_components(item: LivingRoomObjectSpec) -> tuple[SceneMeshComponent, ...]:
    width, height, depth = item.size_xyz
    x, y, z = item.center_xyz
    top = _make_box_component(
        f"{item.object_id}_top",
        item.semantic_label,
        center_xyz=(x, y + 0.18, z),
        size_xyz=(width * 0.96, 0.08, depth * 0.82),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    shell = _make_box_component(
        f"{item.object_id}_shell",
        item.semantic_label,
        center_xyz=(x, y + 0.01, z),
        size_xyz=(width * 0.94, 0.28, depth * 0.88),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    side_left = _make_box_component(
        f"{item.object_id}_left_side",
        item.semantic_label,
        center_xyz=(x - width * 0.45, y + 0.14, z),
        size_xyz=(0.10, 0.60, depth * 0.82),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    side_right = _make_box_component(
        f"{item.object_id}_right_side",
        item.semantic_label,
        center_xyz=(x + width * 0.45, y + 0.14, z),
        size_xyz=(0.10, 0.60, depth * 0.82),
        material_key=item.material_key,
        yaw_deg=item.yaw_deg,
    )

    return (top, shell, side_left, side_right)


def _make_box_component(
    component_id: str,
    semantic_label: str,
    *,
    center_xyz: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    material_key: str,
    yaw_deg: float = 0.0,
) -> SceneMeshComponent:
    vertices_xyz, triangles = _box_mesh_arrays(size_xyz=size_xyz)
    rotation = _rotation_y(yaw_deg)
    transformed_vertices = (vertices_xyz @ rotation.T) + np.asarray(center_xyz, dtype=np.float32)
    color = np.asarray(material_color_rgb(material_key), dtype=np.float32)
    vertex_colors = np.tile(color.reshape(1, 3), (transformed_vertices.shape[0], 1))
    return SceneMeshComponent(
        component_id=component_id,
        semantic_label=semantic_label,
        material_key=material_key,
        vertices_xyz=transformed_vertices,
        triangles=triangles,
        vertex_colors=vertex_colors,
    )


def _box_mesh_arrays(*, size_xyz: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = (float(value) * 0.5 for value in size_xyz)
    vertices = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=np.float32,
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return vertices, triangles


def _rotation_y(yaw_deg: float) -> np.ndarray:
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
