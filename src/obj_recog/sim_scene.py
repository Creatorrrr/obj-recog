from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

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
    )


def build_scene_mesh_components(scene: LivingRoomSceneSpec) -> tuple[SceneMeshComponent, ...]:
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
            "ceiling",
            "ceiling",
            center_xyz=(0.0, room_height + 0.03, 0.0),
            size_xyz=(room_width, 0.06, room_depth),
            material_key="matte_ceiling",
        ),
    ]

    for item in scene.objects:
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


def pose_distance_to_goal(scene: LivingRoomSceneSpec, pose: RobotPose) -> float:
    goal = np.asarray(scene.hidden_goal_pose_xyz, dtype=np.float32)
    robot = np.asarray((pose.x, pose.y, pose.z), dtype=np.float32)
    return float(np.linalg.norm(goal - robot))


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
