from __future__ import annotations

import math

import numpy as np
import pytest

from obj_recog.blend_scene_loader import (
    BlendSceneManifest,
    BlendSceneObject,
    normalize_blender_xyz_to_sim_xyz,
    tv_front_goal_from_manifest,
)
from obj_recog.sim_scene import (
    build_interior_test_tv_scene_spec,
    build_living_room_scene_spec,
    build_scene_mesh_components,
)


def test_living_room_scene_has_expected_room_dimensions() -> None:
    scene = build_living_room_scene_spec()

    assert scene.scene_id == "living_room_navigation_v1"
    assert scene.room_size_xyz == (7.2, 2.5, 5.4)
    assert scene.window_wall == "front"


def test_living_room_scene_places_hidden_goal_in_front_of_dining_table() -> None:
    scene = build_living_room_scene_spec()
    dining_table = next(item for item in scene.objects if item.object_id == "dining_table")
    goal = np.asarray(scene.hidden_goal_pose_xyz, dtype=np.float32)
    near_edge_z = float(dining_table.center_xyz[2] - (dining_table.size_xyz[2] * 0.5))

    assert goal[0] == dining_table.center_xyz[0]
    assert math.isclose(near_edge_z - float(goal[2]), 0.75, rel_tol=0.0, abs_tol=1e-6)


def test_living_room_scene_builds_multi_component_furniture() -> None:
    components = build_scene_mesh_components(build_living_room_scene_spec())

    sofa_components = [c for c in components if c.semantic_label == "sofa"]
    dining_table_components = [c for c in components if c.semantic_label == "dining_table"]
    chair_components = [c for c in components if c.semantic_label == "dining_chair"]
    tv_console_components = [c for c in components if c.semantic_label == "tv_console"]

    assert len(sofa_components) >= 3
    assert len(dining_table_components) >= 3
    assert len(chair_components) >= 6
    assert len(tv_console_components) >= 3


def test_living_room_scene_sofa_expands_into_parts() -> None:
    components = build_scene_mesh_components(build_living_room_scene_spec())

    sofa_parts = [c for c in components if c.semantic_label == "sofa"]
    assert 7 <= len(sofa_parts) <= 9
    sofa_heights = [float(np.max(part.vertices_xyz[:, 1]) - np.min(part.vertices_xyz[:, 1])) for part in sofa_parts]

    assert len(sofa_parts) >= 3
    assert max(sofa_heights) - min(sofa_heights) > 0.15


def test_living_room_scene_sofa_has_plural_cushions_including_back_cushions() -> None:
    sofa_parts = [c for c in build_scene_mesh_components(build_living_room_scene_spec()) if c.semantic_label == "sofa"]
    if not sofa_parts:
        raise AssertionError("No sofa components found")

    all_points = np.concatenate([part.vertices_xyz for part in sofa_parts], axis=0)
    sofa_min = np.min(all_points, axis=0)
    sofa_max = np.max(all_points, axis=0)
    sofa_mid_y = float((sofa_min[1] + sofa_max[1]) * 0.5)

    thin_parts = []
    for part in sofa_parts:
        extent = np.max(part.vertices_xyz, axis=0) - np.min(part.vertices_xyz, axis=0)
        if float(extent[1]) < 0.14 and float(extent[0]) > 0.3 and float(extent[2]) > 0.2:
            thin_parts.append(part)

    seat_like = []
    back_like = []
    for part in thin_parts:
        part_min = np.min(part.vertices_xyz, axis=0)
        part_max = np.max(part.vertices_xyz, axis=0)
        center_y = float((part_min[1] + part_max[1]) * 0.5)
        if center_y <= sofa_mid_y:
            seat_like.append(part)
        else:
            back_like.append(part)

    assert len(thin_parts) >= 4
    assert len(seat_like) >= 2
    assert len(back_like) >= 2


def test_living_room_scene_dining_table_has_support_and_top_parts() -> None:
    scene = build_living_room_scene_spec()
    components = build_scene_mesh_components(scene)

    table_components = [c for c in components if c.semantic_label == "dining_table"]
    table_heights = [
        float(np.max(c.vertices_xyz[:, 1]) - np.min(c.vertices_xyz[:, 1])) for c in table_components
    ]

    assert len(table_components) >= 3
    assert max(table_heights) - min(table_heights) > 0.2
    assert any(height <= 0.35 for height in table_heights)


def test_living_room_scene_dining_chairs_are_multi_component() -> None:
    scene = build_living_room_scene_spec()
    components = build_scene_mesh_components(scene)

    dining_chairs = [obj for obj in scene.objects if obj.object_id.startswith("dining_chair")]

    def _parts_near_object(obj) -> list:
        result = []
        obj_center = np.asarray(obj.center_xyz, dtype=np.float32)
        obj_half = np.asarray(obj.size_xyz, dtype=np.float32) * 0.5
        search_xy_radius = float(np.linalg.norm(obj_half[[0, 2]]) * 1.4)
        search_y_overlap = obj_half[1] * 1.2
        search_z_extent = obj.size_xyz[2] * 0.8

        for part in [c for c in components if c.semantic_label == "dining_chair"]:
            part_min = np.min(part.vertices_xyz, axis=0)
            part_max = np.max(part.vertices_xyz, axis=0)
            part_center = (part_min + part_max) / 2.0
            xy_dist = float(np.linalg.norm(part_center[[0, 2]] - obj_center[[0, 2]]))
            z_within = part_min[2] < obj_center[2] + search_z_extent and part_max[2] > obj_center[2] - search_z_extent
            y_overlap = min(part_max[1], obj_center[1] + search_y_overlap) - max(part_min[1], obj_center[1] - search_y_overlap)
            if xy_dist <= search_xy_radius and z_within and y_overlap > 0:
                result.append(part)

        return result

    for chair in dining_chairs:
        chair_parts = _parts_near_object(chair)
        assert len(chair_parts) >= 2


def test_living_room_scene_tv_console_expands_into_parts() -> None:
    components = build_scene_mesh_components(build_living_room_scene_spec())

    tv_console_parts = [c for c in components if c.semantic_label == "tv_console"]
    tv_console_heights = [float(np.max(part.vertices_xyz[:, 1]) - np.min(part.vertices_xyz[:, 1])) for part in tv_console_parts]

    assert len(tv_console_parts) >= 3
    assert max(tv_console_heights) - min(tv_console_heights) > 0.05


def test_living_room_scene_front_window_has_structural_frame() -> None:
    scene = build_living_room_scene_spec()
    components = build_scene_mesh_components(scene)

    window_front_z = scene.room_size_xyz[2] / 2
    front_band_min = window_front_z - 0.35
    front_band_max = window_front_z + 0.20
    window_glass = [
        c
        for c in components
        if c.semantic_label == "glass"
        and np.max(c.vertices_xyz[:, 2]) > front_band_min
        and np.min(c.vertices_xyz[:, 2]) < front_band_max
    ]

    def _aabb_overlap_1d(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
        return max(0.0, min(a_max, b_max) - max(a_min, b_min))

    def _is_adjacent_box(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray, *, gap_tol: float) -> bool:
        x_overlap = _aabb_overlap_1d(a_min[0], a_max[0], b_min[0], b_max[0])
        z_overlap = _aabb_overlap_1d(a_min[2], a_max[2], b_min[2], b_max[2])
        z_gap = max(0.0, a_min[2] - b_max[2], b_min[2] - a_max[2])
        return (x_overlap > 0.0 and z_overlap > 0.0) or (z_gap <= gap_tol)

    assert len(window_glass) == 1
    glass = window_glass[0]
    glass_min = np.min(glass.vertices_xyz, axis=0)
    glass_max = np.max(glass.vertices_xyz, axis=0)
    glass_size = glass_max - glass_min
    glass_mid_y = (glass_min[1] + glass_max[1]) / 2.0
    size_y = glass_size[1]
    adjacent_structure = [
        c
        for c in components
        if c.semantic_label != "glass"
        and np.max(c.vertices_xyz[:, 2]) > front_band_min
        and np.min(c.vertices_xyz[:, 2]) < front_band_max
        and abs((np.min(c.vertices_xyz[:, 1]) + np.max(c.vertices_xyz[:, 1])) / 2.0 - glass_mid_y) <= size_y
    ]
    frame_like = [
        c
        for c in adjacent_structure
        if _is_adjacent_box(
            glass_min,
            glass_max,
            np.min(c.vertices_xyz, axis=0),
            np.max(c.vertices_xyz, axis=0),
            gap_tol=0.25,
        )
    ]
    front_window_frames = [c for c in frame_like if c.semantic_label == "window_frame"]

    assert len(front_window_frames) >= 4
    left = [c for c in front_window_frames if np.max(c.vertices_xyz[:, 0]) <= glass_min[0] + 0.15]
    right = [c for c in front_window_frames if np.min(c.vertices_xyz[:, 0]) >= glass_max[0] - 0.15]
    top = [c for c in front_window_frames if np.min(c.vertices_xyz[:, 1]) >= glass_max[1] - 0.30]
    bottom = [c for c in front_window_frames if np.max(c.vertices_xyz[:, 1]) <= glass_min[1] + 0.12]

    assert len(left) >= 1
    assert len(right) >= 1
    assert len(top) >= 1
    assert len(bottom) >= 1


def test_living_room_scene_builds_expected_mesh_component_groups() -> None:
    scene = build_living_room_scene_spec()
    components = build_scene_mesh_components(scene)
    labels = [component.semantic_label for component in components]

    assert labels.count("floor") == 1
    assert labels.count("wall") == 3
    assert labels.count("ceiling") == 1
    assert labels.count("coffee_table") == 1
    assert labels.count("tv_panel") == 1

    assert all(component.vertices_xyz.shape[1] == 3 for component in components)
    assert all(component.triangles.shape[1] == 3 for component in components)


def test_interior_test_tv_scene_has_expected_spec_metadata() -> None:
    scene = build_interior_test_tv_scene_spec()

    assert scene.scene_id == "interior_test_tv_navigation_v1"
    assert scene.blend_file_path == "/Users/chasoik/Downloads/InteriorTest.blend"
    assert scene.semantic_target_class == "tv"
    assert "TV" in scene.goal_description
    assert scene.start_pose == scene.start_pose.__class__(x=0.0, y=1.25, z=-2.8, yaw_deg=0.0, camera_pan_deg=0.0)


def test_blend_coordinate_normalization_maps_blender_z_to_sim_height() -> None:
    sim_xyz = normalize_blender_xyz_to_sim_xyz((1.5, 3.25, 0.8))

    assert sim_xyz == pytest.approx((1.5, 0.8, 3.25))


def test_interior_test_tv_goal_is_derived_from_tv_anchor() -> None:
    manifest = BlendSceneManifest(
        blend_file_path="/Users/chasoik/Downloads/InteriorTest.blend",
        room_size_xyz=(5.0, 3.0, 8.0),
        objects=(
            BlendSceneObject(
                object_id="TV",
                object_type="MESH",
                semantic_label="tv",
                center_xyz=(0.0, 1.3221, 3.9260),
                size_xyz=(1.5, 0.8, 0.05),
                yaw_deg=0.0,
                vertices_xyz=np.empty((0, 3), dtype=np.float32),
                triangles=np.empty((0, 3), dtype=np.int32),
                collider=False,
            ),
        ),
    )

    goal = tv_front_goal_from_manifest(manifest, tv_object_id="TV", offset_m=0.8)

    assert goal == pytest.approx((0.0, 1.25, 3.1260), abs=1e-4)


def test_build_scene_mesh_components_uses_authored_manifest_geometry_for_interior_scene() -> None:
    manifest = BlendSceneManifest(
        blend_file_path="/Users/chasoik/Downloads/InteriorTest.blend",
        room_size_xyz=(5.0, 3.0, 8.0),
        objects=(
            BlendSceneObject(
                object_id="Floor",
                object_type="MESH",
                semantic_label="floor",
                center_xyz=(0.0, 0.0, 0.0),
                size_xyz=(5.0, 0.01, 8.0),
                yaw_deg=0.0,
                vertices_xyz=np.asarray(
                    [
                        [-2.5, 0.0, -4.0],
                        [2.5, 0.0, -4.0],
                        [2.5, 0.0, 4.0],
                        [-2.5, 0.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
                triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                collider=True,
            ),
            BlendSceneObject(
                object_id="TV",
                object_type="MESH",
                semantic_label="tv",
                center_xyz=(0.0, 1.3221, 3.9260),
                size_xyz=(1.5, 0.8, 0.05),
                yaw_deg=0.0,
                vertices_xyz=np.asarray(
                    [
                        [-0.75, 0.9, 3.90],
                        [0.75, 0.9, 3.90],
                        [0.75, 1.7, 3.95],
                        [-0.75, 1.7, 3.95],
                    ],
                    dtype=np.float32,
                ),
                triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                collider=False,
            ),
        ),
    )

    scene = build_interior_test_tv_scene_spec(manifest)
    components = build_scene_mesh_components(scene)
    floor_component = next(component for component in components if component.component_id == "Floor")
    tv_component = next(component for component in components if component.component_id == "TV")

    assert floor_component.vertices_xyz.shape == (4, 3)
    assert floor_component.triangles.shape == (2, 3)
    assert tv_component.vertices_xyz.shape == (4, 3)
    assert tv_component.triangles.shape == (2, 3)
