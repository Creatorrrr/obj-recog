from __future__ import annotations

import math

import numpy as np

from obj_recog.sim_scene import build_living_room_scene_spec, build_scene_mesh_components


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


def test_living_room_scene_builds_expected_mesh_component_groups() -> None:
    scene = build_living_room_scene_spec()
    components = build_scene_mesh_components(scene)
    labels = [component.semantic_label for component in components]

    assert labels.count("floor") == 1
    assert labels.count("wall") == 3
    assert labels.count("glass") == 1
    assert labels.count("ceiling") == 1
    assert labels.count("sofa") == 1
    assert labels.count("coffee_table") == 1
    assert labels.count("tv_console") == 1
    assert labels.count("tv_panel") == 1
    assert labels.count("dining_table") == 1
    assert labels.count("dining_chair") == 4

    assert all(component.vertices_xyz.shape[1] == 3 for component in components)
    assert all(component.triangles.shape[1] == 3 for component in components)
