from __future__ import annotations

import importlib.util
import io
import json
import types
from pathlib import Path
import sys

import pytest
import numpy as np

from obj_recog.sim_scene import build_living_room_scene_spec, build_scene_mesh_components


def _load_worker_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "blender" / "realtime_worker.py"
    spec = importlib.util.spec_from_file_location("realtime_worker_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scene_payload(scene) -> dict[str, object]:
    return {
        "scene_id": scene.scene_id,
        "room_size_xyz": list(scene.room_size_xyz),
        "wall_thickness_m": scene.wall_thickness_m,
        "window_wall": scene.window_wall,
        "start_pose": {
            "x": scene.start_pose.x,
            "y": scene.start_pose.y,
            "z": scene.start_pose.z,
            "yaw_deg": scene.start_pose.yaw_deg,
            "camera_pan_deg": scene.start_pose.camera_pan_deg,
        },
        "hidden_goal_pose_xyz": list(scene.hidden_goal_pose_xyz),
        "objects": [
            {
                "object_id": item.object_id,
                "semantic_label": item.semantic_label,
                "center_xyz": list(item.center_xyz),
                "size_xyz": list(item.size_xyz),
                "yaw_deg": item.yaw_deg,
                "material_key": item.material_key,
                "collider": item.collider,
            }
            for item in scene.objects
        ],
        "lights": [
            {
                "light_id": light.light_id,
                "light_type": light.light_type,
                "location_xyz": list(light.location_xyz),
                "rotation_deg_xyz": list(light.rotation_deg_xyz),
                "color_rgb": list(light.color_rgb),
                "energy": light.energy,
            }
            for light in scene.lights
        ],
    }


def test_realtime_worker_python_runtime_handles_build_then_render(tmp_path: Path) -> None:
    module = _load_worker_module()
    runtime = module.create_worker_runtime(output_root=tmp_path, force_python_fallback=True)
    scene = build_living_room_scene_spec()

    build_response = runtime.process_request(
        {
            "kind": "build_scene",
            "scene_spec": _scene_payload(scene),
            "image_width": 32,
            "image_height": 24,
            "horizontal_fov_deg": 72.0,
            "near_plane_m": 0.2,
            "far_plane_m": 8.0,
        }
    )

    assert build_response["status"] == "ready"
    render_response = runtime.process_request(
        {
            "kind": "render_frame",
            "frame_index": 2,
            "timestamp_sec": 1.0,
            "robot_pose": {"x": -2.0, "y": 1.25, "z": -1.5, "yaw_deg": 0.0, "camera_pan_deg": 0.0},
            "camera_pose_world": np.eye(4, dtype=np.float32).tolist(),
        }
    )

    assert Path(render_response["rgb_path"]).is_file()
    assert Path(render_response["depth_path"]).is_file()
    assert Path(render_response["semantic_mask_path"]).is_file()
    assert Path(render_response["instance_mask_path"]).is_file()
    assert np.load(render_response["rgb_path"]).shape == (24, 32, 3)


def test_realtime_worker_python_runtime_build_scene_uses_shared_mesh_components(tmp_path: Path) -> None:
    module = _load_worker_module()
    runtime = module.create_worker_runtime(output_root=tmp_path, force_python_fallback=True)
    scene = build_living_room_scene_spec()
    components = build_scene_mesh_components(scene)

    runtime.process_request(
        {
            "kind": "build_scene",
            "scene_spec": _scene_payload(scene),
            "image_width": 32,
            "image_height": 24,
            "horizontal_fov_deg": 72.0,
            "near_plane_m": 0.2,
            "far_plane_m": 8.0,
        }
    )

    primitive_ids = [primitive.primitive_id for primitive in runtime._primitives]
    component_ids = [component.component_id for component in components]

    assert len(runtime._primitives) == len(components)
    assert set(primitive_ids) == set(component_ids)
    assert len(primitive_ids) > len(scene.objects) + 3
    assert any(primitive.semantic_label == "window_frame" for primitive in runtime._primitives)
    assert len({primitive.semantic_label for primitive in runtime._primitives}) >= 8
    assert max(int(primitive.instance_id) for primitive in runtime._primitives) == len(runtime._primitives)


def test_realtime_worker_rejects_non_box_mesh_component(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_worker_module()
    scene = build_living_room_scene_spec()
    components = list(build_scene_mesh_components(scene))
    components[0] = types.SimpleNamespace(
        component_id="broken_panel",
        semantic_label="wall",
        material_key="painted_wall",
        vertices_xyz=np.array(
            [
                [-1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [1.0, 1.0, -1.0],
            ],
            dtype=np.float32,
        ),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        vertex_colors=np.zeros((3, 3), dtype=np.float32),
    )
    monkeypatch.setattr(module, "build_scene_mesh_components", lambda _scene_spec: tuple(components))

    runtime = module.create_worker_runtime(output_root=tmp_path, force_python_fallback=True)
    with pytest.raises(RuntimeError, match="box/slab/panel contract"):
        runtime.process_request(
            {
                "kind": "build_scene",
                "scene_spec": _scene_payload(scene),
                "image_width": 32,
                "image_height": 24,
                "horizontal_fov_deg": 72.0,
                "near_plane_m": 0.2,
                "far_plane_m": 8.0,
            }
        )


def test_realtime_worker_render_outputs_use_shared_component_semantics(tmp_path: Path) -> None:
    module = _load_worker_module()
    runtime = module.create_worker_runtime(output_root=tmp_path, force_python_fallback=True)
    scene = build_living_room_scene_spec()
    runtime.process_request(
        {
            "kind": "build_scene",
            "scene_spec": _scene_payload(scene),
            "image_width": 64,
            "image_height": 48,
            "horizontal_fov_deg": 72.0,
            "near_plane_m": 0.2,
            "far_plane_m": 8.0,
        }
    )

    render_response = runtime.process_request(
        {
            "kind": "render_frame",
            "frame_index": 0,
            "timestamp_sec": 0.0,
            "robot_pose": {
                "x": scene.start_pose.x,
                "y": scene.start_pose.y,
                "z": scene.start_pose.z,
                "yaw_deg": scene.start_pose.yaw_deg,
                "camera_pan_deg": scene.start_pose.camera_pan_deg,
            },
            "camera_pose_world": np.eye(4, dtype=np.float32).tolist(),
        }
    )

    semantic = np.load(render_response["semantic_mask_path"])
    instance = np.load(render_response["instance_mask_path"])

    assert float(semantic.max()) > 1.0
    assert np.unique(instance).size > 2
    assert np.max(instance) <= len(runtime._primitives)
    runtime_semantic_ids = {primitive.semantic_id for primitive in runtime._primitives}
    observed_semantics = set(np.unique(semantic).tolist())
    assert observed_semantics.issubset(runtime_semantic_ids | {0})


def test_realtime_worker_python_runtime_renders_non_uniform_room_observations(tmp_path: Path) -> None:
    module = _load_worker_module()
    runtime = module.create_worker_runtime(output_root=tmp_path, force_python_fallback=True)
    scene = build_living_room_scene_spec()
    runtime.process_request(
        {
            "kind": "build_scene",
            "scene_spec": _scene_payload(scene),
            "image_width": 64,
            "image_height": 48,
            "horizontal_fov_deg": 72.0,
            "near_plane_m": 0.2,
            "far_plane_m": 8.0,
        }
    )

    render_response = runtime.process_request(
        {
            "kind": "render_frame",
            "frame_index": 0,
            "timestamp_sec": 0.0,
            "robot_pose": {
                "x": scene.start_pose.x,
                "y": scene.start_pose.y,
                "z": scene.start_pose.z,
                "yaw_deg": scene.start_pose.yaw_deg,
                "camera_pan_deg": scene.start_pose.camera_pan_deg,
            },
            "camera_pose_world": np.eye(4, dtype=np.float32).tolist(),
        }
    )

    rgb = np.load(render_response["rgb_path"])
    depth = np.load(render_response["depth_path"])
    semantic = np.load(render_response["semantic_mask_path"])

    assert float(rgb.std()) > 1.0
    assert float(depth.max() - depth.min()) > 0.1
    assert np.unique(semantic).size > 1


def test_realtime_worker_camera_pan_changes_rendered_view(tmp_path: Path) -> None:
    module = _load_worker_module()
    runtime = module.create_worker_runtime(output_root=tmp_path, force_python_fallback=True)
    scene = build_living_room_scene_spec()
    runtime.process_request(
        {
            "kind": "build_scene",
            "scene_spec": _scene_payload(scene),
            "image_width": 64,
            "image_height": 48,
            "horizontal_fov_deg": 72.0,
            "near_plane_m": 0.2,
            "far_plane_m": 8.0,
        }
    )

    left_response = runtime.process_request(
        {
            "kind": "render_frame",
            "frame_index": 0,
            "timestamp_sec": 0.0,
            "robot_pose": {
                "x": scene.start_pose.x,
                "y": scene.start_pose.y,
                "z": scene.start_pose.z,
                "yaw_deg": scene.start_pose.yaw_deg,
                "camera_pan_deg": -30.0,
            },
            "camera_pose_world": np.eye(4, dtype=np.float32).tolist(),
        }
    )
    right_response = runtime.process_request(
        {
            "kind": "render_frame",
            "frame_index": 1,
            "timestamp_sec": 0.5,
            "robot_pose": {
                "x": scene.start_pose.x,
                "y": scene.start_pose.y,
                "z": scene.start_pose.z,
                "yaw_deg": scene.start_pose.yaw_deg,
                "camera_pan_deg": 30.0,
            },
            "camera_pose_world": np.eye(4, dtype=np.float32).tolist(),
        }
    )

    left_rgb = np.load(left_response["rgb_path"])
    right_rgb = np.load(right_response["rgb_path"])

    assert not np.array_equal(left_rgb, right_rgb)


def test_realtime_worker_glass_tints_rgb_but_keeps_opaque_depth_and_semantics(tmp_path: Path) -> None:
    module = _load_worker_module()
    runtime = module.create_worker_runtime(output_root=tmp_path, force_python_fallback=True)
    runtime._image_width = 1
    runtime._image_height = 1
    runtime._near_plane_m = 0.2
    runtime._far_plane_m = 8.0
    runtime._intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    runtime._scene_center = np.zeros(3, dtype=np.float32)
    runtime._lights = ()

    identity = np.eye(3, dtype=np.float32)
    wall_primitive = module._RenderPrimitive(
        primitive_id="wall",
        semantic_label="wall",
        semantic_id=2,
        instance_id=2,
        material_key="painted_wall",
        center_xyz=np.array([0.0, 0.0, 2.0], dtype=np.float32),
        half_size_xyz=np.array([1.0, 1.0, 0.2], dtype=np.float32),
        rotation_world_to_local=identity,
        rotation_local_to_world=identity,
    )
    glass_primitive = module._RenderPrimitive(
        primitive_id="glass",
        semantic_label="glass",
        semantic_id=1,
        instance_id=1,
        material_key="tinted_glass",
        center_xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        half_size_xyz=np.array([1.0, 1.0, 0.05], dtype=np.float32),
        rotation_world_to_local=identity,
        rotation_local_to_world=identity,
    )
    runtime._primitives = (wall_primitive,)
    dirs_camera, dirs_world = runtime._camera_rays(0.0)
    wall_rgb, wall_depth, wall_semantic, wall_instance = runtime._trace_scene(
        origin_world=np.zeros(3, dtype=np.float32),
        dirs_camera=dirs_camera,
        dirs_world=dirs_world,
    )

    runtime._primitives = (glass_primitive, wall_primitive)
    glass_rgb, glass_depth, glass_semantic, glass_instance = runtime._trace_scene(
        origin_world=np.zeros(3, dtype=np.float32),
        dirs_camera=dirs_camera,
        dirs_world=dirs_world,
    )

    assert int(wall_semantic[0, 0]) == 2
    assert int(glass_semantic[0, 0]) == 2
    assert float(glass_depth[0, 0]) == pytest.approx(float(wall_depth[0, 0]), abs=1e-4)
    assert int(glass_instance[0, 0]) == 2
    assert not np.array_equal(glass_rgb[0, 0], wall_rgb[0, 0])
    assert float(glass_rgb[0, 0].mean()) < float(wall_rgb[0, 0].mean())


def test_realtime_worker_loop_writes_json_lines(tmp_path: Path) -> None:
    module = _load_worker_module()
    scene = build_living_room_scene_spec()
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "build_scene",
                        "scene_spec": _scene_payload(scene),
                        "image_width": 16,
                        "image_height": 12,
                        "horizontal_fov_deg": 72.0,
                        "near_plane_m": 0.2,
                        "far_plane_m": 8.0,
                    }
                ),
                json.dumps(
                    {
                        "kind": "render_frame",
                        "frame_index": 0,
                        "timestamp_sec": 0.0,
                        "robot_pose": {"x": -2.0, "y": 1.25, "z": -1.5, "yaw_deg": 0.0, "camera_pan_deg": 0.0},
                        "camera_pose_world": np.eye(4, dtype=np.float32).tolist(),
                    }
                ),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = module.run_worker_loop(
        argv=["realtime_worker.py", "--output-root", str(tmp_path)],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        force_python_fallback=True,
    )

    assert exit_code == 0
    output_lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert output_lines[0]["status"] == "ready"
    assert output_lines[1]["worker_state"] == "ready"
    assert stderr.getvalue().strip()
