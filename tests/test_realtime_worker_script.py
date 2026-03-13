from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

from obj_recog.sim_scene import build_living_room_scene_spec


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
