from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from obj_recog.blend_scene_loader import BlendSceneManifest, BlendSceneObject
from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.sim_protocol import ActionPrimitive, ActionSchedule, ActionStep, EpisodePhase, RobotPose, SensorFrame
from obj_recog.sim_scene import build_interior_test_tv_scene_spec, build_living_room_scene_spec
from obj_recog.simulation import LivingRoomEpisodeRunner, LivingRoomSimulationRuntime


class _FakePlanner:
    def __init__(self, schedules: list[ActionSchedule]) -> None:
        self._schedules = list(schedules)
        self.calls = []
        self.model = "fake-planner"

    def plan(self, *, context):
        self.calls.append(context)
        if not self._schedules:
            return ActionSchedule(steps=(), rationale="empty", model=self.model, issued_at_frame=context.frame_index)
        return self._schedules.pop(0)


class _FakeSensorBackend:
    def __init__(self) -> None:
        self.build_calls = []
        self.render_calls = []

    def build_scene(self, scene_spec) -> None:
        self.build_calls.append(scene_spec.scene_id)

    def render_frame(self, *, world_state, frame_index: int, timestamp_sec: float) -> SensorFrame:
        self.render_calls.append((frame_index, world_state.phase.value))
        frame = np.full((12, 16, 3), 80 + frame_index, dtype=np.uint8)
        depth = np.full((12, 16), 1.5 + (0.1 * frame_index), dtype=np.float32)
        semantic = np.zeros((12, 16), dtype=np.uint8)
        instance = np.zeros((12, 16), dtype=np.uint8)
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = world_state.robot_pose.x
        pose[1, 3] = world_state.robot_pose.y
        pose[2, 3] = world_state.robot_pose.z
        return SensorFrame(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            frame_bgr=frame,
            depth_map=depth,
            semantic_mask=semantic,
            instance_mask=instance,
            camera_pose_world=pose,
            intrinsics={"fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 6.0},
            render_time_ms=5.0,
        )


def _config(**overrides: object) -> AppConfig:
    values = dict(
        camera_index=0,
        width=16,
        height=12,
        device="cpu",
        conf_threshold=0.35,
        point_stride=1,
        max_points=128,
        input_source="sim",
        sim_headless=True,
        sim_camera_fps=2.0,
        sim_selfcal_max_sec=6.0,
        sim_action_batch_size=2,
        eval_budget_sec=20.0,
    )
    values.update(overrides)
    return AppConfig(**values)


def _artifacts(packet: FramePacket):
    return type(
        "Artifacts",
        (),
        {
            "detections": [],
            "segments": [],
            "scene_graph_snapshot": None,
            "mesh_vertices_xyz": np.zeros((4, 3), dtype=np.float32),
            "depth_map": np.asarray(packet.depth_map, dtype=np.float32),
            "slam_tracking_state": "TRACKING",
        },
    )()


def _interior_manifest() -> BlendSceneManifest:
    return BlendSceneManifest(
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
                    [[-2.5, 0.0, -4.0], [2.5, 0.0, -4.0], [2.5, 0.0, 4.0], [-2.5, 0.0, 4.0]],
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
                    [[-0.75, 0.9, 3.90], [0.75, 0.9, 3.90], [0.75, 1.7, 3.95], [-0.75, 1.7, 3.95]],
                    dtype=np.float32,
                ),
                triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                collider=False,
            ),
        ),
    )


def test_living_room_runtime_self_calibrates_before_first_planner_turn(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.MOVE_FORWARD, 0.5),),
                rationale="advance",
                model="fake-planner",
                issued_at_frame=0,
            )
        ]
    )
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=planner,
        sensor_backend=_FakeSensorBackend(),
    )

    for _ in range(6):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert planner.calls
    assert all(context.phase != EpisodePhase.SELF_CALIBRATING for context in planner.calls)


def test_living_room_runtime_recovers_from_empty_schedule_with_pause_fallback(tmp_path: Path) -> None:
    planner = _FakePlanner([])
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=planner,
        sensor_backend=_FakeSensorBackend(),
    )

    for _ in range(7):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert runner.current_phase in {EpisodePhase.EXECUTING_SCHEDULE, EpisodePhase.REASSESS}
    assert runner.current_schedule is not None
    assert runner.current_schedule.steps
    assert runner.current_schedule.steps[0].primitive == ActionPrimitive.PAUSE


def test_living_room_runtime_marks_success_when_robot_reaches_hidden_goal(tmp_path: Path) -> None:
    base_scene = build_living_room_scene_spec()
    scene = replace(base_scene, objects=())
    goal_x, goal_y, goal_z = scene.hidden_goal_pose_xyz
    planner = _FakePlanner(
        [
            ActionSchedule(
                steps=(
                    ActionStep(ActionPrimitive.STRAFE_RIGHT, goal_x - scene.start_pose.x),
                    ActionStep(ActionPrimitive.MOVE_FORWARD, goal_z - scene.start_pose.z),
                ),
                rationale="drive to dining goal",
                model="fake-planner",
                issued_at_frame=0,
            )
        ]
    )
    runner = LivingRoomEpisodeRunner(
        config=_config(sim_camera_fps=10.0),
        report_path=tmp_path / "episode.json",
        planner=planner,
        sensor_backend=_FakeSensorBackend(),
        scene_spec=scene,
    )

    for _ in range(80):
        packet = runner.next_frame()
        if packet is None:
            break
        runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))
        if runner.current_phase == EpisodePhase.SUCCEEDED:
            break

    assert runner.current_phase == EpisodePhase.SUCCEEDED
    report = json.loads((tmp_path / "episode_report.json").read_text(encoding="utf-8"))
    assert report["success"] is True


def test_living_room_runtime_writes_episode_artifacts(tmp_path: Path) -> None:
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend=_FakeSensorBackend(),
    )

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))
    runner.close()

    assert (tmp_path / "planner_turns.jsonl").is_file()
    assert (tmp_path / "self_calibration.json").is_file()
    assert (tmp_path / "episode_report.json").is_file()


def test_living_room_runtime_uses_tracking_safe_self_calibration_motion_steps(tmp_path: Path) -> None:
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend=_FakeSensorBackend(),
    )

    selfcal_steps = list(runner._selfcal_actions)

    assert [step.primitive.value for step in selfcal_steps] == [
        "camera_pan_left",
        "camera_pan_right",
        "turn_left",
        "turn_right",
        "move_forward",
    ]
    assert float(selfcal_steps[0].value) <= 8.0
    assert float(selfcal_steps[1].value) <= 8.0
    assert float(selfcal_steps[2].value) <= 8.0
    assert float(selfcal_steps[3].value) <= 8.0
    assert float(selfcal_steps[4].value) <= 0.15


def test_living_room_runtime_chunks_large_schedule_steps_into_tracking_safe_substeps(tmp_path: Path) -> None:
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend=_FakeSensorBackend(),
    )
    runner._state.phase = EpisodePhase.EXECUTING_SCHEDULE
    runner._current_schedule = ActionSchedule(
        steps=(
            ActionStep(ActionPrimitive.CAMERA_PAN_LEFT, 30.0),
            ActionStep(ActionPrimitive.MOVE_FORWARD, 0.5),
        ),
        rationale="chunk test",
        model="fake-planner",
        issued_at_frame=0,
    )
    runner._state.schedule_cursor = 0

    runner._execute_one_step()

    assert runner._state.robot_pose.camera_pan_deg == 6.0
    assert runner.current_schedule is not None
    assert runner.current_schedule.steps[0].primitive == ActionPrimitive.CAMERA_PAN_LEFT
    assert float(runner.current_schedule.steps[0].value) == 24.0
    assert runner.current_phase == EpisodePhase.EXECUTING_SCHEDULE


def test_living_room_runtime_moves_forward_relative_to_robot_yaw(tmp_path: Path) -> None:
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend=_FakeSensorBackend(),
    )
    runner._state.robot_pose = RobotPose(x=0.0, y=1.25, z=-2.0, yaw_deg=90.0, camera_pan_deg=0.0)

    runner._apply_step(ActionStep(ActionPrimitive.MOVE_FORWARD, 0.5))

    assert runner._state.robot_pose.x == pytest.approx(0.5, abs=1e-4)
    assert runner._state.robot_pose.z == pytest.approx(-2.0, abs=1e-4)


def test_living_room_runtime_blocks_motion_into_hidden_collider(tmp_path: Path) -> None:
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
                vertices_xyz=np.empty((0, 3), dtype=np.float32),
                triangles=np.empty((0, 3), dtype=np.int32),
                collider=False,
            ),
            BlendSceneObject(
                object_id="BlockerTable",
                object_type="MESH",
                semantic_label="table",
                center_xyz=(0.0, 0.45, -2.35),
                size_xyz=(0.80, 0.90, 0.80),
                yaw_deg=0.0,
                vertices_xyz=np.empty((0, 3), dtype=np.float32),
                triangles=np.empty((0, 3), dtype=np.int32),
                collider=True,
            ),
        ),
    )
    scene = build_interior_test_tv_scene_spec(manifest)
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend=_FakeSensorBackend(),
        scene_spec=scene,
    )

    original_pose = runner._state.robot_pose
    runner._apply_step(ActionStep(ActionPrimitive.MOVE_FORWARD, 0.5))

    assert runner._state.robot_pose.x == pytest.approx(original_pose.x, abs=1e-4)
    assert runner._state.robot_pose.z == pytest.approx(original_pose.z, abs=1e-4)


def test_simulation_runtime_selects_interior_test_tv_scene_from_config(tmp_path: Path) -> None:
    sensor_backend = _FakeSensorBackend()
    runtime = LivingRoomSimulationRuntime(
        config=_config(scenario="interior_test_tv_navigation_v1"),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend_factory=lambda _config, _camera_rig: sensor_backend,
    )

    runner = runtime.create_frame_source()

    assert runner._scene_spec.scene_id == "interior_test_tv_navigation_v1"
    assert sensor_backend.build_calls == ["interior_test_tv_navigation_v1"]


def test_episode_runner_uses_scene_semantic_target_class_in_operator_state(tmp_path: Path) -> None:
    scene = build_interior_test_tv_scene_spec()
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend=_FakeSensorBackend(),
        scene_spec=scene,
    )

    packet = runner.next_frame()

    assert packet is not None
    assert packet.scenario_state is not None
    assert packet.scenario_state.semantic_target_class == "tv"


def test_simulation_runtime_loads_blend_manifest_for_interior_test_scene(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sensor_backend = _FakeSensorBackend()
    runtime = LivingRoomSimulationRuntime(
        config=_config(
            scenario="interior_test_tv_navigation_v1",
            blender_exec="/Applications/Blender.app/Contents/MacOS/Blender",
        ),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend_factory=lambda _config, _camera_rig: sensor_backend,
    )
    manifest = _interior_manifest()
    monkeypatch.setattr("obj_recog.simulation.load_blend_scene_manifest", lambda **_kwargs: manifest)

    runner = runtime.create_frame_source()

    assert runner._scene_spec.scene_metadata["blend_manifest"] is manifest
    assert runner._scene_spec.hidden_goal_pose_xyz == pytest.approx((0.0, 1.25, 3.1260), abs=1e-4)
