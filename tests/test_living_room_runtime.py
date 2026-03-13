from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.sim_protocol import ActionPrimitive, ActionSchedule, ActionStep, EpisodePhase, SensorFrame
from obj_recog.sim_scene import build_living_room_scene_spec
from obj_recog.simulation import LivingRoomEpisodeRunner


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
    scene = build_living_room_scene_spec()
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
