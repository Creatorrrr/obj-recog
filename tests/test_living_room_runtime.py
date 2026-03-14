from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.sim_protocol import ActionPrimitive, ActionSchedule, ActionStep, EpisodePhase, SensorFrame
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
        self.reset_calls: list[str] = []
        self.apply_calls: list[tuple[str, float]] = []
        self.closed = False
        self._frame_index = 0

    def reset_episode(self, *, scene_spec) -> SensorFrame:
        self.reset_calls.append(scene_spec.scene_id)
        self._frame_index = 0
        return self._frame()

    def apply_action(self, command) -> SensorFrame:
        self.apply_calls.append((command.primitive.value, float(command.value)))
        self._frame_index += 1
        return self._frame()

    def close(self) -> None:
        self.closed = True

    def _frame(self) -> SensorFrame:
        frame = np.full((12, 16, 3), 80 + self._frame_index, dtype=np.uint8)
        return SensorFrame(
            frame_index=self._frame_index,
            timestamp_sec=float(self._frame_index) * 0.5,
            frame_bgr=frame,
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


def _artifacts(_packet: FramePacket):
    return type(
        "Artifacts",
        (),
        {
            "detections": [],
            "segments": [],
            "scene_graph_snapshot": None,
            "mesh_vertices_xyz": np.zeros((4, 3), dtype=np.float32),
            "depth_map": np.full((12, 16), 1.5, dtype=np.float32),
            "slam_tracking_state": "TRACKING",
        },
    )()


def _goal_artifacts(_packet: FramePacket):
    return type(
        "Artifacts",
        (),
        {
            "frame_bgr": np.full((12, 16, 3), 120, dtype=np.uint8),
            "detections": [
                type(
                    "Det",
                    (),
                    {
                        "label": "tv",
                        "xyxy": (1, 1, 15, 11),
                        "confidence": 0.92,
                    },
                )()
            ],
            "segments": [],
            "scene_graph_snapshot": None,
            "mesh_vertices_xyz": np.zeros((4, 3), dtype=np.float32),
            "depth_map": np.full((12, 16), 1.6, dtype=np.float32),
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
    backend = _FakeSensorBackend()
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=planner,
        sensor_backend=backend,
    )

    for _ in range(6):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert planner.calls
    assert all(context.phase != EpisodePhase.SELF_CALIBRATING for context in planner.calls)
    assert [primitive for primitive, _value in backend.apply_calls[:5]] == [
        "camera_pan_left",
        "camera_pan_right",
        "turn_left",
        "turn_right",
        "move_forward",
    ]


def test_living_room_runtime_recovers_from_empty_schedule_with_pause_fallback(tmp_path: Path) -> None:
    backend = _FakeSensorBackend()
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend=backend,
    )

    for _ in range(7):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert ("pause", 0.5) in backend.apply_calls
    assert runner.current_phase in {EpisodePhase.EXECUTING_SCHEDULE, EpisodePhase.REASSESS}


def test_living_room_runtime_writes_rgb_only_episode_artifacts(tmp_path: Path) -> None:
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

    report_payload = json.loads((tmp_path / "episode_report.json").read_text(encoding="utf-8"))
    assert (tmp_path / "planner_turns.jsonl").is_file()
    assert (tmp_path / "self_calibration.json").is_file()
    assert report_payload["success"] is None
    assert report_payload["offline_evaluation_required"] is True
    assert report_payload["sim_interface_mode"] == "rgb_only"


def test_living_room_runtime_marks_success_when_tv_is_visually_reached(tmp_path: Path) -> None:
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend=_FakeSensorBackend(),
    )
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(frame_packet=packet, artifacts=_goal_artifacts(packet))

    report_payload = json.loads((tmp_path / "episode_report.json").read_text(encoding="utf-8"))
    assert runner.next_frame() is None
    assert report_payload["success"] is True
    assert report_payload["final_phase"] == EpisodePhase.SUCCEEDED.value


def test_living_room_runtime_does_not_fail_early_when_soft_budgets_are_small(tmp_path: Path) -> None:
    runner = LivingRoomEpisodeRunner(
        config=_config(eval_budget_sec=0.1, sim_max_steps=1),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend=_FakeSensorBackend(),
    )

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    report_payload = json.loads((tmp_path / "episode_report.json").read_text(encoding="utf-8"))
    assert runner.next_frame() is not None
    assert report_payload["final_phase"] != EpisodePhase.FAILED.value
    assert report_payload["success"] is None


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

    step = runner._next_tracking_safe_step()

    assert step is not None
    assert step.primitive == ActionPrimitive.CAMERA_PAN_LEFT
    assert float(step.value) == 6.0
    assert runner.current_schedule is not None
    assert float(runner.current_schedule.steps[0].value) == 24.0


def test_living_room_runtime_forwards_planner_actions_to_backend(tmp_path: Path) -> None:
    backend = _FakeSensorBackend()
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner(
            [
                ActionSchedule(
                    steps=(
                        ActionStep(ActionPrimitive.MOVE_FORWARD, 0.5),
                        ActionStep(ActionPrimitive.TURN_RIGHT, 12.0),
                    ),
                    rationale="advance and turn",
                    model="fake-planner",
                    issued_at_frame=0,
                )
            ]
        ),
        sensor_backend=backend,
    )

    for _ in range(8):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert ("move_forward", 0.12) in backend.apply_calls
    assert any(primitive == "turn_right" for primitive, _value in backend.apply_calls)


def test_simulation_runtime_uses_living_room_scene_and_resets_backend(tmp_path: Path) -> None:
    backend = _FakeSensorBackend()
    runtime = LivingRoomSimulationRuntime(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend_factory=lambda _config, _camera_rig: backend,
    )

    runner = runtime.create_frame_source()

    assert runner._scene_spec.scene_id == "living_room_navigation_v1"
    assert backend.reset_calls == ["living_room_navigation_v1"]
