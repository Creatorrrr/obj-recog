from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.sim_protocol import (
    ActionSchedule,
    CommandKind,
    EpisodePhase,
    MotionCommand,
    RigCapabilities,
    SensorFrame,
)
from obj_recog.simulation import (
    LivingRoomEpisodeRunner,
    LivingRoomSimulationRuntime,
    _INITIALIZING_RECOVERY_FRAME_THRESHOLD,
    _build_unity_rgb_sensor_backend,
)


class _FakePlanner:
    def __init__(self, schedules: list[ActionSchedule]) -> None:
        self._schedules = list(schedules)
        self.calls = []
        self.model = "fake-planner"

    def plan(self, *, context, frame_bgr=None):
        _ = frame_bgr
        self.calls.append(context)
        if not self._schedules:
            return ActionSchedule(commands=(), rationale="empty", model=self.model, issued_at_frame=context.frame_index)
        return self._schedules.pop(0)


class _FakeSensorBackend:
    def __init__(self) -> None:
        self.reset_calls: list[str] = []
        self.apply_calls: list[object] = []
        self.closed = False
        self._frame_index = 0

    def reset_episode(self, *, scene_spec) -> SensorFrame:
        self.reset_calls.append(scene_spec.scene_id)
        self._frame_index = 0
        return self._frame()

    def apply_action(self, command) -> SensorFrame:
        self.apply_calls.append(command)
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


class _AsyncFakeSensorBackend(_FakeSensorBackend):
    def __init__(self, *, pending_polls: int = 1) -> None:
        super().__init__()
        self._pending_polls = int(pending_polls)
        self._remaining_pending_polls = 0
        self._pending_frame: SensorFrame | None = None
        self._waiting = False

    def submit_action(self, command) -> None:
        self.apply_calls.append(command)
        self._frame_index += 1
        self._pending_frame = self._frame()
        self._remaining_pending_polls = self._pending_polls
        self._waiting = True

    def poll_action_frame(self, *, timeout_sec: float | None = 0.0) -> SensorFrame | None:
        _ = timeout_sec
        if not self._waiting:
            return None
        if self._remaining_pending_polls > 0:
            self._remaining_pending_polls -= 1
            return None
        frame = self._pending_frame
        self._pending_frame = None
        self._waiting = False
        return frame

    def is_waiting_for_frame(self) -> bool:
        return self._waiting


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


def _detection(*, label: str, xyxy: tuple[int, int, int, int], confidence: float = 0.92):
    return type(
        "Det",
        (),
        {
            "label": label,
            "xyxy": xyxy,
            "confidence": confidence,
        },
    )()


def _artifacts(
    packet: FramePacket,
    *,
    detections: list[object] | None = None,
    slam_tracking_state: str = "TRACKING",
    depth_m: float = 1.5,
    camera_pose_world: np.ndarray | None = None,
    tracked_feature_count: int = 0,
    median_reprojection_error: float | None = None,
):
    return type(
        "Artifacts",
        (),
        {
            "frame_bgr": np.asarray(packet.frame_bgr, dtype=np.uint8),
            "detections": list(detections or []),
            "segments": [],
            "scene_graph_snapshot": None,
            "mesh_vertices_xyz": np.zeros((4, 3), dtype=np.float32),
            "mesh_triangles": np.empty((0, 3), dtype=np.int32),
            "depth_map": np.full((12, 16), depth_m, dtype=np.float32),
            "slam_tracking_state": slam_tracking_state,
            "camera_pose_world": (
                np.asarray(camera_pose_world, dtype=np.float32)
                if camera_pose_world is not None
                else np.eye(4, dtype=np.float32)
            ),
            "tracked_feature_count": int(tracked_feature_count),
            "median_reprojection_error": median_reprojection_error,
        },
    )()


def _goal_artifacts(packet: FramePacket):
    return _artifacts(
        packet,
        detections=[_detection(label="tv", xyxy=(1, 1, 15, 11))],
        depth_m=1.6,
        camera_pose_world=np.eye(4, dtype=np.float32),
    )


def test_living_room_runtime_self_calibrates_with_explicit_macro_commands_before_planning(
    tmp_path: Path,
) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                commands=(
                    MotionCommand(
                        kind=CommandKind.TRANSLATE,
                        direction="forward",
                        mode="distance_m",
                        value=0.5,
                        intent="advance",
                    ),
                ),
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

    for _ in range(40):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))
        if planner.calls:
            break

    assert planner.calls
    assert all(context.phase != EpisodePhase.SELF_CALIBRATING for context in planner.calls)
    assert any(command.camera_yaw_delta_deg != 0.0 for command in backend.apply_calls)
    assert any(command.camera_pitch_delta_deg != 0.0 for command in backend.apply_calls)
    assert any(command.body_yaw_deg != 0.0 for command in backend.apply_calls)
    assert any(command.translate_forward_m != 0.0 for command in backend.apply_calls)
    assert any(command.translate_right_m != 0.0 for command in backend.apply_calls)


def test_living_room_runtime_self_calibration_executes_body_motion_while_tracking_initializes(
    tmp_path: Path,
) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                commands=(
                    MotionCommand(
                        kind=CommandKind.AIM_CAMERA,
                        yaw_deg=0.0,
                        pitch_deg=0.0,
                        intent="noop",
                    ),
                ),
                rationale="noop",
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

    for _ in range(40):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(
            frame_packet=packet,
            artifacts=_artifacts(packet, slam_tracking_state="INITIALIZING"),
        )
        if planner.calls:
            break

    assert planner.calls
    assert any(command.body_yaw_deg != 0.0 for command in backend.apply_calls)
    assert any(command.translate_forward_m != 0.0 for command in backend.apply_calls)
    assert any(command.translate_right_m != 0.0 for command in backend.apply_calls)


def test_living_room_runtime_recovers_from_empty_schedule_with_pause_fallback(tmp_path: Path) -> None:
    backend = _FakeSensorBackend()
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend=backend,
    )
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert backend.apply_calls[0].pause_sec == pytest.approx(0.5)
    assert runner.current_schedule is not None
    assert [command.kind for command in runner.current_schedule.commands] == [CommandKind.PAUSE]


def test_living_room_runtime_logs_target_detection_with_relative_xyz_and_camera_state(
    tmp_path: Path,
) -> None:
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner(
            [
                ActionSchedule(
                    commands=(
                        MotionCommand(
                            kind=CommandKind.TRANSLATE,
                            direction="forward",
                            mode="distance_m",
                            value=0.24,
                            intent="approach visible tv",
                        ),
                    ),
                    rationale="approach visible tv",
                    model="fake-planner",
                    issued_at_frame=0,
                )
            ]
        ),
        sensor_backend=_FakeSensorBackend(),
    )
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(
        frame_packet=packet,
        artifacts=_artifacts(packet, detections=[_detection(label="tv", xyxy=(5, 2, 9, 6))], depth_m=3.2),
    )

    turns = (tmp_path / "planner_turns.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(turns[-1])
    assert payload["prompt"]["target_detection"]["label"] == "tv"
    assert payload["prompt"]["target_detection"]["horizontal_position"] == "center"
    assert payload["prompt"]["target_detection"]["median_depth_m"] == pytest.approx(3.2)
    assert payload["prompt"]["target_detection"]["relative_xyz"] is not None
    assert payload["prompt"]["robot_state"]["current_camera_state"] == {"yaw_deg": 0.0, "pitch_deg": 0.0}


def test_living_room_runtime_marks_success_when_target_is_visually_reached(tmp_path: Path) -> None:
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


def test_living_room_runtime_logs_macro_actions_once_even_when_microstepped(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                commands=(
                    MotionCommand(
                        kind=CommandKind.TRANSLATE,
                        direction="forward",
                        mode="distance_m",
                        value=0.12,
                        intent="advance",
                    ),
                ),
                rationale="advance",
                model="fake-planner",
                issued_at_frame=0,
            ),
            ActionSchedule(
                commands=(MotionCommand(kind=CommandKind.PAUSE, duration_sec=0.5, intent="reassess"),),
                rationale="reassess",
                model="fake-planner",
                issued_at_frame=0,
            ),
        ]
    )
    backend = _FakeSensorBackend()
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=planner,
        sensor_backend=backend,
    )
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)

    for _ in range(2):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(
            frame_packet=packet,
            artifacts=_artifacts(packet, depth_m=0.6, camera_pose_world=np.eye(4, dtype=np.float32)),
        )

    assert len(planner.calls) >= 2
    assert backend.apply_calls[0].translate_forward_m == pytest.approx(0.08)
    effect = planner.calls[1].recent_action_effects[0]
    search = planner.calls[1].memory.recent_searches[0]
    assert effect.action == "translate:forward:distance_m:0.12"
    assert effect.likely_blocked is True
    assert effect.aborted is True
    assert search.executed_actions == ("translate:forward:distance_m:0.12",)
    assert search.likely_blocked is True
    assert search.pose_progress_m == pytest.approx(0.0)


def test_living_room_runtime_uses_escape_schedule_for_contradictory_commands(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                commands=(
                    MotionCommand(
                        kind=CommandKind.ROTATE_BODY,
                        direction="left",
                        mode="angle_deg",
                        value=6.0,
                        intent="scan left",
                    ),
                    MotionCommand(
                        kind=CommandKind.ROTATE_BODY,
                        direction="right",
                        mode="angle_deg",
                        value=6.0,
                        intent="scan right",
                    ),
                ),
                rationale="contradictory",
                model="fake-planner",
                issued_at_frame=0,
                situation_summary="Confused plan",
                behavior_mode="scan",
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
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet, depth_m=0.4))

    assert runner.current_schedule is not None
    assert runner.current_schedule.behavior_mode == "escape"
    assert runner.current_schedule.safety_flags is not None
    assert runner.current_schedule.safety_flags.replan_reason == "contradictory_actions"
    assert backend.apply_calls
    assert (
        backend.apply_calls[0].translate_forward_m < 0.0
        or backend.apply_calls[0].translate_right_m != 0.0
    )


def test_living_room_runtime_replaces_body_motion_with_pause_when_tracking_is_lost(
    tmp_path: Path,
) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                commands=(
                    MotionCommand(
                        kind=CommandKind.TRANSLATE,
                        direction="forward",
                        mode="distance_m",
                        value=0.12,
                        intent="advance",
                    ),
                ),
                rationale="advance",
                model="fake-planner",
                issued_at_frame=0,
                situation_summary="Move ahead",
                behavior_mode="approach",
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
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(
        frame_packet=packet,
        artifacts=_artifacts(packet, slam_tracking_state="LOST"),
    )

    assert runner._latest_planner_schedule is not None
    assert [command.kind for command in runner._latest_planner_schedule.commands] == [
        CommandKind.AIM_CAMERA,
        CommandKind.PAUSE,
    ]
    assert backend.apply_calls[0].pause_sec == pytest.approx(0.5)
    assert runner._latest_planner_schedule.safety_flags is not None
    assert runner._latest_planner_schedule.safety_flags.replan_reason == "tracking_risk"


def test_living_room_runtime_restricts_planner_allowed_command_kinds_when_tracking_is_initializing(
    tmp_path: Path,
) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                commands=(
                    MotionCommand(
                        kind=CommandKind.AIM_CAMERA,
                        yaw_deg=-12.0,
                        pitch_deg=0.0,
                        intent="scan left",
                    ),
                ),
                rationale="scan left",
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
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(
        frame_packet=packet,
        artifacts=_artifacts(packet, slam_tracking_state="INITIALIZING"),
    )

    assert planner.calls
    assert planner.calls[0].constraints.allowed_command_kinds == (
        CommandKind.AIM_CAMERA.value,
        CommandKind.PAUSE.value,
    )


def test_living_room_runtime_enables_bootstrap_body_motion_after_persistent_initializing(
    tmp_path: Path,
) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                commands=(
                    MotionCommand(
                        kind=CommandKind.AIM_CAMERA,
                        yaw_deg=-12.0,
                        pitch_deg=0.0,
                        intent="keep scanning",
                    ),
                ),
                rationale="scan only",
                model="fake-planner",
                issued_at_frame=0,
                situation_summary="Tracking is still bootstrapping",
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
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)
    runner._initializing_streak_frames = _INITIALIZING_RECOVERY_FRAME_THRESHOLD

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(
        frame_packet=packet,
        artifacts=_artifacts(packet, slam_tracking_state="INITIALIZING"),
    )

    assert planner.calls
    assert planner.calls[0].constraints.allowed_command_kinds == (
        CommandKind.TRANSLATE.value,
        CommandKind.ROTATE_BODY.value,
        CommandKind.AIM_CAMERA.value,
        CommandKind.PAUSE.value,
    )
    assert runner._latest_planner_schedule is not None
    assert runner._latest_planner_schedule.safety_flags is not None
    assert runner._latest_planner_schedule.safety_flags.replan_reason == "tracking_bootstrap"
    assert backend.apply_calls
    assert (
        backend.apply_calls[0].translate_forward_m != 0.0
        or backend.apply_calls[0].translate_right_m != 0.0
        or backend.apply_calls[0].body_yaw_deg != 0.0
    )


def test_living_room_runtime_converts_hold_sec_commands_using_rig_capabilities(
    tmp_path: Path,
) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                commands=(
                    MotionCommand(
                        kind=CommandKind.TRANSLATE,
                        direction="forward",
                        mode="hold_sec",
                        value=0.1,
                        intent="timed advance",
                    ),
                ),
                rationale="timed advance",
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
    runner._rig_capabilities = RigCapabilities(
        move_speed_mps=2.0,
        turn_speed_deg_per_sec=100.0,
        camera_yaw_speed_deg_per_sec=90.0,
        camera_pitch_speed_deg_per_sec=90.0,
        camera_yaw_limit_deg=70.0,
        camera_pitch_limit_deg=55.0,
    )
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert backend.apply_calls[0].translate_forward_m == pytest.approx(0.08)
    assert runner._active_macro_execution is not None
    assert runner._active_macro_execution.requested_translation_m == pytest.approx(0.2)


def test_living_room_runtime_writes_command_based_planner_turn_payload(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                commands=(
                    MotionCommand(
                        kind=CommandKind.AIM_CAMERA,
                        yaw_deg=-12.0,
                        pitch_deg=6.0,
                        intent="center target",
                    ),
                ),
                rationale="center target",
                model="fake-planner",
                issued_at_frame=0,
                behavior_mode="scan",
            )
        ]
    )
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=planner,
        sensor_backend=_FakeSensorBackend(),
    )
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    turns = (tmp_path / "planner_turns.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(turns[-1])
    assert "commands" in payload["schedule"]
    assert "steps" not in payload["schedule"]
    assert payload["schedule"]["commands"][0]["kind"] == "aim_camera"
    assert "allowed_command_kinds" in payload["prompt"]["constraints"]
    assert "max_commands_per_schedule" in payload["prompt"]["constraints"]


def test_living_room_runtime_writes_slam_diagnostics_into_planner_turn_payload(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                commands=(MotionCommand(kind=CommandKind.PAUSE, duration_sec=0.5, intent="wait"),),
                rationale="wait",
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
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(
        frame_packet=packet,
        artifacts=_artifacts(
            packet,
            tracked_feature_count=57,
            median_reprojection_error=1.25,
        ),
    )

    turns = (tmp_path / "planner_turns.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(turns[-1])
    assert payload["prompt"]["reconstruction"]["tracked_feature_count"] == 57
    assert payload["prompt"]["reconstruction"]["median_reprojection_error"] == pytest.approx(1.25)
    assert payload["prompt"]["legacy_summary"]["reconstruction_summary"]["tracked_feature_count"] == 57
    assert payload["prompt"]["legacy_summary"]["reconstruction_summary"]["median_reprojection_error"] == pytest.approx(
        1.25
    )


def test_living_room_runtime_promotes_async_backend_frame_without_reprocessing_stale_frame(
    tmp_path: Path,
) -> None:
    backend = _AsyncFakeSensorBackend(pending_polls=1)
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner([]),
        sensor_backend=backend,
    )

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert runner.is_waiting_for_frame() is True
    assert runner.next_frame(timeout_sec=0.0) is None

    next_packet = runner.next_frame(timeout_sec=0.0)

    assert next_packet is not None
    assert runner.is_waiting_for_frame() is False
    assert runner._state.frame_index == 1
    assert next_packet.timestamp_sec == pytest.approx(0.5)
    assert backend.apply_calls[0].camera_yaw_delta_deg == pytest.approx(-4.0)


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


def test_unity_rgb_sensor_backend_validates_vendor_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setattr("obj_recog.simulation.validate_unity_vendor_setup", lambda: calls.append("checked"))
    monkeypatch.setattr("obj_recog.simulation.UnityRgbClient", _FakeClient)

    backend = _build_unity_rgb_sensor_backend(
        config=_config(
            unity_player_path="/tmp/obj-recog-unity.app",
            unity_host="127.0.0.2",
            unity_port=9001,
        ),
        camera_rig=object(),
    )

    assert calls == ["checked"]
    assert backend._client.kwargs["unity_player_path"] == "/tmp/obj-recog-unity.app"
    assert backend._client.kwargs["host"] == "127.0.0.2"
    assert backend._client.kwargs["port"] == 9001


def test_unity_rgb_sensor_backend_requires_rig_capabilities_in_reset_response() -> None:
    class _ClientWithoutCapabilities:
        def reset_episode(self, *, scenario_id):
            _ = scenario_id
            return type(
                "Frame",
                (),
                {
                    "frame_bgr": np.full((12, 16, 3), 64, dtype=np.uint8),
                    "timestamp_sec": 0.0,
                    "metadata": {},
                },
            )()

        def close(self) -> None:
            return None

    from obj_recog.simulation import UnityRgbSensorBackend

    backend = UnityRgbSensorBackend(client=_ClientWithoutCapabilities())

    with pytest.raises(RuntimeError, match="did not advertise rig capabilities"):
        backend.reset_episode(scene_spec=type("SceneSpec", (), {"scene_id": "living_room_navigation_v1"})())
