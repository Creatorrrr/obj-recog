from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.scene_graph import GraphNode, SceneGraphSnapshot
from obj_recog.sim_protocol import ActionPrimitive, ActionSchedule, ActionStep, EpisodePhase, SensorFrame
from obj_recog.simulation import (
    LivingRoomEpisodeRunner,
    LivingRoomSimulationRuntime,
    _build_unity_rgb_sensor_backend,
)
from obj_recog.types import PanopticSegment


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


def _artifacts(
    packet: FramePacket,
    *,
    detections: list[object] | None = None,
    scene_graph_snapshot: SceneGraphSnapshot | None = None,
    slam_tracking_state: str = "TRACKING",
    depth_m: float = 1.5,
    camera_pose_world: np.ndarray | None = None,
    segments: list[PanopticSegment] | None = None,
):
    return type(
        "Artifacts",
        (),
        {
            "frame_bgr": np.asarray(packet.frame_bgr, dtype=np.uint8),
            "detections": list(detections or []),
            "segments": list(segments or []),
            "scene_graph_snapshot": scene_graph_snapshot,
            "mesh_vertices_xyz": np.zeros((4, 3), dtype=np.float32),
            "mesh_triangles": np.empty((0, 3), dtype=np.int32),
            "depth_map": np.full((12, 16), depth_m, dtype=np.float32),
            "slam_tracking_state": slam_tracking_state,
            "camera_pose_world": (
                np.asarray(camera_pose_world, dtype=np.float32)
                if camera_pose_world is not None
                else np.eye(4, dtype=np.float32)
            ),
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


def _visible_tv_artifacts(_packet: FramePacket):
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
                        "xyxy": (5, 2, 9, 6),
                        "confidence": 0.92,
                    },
                )()
            ],
            "segments": [],
            "scene_graph_snapshot": None,
            "mesh_vertices_xyz": np.zeros((4, 3), dtype=np.float32),
            "depth_map": np.full((12, 16), 3.2, dtype=np.float32),
            "slam_tracking_state": "TRACKING",
        },
    )()


def _memory_snapshot_for_target(
    *,
    target_label: str,
    frame_index: int,
    state: str = "occluded",
    last_seen_frame: int | None = None,
    last_seen_direction: str = "front-right",
) -> SceneGraphSnapshot:
    resolved_last_seen_frame = frame_index - 1 if last_seen_frame is None else last_seen_frame
    return SceneGraphSnapshot(
        frame_index=frame_index,
        camera_pose_world=np.eye(4, dtype=np.float32),
        nodes=(
            GraphNode(
                id="ego",
                type="ego",
                label="camera",
                state="visible",
                confidence=1.0,
                world_centroid=np.zeros(3, dtype=np.float32),
                last_seen_frame=frame_index,
                last_seen_direction="front",
                source_track_id=None,
            ),
            GraphNode(
                id="target-1",
                type="object",
                label=target_label,
                state=state,
                confidence=0.92,
                world_centroid=np.array([0.5, 0.0, 1.5], dtype=np.float32),
                last_seen_frame=resolved_last_seen_frame,
                last_seen_direction=last_seen_direction,
                source_track_id=1,
            ),
        ),
        edges=(),
        visible_node_ids=("ego",),
        visible_edge_keys=(),
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
    assert planner.calls[0].memory.recent_searches == ()
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


def test_living_room_runtime_logs_target_detection_summary_for_visible_tv(tmp_path: Path) -> None:
    runner = LivingRoomEpisodeRunner(
        config=_config(),
        report_path=tmp_path / "episode.json",
        planner=_FakePlanner(
            [
                ActionSchedule(
                    steps=(ActionStep(ActionPrimitive.MOVE_FORWARD, 0.24),),
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
    runner.record_runtime_observation(frame_packet=packet, artifacts=_visible_tv_artifacts(packet))

    turns = (tmp_path / "planner_turns.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(turns[-1])
    assert payload["prompt"]["target_detection"]["label"] == "tv"
    assert payload["prompt"]["target_detection"]["horizontal_position"] == "center"
    assert payload["prompt"]["target_detection"]["median_depth_m"] == pytest.approx(3.2)


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


def test_living_room_runtime_exposes_target_memory_on_first_planner_turn(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.MOVE_FORWARD, 0.12),),
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
    target_label = runner._scene_spec.semantic_target_class

    for _ in range(6):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(
            frame_packet=packet,
            artifacts=_artifacts(
                packet,
                scene_graph_snapshot=_memory_snapshot_for_target(
                    target_label=target_label,
                    frame_index=runner._state.frame_index,
                ),
            ),
        )

    assert planner.calls[0].memory.target_memory is not None
    assert planner.calls[0].memory.target_memory.label == target_label
    assert planner.calls[0].memory.recent_searches == ()


def test_living_room_runtime_includes_completed_search_in_next_planner_turn(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.MOVE_FORWARD, 0.12),),
                rationale="advance",
                model="fake-planner",
                issued_at_frame=0,
            ),
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.TURN_RIGHT, 6.0),),
                rationale="scan",
                model="fake-planner",
                issued_at_frame=0,
            ),
        ]
    )
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

    assert len(planner.calls) >= 2
    first_search = planner.calls[1].memory.recent_searches[0]
    assert planner.calls[0].memory.recent_searches == ()
    assert first_search.start_frame == planner.calls[0].frame_index
    assert first_search.end_frame == planner.calls[1].frame_index
    assert first_search.executed_actions == ("move_forward:0.12",)
    assert first_search.target_visible_after_search is False
    assert first_search.tracking_status == "TRACKING"


def test_living_room_runtime_records_chunked_actions_in_search_history(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.CAMERA_PAN_LEFT, 18.0),),
                rationale="wide scan",
                model="fake-planner",
                issued_at_frame=0,
            ),
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.PAUSE, 0.5),),
                rationale="pause",
                model="fake-planner",
                issued_at_frame=0,
            ),
        ]
    )
    runner = LivingRoomEpisodeRunner(
        config=_config(sim_action_batch_size=3),
        report_path=tmp_path / "episode.json",
        planner=planner,
        sensor_backend=_FakeSensorBackend(),
    )

    for _ in range(9):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert len(planner.calls) >= 2
    assert planner.calls[1].memory.recent_searches[0].executed_actions == (
        "camera_pan_left:6.0",
        "camera_pan_left:6.0",
        "camera_pan_left:6.0",
    )


def test_living_room_runtime_limits_search_history_to_four_items(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.MOVE_FORWARD, 0.12),),
                rationale=f"step-{index}",
                model="fake-planner",
                issued_at_frame=0,
            )
            for index in range(6)
        ]
    )
    runner = LivingRoomEpisodeRunner(
        config=_config(sim_action_batch_size=1),
        report_path=tmp_path / "episode.json",
        planner=planner,
        sensor_backend=_FakeSensorBackend(),
    )

    iterations = 0
    while len(planner.calls) < 6 and iterations < 20:
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))
        iterations += 1

    assert len(planner.calls) >= 6
    recent_searches = planner.calls[-1].memory.recent_searches
    assert len(recent_searches) == 4
    assert [item.start_frame for item in recent_searches] == [
        planner.calls[index].frame_index for index in range(1, 5)
    ]


def test_living_room_runtime_writes_planner_memory_to_turn_log(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.MOVE_FORWARD, 0.12),),
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
    target_label = runner._scene_spec.semantic_target_class

    for _ in range(6):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(
            frame_packet=packet,
            artifacts=_artifacts(
                packet,
                scene_graph_snapshot=_memory_snapshot_for_target(
                    target_label=target_label,
                    frame_index=runner._state.frame_index,
                ),
            ),
        )

    turns = (tmp_path / "planner_turns.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert turns
    payload = json.loads(turns[0])
    assert "memory" in payload["prompt"]
    assert payload["prompt"]["memory"]["target_memory"]["label"] == target_label
    assert payload["prompt"]["memory"]["recent_searches"] == []


def test_living_room_runtime_records_action_effects_and_search_metadata(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.MOVE_FORWARD, 0.12),),
                rationale="advance",
                model="fake-planner",
                issued_at_frame=0,
            ),
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.PAUSE, 0.5),),
                rationale="reassess",
                model="fake-planner",
                issued_at_frame=0,
            ),
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
        artifacts=_artifacts(packet, depth_m=0.6, camera_pose_world=np.eye(4, dtype=np.float32)),
    )

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(
        frame_packet=packet,
        artifacts=_artifacts(packet, depth_m=0.6, camera_pose_world=np.eye(4, dtype=np.float32)),
    )

    assert len(planner.calls) >= 2
    effect = planner.calls[1].recent_action_effects[0]
    search = planner.calls[1].memory.recent_searches[0]
    assert effect.action == "move_forward:0.12"
    assert effect.likely_blocked is True
    assert search.likely_blocked is True
    assert search.pose_progress_m == pytest.approx(0.0)
    assert search.entered_new_view is False
    assert search.evidence_gained is False


def test_living_room_runtime_applies_replan_frame_budget(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                steps=(
                    ActionStep(ActionPrimitive.TURN_LEFT, 6.0),
                    ActionStep(ActionPrimitive.TURN_LEFT, 6.0),
                    ActionStep(ActionPrimitive.MOVE_FORWARD, 0.12),
                ),
                rationale="scan then advance",
                model="fake-planner",
                issued_at_frame=0,
            ),
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.PAUSE, 0.5),),
                rationale="replan",
                model="fake-planner",
                issued_at_frame=0,
            ),
        ]
    )
    runner = LivingRoomEpisodeRunner(
        config=_config(sim_action_batch_size=4, sim_replan_interval_sec=0.4, sim_camera_fps=2.0),
        report_path=tmp_path / "episode.json",
        planner=planner,
        sensor_backend=_FakeSensorBackend(),
    )
    runner._state.phase = EpisodePhase.REASSESS
    runner._state.selfcal_step_index = len(runner._selfcal_actions)

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert runner.current_phase == EpisodePhase.REASSESS
    assert len(planner.calls) == 1

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert len(planner.calls) >= 2


def test_living_room_runtime_writes_enhanced_planner_turn_payload(tmp_path: Path) -> None:
    planner = _FakePlanner(
        [
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.MOVE_FORWARD, 0.12),),
                rationale="advance",
                model="fake-planner",
                issued_at_frame=0,
            ),
            ActionSchedule(
                steps=(ActionStep(ActionPrimitive.PAUSE, 0.5),),
                rationale="reassess",
                model="fake-planner",
                issued_at_frame=0,
            ),
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

    for _ in range(2):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(
            frame_packet=packet,
            artifacts=_artifacts(packet, depth_m=0.6, camera_pose_world=np.eye(4, dtype=np.float32)),
        )

    turns = (tmp_path / "planner_turns.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(turns[-1])
    assert "goal_estimate" in payload["prompt"]
    assert "navigation_affordances" in payload["prompt"]
    assert "recent_action_effects" in payload["prompt"]
    assert "constraints" in payload["prompt"]
    assert payload["prompt"]["recent_action_effects"][0]["likely_blocked"] is True
    assert payload["prompt"]["recent_searches"][0]["likely_blocked"] is True


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
        config=_config(unity_player_path="C:/UnityBuild/obj-recog.exe", unity_host="127.0.0.2", unity_port=9001),
        camera_rig=object(),
    )

    assert calls == ["checked"]
    assert backend._client.kwargs["unity_player_path"] == "C:/UnityBuild/obj-recog.exe"
    assert backend._client.kwargs["host"] == "127.0.0.2"
    assert backend._client.kwargs["port"] == 9001
