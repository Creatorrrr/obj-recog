from __future__ import annotations

import os

import numpy as np
import pytest

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.sim_planner import OpenAILivingRoomPlanner
from obj_recog.sim_protocol import ActionPrimitive, ActionSchedule, ActionStep, SensorFrame
from obj_recog.simulation import LivingRoomEpisodeRunner


class _StaticSensorBackend:
    def __init__(self) -> None:
        self._frame_index = 0
        self.commands: list[object] = []

    def reset_episode(self, *, scene_spec) -> SensorFrame:
        _ = scene_spec
        self._frame_index = 0
        return self._frame()

    def apply_action(self, command) -> SensorFrame:
        self.commands.append(command)
        self._frame_index += 1
        return self._frame()

    def close(self) -> None:
        return None

    def _frame(self) -> SensorFrame:
        return SensorFrame(
            frame_index=self._frame_index,
            timestamp_sec=float(self._frame_index) * 0.5,
            frame_bgr=np.full((12, 16, 3), 90, dtype=np.uint8),
            render_time_ms=1.0,
        )


def _artifacts(_packet: FramePacket):
    return type(
        "Artifacts",
        (),
        {
            "detections": [
                type(
                    "Det",
                    (),
                    {
                        "label": "dining_table",
                    },
                )()
            ],
            "segments": [],
            "scene_graph_snapshot": None,
            "mesh_vertices_xyz": np.zeros((4, 3), dtype=np.float32),
            "depth_map": np.full((12, 16), 1.7, dtype=np.float32),
            "slam_tracking_state": "TRACKING",
        },
    )()


class _FixedPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, *, context) -> ActionSchedule:
        self.calls += 1
        return ActionSchedule(
            steps=(
                ActionStep(ActionPrimitive.MOVE_FORWARD, 0.5),
                ActionStep(ActionPrimitive.TURN_LEFT, 12.0),
            ),
            rationale="advance carefully toward the visible TV",
            model="fake-planner",
            issued_at_frame=context.frame_index,
        )


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is required")
def test_real_llm_planner_returns_a_valid_schedule_in_episode_loop(tmp_path) -> None:
    planner = OpenAILivingRoomPlanner(model="gpt-5-mini", timeout_sec=10.0)
    runner = LivingRoomEpisodeRunner(
        config=AppConfig(
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
        ),
        report_path=tmp_path / "episode_report.json",
        planner=planner,
        sensor_backend=_StaticSensorBackend(),
    )

    for _ in range(8):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))
        if runner.current_schedule is not None:
            break

    assert runner.current_schedule is not None
    assert runner.current_schedule.steps


def test_episode_runner_executes_only_one_llm_step_before_replanning(tmp_path) -> None:
    planner = _FixedPlanner()
    backend = _StaticSensorBackend()
    runner = LivingRoomEpisodeRunner(
        config=AppConfig(
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
            sim_action_batch_size=1,
        ),
        report_path=tmp_path / "episode_report.json",
        planner=planner,
        sensor_backend=backend,
    )

    for _ in range(6):
        packet = runner.next_frame()
        assert packet is not None
        runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert planner.calls == 1
    assert len(backend.commands) == 6
    assert backend.commands[-1].primitive is ActionPrimitive.MOVE_FORWARD
    assert backend.commands[-1].value == pytest.approx(0.12)
    assert runner.current_schedule is None

    packet = runner.next_frame()
    assert packet is not None
    runner.record_runtime_observation(frame_packet=packet, artifacts=_artifacts(packet))

    assert planner.calls == 2
