from __future__ import annotations

import os

import numpy as np
import pytest

from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.sim_planner import OpenAILivingRoomPlanner
from obj_recog.sim_protocol import SensorFrame
from obj_recog.simulation import LivingRoomEpisodeRunner


class _StaticSensorBackend:
    def build_scene(self, scene_spec) -> None:
        self.scene_spec = scene_spec

    def render_frame(self, *, world_state, frame_index: int, timestamp_sec: float) -> SensorFrame:
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = world_state.robot_pose.x
        pose[1, 3] = world_state.robot_pose.y
        pose[2, 3] = world_state.robot_pose.z
        return SensorFrame(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            frame_bgr=np.full((12, 16, 3), 90, dtype=np.uint8),
            depth_map=np.full((12, 16), 1.7, dtype=np.float32),
            semantic_mask=np.zeros((12, 16), dtype=np.uint8),
            instance_mask=np.zeros((12, 16), dtype=np.uint8),
            camera_pose_world=pose,
            intrinsics={"fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 6.0},
            render_time_ms=1.0,
        )

    def close(self) -> None:
        return None


def _artifacts(packet: FramePacket):
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
            "depth_map": np.asarray(packet.depth_map, dtype=np.float32),
            "slam_tracking_state": "TRACKING",
        },
    )()


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
