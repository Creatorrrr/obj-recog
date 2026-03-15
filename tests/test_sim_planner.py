from __future__ import annotations

import json

import numpy as np
import pytest

from obj_recog.reconstruct import intrinsics_for_frame
from obj_recog.scene_graph import GraphEdge, GraphNode, SceneGraphSnapshot
from obj_recog.sim_planner import OpenAILivingRoomPlanner, build_planner_context, planner_prompt_from_context
from obj_recog.sim_protocol import (
    CommandKind,
    EpisodePhase,
    PlannerActionEffectSummary,
    PlannerCameraState,
    PlannerSearchOutcome,
)
from obj_recog.types import Detection, PanopticSegment


def _node(
    *,
    node_id: str,
    label: str,
    node_type: str = "object",
    state: str = "visible",
    last_seen_frame: int = 3,
    last_seen_direction: str | None = None,
) -> GraphNode:
    return GraphNode(
        id=node_id,
        type=node_type,
        label=label,
        state=state,
        confidence=0.93,
        world_centroid=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        last_seen_frame=last_seen_frame,
        last_seen_direction=last_seen_direction,
        source_track_id=1,
    )


def _graph_snapshot(
    *,
    nodes: tuple[GraphNode, ...],
    visible_node_ids: tuple[str, ...],
    edges: tuple[GraphEdge, ...] = (),
) -> SceneGraphSnapshot:
    return SceneGraphSnapshot(
        frame_index=8,
        camera_pose_world=np.eye(4, dtype=np.float32),
        nodes=nodes,
        edges=edges,
        visible_node_ids=visible_node_ids,
        visible_edge_keys=tuple((edge.source, edge.target, edge.relation) for edge in edges),
    )


def _ego_node() -> GraphNode:
    return GraphNode(
        id="ego",
        type="ego",
        label="ego",
        state="visible",
        confidence=1.0,
        world_centroid=None,
        last_seen_frame=8,
        last_seen_direction=None,
        source_track_id=None,
    )


def test_planner_context_redacts_hidden_pose_and_unseen_authored_objects() -> None:
    context = build_planner_context(
        phase=EpisodePhase.PERCEIVE_AND_PLAN,
        frame_index=8,
        detections=[
            Detection(
                xyxy=(1, 1, 10, 10),
                class_id=1,
                label="sofa",
                confidence=0.8,
                color=(0, 255, 0),
            )
        ],
        scene_graph_snapshot=_graph_snapshot(
            nodes=(
                _ego_node(),
                _node(node_id="chair-1", label="chair", last_seen_direction="front-right"),
            ),
            visible_node_ids=("ego",),
        ),
        reconstruction_summary={"mesh_vertices": 1200, "tracked_points": 480},
        depth_summary={"min_depth_m": 0.6, "median_depth_m": 1.8},
        recent_actions=("rotate_body:left:angle_deg:15.00",),
        calibration_status="converged",
        tracking_status="TRACKING",
    )
    prompt = planner_prompt_from_context(context)

    assert "pose_world_gt" not in prompt
    assert "hidden_goal_pose_xyz" not in prompt
    assert '"chair"' not in prompt
    assert '"sofa"' in prompt


def test_planner_context_includes_relative_xyz_camera_state_and_coordinate_convention() -> None:
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=18,
        detections=[
            Detection(
                xyxy=(1, 1, 7, 8),
                class_id=1,
                label="chair",
                confidence=0.84,
                color=(255, 255, 0),
            )
        ],
        scene_graph_snapshot=_graph_snapshot(nodes=(_ego_node(),), visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 1200, "tracked_points": 900, "mesh_triangles": 340},
        depth_summary={"min_depth_m": 0.7, "median_depth_m": 1.8},
        recent_actions=("translate:forward:distance_m:0.12",),
        calibration_status="converged",
        tracking_status="TRACKING",
        depth_map=np.full((10, 10), 1.8, dtype=np.float32),
        frame_shape=(10, 10),
        camera_intrinsics=intrinsics_for_frame(10, 10),
        current_camera_state=PlannerCameraState(yaw_deg=-12.0, pitch_deg=6.0),
        segments=(
            PanopticSegment(
                segment_id=1,
                label_id=1,
                label="floor",
                color_rgb=(0, 255, 0),
                mask=np.ones((10, 10), dtype=bool),
                area_pixels=100,
            ),
        ),
    )
    prompt_payload = json.loads(planner_prompt_from_context(context))

    assert context.perception.objects
    assert context.perception.objects[0].relative_xyz is not None
    assert prompt_payload["coordinate_convention"]["camera_frame"] == {
        "x": "right",
        "y": "up",
        "z": "forward",
    }
    assert prompt_payload["robot_state"]["current_camera_state"] == {
        "yaw_deg": -12.0,
        "pitch_deg": 6.0,
    }
    assert prompt_payload["objects"][0]["relative_xyz"] is not None


def test_planner_context_serializes_recent_searches_and_aborted_macro_effects() -> None:
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=17,
        detections=[],
        scene_graph_snapshot=_graph_snapshot(nodes=(_ego_node(),), visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 800, "tracked_points": 300},
        depth_summary={"min_depth_m": 0.7, "median_depth_m": 1.7},
        recent_actions=("rotate_body:left:angle_deg:6.00",),
        recent_searches=(
            PlannerSearchOutcome(
                start_frame=10,
                end_frame=13,
                executed_actions=(
                    "rotate_body:left:angle_deg:6.00",
                    "translate:forward:distance_m:0.12",
                ),
                target_visible_after_search=False,
                tracking_status="TRACKING",
            ),
        ),
        recent_action_effects=(
            PlannerActionEffectSummary(
                action="translate:forward:distance_m:0.12",
                estimated_motion="translation=0.00m",
                clearance_change="forward:stable",
                target_evidence_change="none",
                likely_blocked=True,
                aborted=True,
            ),
        ),
        calibration_status="converged",
        tracking_status="TRACKING",
    )
    prompt_payload = json.loads(planner_prompt_from_context(context))

    assert prompt_payload["recent_searches"][0]["executed_actions"] == [
        "rotate_body:left:angle_deg:6.00",
        "translate:forward:distance_m:0.12",
    ]
    assert prompt_payload["robot_state"]["recent_action_effects"][0]["aborted"] is True
    assert prompt_payload["robot_state"]["recent_action_effects"][0]["likely_blocked"] is True


def test_planner_context_uses_command_constraints() -> None:
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=12,
        detections=[],
        scene_graph_snapshot=_graph_snapshot(nodes=(_ego_node(),), visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 400, "tracked_points": 200},
        depth_summary={"min_depth_m": 0.8, "median_depth_m": 1.9},
        recent_actions=(),
        calibration_status="converged",
        tracking_status="TRACKING",
        max_commands_per_schedule=2,
        execution_capabilities={
            "move_speed_mps": 1.6,
            "turn_speed_deg_per_sec": 100.0,
            "camera_yaw_speed_deg_per_sec": 90.0,
            "camera_pitch_speed_deg_per_sec": 90.0,
            "camera_yaw_limit_deg": 70.0,
            "camera_pitch_limit_deg": 55.0,
            "max_translate_distance_m": 0.72,
            "max_rotate_body_deg": 36.0,
        },
        microstep_limits={
            "translate_distance_m": 0.08,
            "body_yaw_deg": 4.0,
            "camera_yaw_deg": 4.0,
            "camera_pitch_deg": 4.0,
        },
    )
    prompt_payload = json.loads(planner_prompt_from_context(context))

    assert context.constraints.allowed_command_kinds == tuple(item.value for item in CommandKind)
    assert context.constraints.max_commands_per_schedule == 2
    assert prompt_payload["constraints"]["allowed_command_kinds"] == [item.value for item in CommandKind]
    assert prompt_payload["constraints"]["max_commands_per_schedule"] == 2
    assert "absolute yaw/pitch setpoints" in " ".join(context.constraints.notes)


def test_planner_context_includes_target_detection_summary_when_target_is_visible() -> None:
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=14,
        detections=[
            Detection(
                xyxy=(2, 1, 12, 10),
                class_id=1,
                label="tv_panel",
                confidence=0.91,
                color=(255, 0, 0),
            )
        ],
        scene_graph_snapshot=_graph_snapshot(nodes=(_ego_node(),), visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 1500, "tracked_points": 520},
        depth_summary={"min_depth_m": 0.8, "median_depth_m": 2.0},
        recent_actions=("rotate_body:left:angle_deg:6.00",),
        target_label="tv_panel",
        calibration_status="converged",
        tracking_status="TRACKING",
        target_detection={
            "label": "tv_panel",
            "confidence": 0.91,
            "area_ratio": 0.12,
            "center_offset_ratio": 0.08,
            "horizontal_position": "center",
            "median_depth_m": 1.9,
            "relative_xyz": (0.1, -0.2, 1.9),
            "center_ray_xyz": (0.1, -0.2, 1.9),
        },
    )
    prompt_payload = json.loads(planner_prompt_from_context(context))

    assert context.perception.target_detection is not None
    assert prompt_payload["target_detection"]["relative_xyz"] == [0.1, -0.2, 1.9]
    assert prompt_payload["target_detection"]["center_ray_xyz"] == [0.1, -0.2, 1.9]


def test_openai_planner_sends_rgb_image_and_parses_structured_commands() -> None:
    class _FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            return type(
                "Response",
                (),
                {
                    "output_text": (
                        '{"situation_summary":"Target visible ahead.","goal_hypothesis":{"status":"visible",'
                        '"bearing_hint":"front","distance_hint":"1.8m","evidence":["target_detection"],'
                        '"confidence":0.94},"behavior_mode":"approach","commands":['
                        '{"kind":"rotate_body","direction":"left","mode":"angle_deg","value":18.0,"intent":"face target"},'
                        '{"kind":"aim_camera","yaw_deg":-12.0,"pitch_deg":6.0,"intent":"center view"},'
                        '{"kind":"translate","direction":"forward","mode":"distance_m","value":0.36,"intent":"close distance"}],'
                        '"safety_flags":{"front_blocked":false,"dead_end_risk":0.1,"tracking_risk":"low",'
                        '"replan_reason":"target_visible"},"confidence":0.91}'
                    )
                },
            )()

    fake_responses = _FakeResponses()
    client = type("Client", (), {"responses": fake_responses})()
    planner = OpenAILivingRoomPlanner(model="gpt-5-mini", timeout_sec=5.0, client=client)
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=23,
        detections=[],
        scene_graph_snapshot=_graph_snapshot(nodes=(_ego_node(),), visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 500, "tracked_points": 300},
        depth_summary={"min_depth_m": 0.8, "median_depth_m": 1.8},
        recent_actions=(),
        target_label="tv_panel",
        calibration_status="converged",
        tracking_status="TRACKING",
    )

    schedule = planner.plan(context=context, frame_bgr=np.full((8, 8, 3), 127, dtype=np.uint8))

    assert fake_responses.calls
    content = fake_responses.calls[0]["input"][0]["content"]
    assert any(item["type"] == "input_text" for item in content)
    assert any(item["type"] == "input_image" for item in content)
    assert schedule.situation_summary == "Target visible ahead."
    assert schedule.behavior_mode == "approach"
    assert schedule.goal_hypothesis is not None
    assert schedule.goal_hypothesis.status == "visible"
    assert [command.kind for command in schedule.commands] == [
        CommandKind.ROTATE_BODY,
        CommandKind.AIM_CAMERA,
        CommandKind.TRANSLATE,
    ]
    assert schedule.commands[0].intent == "face target"
    assert schedule.commands[1].yaw_deg == pytest.approx(-12.0)
    assert schedule.commands[2].value == pytest.approx(0.36)
