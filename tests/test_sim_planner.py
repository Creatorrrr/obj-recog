from __future__ import annotations

import numpy as np

from obj_recog.scene_graph import GraphEdge, GraphNode, SceneGraphSnapshot
from obj_recog.sim_planner import build_planner_context, planner_prompt_from_context
from obj_recog.sim_protocol import EpisodePhase
from obj_recog.types import Detection


def _graph_snapshot(*, visible_node_ids: tuple[str, ...]) -> SceneGraphSnapshot:
    return SceneGraphSnapshot(
        frame_index=3,
        camera_pose_world=np.eye(4, dtype=np.float32),
        nodes=(
            GraphNode(
                id="ego",
                type="ego",
                label="ego",
                state="visible",
                confidence=1.0,
                world_centroid=None,
                last_seen_frame=3,
                last_seen_direction=None,
                source_track_id=None,
            ),
            GraphNode(
                id="chair-1",
                type="object",
                label="chair",
                state="visible",
                confidence=0.93,
                world_centroid=np.array([1.0, 0.0, 1.0], dtype=np.float32),
                last_seen_frame=3,
                last_seen_direction="front-right",
                source_track_id=1,
            ),
        ),
        edges=(
            GraphEdge(
                source="ego",
                target="chair-1",
                relation="front-right-of",
                confidence=0.88,
                last_updated_frame=3,
                distance_bucket="mid",
                source_kind="detection",
            ),
        ),
        visible_node_ids=visible_node_ids,
        visible_edge_keys=(("ego", "chair-1", "front-right-of"),) if "chair-1" in visible_node_ids else (),
    )


def test_planner_context_redacts_hidden_pose_and_occluded_objects() -> None:
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
        scene_graph_snapshot=_graph_snapshot(visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 1200, "tracked_points": 480},
        depth_summary={"min_depth_m": 0.6, "median_depth_m": 1.8},
        recent_actions=("turn_left:15",),
        calibration_status="converged",
        tracking_status="TRACKING",
    )
    prompt = planner_prompt_from_context(context)

    assert "pose_world_gt" not in prompt
    assert "hidden_goal_pose_xyz" not in prompt
    assert "chair" not in prompt
    assert "sofa" in prompt


def test_planner_context_includes_visible_graph_nodes_once_they_are_seen() -> None:
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=11,
        detections=[],
        scene_graph_snapshot=_graph_snapshot(visible_node_ids=("ego", "chair-1")),
        reconstruction_summary={"mesh_vertices": 2200, "tracked_points": 980},
        depth_summary={"min_depth_m": 0.7, "median_depth_m": 2.1},
        recent_actions=("move_forward:0.5",),
        calibration_status="converged",
        tracking_status="TRACKING",
    )
    prompt = planner_prompt_from_context(context)

    assert "chair" in prompt
    assert "front-right-of" in prompt


def test_planner_context_uses_scene_specific_goal_description() -> None:
    context = build_planner_context(
        phase=EpisodePhase.PERCEIVE_AND_PLAN,
        frame_index=13,
        goal_description="Reach the front position of the TV using only current visible evidence.",
        detections=[],
        scene_graph_snapshot=_graph_snapshot(visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 640, "tracked_points": 220},
        depth_summary={"min_depth_m": 0.9, "median_depth_m": 2.6},
        recent_actions=("turn_right:12",),
        calibration_status="converged",
        tracking_status="TRACKING",
    )

    assert context.goal_description == "Reach the front position of the TV using only current visible evidence."
