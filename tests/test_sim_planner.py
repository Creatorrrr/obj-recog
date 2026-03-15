from __future__ import annotations

import numpy as np
import pytest

from obj_recog.scene_graph import GraphEdge, GraphNode, SceneGraphSnapshot
from obj_recog.sim_planner import build_planner_context, planner_prompt_from_context
from obj_recog.sim_protocol import EpisodePhase, PlannerActionEffectSummary, PlannerSearchOutcome
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
        recent_actions=("turn_left:15",),
        calibration_status="converged",
        tracking_status="TRACKING",
    )
    prompt = planner_prompt_from_context(context)

    assert "pose_world_gt" not in prompt
    assert "hidden_goal_pose_xyz" not in prompt
    assert '"chair"' not in prompt
    assert '"sofa"' in prompt


def test_planner_context_includes_visible_graph_nodes_once_they_are_seen() -> None:
    edge = GraphEdge(
        source="ego",
        target="chair-1",
        relation="front-right-of",
        confidence=0.88,
        last_updated_frame=8,
        distance_bucket="mid",
        source_kind="detection",
    )
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=11,
        detections=[],
        scene_graph_snapshot=_graph_snapshot(
            nodes=(
                _ego_node(),
                _node(node_id="chair-1", label="chair", last_seen_direction="front-right"),
            ),
            visible_node_ids=("ego", "chair-1"),
            edges=(edge,),
        ),
        reconstruction_summary={"mesh_vertices": 2200, "tracked_points": 980},
        depth_summary={"min_depth_m": 0.7, "median_depth_m": 2.1},
        recent_actions=("move_forward:0.5",),
        calibration_status="converged",
        tracking_status="TRACKING",
    )
    prompt = planner_prompt_from_context(context)

    assert 'ego->front-right-of->chair-1' in prompt


def test_planner_context_uses_scene_specific_goal_description() -> None:
    context = build_planner_context(
        phase=EpisodePhase.PERCEIVE_AND_PLAN,
        frame_index=13,
        goal_description="Reach the front position of the TV using only current visible evidence.",
        detections=[],
        scene_graph_snapshot=_graph_snapshot(nodes=(_ego_node(),), visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 640, "tracked_points": 220},
        depth_summary={"min_depth_m": 0.9, "median_depth_m": 2.6},
        recent_actions=("turn_right:12",),
        calibration_status="converged",
        tracking_status="TRACKING",
    )

    assert context.goal_description == "Reach the front position of the TV using only current visible evidence."


def test_planner_context_places_nonvisible_target_only_in_memory() -> None:
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=14,
        detections=[],
        scene_graph_snapshot=_graph_snapshot(
            nodes=(
                _ego_node(),
                _node(
                    node_id="tv-1",
                    label="tv_panel",
                    state="occluded",
                    last_seen_frame=12,
                    last_seen_direction="front-right",
                ),
            ),
            visible_node_ids=("ego",),
        ),
        reconstruction_summary={"mesh_vertices": 1500, "tracked_points": 520},
        depth_summary={"min_depth_m": 0.8, "median_depth_m": 2.0},
        recent_actions=("turn_left:6",),
        target_label="tv_panel",
        calibration_status="converged",
        tracking_status="TRACKING",
    )
    prompt = planner_prompt_from_context(context)

    assert context.perception.visible_detections == ()
    assert context.memory.target_memory is not None
    assert context.memory.target_memory.label == "tv_panel"
    assert context.memory.target_memory.state == "occluded"
    assert context.memory.target_memory.age_frames == 2
    assert context.goal_estimate.status == "remembered"
    assert '"target_memory"' in prompt
    assert '"visible_detections": []' in prompt


def test_planner_context_uses_current_target_visibility_over_memory() -> None:
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
        scene_graph_snapshot=_graph_snapshot(
            nodes=(
                _ego_node(),
                _node(
                    node_id="tv-1",
                    label="tv_panel",
                    state="occluded",
                    last_seen_frame=12,
                    last_seen_direction="left",
                ),
            ),
            visible_node_ids=("ego",),
        ),
        reconstruction_summary={"mesh_vertices": 1500, "tracked_points": 520},
        depth_summary={"min_depth_m": 0.8, "median_depth_m": 2.0},
        recent_actions=("turn_left:6",),
        target_label="tv_panel",
        calibration_status="converged",
        tracking_status="TRACKING",
    )

    assert context.perception.visible_detections == ("tv_panel",)
    assert context.memory.target_memory is None


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
        recent_actions=("turn_left:6",),
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
        },
    )
    prompt = planner_prompt_from_context(context)

    assert context.perception.target_detection is not None
    assert context.perception.target_detection["horizontal_position"] == "center"
    assert '"target_detection"' in prompt
    assert '"median_depth_m": 1.9' in prompt


def test_planner_context_excludes_floor_and_ceiling_from_nonvisible_memory() -> None:
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=10,
        detections=[],
        scene_graph_snapshot=_graph_snapshot(
            nodes=(
                _ego_node(),
                _node(node_id="seg-floor-1", label="floor", node_type="segment", state="occluded", last_seen_frame=9),
                _node(node_id="seg-wall-1", label="wall", node_type="segment", state="occluded", last_seen_frame=9),
                _node(node_id="seg-ceiling-1", label="ceiling", node_type="segment", state="lost", last_seen_frame=8),
            ),
            visible_node_ids=("ego",),
        ),
        reconstruction_summary={"mesh_vertices": 300, "tracked_points": 120},
        depth_summary={"min_depth_m": 0.5, "median_depth_m": 1.2},
        recent_actions=(),
        calibration_status="converged",
        tracking_status="TRACKING",
    )

    assert [item.label for item in context.memory.nonvisible_observations] == ["wall"]


def test_planner_context_serializes_recent_searches_without_raw_rationale() -> None:
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=17,
        detections=[],
        scene_graph_snapshot=_graph_snapshot(nodes=(_ego_node(),), visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 800, "tracked_points": 300},
        depth_summary={"min_depth_m": 0.7, "median_depth_m": 1.7},
        recent_actions=("turn_right:6",),
        recent_searches=(
            PlannerSearchOutcome(
                start_frame=10,
                end_frame=13,
                executed_actions=("turn_left:6", "move_forward:0.12"),
                target_visible_after_search=False,
                tracking_status="TRACKING",
            ),
        ),
        calibration_status="converged",
        tracking_status="TRACKING",
    )
    prompt = planner_prompt_from_context(context)

    assert '"recent_searches"' in prompt
    assert '"executed_actions": ["turn_left:6", "move_forward:0.12"]' in prompt
    assert '"target_visible_after_search": false' in prompt
    assert '"rationale"' not in prompt


def test_planner_context_serializes_structured_navigation_sections() -> None:
    mask = np.ones((10, 10), dtype=bool)
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
        scene_graph_snapshot=_graph_snapshot(
            nodes=(
                _ego_node(),
                _node(node_id="chair-1", label="chair", last_seen_direction="front-left"),
            ),
            visible_node_ids=("ego", "chair-1"),
            edges=(
                GraphEdge(
                    source="ego",
                    target="chair-1",
                    relation="front-left",
                    confidence=0.91,
                    last_updated_frame=18,
                    distance_bucket="mid",
                    source_kind="detection",
                ),
            ),
        ),
        reconstruction_summary={"mesh_vertices": 1200, "tracked_points": 900, "mesh_triangles": 340},
        depth_summary={"min_depth_m": 0.7, "median_depth_m": 1.8},
        recent_actions=("turn_left:6", "move_forward:0.12"),
        calibration_status="converged",
        tracking_status="TRACKING",
        depth_map=np.full((10, 10), 1.8, dtype=np.float32),
        frame_shape=(10, 10),
        segments=(
            PanopticSegment(
                segment_id=1,
                label_id=1,
                label="floor",
                color_rgb=(0, 255, 0),
                mask=mask,
                area_pixels=int(mask.sum()),
            ),
        ),
        recent_action_effects=(
            PlannerActionEffectSummary(
                action="move_forward:0.12",
                estimated_motion="translation=0.10m",
                clearance_change="forward:+0.12m",
                target_evidence_change="none",
                likely_blocked=False,
            ),
        ),
        batch_step_limit=3,
        replan_frame_budget=6,
    )
    prompt = planner_prompt_from_context(context)

    assert context.perception.objects
    assert context.perception.objects[0].depth_m == pytest.approx(1.8)
    assert context.perception.structural_segments[0].label == "floor"
    assert context.perception.navigation_affordances is not None
    assert context.constraints.batch_step_limit == 3
    assert context.recent_action_effects[0].action == "move_forward:0.12"
    assert '"objects"' in prompt
    assert '"segments"' in prompt
    assert '"navigation_affordances"' in prompt
    assert '"recent_action_effects"' in prompt
    assert '"constraints"' in prompt


def test_planner_context_goal_estimate_status_visible() -> None:
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=12,
        detections=[],
        scene_graph_snapshot=_graph_snapshot(nodes=(_ego_node(),), visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 400, "tracked_points": 200},
        depth_summary={"min_depth_m": 0.8, "median_depth_m": 1.9},
        recent_actions=(),
        target_label="tv_panel",
        calibration_status="converged",
        tracking_status="TRACKING",
        target_detection={"label": "tv_panel", "horizontal_position": "right", "median_depth_m": 2.1},
    )

    assert context.goal_estimate.status == "visible"
    assert context.goal_estimate.bearing_hint == "right"


def test_planner_context_goal_estimate_status_inferred_for_tv_cues() -> None:
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=12,
        detections=[
            Detection(
                xyxy=(1, 1, 7, 8),
                class_id=1,
                label="sofa",
                confidence=0.92,
                color=(255, 0, 0),
            )
        ],
        scene_graph_snapshot=_graph_snapshot(nodes=(_ego_node(),), visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 600, "tracked_points": 300},
        depth_summary={"min_depth_m": 0.8, "median_depth_m": 2.1},
        recent_actions=(),
        target_label="tv_panel",
        calibration_status="converged",
        tracking_status="TRACKING",
        depth_map=np.full((10, 10), 2.2, dtype=np.float32),
        frame_shape=(10, 10),
        segments=(
            PanopticSegment(
                segment_id=1,
                label_id=1,
                label="wall",
                color_rgb=(255, 255, 255),
                mask=np.ones((10, 10), dtype=bool),
                area_pixels=100,
            ),
        ),
    )

    assert context.goal_estimate.status == "inferred"
    assert "object:sofa" in context.goal_estimate.evidence_sources


def test_planner_context_goal_estimate_status_unknown_without_target_cues() -> None:
    context = build_planner_context(
        phase=EpisodePhase.REASSESS,
        frame_index=12,
        detections=[],
        scene_graph_snapshot=_graph_snapshot(nodes=(_ego_node(),), visible_node_ids=("ego",)),
        reconstruction_summary={"mesh_vertices": 200, "tracked_points": 80},
        depth_summary={"min_depth_m": 0.9, "median_depth_m": 2.8},
        recent_actions=(),
        target_label="lamp",
        calibration_status="converged",
        tracking_status="TRACKING",
        depth_map=np.full((10, 10), 2.8, dtype=np.float32),
        frame_shape=(10, 10),
    )

    assert context.goal_estimate.status == "unknown"
