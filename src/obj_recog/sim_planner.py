from __future__ import annotations

import base64
import json
import os
from collections import Counter
from typing import Iterable

import numpy as np

from obj_recog.reconstruct import CameraIntrinsics, back_project_pixels
from obj_recog.scene_graph import GraphNode, SceneGraphSnapshot
from obj_recog.sim_protocol import (
    ActionSchedule,
    CommandKind,
    EpisodePhase,
    MotionCommand,
    PerceptionSnapshot,
    PlannerActionEffectSummary,
    PlannerCameraState,
    PlannerConstraintSummary,
    PlannerContext,
    PlannerGoalCompletion,
    PlannerGoalEstimate,
    PlannerMemoryObservation,
    PlannerMemorySnapshot,
    PlannerNavigationAffordances,
    PlannerNavigationSectorObservation,
    PlannerObjectObservation,
    PlannerReconstructionBrief,
    PlannerSafetyFlags,
    PlannerSceneEdgeObservation,
    PlannerSceneNodeObservation,
    PlannerSearchOutcome,
    PlannerSegmentObservation,
)
from obj_recog.types import Detection, PanopticSegment


_PLANNER_SYSTEM_INSTRUCTIONS = (
    "You are a multimodal navigation planner for a wheeled indoor robot operating from the current egocentric RGB image, "
    "depth-derived navigation sectors, segmentation summaries, scene-graph memory, current camera yaw/pitch state, "
    "camera-frame 3D object coordinates, and short-term action history. "
    "Return JSON only. "
    "Use only planner-visible evidence from the prompt. Never invent hidden coordinates, authored scene truth, "
    "or unseen furniture. "
    "Allowed command kinds: "
    + ", ".join(item.value for item in CommandKind)
    + ". "
    "Use the coordinate convention camera-frame = +x right, +y up, +z forward. "
    "For aim_camera, yaw_deg > 0 means right, yaw_deg < 0 means left, pitch_deg > 0 means up, pitch_deg < 0 means down. "
    "aim_camera sets an absolute camera yaw/pitch target, not a relative pan. "
    "rotate_body is always relative body yaw, and translate is always relative body-frame motion. "
    "Judge final goal completion from the current evidence, not from hidden ground truth. "
    "Use detections, segmentation labels, scene-graph memory, relative_xyz, depth hints, and recent action effects together. "
    "If the goal already appears achieved, set goal_completion.reached=true, explain why in goal_completion.rationale, "
    "and do not request extra motion. "
    "Behavior modes: "
    "If goal_hypothesis.status is visible, center the target first and then approach it. "
    "Do not mix opposing body rotations or contradictory camera aims in the same batch. "
    "If goal_hypothesis.status is remembered, search using the remembered bearing and avoid repeating a recently "
    "failed batch when better options exist. "
    "If goal_hypothesis.status is inferred, infer promising target regions from visible cues such as chairs, sofas, "
    "walls, consoles, shelves, cabinets, open floor, visible relations, and relative_xyz. "
    "If recent_action_effects or navigation_map indicate blockage, dead end, or no progress, switch to "
    "escape mode: include translation or rotation plus camera aim, and do not repeat pure forward pushes. "
    "Return at most 3 commands. Favor committed batches over oscillating. "
    "Use aim_camera to set a clear yaw/pitch target, and pair scanning with viewpoint change when recent pure scans failed. "
    "Respect constraints.allowed_command_kinds, constraints.execution_capabilities, constraints.microstep_limits, "
    "and constraints.max_commands_per_schedule. "
    "Output keys must be exactly situation_summary, goal_hypothesis, goal_completion, behavior_mode, commands, safety_flags, confidence. "
    "Each command must include kind and intent, with translate/rotate_body using direction+mode+value, "
    "aim_camera using yaw_deg+pitch_deg, and pause using duration_sec."
)

_TV_CUE_LABELS = {
    "console",
    "tv_console",
    "media_console",
    "cabinet",
    "shelf",
    "sofa",
    "couch",
    "wall",
    "window-like",
}
_TABLE_CUE_LABELS = {
    "chair",
    "stool",
    "bench",
    "floor",
    "table",
}
_OPENING_LABELS = {"floor", "door", "window-like"}


def _normalize_label(label: object) -> str:
    return str(label or "").strip().lower().replace(" ", "_")


def _label_matches_target(*, detection_label: object, target_label: str) -> bool:
    normalized_detection = _normalize_label(detection_label)
    normalized_target = _normalize_label(target_label)
    if not normalized_detection or not normalized_target:
        return False
    return (
        normalized_detection == normalized_target
        or normalized_target in normalized_detection
        or normalized_detection in normalized_target
    )


def _visible_target_present(
    *,
    detections: Iterable[Detection],
    scene_graph_snapshot: SceneGraphSnapshot | None,
    target_label: str,
) -> bool:
    if not str(target_label).strip():
        return False
    for detection in detections:
        if _label_matches_target(detection_label=getattr(detection, "label", ""), target_label=target_label):
            return True
    if scene_graph_snapshot is None:
        return False
    for node in scene_graph_snapshot.visible_nodes:
        if node.id == "ego":
            continue
        if _label_matches_target(detection_label=node.label, target_label=target_label):
            return True
    return False


def _memory_observation_from_node(node: GraphNode, *, frame_index: int) -> PlannerMemoryObservation:
    return PlannerMemoryObservation(
        label=str(node.label),
        kind=str(node.type),
        state=str(node.state),
        last_seen_direction=node.last_seen_direction,
        age_frames=max(0, int(frame_index) - int(node.last_seen_frame)),
    )


def _planner_memory_snapshot(
    *,
    frame_index: int,
    detections: Iterable[Detection],
    scene_graph_snapshot: SceneGraphSnapshot | None,
    target_label: str,
    recent_searches: Iterable[PlannerSearchOutcome],
) -> PlannerMemorySnapshot:
    if scene_graph_snapshot is None:
        return PlannerMemorySnapshot(
            target_memory=None,
            nonvisible_observations=(),
            recent_searches=tuple(recent_searches),
        )

    recent_searches_tuple = tuple(recent_searches)
    visible_target_present = _visible_target_present(
        detections=detections,
        scene_graph_snapshot=scene_graph_snapshot,
        target_label=target_label,
    )
    target_candidates: list[GraphNode] = []
    nonvisible_by_kind_label: dict[tuple[str, str], GraphNode] = {}
    for node in scene_graph_snapshot.nodes:
        if node.id == "ego" or str(node.state) not in {"occluded", "lost"}:
            continue
        if _label_matches_target(detection_label=node.label, target_label=target_label):
            target_candidates.append(node)
            continue
        if node.type == "segment" and _normalize_label(node.label) in {"floor", "ceiling"}:
            continue
        key = (str(node.type), str(node.label))
        previous = nonvisible_by_kind_label.get(key)
        if previous is None or int(node.last_seen_frame) > int(previous.last_seen_frame):
            nonvisible_by_kind_label[key] = node

    target_memory = None
    if not visible_target_present and target_candidates:
        latest_target = max(target_candidates, key=lambda item: (int(item.last_seen_frame), float(item.confidence)))
        target_memory = _memory_observation_from_node(latest_target, frame_index=frame_index)

    nonvisible_nodes = sorted(
        nonvisible_by_kind_label.values(),
        key=lambda item: (int(item.last_seen_frame), float(item.confidence)),
        reverse=True,
    )[:6]
    nonvisible_observations = tuple(
        _memory_observation_from_node(node, frame_index=frame_index)
        for node in nonvisible_nodes
    )
    return PlannerMemorySnapshot(
        target_memory=target_memory,
        nonvisible_observations=nonvisible_observations,
        recent_searches=recent_searches_tuple,
    )


def _frame_shape_from_inputs(
    *,
    frame_shape: tuple[int, int] | None,
    depth_map: np.ndarray | None,
) -> tuple[int, int]:
    if frame_shape is not None and len(frame_shape) >= 2:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if height > 0 and width > 0:
            return height, width
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim == 2 and depth.shape[0] > 0 and depth.shape[1] > 0:
        return int(depth.shape[0]), int(depth.shape[1])
    return 0, 0


def _bbox_depth_m(depth_map: np.ndarray | None, xyxy: tuple[int, int, int, int]) -> float | None:
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2 or depth.size == 0:
        return None
    height, width = depth.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in xyxy)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    crop = depth[y1:y2, x1:x2]
    valid = np.isfinite(crop) & (crop > 0.05)
    if not np.any(valid):
        return None
    return float(np.nanmedian(crop[valid]))


def _distance_bucket(depth_m: float | None) -> str | None:
    if depth_m is None:
        return None
    if float(depth_m) <= 1.5:
        return "near"
    if float(depth_m) <= 3.5:
        return "mid"
    return "far"


def _relative_xyz(
    *,
    world_centroid: np.ndarray | None,
    camera_pose_world: np.ndarray | None,
) -> tuple[float, float, float] | None:
    centroid = np.asarray(world_centroid, dtype=np.float32)
    pose = np.asarray(camera_pose_world, dtype=np.float32)
    if centroid.shape != (3,) or pose.shape != (4, 4):
        return None
    pose_inv = np.linalg.inv(pose)
    homogeneous = np.concatenate((centroid, np.array([1.0], dtype=np.float32)))
    camera_xyz = pose_inv @ homogeneous
    return tuple(round(float(value), 3) for value in camera_xyz[:3])


def _bbox_relative_xyz(
    *,
    depth_map: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    xyxy: tuple[int, int, int, int],
) -> tuple[float, float, float] | None:
    depth_m = _bbox_depth_m(depth_map, xyxy)
    if depth_m is None or intrinsics is None:
        return None
    x1, y1, x2, y2 = (int(value) for value in xyxy)
    center_xy = np.asarray([[(x1 + x2) * 0.5, (y1 + y2) * 0.5]], dtype=np.float32)
    center_xyz = back_project_pixels(center_xy, np.asarray([depth_m], dtype=np.float32), intrinsics)
    if center_xyz.shape != (1, 3):
        return None
    return tuple(round(float(value), 3) for value in center_xyz[0])


def _direction_from_center(center_x_ratio: float) -> str:
    if center_x_ratio <= 0.18:
        return "left"
    if center_x_ratio <= 0.38:
        return "front-left"
    if center_x_ratio < 0.62:
        return "front"
    if center_x_ratio < 0.82:
        return "front-right"
    return "right"


def _bbox_area_ratio(
    *,
    xyxy: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> float:
    x1, y1, x2, y2 = (int(value) for value in xyxy)
    bbox_width = max(1, x2 - x1)
    bbox_height = max(1, y2 - y1)
    return float(bbox_width * bbox_height) / float(max(1, frame_width * frame_height))


def _object_observations_from_detections(
    *,
    detections: Iterable[Detection],
    depth_map: np.ndarray | None,
    frame_shape: tuple[int, int] | None,
    camera_intrinsics: CameraIntrinsics | None,
    target_label: str,
    limit: int = 8,
) -> tuple[PlannerObjectObservation, ...]:
    frame_height, frame_width = _frame_shape_from_inputs(frame_shape=frame_shape, depth_map=depth_map)
    if frame_height <= 0 or frame_width <= 0:
        return ()
    observations: list[tuple[float, float, PlannerObjectObservation]] = []
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in getattr(detection, "xyxy", (0, 0, 0, 0)))
        area_ratio = _bbox_area_ratio(
            xyxy=(x1, y1, x2, y2),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        center_x_ratio = ((x1 + x2) * 0.5) / float(max(1, frame_width))
        confidence = float(getattr(detection, "confidence", 0.0))
        depth_m = _bbox_depth_m(depth_map, (x1, y1, x2, y2))
        observation = PlannerObjectObservation(
            label=str(getattr(detection, "label", "")),
            confidence=confidence,
            direction=_direction_from_center(center_x_ratio),
            bbox_area_ratio=area_ratio,
            depth_m=depth_m,
            is_target_match=_label_matches_target(
                detection_label=getattr(detection, "label", ""),
                target_label=target_label,
            ),
            distance_bucket=_distance_bucket(depth_m),
            relative_xyz=_bbox_relative_xyz(
                depth_map=depth_map,
                intrinsics=camera_intrinsics,
                xyxy=(x1, y1, x2, y2),
            ),
        )
        observations.append((area_ratio, confidence, observation))
    observations.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return tuple(item[2] for item in observations[:max(1, int(limit))])


def _dominant_segment_direction(mask: np.ndarray) -> str:
    if mask.ndim != 2 or mask.size == 0:
        return "front"
    pixel_yx = np.argwhere(mask)
    if pixel_yx.size == 0:
        return "front"
    mean_x = float(np.mean(pixel_yx[:, 1]))
    width = max(1, int(mask.shape[1]))
    return _direction_from_center(mean_x / float(width))


def _segment_opening_hint(segment: PanopticSegment, *, coverage_ratio: float) -> str | None:
    label = _normalize_label(segment.label)
    if label == "floor":
        return "walkable_floor"
    if label in {"door", "window_like", "window-like"}:
        return "possible_opening"
    if coverage_ratio >= 0.35 and label in {"wall", "cabinet", "shelf"}:
        return "broad_obstacle"
    if coverage_ratio <= 0.08 and label in _OPENING_LABELS:
        return "thin_open_boundary"
    return None


def _segment_observations_from_segments(
    *,
    segments: Iterable[PanopticSegment],
    frame_shape: tuple[int, int] | None,
    limit: int = 6,
) -> tuple[PlannerSegmentObservation, ...]:
    frame_height, frame_width = _frame_shape_from_inputs(frame_shape=frame_shape, depth_map=None)
    if frame_height <= 0 or frame_width <= 0:
        return ()
    observations: list[tuple[float, PlannerSegmentObservation]] = []
    for segment in segments:
        area_pixels = int(getattr(segment, "area_pixels", 0))
        if area_pixels <= 0:
            continue
        coverage_ratio = float(area_pixels) / float(max(1, frame_width * frame_height))
        observation = PlannerSegmentObservation(
            label=str(getattr(segment, "label", "")),
            coverage_ratio=coverage_ratio,
            dominant_direction=_dominant_segment_direction(np.asarray(segment.mask, dtype=bool)),
            opening_hint=_segment_opening_hint(segment, coverage_ratio=coverage_ratio),
        )
        observations.append((coverage_ratio, observation))
    observations.sort(key=lambda item: item[0], reverse=True)
    return tuple(item[1] for item in observations[:max(1, int(limit))])


def _scene_node_observations(
    *,
    scene_graph_snapshot: SceneGraphSnapshot | None,
    limit: int = 12,
) -> tuple[PlannerSceneNodeObservation, ...]:
    if scene_graph_snapshot is None:
        return ()
    visible_ids = set(scene_graph_snapshot.visible_node_ids)
    observations: list[PlannerSceneNodeObservation] = []
    for node in scene_graph_snapshot.nodes:
        if node.id == "ego":
            continue
        if node.id not in visible_ids and str(node.state) == "visible":
            continue
        relative_xyz = _relative_xyz(
            world_centroid=node.world_centroid,
            camera_pose_world=scene_graph_snapshot.camera_pose_world,
        )
        depth_m = None if relative_xyz is None else max(0.0, float(relative_xyz[2]))
        observations.append(
            PlannerSceneNodeObservation(
                label=str(node.label),
                kind=str(node.type),
                state=str(node.state),
                bearing=node.last_seen_direction,
                distance_bucket=_distance_bucket(depth_m),
                confidence=float(node.confidence),
                relative_xyz=relative_xyz,
            )
        )
    observations.sort(
        key=lambda item: (
            0 if item.state == "visible" else 1,
            0 if item.kind == "object" else 1,
            -float(item.confidence),
        )
    )
    return tuple(observations[: max(1, int(limit))])


def _scene_edge_observations(
    *,
    scene_graph_snapshot: SceneGraphSnapshot | None,
    limit: int = 16,
) -> tuple[PlannerSceneEdgeObservation, ...]:
    if scene_graph_snapshot is None:
        return ()
    node_labels = {node.id: str(node.label) for node in scene_graph_snapshot.nodes}
    observations = [
        PlannerSceneEdgeObservation(
            source_label=node_labels.get(edge.source, str(edge.source)),
            target_label=node_labels.get(edge.target, str(edge.target)),
            relation=str(edge.relation),
            confidence=float(edge.confidence),
            source_kind=str(edge.source_kind),
            distance_bucket=edge.distance_bucket,
        )
        for edge in scene_graph_snapshot.edges
    ]
    observations.sort(key=lambda item: (-float(item.confidence), item.relation))
    return tuple(observations[: max(1, int(limit))])


def _sector_clearance(depth_map: np.ndarray | None, *, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> float | None:
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2 or depth.size == 0:
        return None
    height, width = depth.shape[:2]
    x1 = int(max(0, min(width - 1, round(width * float(x_bounds[0])))))
    x2 = int(max(x1 + 1, min(width, round(width * float(x_bounds[1])))))
    y1 = int(max(0, min(height - 1, round(height * float(y_bounds[0])))))
    y2 = int(max(y1 + 1, min(height, round(height * float(y_bounds[1])))))
    crop = depth[y1:y2, x1:x2]
    valid = crop[np.isfinite(crop) & (crop > 0.05)]
    if valid.size == 0:
        return None
    return float(np.nanpercentile(valid, 30))


def _sector_specs() -> tuple[tuple[str, tuple[float, float], tuple[float, float]], ...]:
    return (
        ("left", (0.00, 0.22), (0.45, 0.95)),
        ("front-left", (0.18, 0.42), (0.42, 0.90)),
        ("front", (0.35, 0.65), (0.42, 0.95)),
        ("front-right", (0.58, 0.82), (0.42, 0.90)),
        ("right", (0.78, 1.00), (0.45, 0.95)),
    )


def _sector_map_from_depth(
    *,
    depth_map: np.ndarray | None,
    recent_failures: Iterable[str] = (),
) -> tuple[PlannerNavigationSectorObservation, ...]:
    failed = {str(item) for item in recent_failures if str(item)}
    observations: list[PlannerNavigationSectorObservation] = []
    for sector_name, x_bounds, y_bounds in _sector_specs():
        clearance_m = _sector_clearance(depth_map, x_bounds=x_bounds, y_bounds=y_bounds)
        traversable = clearance_m is not None and clearance_m >= 0.95
        obstacle_likelihood = 1.0
        frontier_score = 0.0
        if clearance_m is not None:
            obstacle_likelihood = float(np.clip(1.0 - (float(clearance_m) / 2.5), 0.0, 1.0))
            frontier_score = float(np.clip(float(clearance_m) / 2.5, 0.0, 1.0))
        observations.append(
            PlannerNavigationSectorObservation(
                sector=sector_name,
                clearance_m=clearance_m,
                traversable=bool(traversable),
                obstacle_likelihood=obstacle_likelihood,
                frontier_score=frontier_score,
                recently_failed=sector_name in failed,
            )
        )
    return tuple(observations)


def _default_navigation_affordances(
    *,
    depth_map: np.ndarray | None,
    segment_observations: Iterable[PlannerSegmentObservation],
) -> PlannerNavigationAffordances:
    sector_map = _sector_map_from_depth(depth_map=depth_map)
    forward_clearance_m = _sector_clearance(depth_map, x_bounds=(0.35, 0.65), y_bounds=(0.45, 0.95))
    left_clearance_m = _sector_clearance(depth_map, x_bounds=(0.0, 0.35), y_bounds=(0.45, 0.95))
    right_clearance_m = _sector_clearance(depth_map, x_bounds=(0.65, 1.0), y_bounds=(0.45, 0.95))
    direction_scores = {
        "front": -1.0 if forward_clearance_m is None else float(forward_clearance_m),
        "left": -1.0 if left_clearance_m is None else float(left_clearance_m),
        "right": -1.0 if right_clearance_m is None else float(right_clearance_m),
    }
    candidate_open_directions = tuple(
        direction
        for direction, clearance in sorted(direction_scores.items(), key=lambda item: item[1], reverse=True)
        if clearance >= 1.15
    )
    front_blocked = forward_clearance_m is not None and forward_clearance_m < 0.8
    side_clearances = [value for value in (left_clearance_m, right_clearance_m) if value is not None]
    side_best = max(side_clearances) if side_clearances else None
    dead_end_likelihood = 0.15
    if front_blocked and (side_best is None or side_best < 0.85):
        dead_end_likelihood = 0.9
    elif front_blocked:
        dead_end_likelihood = 0.55
    elif not candidate_open_directions:
        dead_end_likelihood = 0.45
    if any(item.opening_hint == "possible_opening" for item in segment_observations):
        dead_end_likelihood = max(0.0, dead_end_likelihood - 0.15)
    best_exploration_direction = None
    if candidate_open_directions:
        best_exploration_direction = candidate_open_directions[0]
    else:
        best_exploration_direction = max(direction_scores.items(), key=lambda item: item[1])[0]
    return PlannerNavigationAffordances(
        forward_clearance_m=forward_clearance_m,
        left_clearance_m=left_clearance_m,
        right_clearance_m=right_clearance_m,
        rear_clearance_m=None,
        front_blocked=bool(front_blocked),
        candidate_open_directions=candidate_open_directions,
        dead_end_likelihood=float(np.clip(dead_end_likelihood, 0.0, 1.0)),
        best_exploration_direction=best_exploration_direction,
        sector_map=sector_map,
        recently_failed_directions=(),
    )


def _default_reconstruction_brief(
    *,
    reconstruction_summary: dict[str, float | int],
    navigation_affordances: PlannerNavigationAffordances | None,
) -> PlannerReconstructionBrief:
    mesh_vertex_count = int(reconstruction_summary.get("mesh_vertices", reconstruction_summary.get("tracked_points", 0)) or 0)
    mesh_triangle_count = int(reconstruction_summary.get("mesh_triangles", 0) or 0)
    mesh_growth_delta = int(reconstruction_summary.get("mesh_growth_delta", 0) or 0)
    frontier_directions = ()
    if navigation_affordances is not None:
        frontier_directions = tuple(navigation_affordances.candidate_open_directions[:3])
    explored_directions = ()
    unexplored_directions = ()
    recently_failed_directions = ()
    if navigation_affordances is not None:
        explored_directions = tuple(
            item.sector
            for item in navigation_affordances.sector_map
            if item.clearance_m is not None
        )
        unexplored_directions = tuple(
            item.sector
            for item in navigation_affordances.sector_map
            if item.clearance_m is None or item.frontier_score >= 0.45
        )
        recently_failed_directions = tuple(navigation_affordances.recently_failed_directions)
    return PlannerReconstructionBrief(
        pose_delta_m=None,
        yaw_delta_deg=None,
        mesh_vertex_count=mesh_vertex_count,
        mesh_triangle_count=mesh_triangle_count,
        mesh_growth_delta=mesh_growth_delta,
        frontier_directions=frontier_directions,
        explored_directions=explored_directions,
        unexplored_directions=unexplored_directions,
        recently_failed_directions=recently_failed_directions,
        tracked_feature_count=int(reconstruction_summary.get("tracked_feature_count", 0) or 0),
        median_reprojection_error=(
            None
            if reconstruction_summary.get("median_reprojection_error") is None
            else float(reconstruction_summary["median_reprojection_error"])
        ),
    )


def _relation_bearing_votes(visible_graph_relations: Iterable[str]) -> list[str]:
    votes: list[str] = []
    for relation in visible_graph_relations:
        relation_text = str(relation)
        for direction in ("front-right", "front-left", "front", "right", "left"):
            if direction in relation_text:
                votes.append(direction)
                break
    return votes


def _most_common_direction(candidates: Iterable[str]) -> str | None:
    filtered = [str(item) for item in candidates if str(item)]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def _target_family(target_label: str) -> str:
    normalized_target = _normalize_label(target_label)
    if "tv" in normalized_target or "monitor" in normalized_target:
        return "tv"
    if "table" in normalized_target or "desk" in normalized_target:
        return "table"
    return "generic"


def _goal_estimate_from_context(
    *,
    target_label: str,
    target_detection: dict[str, float | str | None] | None,
    memory: PlannerMemorySnapshot,
    object_observations: Iterable[PlannerObjectObservation],
    segment_observations: Iterable[PlannerSegmentObservation],
    visible_graph_relations: Iterable[str],
    navigation_affordances: PlannerNavigationAffordances | None,
) -> PlannerGoalEstimate:
    if target_detection is not None:
        distance_hint = None
        depth_value = target_detection.get("median_depth_m")
        if depth_value is not None:
            distance_hint = f"{float(depth_value):.1f}m"
        return PlannerGoalEstimate(
            status="visible",
            bearing_hint=str(target_detection.get("horizontal_position") or "front"),
            distance_hint=distance_hint,
            evidence_sources=("target_detection",),
            confidence=0.95,
        )

    if memory.target_memory is not None:
        evidence_sources = ["target_memory"]
        if memory.target_memory.last_seen_direction:
            evidence_sources.append(f"memory_direction:{memory.target_memory.last_seen_direction}")
        return PlannerGoalEstimate(
            status="remembered",
            bearing_hint=memory.target_memory.last_seen_direction,
            distance_hint=None if memory.target_memory.age_frames <= 6 else "stale_memory",
            evidence_sources=tuple(evidence_sources),
            confidence=0.72 if memory.target_memory.age_frames <= 6 else 0.5,
        )

    object_items = tuple(object_observations)
    segment_items = tuple(segment_observations)
    graph_relations = tuple(visible_graph_relations)
    family = _target_family(target_label)
    cue_directions: list[str] = []
    evidence_sources: list[str] = []

    if family == "tv":
        for item in object_items:
            label = _normalize_label(item.label)
            if label in _TV_CUE_LABELS:
                cue_directions.append(item.direction)
                evidence_sources.append(f"object:{item.label}")
        for item in segment_items:
            label = _normalize_label(item.label)
            if label in {"wall", "window-like"}:
                cue_directions.append(item.dominant_direction)
                evidence_sources.append(f"segment:{item.label}")
        for relation in graph_relations:
            normalized_relation = _normalize_label(relation)
            if any(keyword in normalized_relation for keyword in ("attached_to", "wall", "shelf", "cabinet", "sofa")):
                cue_directions.extend(_relation_bearing_votes((relation,)))
                evidence_sources.append(f"relation:{relation}")
    elif family == "table":
        for item in object_items:
            label = _normalize_label(item.label)
            if label in _TABLE_CUE_LABELS:
                cue_directions.append(item.direction)
                evidence_sources.append(f"object:{item.label}")
        for item in segment_items:
            label = _normalize_label(item.label)
            if label == "floor" and item.coverage_ratio >= 0.12:
                cue_directions.append(item.dominant_direction)
                evidence_sources.append("segment:floor")
        for relation in graph_relations:
            normalized_relation = _normalize_label(relation)
            if any(keyword in normalized_relation for keyword in ("chair", "on", "near", "table")):
                cue_directions.extend(_relation_bearing_votes((relation,)))
                evidence_sources.append(f"relation:{relation}")

    if family in {"tv", "table"} and not evidence_sources and navigation_affordances is not None:
        open_direction = navigation_affordances.best_exploration_direction
        if open_direction is not None and (
            navigation_affordances.front_blocked or navigation_affordances.candidate_open_directions
        ):
            cue_directions.append(open_direction)
            evidence_sources.append("navigation_affordances")

    if evidence_sources:
        distance_hint = None
        if navigation_affordances is not None and navigation_affordances.forward_clearance_m is not None:
            if navigation_affordances.forward_clearance_m >= 2.0:
                distance_hint = "mid-to-far"
            else:
                distance_hint = "nearby_constrained"
        return PlannerGoalEstimate(
            status="inferred",
            bearing_hint=_most_common_direction(cue_directions),
            distance_hint=distance_hint,
            evidence_sources=tuple(dict.fromkeys(evidence_sources)),
            confidence=0.58 if len(evidence_sources) == 1 else 0.66,
        )

    return PlannerGoalEstimate(
        status="unknown",
        bearing_hint=(
            None
            if navigation_affordances is None
            else navigation_affordances.best_exploration_direction
        ),
        distance_hint=None,
        evidence_sources=(),
        confidence=0.2,
    )


def _default_constraints(
    *,
    allowed_command_kinds: Iterable[str] | None = None,
    max_commands_per_schedule: int | None = None,
    execution_capabilities: dict[str, float] | None = None,
    microstep_limits: dict[str, float] | None = None,
    batch_step_limit: int | None,
    replan_frame_budget: int | None,
    tracking_safe_limits: dict[str, float] | None,
) -> PlannerConstraintSummary:
    resolved_microstep_limits = dict(microstep_limits or tracking_safe_limits or {})
    if not resolved_microstep_limits:
        resolved_microstep_limits = {
            "translate_distance_m": 0.08,
            "body_yaw_deg": 4.0,
            "camera_yaw_deg": 4.0,
            "camera_pitch_deg": 4.0,
        }
    resolved_execution_capabilities = dict(execution_capabilities or {})
    if not resolved_execution_capabilities:
        resolved_execution_capabilities = {
            "move_speed_mps": 1.6,
            "turn_speed_deg_per_sec": 100.0,
            "camera_yaw_speed_deg_per_sec": 90.0,
            "camera_pitch_speed_deg_per_sec": 90.0,
            "camera_yaw_limit_deg": 70.0,
            "camera_pitch_limit_deg": 55.0,
            "max_translate_distance_m": 0.72,
            "max_rotate_body_deg": 36.0,
        }
    resolved_max_commands = min(3, max(1, int(max_commands_per_schedule or batch_step_limit or 3)))
    resolved_allowed_command_kinds = tuple(str(item) for item in (allowed_command_kinds or (item.value for item in CommandKind)))
    _ = replan_frame_budget
    return PlannerConstraintSummary(
        allowed_command_kinds=resolved_allowed_command_kinds,
        execution_capabilities=resolved_execution_capabilities,
        microstep_limits=resolved_microstep_limits,
        max_commands_per_schedule=resolved_max_commands,
        notes=(
            "Prefer committed batches over oscillation.",
            "Escape blocked states with translation or rotation before re-centering.",
            "Use camera-frame coordinates where +x is right, +y is up, and +z is forward.",
            "Use aim_camera for absolute yaw/pitch setpoints; do not express camera motion as relative pans.",
        ),
    )


def build_planner_context(
    *,
    phase: EpisodePhase,
    frame_index: int,
    goal_description: str = "Reach the front position of the dining table using only current visible evidence.",
    detections: Iterable[Detection],
    scene_graph_snapshot: SceneGraphSnapshot | None,
    reconstruction_summary: dict[str, float | int],
    depth_summary: dict[str, float | int],
    recent_actions: Iterable[str],
    recent_searches: Iterable[PlannerSearchOutcome] = (),
    target_label: str = "",
    calibration_status: str,
    tracking_status: str,
    target_detection: dict[str, float | str | None] | None = None,
    segments: Iterable[PanopticSegment] = (),
    depth_map: np.ndarray | None = None,
    frame_shape: tuple[int, int] | None = None,
    camera_pose_world: np.ndarray | None = None,
    camera_intrinsics: CameraIntrinsics | None = None,
    current_camera_state: PlannerCameraState | None = None,
    object_observations: Iterable[PlannerObjectObservation] = (),
    segment_observations: Iterable[PlannerSegmentObservation] = (),
    navigation_affordances: PlannerNavigationAffordances | None = None,
    reconstruction_brief: PlannerReconstructionBrief | None = None,
    goal_estimate: PlannerGoalEstimate | None = None,
    recent_action_effects: Iterable[PlannerActionEffectSummary] = (),
    constraints: PlannerConstraintSummary | None = None,
    allowed_command_kinds: Iterable[str] | None = None,
    max_commands_per_schedule: int | None = None,
    execution_capabilities: dict[str, float] | None = None,
    microstep_limits: dict[str, float] | None = None,
    batch_step_limit: int | None = None,
    replan_frame_budget: int | None = None,
    tracking_safe_limits: dict[str, float] | None = None,
) -> PlannerContext:
    detections_tuple = tuple(detections)
    segments_tuple = tuple(segments)
    frame_height, frame_width = _frame_shape_from_inputs(frame_shape=frame_shape, depth_map=depth_map)
    visible_detections = tuple(str(item.label) for item in detections_tuple)

    graph_visible_segments = {
        str(node.label)
        for node in ((scene_graph_snapshot.visible_nodes if scene_graph_snapshot is not None else ()) or ())
        if node.id != "ego" and node.type != "object"
    }
    visible_segments = tuple(
        sorted(
            graph_visible_segments.union(str(item.label) for item in segments_tuple)
        )
    )
    visible_graph_relations = tuple(
        f"{edge.source}->{edge.relation}->{edge.target}"
        for edge in ((scene_graph_snapshot.visible_edges if scene_graph_snapshot is not None else ()) or ())
    )
    scene_node_observations = _scene_node_observations(scene_graph_snapshot=scene_graph_snapshot)
    scene_edge_observations = _scene_edge_observations(scene_graph_snapshot=scene_graph_snapshot)
    resolved_object_observations = tuple(object_observations) or _object_observations_from_detections(
        detections=detections_tuple,
        depth_map=depth_map,
        frame_shape=frame_shape,
        camera_intrinsics=camera_intrinsics,
        target_label=str(target_label),
    )
    resolved_segment_observations = tuple(segment_observations) or _segment_observations_from_segments(
        segments=segments_tuple,
        frame_shape=frame_shape,
    )
    resolved_navigation_affordances = navigation_affordances or _default_navigation_affordances(
        depth_map=depth_map,
        segment_observations=resolved_segment_observations,
    )
    resolved_reconstruction_brief = reconstruction_brief or _default_reconstruction_brief(
        reconstruction_summary=reconstruction_summary,
        navigation_affordances=resolved_navigation_affordances,
    )
    memory = _planner_memory_snapshot(
        frame_index=int(frame_index),
        detections=detections_tuple,
        scene_graph_snapshot=scene_graph_snapshot,
        target_label=str(target_label),
        recent_searches=tuple(recent_searches),
    )
    resolved_goal_estimate = goal_estimate or _goal_estimate_from_context(
        target_label=str(target_label),
        target_detection=target_detection,
        memory=memory,
        object_observations=resolved_object_observations,
        segment_observations=resolved_segment_observations,
        visible_graph_relations=visible_graph_relations,
        navigation_affordances=resolved_navigation_affordances,
    )
    resolved_constraints = constraints or _default_constraints(
        allowed_command_kinds=allowed_command_kinds,
        max_commands_per_schedule=max_commands_per_schedule,
        execution_capabilities=execution_capabilities,
        microstep_limits=microstep_limits,
        batch_step_limit=batch_step_limit,
        replan_frame_budget=replan_frame_budget,
        tracking_safe_limits=tracking_safe_limits,
    )
    return PlannerContext(
        phase=phase,
        frame_index=int(frame_index),
        goal_description=str(goal_description),
        target_label=str(target_label),
        perception=PerceptionSnapshot(
            visible_detections=visible_detections,
            visible_segments=visible_segments,
            visible_graph_relations=visible_graph_relations,
            reconstruction_summary=dict(reconstruction_summary),
            depth_summary=dict(depth_summary),
            target_detection=None if target_detection is None else dict(target_detection),
            calibration_status=str(calibration_status),
            tracking_status=str(tracking_status),
            objects=resolved_object_observations,
            structural_segments=resolved_segment_observations,
            navigation_affordances=resolved_navigation_affordances,
            reconstruction_brief=resolved_reconstruction_brief,
            frame_size=(frame_height, frame_width),
            scene_nodes=scene_node_observations,
            scene_edges=scene_edge_observations,
            current_camera_state=current_camera_state,
        ),
        memory=memory,
        goal_estimate=resolved_goal_estimate,
        recent_action_effects=tuple(recent_action_effects)[-4:],
        constraints=resolved_constraints,
        recent_actions=tuple(str(item) for item in recent_actions),
        image_metadata={
            "width": int(frame_width),
            "height": int(frame_height),
            "channels": 3,
            "source": "current_rgb_frame",
        },
    )


def _memory_prompt_payload(memory: PlannerMemorySnapshot) -> dict[str, object]:
    return {
        "target_memory": (
            None
            if memory.target_memory is None
            else {
                "label": memory.target_memory.label,
                "kind": memory.target_memory.kind,
                "state": memory.target_memory.state,
                "last_seen_direction": memory.target_memory.last_seen_direction,
                "age_frames": int(memory.target_memory.age_frames),
            }
        ),
        "nonvisible_observations": [
            {
                "label": item.label,
                "kind": item.kind,
                "state": item.state,
                "last_seen_direction": item.last_seen_direction,
                "age_frames": int(item.age_frames),
            }
            for item in memory.nonvisible_observations
        ],
        "recent_searches": [
            {
                "start_frame": int(item.start_frame),
                "end_frame": int(item.end_frame),
                "executed_actions": list(item.executed_actions),
                "target_visible_after_search": bool(item.target_visible_after_search),
                "tracking_status": item.tracking_status,
                "entered_new_view": bool(item.entered_new_view),
                "pose_progress_m": item.pose_progress_m,
                "likely_blocked": bool(item.likely_blocked),
                "evidence_gained": bool(item.evidence_gained),
            }
            for item in memory.recent_searches
        ],
    }


def _goal_hypothesis_payload(goal_estimate: PlannerGoalEstimate) -> dict[str, object]:
    return {
        "status": goal_estimate.status,
        "bearing_hint": goal_estimate.bearing_hint,
        "distance_hint": goal_estimate.distance_hint,
        "evidence": list(goal_estimate.evidence_sources),
        "confidence": float(goal_estimate.confidence),
    }


def _planner_request_payload(context: PlannerContext) -> dict[str, object]:
    navigation_affordances = context.perception.navigation_affordances
    reconstruction_brief = context.perception.reconstruction_brief
    memory_payload = _memory_prompt_payload(context.memory)
    current_camera_state = context.perception.current_camera_state
    return {
        "request_type": "multimodal_planner_core",
        "phase": context.phase.value,
        "frame_index": int(context.frame_index),
        "image_metadata": dict(context.image_metadata),
        "coordinate_convention": {
            "camera_frame": {
                "x": "right",
                "y": "up",
                "z": "forward",
            },
            "aim_camera": {
                "yaw_deg_positive": "right",
                "yaw_deg_negative": "left",
                "pitch_deg_positive": "up",
                "pitch_deg_negative": "down",
            },
            "rotate_body": {
                "type": "relative_yaw_delta",
                "direction_left": "positive",
                "direction_right": "negative",
            },
        },
        "visible_detections": list(context.perception.visible_detections),
        "visible_segments": list(context.perception.visible_segments),
        "visible_graph_relations": list(context.perception.visible_graph_relations),
        "target_detection": (
            None
            if context.perception.target_detection is None
            else dict(context.perception.target_detection)
        ),
        "goal_estimate": {
            "status": context.goal_estimate.status,
            "bearing_hint": context.goal_estimate.bearing_hint,
            "distance_hint": context.goal_estimate.distance_hint,
            "evidence_sources": list(context.goal_estimate.evidence_sources),
            "confidence": float(context.goal_estimate.confidence),
        },
        "robot_state": {
            "tracking_status": context.perception.tracking_status,
            "calibration_status": context.perception.calibration_status,
            "current_camera_state": (
                None
                if current_camera_state is None
                else {
                    "yaw_deg": float(current_camera_state.yaw_deg),
                    "pitch_deg": float(current_camera_state.pitch_deg),
                }
            ),
            "recent_actions": list(context.recent_actions),
            "recent_action_effects": [
                {
                    "action": item.action,
                    "estimated_motion": item.estimated_motion,
                    "clearance_change": item.clearance_change,
                    "target_evidence_change": item.target_evidence_change,
                    "likely_blocked": bool(item.likely_blocked),
                    "aborted": bool(item.aborted),
                    "commanded_progress_m": item.commanded_progress_m,
                    "vision_progress_m": item.vision_progress_m,
                    "fused_progress_m": item.fused_progress_m,
                    "progress_source": item.progress_source,
                }
                for item in context.recent_action_effects
            ],
        },
        "goal": {
            "description": context.goal_description,
            "target_label": context.target_label,
            "goal_hypothesis": _goal_hypothesis_payload(context.goal_estimate),
            "target_detection": (
                None
                if context.perception.target_detection is None
                else dict(context.perception.target_detection)
            ),
        },
        "visible_objects": [
            {
                "label": item.label,
                "confidence": float(item.confidence),
                "direction": item.direction,
                "distance_bucket": item.distance_bucket,
                "bbox_area_ratio": float(item.bbox_area_ratio),
                "depth_m": item.depth_m,
                "relative_xyz": item.relative_xyz,
                "is_target_match": bool(item.is_target_match),
            }
            for item in context.perception.objects
        ],
        "objects": [
            {
                "label": item.label,
                "confidence": float(item.confidence),
                "direction": item.direction,
                "distance_bucket": item.distance_bucket,
                "bbox_area_ratio": float(item.bbox_area_ratio),
                "depth_m": item.depth_m,
                "relative_xyz": item.relative_xyz,
                "is_target_match": bool(item.is_target_match),
            }
            for item in context.perception.objects
        ],
        "structural_segments": [
            {
                "label": item.label,
                "coverage_ratio": float(item.coverage_ratio),
                "dominant_direction": item.dominant_direction,
                "opening_hint": item.opening_hint,
            }
            for item in context.perception.structural_segments
        ],
        "segments": [
            {
                "label": item.label,
                "coverage_ratio": float(item.coverage_ratio),
                "dominant_direction": item.dominant_direction,
                "opening_hint": item.opening_hint,
            }
            for item in context.perception.structural_segments
        ],
        "scene_graph": {
            "visible_nodes": [
                {
                    "label": item.label,
                    "kind": item.kind,
                    "state": item.state,
                    "bearing": item.bearing,
                    "distance_bucket": item.distance_bucket,
                    "confidence": float(item.confidence),
                    "relative_xyz": item.relative_xyz,
                }
                for item in context.perception.scene_nodes
                if item.state == "visible"
            ],
            "memory_nodes": [
                {
                    "label": item.label,
                    "kind": item.kind,
                    "state": item.state,
                    "bearing": item.bearing,
                    "distance_bucket": item.distance_bucket,
                    "confidence": float(item.confidence),
                    "relative_xyz": item.relative_xyz,
                }
                for item in context.perception.scene_nodes
                if item.state != "visible"
            ],
            "visible_edges": [
                {
                    "source_label": item.source_label,
                    "target_label": item.target_label,
                    "relation": item.relation,
                    "confidence": float(item.confidence),
                    "source_kind": item.source_kind,
                    "distance_bucket": item.distance_bucket,
                }
                for item in context.perception.scene_edges
            ],
            "visible_relations": list(context.perception.visible_graph_relations),
        },
        "navigation_map": (
            None
            if navigation_affordances is None
            else {
                "forward_clearance_m": navigation_affordances.forward_clearance_m,
                "left_clearance_m": navigation_affordances.left_clearance_m,
                "right_clearance_m": navigation_affordances.right_clearance_m,
                "rear_clearance_m": navigation_affordances.rear_clearance_m,
                "front_blocked": bool(navigation_affordances.front_blocked),
                "candidate_open_directions": list(navigation_affordances.candidate_open_directions),
                "dead_end_likelihood": float(navigation_affordances.dead_end_likelihood),
                "best_exploration_direction": navigation_affordances.best_exploration_direction,
                "sector_map": [
                    {
                        "sector": item.sector,
                        "clearance_m": item.clearance_m,
                        "traversable": bool(item.traversable),
                        "obstacle_likelihood": float(item.obstacle_likelihood),
                        "frontier_score": float(item.frontier_score),
                        "recently_failed": bool(item.recently_failed),
                    }
                    for item in navigation_affordances.sector_map
                ],
                "recently_failed_directions": list(navigation_affordances.recently_failed_directions),
            }
        ),
        "navigation_affordances": (
            None
            if navigation_affordances is None
            else {
                "forward_clearance_m": navigation_affordances.forward_clearance_m,
                "left_clearance_m": navigation_affordances.left_clearance_m,
                "right_clearance_m": navigation_affordances.right_clearance_m,
                "rear_clearance_m": navigation_affordances.rear_clearance_m,
                "front_blocked": bool(navigation_affordances.front_blocked),
                "candidate_open_directions": list(navigation_affordances.candidate_open_directions),
                "dead_end_likelihood": float(navigation_affordances.dead_end_likelihood),
                "best_exploration_direction": navigation_affordances.best_exploration_direction,
                "sector_map": [
                    {
                        "sector": item.sector,
                        "clearance_m": item.clearance_m,
                        "traversable": bool(item.traversable),
                        "obstacle_likelihood": float(item.obstacle_likelihood),
                        "frontier_score": float(item.frontier_score),
                        "recently_failed": bool(item.recently_failed),
                    }
                    for item in navigation_affordances.sector_map
                ],
                "recently_failed_directions": list(navigation_affordances.recently_failed_directions),
            }
        ),
        "reconstruction": (
            None
            if reconstruction_brief is None
            else {
                "pose_delta_m": reconstruction_brief.pose_delta_m,
                "yaw_delta_deg": reconstruction_brief.yaw_delta_deg,
                "mesh_vertex_count": int(reconstruction_brief.mesh_vertex_count),
                "mesh_triangle_count": int(reconstruction_brief.mesh_triangle_count),
                "mesh_growth_delta": int(reconstruction_brief.mesh_growth_delta),
                "frontier_directions": list(reconstruction_brief.frontier_directions),
                "explored_directions": list(reconstruction_brief.explored_directions),
                "unexplored_directions": list(reconstruction_brief.unexplored_directions),
                "recently_failed_directions": list(reconstruction_brief.recently_failed_directions),
                "tracked_feature_count": int(reconstruction_brief.tracked_feature_count),
                "median_reprojection_error": reconstruction_brief.median_reprojection_error,
            }
        ),
        "memory": memory_payload,
        "recent_action_effects": [
            {
                "action": item.action,
                "estimated_motion": item.estimated_motion,
                "clearance_change": item.clearance_change,
                "target_evidence_change": item.target_evidence_change,
                "likely_blocked": bool(item.likely_blocked),
                "aborted": bool(item.aborted),
                "commanded_progress_m": item.commanded_progress_m,
                "vision_progress_m": item.vision_progress_m,
                "fused_progress_m": item.fused_progress_m,
                "progress_source": item.progress_source,
            }
            for item in context.recent_action_effects
        ],
        "recent_searches": memory_payload["recent_searches"],
        "target_hypotheses": [_goal_hypothesis_payload(context.goal_estimate)],
        "constraints": {
            "allowed_command_kinds": list(context.constraints.allowed_command_kinds),
            "execution_capabilities": dict(context.constraints.execution_capabilities),
            "microstep_limits": dict(context.constraints.microstep_limits),
            "max_commands_per_schedule": int(context.constraints.max_commands_per_schedule),
            "notes": list(context.constraints.notes),
            "hidden_scene_truth_policy": "forbidden",
        },
        "legacy_summary": {
            "visible_detections": list(context.perception.visible_detections),
            "visible_segments": list(context.perception.visible_segments),
            "visible_graph_relations": list(context.perception.visible_graph_relations),
            "depth_summary": dict(context.perception.depth_summary),
            "reconstruction_summary": dict(context.perception.reconstruction_summary),
            "recent_searches": memory_payload["recent_searches"],
        },
    }


def planner_prompt_from_context(context: PlannerContext) -> str:
    return json.dumps(_planner_request_payload(context), ensure_ascii=False, sort_keys=True)


def _encode_frame_as_data_url(frame_bgr: np.ndarray) -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", np.asarray(frame_bgr, dtype=np.uint8))
    if not ok:
        raise RuntimeError("failed to encode planner frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _planner_input_content(*, context: PlannerContext, frame_bgr: np.ndarray | None) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {"type": "input_text", "text": planner_prompt_from_context(context)}
    ]
    if frame_bgr is not None and np.asarray(frame_bgr).size > 0:
        content.append({"type": "input_image", "image_url": _encode_frame_as_data_url(np.asarray(frame_bgr, dtype=np.uint8))})
    return content


def _command_schema() -> dict[str, object]:
    translate_or_rotate_base = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"type": "string"},
            "direction": {"type": "string"},
            "mode": {"type": "string"},
            "value": {"type": "number"},
            "intent": {"type": "string"},
        },
    }
    return {
        "type": "array",
        "maxItems": 3,
        "items": {
            "anyOf": [
                {
                    **translate_or_rotate_base,
                    "properties": {
                        **translate_or_rotate_base["properties"],
                        "kind": {"type": "string", "const": CommandKind.TRANSLATE.value},
                        "direction": {
                            "type": "string",
                            "enum": ["forward", "backward", "left", "right"],
                        },
                        "mode": {"type": "string", "enum": ["distance_m", "hold_sec"]},
                    },
                    "required": ["kind", "direction", "mode", "value", "intent"],
                },
                {
                    **translate_or_rotate_base,
                    "properties": {
                        **translate_or_rotate_base["properties"],
                        "kind": {"type": "string", "const": CommandKind.ROTATE_BODY.value},
                        "direction": {"type": "string", "enum": ["left", "right"]},
                        "mode": {"type": "string", "enum": ["angle_deg", "hold_sec"]},
                    },
                    "required": ["kind", "direction", "mode", "value", "intent"],
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "const": CommandKind.AIM_CAMERA.value},
                        "yaw_deg": {"type": "number"},
                        "pitch_deg": {"type": "number"},
                        "intent": {"type": "string"},
                    },
                    "required": ["kind", "yaw_deg", "pitch_deg", "intent"],
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "const": CommandKind.PAUSE.value},
                        "duration_sec": {"type": "number"},
                        "intent": {"type": "string"},
                    },
                    "required": ["kind", "duration_sec", "intent"],
                },
            ]
        },
    }


def _motion_command_from_payload(item: dict[str, object]) -> MotionCommand:
    kind = CommandKind(str(item.get("kind") or CommandKind.PAUSE.value))
    if kind in {CommandKind.TRANSLATE, CommandKind.ROTATE_BODY}:
        return MotionCommand(
            kind=kind,
            direction=str(item.get("direction") or ""),
            mode=str(item.get("mode") or ""),
            value=float(item.get("value") or 0.0),
            intent=str(item.get("intent") or ""),
        )
    if kind is CommandKind.AIM_CAMERA:
        return MotionCommand(
            kind=kind,
            yaw_deg=float(item.get("yaw_deg") or 0.0),
            pitch_deg=float(item.get("pitch_deg") or 0.0),
            intent=str(item.get("intent") or ""),
        )
    return MotionCommand(
        kind=CommandKind.PAUSE,
        duration_sec=float(item.get("duration_sec") or 0.0),
        intent=str(item.get("intent") or ""),
    )


class OpenAILivingRoomPlanner:
    def __init__(
        self,
        *,
        model: str,
        timeout_sec: float,
        api_key: str | None = None,
        client=None,
    ) -> None:
        self._model = str(model)
        self._timeout_sec = float(timeout_sec)
        if client is None:
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not resolved_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for living room simulation planning")
            from openai import OpenAI

            client = OpenAI(api_key=resolved_api_key, timeout=self._timeout_sec)
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def plan(self, *, context: PlannerContext, frame_bgr: np.ndarray | None = None) -> ActionSchedule:
        request_kwargs = dict(
            model=self._model,
            instructions=_PLANNER_SYSTEM_INSTRUCTIONS,
            input=[{"role": "user", "content": _planner_input_content(context=context, frame_bgr=frame_bgr)}],
            max_output_tokens=420,
            timeout=self._timeout_sec,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "living_room_action_schedule",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "situation_summary": {"type": "string"},
                            "goal_hypothesis": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "status": {
                                        "type": "string",
                                        "enum": ["visible", "remembered", "inferred", "unknown"],
                                    },
                                    "bearing_hint": {"type": ["string", "null"]},
                                    "distance_hint": {"type": ["string", "null"]},
                                    "evidence": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "confidence": {"type": "number"},
                                },
                                "required": ["status", "bearing_hint", "distance_hint", "evidence", "confidence"],
                            },
                            "goal_completion": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "reached": {"type": "boolean"},
                                    "confidence": {"type": "number"},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["reached", "confidence", "rationale"],
                            },
                            "behavior_mode": {
                                "type": "string",
                                "enum": ["approach", "reacquire", "infer", "escape", "scan"],
                            },
                            "commands": _command_schema(),
                            "safety_flags": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "front_blocked": {"type": "boolean"},
                                    "dead_end_risk": {"type": "number"},
                                    "tracking_risk": {"type": "string"},
                                    "replan_reason": {"type": "string"},
                                },
                                "required": ["front_blocked", "dead_end_risk", "tracking_risk", "replan_reason"],
                            },
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "situation_summary",
                            "goal_hypothesis",
                            "goal_completion",
                            "behavior_mode",
                            "commands",
                            "safety_flags",
                            "confidence",
                        ],
                    },
                },
                "verbosity": "low",
            },
        )
        if self._model.startswith("gpt-5"):
            request_kwargs["reasoning"] = {"effort": "minimal"}
        response = self._client.responses.create(**request_kwargs)
        payload = _response_json(response)
        goal_completion_payload = payload.get("goal_completion") or {}
        return ActionSchedule(
            commands=tuple(
                _motion_command_from_payload(dict(item))
                for item in list(payload.get("commands") or [])
                if isinstance(item, dict)
            ),
            rationale=str(payload.get("situation_summary") or ""),
            model=self._model,
            issued_at_frame=context.frame_index,
            situation_summary=str(payload.get("situation_summary") or ""),
            behavior_mode=str(payload.get("behavior_mode") or "scan"),
            goal_hypothesis=PlannerGoalEstimate(
                status=str((payload.get("goal_hypothesis") or {}).get("status") or "unknown"),
                bearing_hint=(payload.get("goal_hypothesis") or {}).get("bearing_hint"),
                distance_hint=(payload.get("goal_hypothesis") or {}).get("distance_hint"),
                evidence_sources=tuple(str(item) for item in list((payload.get("goal_hypothesis") or {}).get("evidence") or [])),
                confidence=float((payload.get("goal_hypothesis") or {}).get("confidence") or 0.0),
            ),
            goal_completion=PlannerGoalCompletion(
                reached=bool(goal_completion_payload.get("reached")),
                confidence=float(goal_completion_payload.get("confidence") or 0.0),
                rationale=str(goal_completion_payload.get("rationale") or ""),
            ),
            safety_flags=PlannerSafetyFlags(
                front_blocked=bool((payload.get("safety_flags") or {}).get("front_blocked")),
                dead_end_risk=float((payload.get("safety_flags") or {}).get("dead_end_risk") or 0.0),
                tracking_risk=str((payload.get("safety_flags") or {}).get("tracking_risk") or "unknown"),
                replan_reason=str((payload.get("safety_flags") or {}).get("replan_reason") or ""),
            ),
            confidence=float(payload.get("confidence") or 0.0),
        )


def _response_json(response) -> dict[str, object]:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return json.loads(text)
    output = getattr(response, "output", None) or []
    chunks: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
    return json.loads("\n".join(chunks))
