from __future__ import annotations

import json
import os
from typing import Iterable

from obj_recog.scene_graph import GraphNode, SceneGraphSnapshot
from obj_recog.sim_protocol import (
    ActionPrimitive,
    ActionSchedule,
    ActionStep,
    EpisodePhase,
    PlannerMemoryObservation,
    PlannerMemorySnapshot,
    PlannerSearchOutcome,
    PerceptionSnapshot,
    PlannerContext,
)
from obj_recog.types import Detection


_PLANNER_SYSTEM_INSTRUCTIONS = (
    "You are a navigation planner for a wheeled indoor robot. "
    "You know the current camera-derived summaries, planner-visible memory from earlier in the same episode, "
    "and summaries of recent search batches. "
    "Current visible evidence takes priority over memory. "
    "Avoid repeating the same recent unsuccessful search when other unexplored options exist, "
    "but revisiting is allowed if new evidence appears or there is no better alternative. "
    "When the target is not currently visible, return a coordinated search batch of 2 to 4 steps "
    "instead of a single isolated action unless pausing is the only safe option. "
    "Favor committing to one promising direction for a short batch over oscillating left/right every turn. "
    "Use camera pans to confirm visibility, but pair them with translation or heading changes when recent pure scans failed. "
    "When the target is visible, use target_detection to judge whether it is left/right of center, still far away, or already close. "
    "If the target is off-center, prefer small turns or camera pans to center it before adding more forward motion. "
    "When target_detection says left or right, do not include an opposite-direction turn or camera pan in the same batch unless the target is already near center. "
    "If the target is visible and far away, keep the batch committed in the same centering direction and add forward motion instead of restarting a generic search. "
    "If the target is visible while tracking is INITIALIZING or RELOCALIZING, keep motions visually smooth and avoid large repeated pans. "
    "Never infer hidden furniture or exact room coordinates. "
    "Return JSON only with keys rationale and steps. "
    "Each step must include primitive and value. "
    "Allowed primitives: "
    + ", ".join(item.value for item in ActionPrimitive)
    + "."
)


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
) -> PlannerContext:
    detections_tuple = tuple(detections)
    visible_detections = tuple(str(item.label) for item in detections_tuple)
    visible_segments = tuple(
        sorted(
            {
                str(node.label)
                for node in ((scene_graph_snapshot.visible_nodes if scene_graph_snapshot is not None else ()) or ())
                if node.id != "ego" and node.type != "object"
            }
        )
    )
    visible_graph_relations = tuple(
        f"{edge.source}->{edge.relation}->{edge.target}"
        for edge in ((scene_graph_snapshot.visible_edges if scene_graph_snapshot is not None else ()) or ())
    )
    return PlannerContext(
        phase=phase,
        frame_index=int(frame_index),
        goal_description=str(goal_description),
        perception=PerceptionSnapshot(
            visible_detections=visible_detections,
            visible_segments=visible_segments,
            visible_graph_relations=visible_graph_relations,
            reconstruction_summary=dict(reconstruction_summary),
            depth_summary=dict(depth_summary),
            target_detection=None if target_detection is None else dict(target_detection),
            calibration_status=str(calibration_status),
            tracking_status=str(tracking_status),
        ),
        memory=_planner_memory_snapshot(
            frame_index=int(frame_index),
            detections=detections_tuple,
            scene_graph_snapshot=scene_graph_snapshot,
            target_label=str(target_label),
            recent_searches=tuple(recent_searches),
        ),
        recent_actions=tuple(str(item) for item in recent_actions),
    )


def planner_prompt_from_context(context: PlannerContext) -> str:
    payload = {
        "phase": context.phase.value,
        "frame_index": context.frame_index,
        "goal": context.goal_description,
        "visible_detections": list(context.perception.visible_detections),
        "visible_segments": list(context.perception.visible_segments),
        "visible_graph_relations": list(context.perception.visible_graph_relations),
        "reconstruction_summary": dict(context.perception.reconstruction_summary),
        "depth_summary": dict(context.perception.depth_summary),
        "target_detection": (
            None
            if context.perception.target_detection is None
            else dict(context.perception.target_detection)
        ),
        "calibration_status": context.perception.calibration_status,
        "tracking_status": context.perception.tracking_status,
        "memory": {
            "target_memory": (
                None
                if context.memory.target_memory is None
                else {
                    "label": context.memory.target_memory.label,
                    "kind": context.memory.target_memory.kind,
                    "state": context.memory.target_memory.state,
                    "last_seen_direction": context.memory.target_memory.last_seen_direction,
                    "age_frames": int(context.memory.target_memory.age_frames),
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
                for item in context.memory.nonvisible_observations
            ],
            "recent_searches": [
                {
                    "start_frame": int(item.start_frame),
                    "end_frame": int(item.end_frame),
                    "executed_actions": list(item.executed_actions),
                    "target_visible_after_search": bool(item.target_visible_after_search),
                    "tracking_status": item.tracking_status,
                }
                for item in context.memory.recent_searches
            ],
        },
        "recent_actions": list(context.recent_actions),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


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

    def plan(self, *, context: PlannerContext) -> ActionSchedule:
        request_kwargs = dict(
            model=self._model,
            instructions=_PLANNER_SYSTEM_INSTRUCTIONS,
            input=[{"role": "user", "content": [{"type": "input_text", "text": planner_prompt_from_context(context)}]}],
            max_output_tokens=300,
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
                            "rationale": {"type": "string"},
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "primitive": {
                                            "type": "string",
                                            "enum": [item.value for item in ActionPrimitive],
                                        },
                                        "value": {"type": "number"},
                                    },
                                    "required": ["primitive", "value"],
                                },
                            },
                        },
                        "required": ["rationale", "steps"],
                    },
                },
                "verbosity": "low",
            },
        )
        if self._model.startswith("gpt-5"):
            request_kwargs["reasoning"] = {"effort": "minimal"}
        response = self._client.responses.create(**request_kwargs)
        payload = _response_json(response)
        return ActionSchedule(
            steps=tuple(
                ActionStep(
                    primitive=ActionPrimitive(str(item["primitive"])),
                    value=float(item["value"]),
                )
                for item in list(payload.get("steps") or [])
            ),
            rationale=str(payload.get("rationale") or ""),
            model=self._model,
            issued_at_frame=context.frame_index,
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
