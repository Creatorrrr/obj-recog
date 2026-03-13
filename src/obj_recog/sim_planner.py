from __future__ import annotations

import json
import os
from typing import Iterable

from obj_recog.scene_graph import SceneGraphSnapshot
from obj_recog.sim_protocol import (
    ActionPrimitive,
    ActionSchedule,
    ActionStep,
    EpisodePhase,
    PerceptionSnapshot,
    PlannerContext,
)
from obj_recog.types import Detection


_PLANNER_SYSTEM_INSTRUCTIONS = (
    "You are a navigation planner for a wheeled indoor robot. "
    "You only know what is visible in the current camera-derived summaries. "
    "Never infer hidden furniture or exact room coordinates. "
    "Return JSON only with keys rationale and steps. "
    "Each step must include primitive and value. "
    "Allowed primitives: "
    + ", ".join(item.value for item in ActionPrimitive)
    + "."
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
    calibration_status: str,
    tracking_status: str,
) -> PlannerContext:
    visible_detections = tuple(str(item.label) for item in detections)
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
            calibration_status=str(calibration_status),
            tracking_status=str(tracking_status),
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
        "calibration_status": context.perception.calibration_status,
        "tracking_status": context.perception.tracking_status,
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
