from __future__ import annotations

import base64
import os
import textwrap
import threading
import time
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from obj_recog.opencv_runtime import load_cv2
from obj_recog.types import FrameArtifacts


class ExplanationStatus(StrEnum):
    IDLE = "idle"
    CAPTURING = "capturing"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass(slots=True)
class SituationExplanationSnapshot:
    snapshot_id: int
    frame_bgr: np.ndarray
    structured_context: str
    timestamp_label: str


@dataclass(slots=True)
class ExplanationResult:
    text: str
    status: ExplanationStatus
    latency_ms: float | None
    model: str
    error_message: str | None


_SYSTEM_INSTRUCTIONS = (
    "당신은 카메라 장면 설명 도우미다. 반드시 한국어로만 답하라. "
    "현재 프레임에서 실제로 보이는 내용만 우선 설명하고, 구조화된 graph/SLAM 정보는 보조 근거로만 사용하라. "
    "확실하지 않은 내용은 추측하지 말고 불확실성을 명시하라. "
    "보이지 않고 memory에만 있는 정보가 있으면 '이전에 보였음'처럼 구분해서 표현하라. "
    "출력은 정확히 다음 형식을 따르라: "
    "첫 문단은 현재 상황 설명 1문단, 그 다음 줄은 '핵심 객체:', 다음 줄은 '공간 관계:', 다음 줄은 '불확실성:'"
)


def _direction_bucket(x_center: float, width: int) -> str:
    if width <= 0:
        return "front"
    normalized = x_center / float(width)
    if normalized < 0.2:
        return "left"
    if normalized < 0.4:
        return "front-left"
    if normalized < 0.6:
        return "front"
    if normalized < 0.8:
        return "front-right"
    return "right"


def _summarize_detections(frame_artifacts: FrameArtifacts, max_detections: int) -> list[str]:
    frame_width = int(frame_artifacts.frame_bgr.shape[1])
    frame_height = int(frame_artifacts.frame_bgr.shape[0])
    entries: list[str] = []
    for detection in sorted(
        frame_artifacts.detections,
        key=lambda item: float(item.confidence),
        reverse=True,
    )[:max_detections]:
        x1, y1, x2, y2 = detection.xyxy
        center_x = (int(x1) + int(x2)) / 2.0
        direction = _direction_bucket(center_x, frame_width)
        box_width = max(1, int(x2) - int(x1))
        box_height = max(1, int(y2) - int(y1))
        area_ratio = float(box_width * box_height) / float(max(1, frame_width * frame_height))
        entries.append(
            f"- {detection.label} (conf={detection.confidence:.2f}, direction={direction}, "
            f"bbox=({int(x1)},{int(y1)})-({int(x2)},{int(y2)}), area={area_ratio:.3f})"
        )
    return entries


def _summarize_segments(frame_artifacts: FrameArtifacts, max_segments: int = 12) -> list[str]:
    frame_area = max(1, int(frame_artifacts.frame_bgr.shape[0]) * int(frame_artifacts.frame_bgr.shape[1]))
    entries: list[str] = []
    for index, segment in enumerate(
        sorted(frame_artifacts.segments, key=lambda item: int(item.area_pixels), reverse=True)[:max_segments],
        start=1,
    ):
        coverage_ratio = float(int(segment.area_pixels)) / float(frame_area)
        entries.append(f"- {segment.label} (area_rank={index}, coverage={coverage_ratio:.3f})")
    return entries


def _summarize_graph_edges(frame_artifacts: FrameArtifacts, max_graph_edges: int) -> list[str]:
    if frame_artifacts.scene_graph_snapshot is None:
        return []
    ego_edges = [
        edge
        for edge in frame_artifacts.visible_graph_edges
        if edge.source == "ego"
    ]
    nodes_by_id = {node.id: node for node in frame_artifacts.visible_graph_nodes}
    entries: list[str] = []
    for edge in sorted(
        ego_edges,
        key=lambda item: (-float(item.confidence), item.relation, item.target),
    )[:max_graph_edges]:
        target = nodes_by_id.get(edge.target)
        if target is None:
            continue
        suffix = f" ({edge.distance_bucket})" if edge.distance_bucket else ""
        entries.append(f"- ego -> {edge.relation} -> {target.label}{suffix}")
    return entries


def _summarize_graph_nodes(frame_artifacts: FrameArtifacts, max_graph_nodes: int) -> list[str]:
    entries: list[str] = []
    for node in frame_artifacts.visible_graph_nodes:
        if node.id == "ego":
            continue
        entries.append(
            f"{node.label}({node.state}, conf={node.confidence:.2f})"
        )
        if len(entries) >= max_graph_nodes:
            break
    return entries


def build_explanation_snapshot(
    frame_artifacts: FrameArtifacts,
    *,
    snapshot_id: int,
    max_detections: int = 12,
    max_graph_nodes: int = 20,
    max_graph_edges: int = 20,
    timestamp_label: str | None = None,
) -> SituationExplanationSnapshot:
    detections = _summarize_detections(frame_artifacts, max_detections=max_detections)
    graph_edges = _summarize_graph_edges(frame_artifacts, max_graph_edges=max_graph_edges)
    graph_nodes = _summarize_graph_nodes(frame_artifacts, max_graph_nodes=max_graph_nodes)
    segments = _summarize_segments(frame_artifacts)
    frame_height, frame_width = frame_artifacts.frame_bgr.shape[:2]

    sections = [
        "Visible objects",
        f"- image_size={int(frame_width)}x{int(frame_height)}",
        *(detections or ["- none"]),
    ]
    if segments:
        sections.extend(["", "Visible structural segments", *segments])
    if graph_edges:
        sections.extend(["", "Ego-relative graph summary", *graph_edges])

    slam_summary = [
        f"- state={frame_artifacts.slam_tracking_state}",
        f"- keyframe_id={frame_artifacts.keyframe_id if frame_artifacts.keyframe_id is not None else '-'}",
        f"- mesh_triangles={int(frame_artifacts.mesh_triangles.shape[0])}",
        f"- mesh_vertices={int(frame_artifacts.mesh_vertices_xyz.shape[0])}",
    ]
    sections.extend(["", "SLAM/runtime state", *slam_summary])

    notes: list[str] = []
    if graph_nodes:
        notes.append("- visible graph nodes: " + ", ".join(graph_nodes))
    if frame_artifacts.slam_tracking_state in {"INITIALIZING", "LOST"}:
        notes.append(f"- SLAM state is {frame_artifacts.slam_tracking_state}; geometry may be unstable.")
    if not graph_edges:
        notes.append("- Scene graph summary is sparse; rely primarily on visible detections.")
    sections.extend(["", "Notes", *(notes or ["- none"])])

    return SituationExplanationSnapshot(
        snapshot_id=int(snapshot_id),
        frame_bgr=np.asarray(frame_artifacts.frame_bgr, dtype=np.uint8).copy(),
        structured_context="\n".join(sections).strip(),
        timestamp_label=timestamp_label or time.strftime("%H:%M:%S"),
    )


def _response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    def _extract_text_value(value) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        nested_text = getattr(value, "text", None)
        if isinstance(nested_text, str) and nested_text.strip():
            return nested_text.strip()
        nested_value = getattr(value, "value", None)
        if isinstance(nested_value, str) and nested_value.strip():
            return nested_value.strip()
        return ""

    output = getattr(response, "output", None) or []
    chunks: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            extracted = _extract_text_value(value)
            if extracted:
                chunks.append(extracted)
    return "\n".join(chunks).strip()


class OpenAISituationExplainer:
    def __init__(
        self,
        *,
        model: str,
        timeout_sec: float,
        api_key: str | None = None,
        client=None,
        cv2_module=None,
    ) -> None:
        self._model = model
        self._timeout_sec = float(timeout_sec)
        self._cv2_module = cv2_module
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if client is None:
            if not resolved_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for situation explanations")
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on local install.
                raise RuntimeError("openai package is required for situation explanations") from exc
            client = OpenAI(api_key=resolved_api_key, timeout=self._timeout_sec)
        self._client = client

    def _encode_frame_as_data_url(self, frame_bgr: np.ndarray) -> str:
        cv2 = load_cv2(self._cv2_module)
        ok, encoded = cv2.imencode(".jpg", np.asarray(frame_bgr, dtype=np.uint8))
        if not ok:
            raise RuntimeError("failed to encode explanation snapshot image")
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    def explain(self, snapshot: SituationExplanationSnapshot) -> ExplanationResult:
        started_at = time.perf_counter()
        data_url = self._encode_frame_as_data_url(snapshot.frame_bgr)
        request_kwargs = dict(
            model=self._model,
            instructions=_SYSTEM_INSTRUCTIONS,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": snapshot.structured_context},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            max_output_tokens=350,
            timeout=self._timeout_sec,
        )
        if str(self._model).startswith("gpt-5"):
            request_kwargs["reasoning"] = {"effort": "minimal"}
        response = self._client.responses.create(**request_kwargs)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return ExplanationResult(
            text=_response_text(response),
            status=ExplanationStatus.READY,
            latency_ms=latency_ms,
            model=self._model,
            error_message=None,
        )


@dataclass(slots=True)
class _WorkerState:
    pending_snapshot_id: int | None = None
    pending_snapshot: SituationExplanationSnapshot | None = None
    latest_result: tuple[int, ExplanationResult] | None = None
    idle: bool = True


class SituationExplanationWorker:
    def __init__(self, *, explainer) -> None:
        self._explainer = explainer
        self._state = _WorkerState()
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="situation-explainer", daemon=True)
        self._thread.start()

    def is_idle(self) -> bool:
        with self._lock:
            return self._state.idle

    def submit(self, snapshot_id: int, payload: SituationExplanationSnapshot) -> None:
        with self._lock:
            self._state.pending_snapshot_id = int(snapshot_id)
            self._state.pending_snapshot = payload
            self._state.idle = False
        self._event.set()

    def poll(self) -> tuple[int, ExplanationResult] | None:
        with self._lock:
            result = self._state.latest_result
            self._state.latest_result = None
            return result

    def _run(self) -> None:
        while True:
            self._event.wait()
            self._event.clear()
            if self._stop:
                return

            with self._lock:
                snapshot_id = self._state.pending_snapshot_id
                payload = self._state.pending_snapshot
                self._state.pending_snapshot_id = None
                self._state.pending_snapshot = None

            if payload is None or snapshot_id is None:
                continue

            try:
                result = self._explainer.explain(payload)
            except Exception as exc:  # pragma: no cover - exercised with fake explainer tests.
                result = ExplanationResult(
                    text="",
                    status=ExplanationStatus.ERROR,
                    latency_ms=None,
                    model=getattr(self._explainer, "_model", "unknown"),
                    error_message=str(exc),
                )

            with self._lock:
                self._state.latest_result = (int(snapshot_id), result)
                if self._state.pending_snapshot is None:
                    self._state.idle = True
                else:
                    self._event.set()

    def close(self) -> None:
        self._stop = True
        self._event.set()
        self._thread.join(timeout=1.0)


def wrap_explanation_text(
    text: str,
    *,
    width: int = 56,
    max_lines: int | None = 8,
) -> list[str]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    wrapped: list[str] = []
    for paragraph in paragraphs:
        wrapped.extend(textwrap.wrap(paragraph, width=width) or [""])
        if max_lines is not None and len(wrapped) >= max_lines:
            break
    if max_lines is not None and len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
    if (
        max_lines is not None
        and wrapped
        and len(wrapped) == max_lines
        and any(len(line) > width for line in paragraphs)
    ):
        wrapped[-1] = wrapped[-1].rstrip(". ") + "..."
    if max_lines is None:
        return wrapped
    return wrapped[:max_lines]
