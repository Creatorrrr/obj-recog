from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from obj_recog.config import AppConfig, SIM_SCENARIO_CHOICES
from obj_recog.opencv_runtime import load_cv2
from obj_recog.simulation import SCENARIO_SPECS


@dataclass(frozen=True, slots=True)
class SubsystemVerdict:
    status: str
    reason: str
    key_metrics: dict[str, Any]
    first_failure_frame: int | None = None
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "key_metrics": dict(self.key_metrics),
            "first_failure_frame": self.first_failure_frame,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class ScenarioValidationReport:
    scenario: str
    scenario_family: str
    difficulty_level: int
    perception_mode: str
    render_backend: str
    overall_status: str
    frame_count: int
    subsystems: dict[str, SubsystemVerdict]
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "scenario_family": self.scenario_family,
            "difficulty_level": self.difficulty_level,
            "perception_mode": self.perception_mode,
            "render_backend": self.render_backend,
            "overall_status": self.overall_status,
            "frame_count": self.frame_count,
            "error_message": self.error_message,
            "subsystems": {name: verdict.to_dict() for name, verdict in self.subsystems.items()},
        }


@dataclass(frozen=True, slots=True)
class ValidationSummaryReport:
    output_dir: str
    total_runs: int
    pass_runs: int
    warn_runs: int
    fail_runs: int
    skipped_runs: int
    reports: tuple[ScenarioValidationReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "total_runs": self.total_runs,
            "pass_runs": self.pass_runs,
            "warn_runs": self.warn_runs,
            "fail_runs": self.fail_runs,
            "skipped_runs": self.skipped_runs,
            "reports": [report.to_dict() for report in self.reports],
        }


class RuntimeValidationProbe:
    def __init__(self, config: AppConfig, *, output_dir: str | Path | None = None) -> None:
        self._config = config
        self._output_dir = None if output_dir is None else Path(output_dir)
        self._frame_count = 0
        self._first_frame_index: int | None = None
        self._scene_id = str(config.scenario)
        self._scenario_family = SCENARIO_SPECS[str(config.scenario)].scenario_family
        self._difficulty_level = int(SCENARIO_SPECS[str(config.scenario)].difficulty_level)
        self._render_backend = "unknown"
        self._render_profile = str(getattr(config, "render_profile", "fast"))
        self._semantic_target_class = "target"
        self._asset_manifest_id = ""
        self._explanation_api_available = False
        self._error_message: str | None = None

        self._mesh_nonempty_frames = 0
        self._dense_nonempty_frames = 0
        self._max_mesh_vertices = 0
        self._max_mesh_triangles = 0
        self._max_dense_points = 0
        self._max_mesh_z_span = 0.0
        self._mesh_color_nonzero_frames = 0
        self._max_mesh_color_mean = 0.0
        self._first_empty_mesh_frame: int | None = None

        self._calibration_sources: set[str] = set()
        self._selfcal_converged = False
        self._fx_values: list[float] = []
        self._fy_values: list[float] = []

        self._detection_frames = 0
        self._target_detection_frames = 0
        self._max_detection_count = 0
        self._target_confidences: list[float] = []
        self._max_target_crop_unique_colors = 0
        self._preview_shot_path: str | None = None

        self._segmentation_frames = 0
        self._overlay_shape_match_frames = 0
        self._max_segments = 0

        self._graph_frames = 0
        self._max_visible_nodes = 0
        self._max_visible_edges = 0

        self._explanation_ready_frames = 0
        self._explanation_loading_frames = 0
        self._explanation_error_frames = 0
        self._explanation_disabled_frames = 0
        self._explanation_text_frames = 0
        self._first_explanation_error_frame: int | None = None

    def on_start(self, *, explanation_api_available: bool) -> None:
        self._explanation_api_available = bool(explanation_api_available)

    def record_exception(self, error_message: str) -> None:
        self._error_message = str(error_message)

    def record_frame(
        self,
        *,
        frame_index: int,
        frame_packet,
        artifacts,
        explanation_status,
        explanation_result,
        viewer_active: bool,
    ) -> None:
        del viewer_active
        if self._first_frame_index is None:
            self._first_frame_index = int(frame_index)
        self._frame_count += 1

        scenario_state = getattr(frame_packet, "scenario_state", None)
        if scenario_state is not None:
            self._scene_id = str(getattr(scenario_state, "scene_id", self._scene_id))
            self._difficulty_level = int(getattr(scenario_state, "difficulty_level", self._difficulty_level))
            spec = SCENARIO_SPECS.get(self._scene_id)
            if spec is not None:
                self._scenario_family = spec.scenario_family
            self._render_backend = str(getattr(scenario_state, "render_backend", self._render_backend))
            self._render_profile = str(getattr(scenario_state, "render_profile", self._render_profile))
            self._semantic_target_class = str(
                getattr(scenario_state, "semantic_target_class", self._semantic_target_class)
            )
            self._asset_manifest_id = str(getattr(scenario_state, "asset_manifest_id", self._asset_manifest_id))
            self._selfcal_converged = self._selfcal_converged or bool(
                getattr(scenario_state, "selfcal_converged", False)
            )

        mesh_vertices_xyz = np.asarray(getattr(artifacts, "mesh_vertices_xyz", np.empty((0, 3), dtype=np.float32)))
        mesh_triangles = np.asarray(getattr(artifacts, "mesh_triangles", np.empty((0, 3), dtype=np.int32)))
        mesh_vertex_colors = np.asarray(
            getattr(artifacts, "mesh_vertex_colors", np.empty((0, 3), dtype=np.float32))
        )
        dense_map_points_xyz = np.asarray(
            getattr(artifacts, "dense_map_points_xyz", np.empty((0, 3), dtype=np.float32))
        )
        depth_diagnostics = getattr(artifacts, "depth_diagnostics", None)
        calibration_source = getattr(frame_packet, "calibration_source", None)
        if depth_diagnostics is not None:
            calibration_source = getattr(depth_diagnostics, "calibration_source", calibration_source)
            self._max_mesh_z_span = max(
                self._max_mesh_z_span,
                float(getattr(depth_diagnostics, "mesh_z_span", 0.0)),
            )
        if calibration_source:
            self._calibration_sources.add(str(calibration_source))

        self._fx_values.append(float(getattr(getattr(artifacts, "intrinsics", None), "fx", 0.0)))
        self._fy_values.append(float(getattr(getattr(artifacts, "intrinsics", None), "fy", 0.0)))

        mesh_vertex_count = int(mesh_vertices_xyz.shape[0]) if mesh_vertices_xyz.ndim == 2 else 0
        mesh_triangle_count = int(mesh_triangles.shape[0]) if mesh_triangles.ndim == 2 else 0
        dense_point_count = int(dense_map_points_xyz.shape[0]) if dense_map_points_xyz.ndim == 2 else 0
        self._max_mesh_vertices = max(self._max_mesh_vertices, mesh_vertex_count)
        self._max_mesh_triangles = max(self._max_mesh_triangles, mesh_triangle_count)
        self._max_dense_points = max(self._max_dense_points, dense_point_count)
        if mesh_vertex_count > 0:
            self._mesh_nonempty_frames += 1
        elif self._first_empty_mesh_frame is None:
            self._first_empty_mesh_frame = int(frame_index)
        if dense_point_count > 0:
            self._dense_nonempty_frames += 1
        if mesh_vertex_colors.size > 0:
            color_mean = float(np.mean(np.abs(mesh_vertex_colors)))
            self._max_mesh_color_mean = max(self._max_mesh_color_mean, color_mean)
            if color_mean > 1e-3:
                self._mesh_color_nonzero_frames += 1

        detections = list(getattr(artifacts, "detections", []))
        if detections:
            self._detection_frames += 1
        self._max_detection_count = max(self._max_detection_count, len(detections))
        target_detections = [
            item for item in detections if str(getattr(item, "label", "")) == self._semantic_target_class
        ]
        if target_detections:
            self._target_detection_frames += 1
            self._target_confidences.append(
                max(float(getattr(item, "confidence", 0.0)) for item in target_detections)
            )
            self._max_target_crop_unique_colors = max(
                self._max_target_crop_unique_colors,
                self._target_crop_unique_colors(getattr(artifacts, "frame_bgr"), target_detections),
            )
            self._maybe_save_preview_shot(
                frame_bgr=np.asarray(getattr(artifacts, "frame_bgr"), dtype=np.uint8),
                frame_index=frame_index,
                target_detections=target_detections,
            )

        if self._config.segmentation_mode != "off":
            segments = list(getattr(artifacts, "segments", []))
            overlay_bgr = getattr(artifacts, "segmentation_overlay_bgr", None)
            if segments:
                self._segmentation_frames += 1
                self._max_segments = max(self._max_segments, len(segments))
            if (
                overlay_bgr is not None
                and getattr(overlay_bgr, "ndim", 0) == 3
                and tuple(overlay_bgr.shape[:2]) == tuple(getattr(artifacts, "frame_bgr").shape[:2])
                and bool(np.any(np.asarray(overlay_bgr)))
            ):
                self._overlay_shape_match_frames += 1

        if self._config.graph_enabled:
            scene_graph_snapshot = getattr(artifacts, "scene_graph_snapshot", None)
            visible_nodes = list(getattr(artifacts, "visible_graph_nodes", []))
            visible_edges = list(getattr(artifacts, "visible_graph_edges", []))
            if scene_graph_snapshot is not None:
                self._graph_frames += 1
            self._max_visible_nodes = max(self._max_visible_nodes, len(visible_nodes))
            self._max_visible_edges = max(self._max_visible_edges, len(visible_edges))

        explanation_state = None if explanation_status is None else str(explanation_status)
        explanation_text = str(getattr(explanation_result, "text", "") or "")
        if explanation_state == "ready":
            self._explanation_ready_frames += 1
            if explanation_text.strip():
                self._explanation_text_frames += 1
        elif explanation_state == "loading":
            self._explanation_loading_frames += 1
        elif explanation_state == "error":
            self._explanation_error_frames += 1
            if self._first_explanation_error_frame is None:
                self._first_explanation_error_frame = int(frame_index)
        elif explanation_state == "disabled":
            self._explanation_disabled_frames += 1

    def finish(self) -> None:
        return None

    def build_report(self, *, error_message: str | None = None) -> ScenarioValidationReport:
        if error_message:
            self._error_message = str(error_message)
        subsystems = {
            "render_realism": self._render_realism_verdict(),
            "reconstruction": self._reconstruction_verdict(),
            "calibration": self._calibration_verdict(),
            "object_detection": self._object_detection_verdict(),
            "segmentation": self._segmentation_verdict(),
            "scene_graph": self._scene_graph_verdict(),
            "llm_explanation": self._llm_explanation_verdict(),
        }
        overall_status = self._overall_status(subsystems, self._error_message)
        return ScenarioValidationReport(
            scenario=self._scene_id,
            scenario_family=self._scenario_family,
            difficulty_level=self._difficulty_level,
            perception_mode=str(self._config.sim_perception_mode),
            render_backend=self._render_backend,
            overall_status=overall_status,
            frame_count=self._frame_count,
            subsystems=subsystems,
            error_message=self._error_message,
        )

    def _overall_status(
        self,
        subsystems: dict[str, SubsystemVerdict],
        error_message: str | None,
    ) -> str:
        if error_message:
            return "fail"
        statuses = [verdict.status for verdict in subsystems.values()]
        active_statuses = [status for status in statuses if status != "skipped"]
        if not active_statuses:
            return "skipped"
        if any(status == "fail" for status in active_statuses):
            return "fail"
        if any(status == "warn" for status in active_statuses):
            return "warn"
        return "pass"

    def _reconstruction_verdict(self) -> SubsystemVerdict:
        metrics = {
            "max_mesh_vertices": int(self._max_mesh_vertices),
            "max_mesh_triangles": int(self._max_mesh_triangles),
            "max_dense_points": int(self._max_dense_points),
            "max_mesh_z_span": round(float(self._max_mesh_z_span), 4),
            "mesh_nonempty_frames": int(self._mesh_nonempty_frames),
            "mesh_color_nonzero_frames": int(self._mesh_color_nonzero_frames),
        }
        if (
            self._max_mesh_vertices >= 4
            and self._max_mesh_z_span >= 0.1
            and self._mesh_color_nonzero_frames >= 1
        ):
            return SubsystemVerdict(
                status="pass",
                reason="mesh and depth geometry stayed non-empty with non-degenerate z-span",
                key_metrics=metrics,
                sample_count=self._frame_count,
            )
        if self._max_dense_points >= 4:
            return SubsystemVerdict(
                status="warn",
                reason="dense points exist but reconstructed mesh stayed sparse or flat",
                key_metrics=metrics,
                first_failure_frame=self._first_empty_mesh_frame,
                sample_count=self._frame_count,
            )
        return SubsystemVerdict(
            status="fail",
            reason="mesh remained empty or degenerate throughout the run",
            key_metrics=metrics,
            first_failure_frame=self._first_empty_mesh_frame,
            sample_count=self._frame_count,
        )

    def _calibration_verdict(self) -> SubsystemVerdict:
        fx_change_ratio = 0.0
        if self._fx_values and abs(self._fx_values[0]) > 1e-6:
            fx_change_ratio = abs(self._fx_values[-1] - self._fx_values[0]) / abs(self._fx_values[0])
        metrics = {
            "selfcal_converged": bool(self._selfcal_converged),
            "calibration_sources": sorted(self._calibration_sources),
            "fx_change_ratio": round(float(fx_change_ratio), 4),
        }
        if self._selfcal_converged and fx_change_ratio <= 0.6:
            return SubsystemVerdict(
                status="pass",
                reason="self-calibration converged and intrinsics stayed stable enough",
                key_metrics=metrics,
                sample_count=self._frame_count,
            )
        if self._selfcal_converged:
            return SubsystemVerdict(
                status="warn",
                reason="self-calibration converged but focal lengths drifted more than expected",
                key_metrics=metrics,
                sample_count=self._frame_count,
            )
        return SubsystemVerdict(
            status="fail",
            reason="self-calibration never converged",
            key_metrics=metrics,
            first_failure_frame=self._first_frame_index,
            sample_count=self._frame_count,
        )

    def _object_detection_verdict(self) -> SubsystemVerdict:
        avg_target_confidence = (
            0.0
            if not self._target_confidences
            else float(sum(self._target_confidences) / len(self._target_confidences))
        )
        metrics = {
            "detection_frames": int(self._detection_frames),
            "target_detection_frames": int(self._target_detection_frames),
            "avg_target_confidence": round(avg_target_confidence, 4),
            "max_detection_count": int(self._max_detection_count),
            "semantic_target_class": self._semantic_target_class,
        }
        if self._target_detection_frames >= 1 and avg_target_confidence >= float(self._config.conf_threshold):
            return SubsystemVerdict(
                status="pass",
                reason="target was detected with usable confidence",
                key_metrics=metrics,
                sample_count=self._frame_count,
            )
        if self._target_detection_frames >= 1:
            return SubsystemVerdict(
                status="warn",
                reason="target appeared but confidence stayed below the configured threshold",
                key_metrics=metrics,
                sample_count=self._frame_count,
            )
        return SubsystemVerdict(
            status="fail",
            reason="target was never detected",
            key_metrics=metrics,
            first_failure_frame=self._first_frame_index,
            sample_count=self._frame_count,
        )

    def _render_realism_verdict(self) -> SubsystemVerdict:
        metrics = {
            "render_profile": self._render_profile,
            "render_backend": self._render_backend,
            "semantic_target_class": self._semantic_target_class,
            "target_detection_frames": int(self._target_detection_frames),
            "max_target_crop_unique_colors": int(self._max_target_crop_unique_colors),
            "asset_manifest_present": bool(self._asset_manifest_id),
            "preview_shot_path": self._preview_shot_path,
        }
        if (
            self._target_detection_frames >= 1
            and self._max_target_crop_unique_colors >= 12
            and (not self._config.scenario_preview_shots or self._preview_shot_path is not None)
        ):
            return SubsystemVerdict(
                status="pass",
                reason="target-visible frames contained textured object appearance and preview evidence",
                key_metrics=metrics,
                sample_count=self._frame_count,
            )
        if self._target_detection_frames >= 1:
            return SubsystemVerdict(
                status="warn",
                reason="target appeared but render detail stayed sparse or preview evidence was missing",
                key_metrics=metrics,
                first_failure_frame=self._first_frame_index,
                sample_count=self._frame_count,
            )
        return SubsystemVerdict(
            status="fail",
            reason="rendered frames never produced a usable target-visible reference shot",
            key_metrics=metrics,
            first_failure_frame=self._first_frame_index,
            sample_count=self._frame_count,
        )

    def _segmentation_verdict(self) -> SubsystemVerdict:
        if self._config.segmentation_mode == "off":
            return SubsystemVerdict(
                status="skipped",
                reason="segmentation is disabled in the current configuration",
                key_metrics={"segmentation_mode": "off"},
                sample_count=self._frame_count,
            )
        metrics = {
            "segmentation_frames": int(self._segmentation_frames),
            "overlay_shape_match_frames": int(self._overlay_shape_match_frames),
            "max_segments": int(self._max_segments),
        }
        if self._segmentation_frames >= 1 and self._overlay_shape_match_frames >= 1 and self._max_segments >= 1:
            return SubsystemVerdict(
                status="pass",
                reason="segments and segmentation overlays were produced",
                key_metrics=metrics,
                sample_count=self._frame_count,
            )
        if self._overlay_shape_match_frames >= 1:
            return SubsystemVerdict(
                status="warn",
                reason="segmentation overlay was present but the segment set stayed sparse",
                key_metrics=metrics,
                first_failure_frame=self._first_frame_index,
                sample_count=self._frame_count,
            )
        return SubsystemVerdict(
            status="fail",
            reason="segmentation stayed empty while enabled",
            key_metrics=metrics,
            first_failure_frame=self._first_frame_index,
            sample_count=self._frame_count,
        )

    def _scene_graph_verdict(self) -> SubsystemVerdict:
        if not self._config.graph_enabled:
            return SubsystemVerdict(
                status="skipped",
                reason="scene graph generation is disabled in the current configuration",
                key_metrics={"graph_enabled": False},
                sample_count=self._frame_count,
            )
        metrics = {
            "graph_frames": int(self._graph_frames),
            "max_visible_nodes": int(self._max_visible_nodes),
            "max_visible_edges": int(self._max_visible_edges),
        }
        if self._graph_frames >= 1 and self._max_visible_nodes >= 2 and self._max_visible_edges >= 1:
            return SubsystemVerdict(
                status="pass",
                reason="scene graph snapshots contained visible nodes and relations",
                key_metrics=metrics,
                sample_count=self._frame_count,
            )
        if self._graph_frames >= 1 and self._max_visible_nodes >= 1:
            return SubsystemVerdict(
                status="warn",
                reason="scene graph snapshots existed but remained too sparse",
                key_metrics=metrics,
                first_failure_frame=self._first_frame_index,
                sample_count=self._frame_count,
            )
        return SubsystemVerdict(
            status="fail",
            reason="scene graph snapshots never became populated",
            key_metrics=metrics,
            first_failure_frame=self._first_frame_index,
            sample_count=self._frame_count,
        )

    def _llm_explanation_verdict(self) -> SubsystemVerdict:
        if not self._config.explanation_enabled:
            return SubsystemVerdict(
                status="skipped",
                reason="LLM explanation is disabled in the current configuration",
                key_metrics={"explanation_enabled": False},
                sample_count=self._frame_count,
            )
        metrics = {
            "ready_frames": int(self._explanation_ready_frames),
            "loading_frames": int(self._explanation_loading_frames),
            "error_frames": int(self._explanation_error_frames),
            "disabled_frames": int(self._explanation_disabled_frames),
            "text_frames": int(self._explanation_text_frames),
        }
        if not self._explanation_api_available and self._explanation_disabled_frames >= 1:
            return SubsystemVerdict(
                status="skipped",
                reason="OPENAI_API_KEY is unavailable so LLM explanation was skipped",
                key_metrics=metrics,
                sample_count=self._frame_count,
            )
        if self._explanation_ready_frames >= 1 and self._explanation_text_frames >= 1:
            return SubsystemVerdict(
                status="pass",
                reason="LLM explanation returned non-empty text",
                key_metrics=metrics,
                sample_count=self._frame_count,
            )
        if self._explanation_loading_frames >= 1 or self._explanation_error_frames >= 1:
            return SubsystemVerdict(
                status="warn",
                reason="LLM explanation was requested but did not complete cleanly",
                key_metrics=metrics,
                first_failure_frame=self._first_explanation_error_frame,
                sample_count=self._frame_count,
            )
        return SubsystemVerdict(
            status="fail",
            reason="LLM explanation was enabled but no usable response was observed",
            key_metrics=metrics,
            first_failure_frame=self._first_frame_index,
            sample_count=self._frame_count,
        )

    def _target_crop_unique_colors(self, frame_bgr: np.ndarray, target_detections: list[Any]) -> int:
        counts: list[int] = []
        for detection in target_detections:
            x1, y1, x2, y2 = (int(value) for value in getattr(detection, "xyxy"))
            crop = frame_bgr[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
            if crop.size == 0:
                continue
            unique_colors = np.unique(crop.reshape(-1, crop.shape[-1]), axis=0)
            counts.append(int(unique_colors.shape[0]))
        return max(counts, default=0)

    def _maybe_save_preview_shot(
        self,
        *,
        frame_bgr: np.ndarray,
        frame_index: int,
        target_detections: list[Any],
    ) -> None:
        if not self._config.scenario_preview_shots or self._preview_shot_path is not None or self._output_dir is None:
            return
        x1, y1, x2, y2 = (int(value) for value in getattr(target_detections[0], "xyxy"))
        crop = frame_bgr[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        if crop.size == 0:
            return
        preview_path = self._output_dir / f"{self._scene_id}-{self._config.sim_perception_mode}-preview-f{frame_index}.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop[:, :, ::-1], mode="RGB").save(preview_path)
        self._preview_shot_path = str(preview_path)


class _HeadlessCv2Proxy:
    def __init__(self, base_cv2, *, key_sequence: Iterable[int] | None = None) -> None:
        self._base_cv2 = base_cv2
        self._key_sequence = list(key_sequence or [])
        self._mouse_callbacks: dict[str, Callable[..., Any]] = {}
        self._imshow_windows: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self._base_cv2, name)

    def imshow(self, window_name: str, _image) -> None:
        self._imshow_windows.append(str(window_name))

    def waitKey(self, _delay: int = 0) -> int:
        if self._key_sequence:
            return int(self._key_sequence.pop(0))
        return -1

    def setMouseCallback(self, window_name: str, callback) -> None:
        self._mouse_callbacks[str(window_name)] = callback

    def destroyAllWindows(self) -> None:
        return None

    def destroyWindow(self, _window_name: str) -> None:
        return None

    def getWindowProperty(self, _window_name: str, _prop: int) -> float:
        return 1.0


class _ValidationViewer:
    def update(self, *_args, **_kwargs) -> bool:
        return True

    def close(self) -> None:
        return None


def run_validation_suite(
    base_config: AppConfig,
    *,
    output_dir: str | Path | None = None,
    scenarios: Iterable[str] = SIM_SCENARIO_CHOICES,
    perception_modes: Iterable[str] = ("assisted", "runtime"),
    run_fn,
    **run_kwargs,
) -> ValidationSummaryReport:
    output_path = Path(output_dir or base_config.validation_output_dir or (Path.cwd() / "validation"))
    output_path.mkdir(parents=True, exist_ok=True)
    reports: list[ScenarioValidationReport] = []
    scenario_ids = tuple(str(item) for item in scenarios)
    modes = tuple(str(item) for item in perception_modes)

    for scenario_id in scenario_ids:
        if scenario_id not in SCENARIO_SPECS:
            raise ValueError(f"unsupported scenario for validation: {scenario_id}")
        for mode in modes:
            scenario_config = replace(
                base_config,
                input_source="sim",
                list_cameras=False,
                scenario=scenario_id,
                sim_perception_mode=mode,
                validate_all_scenarios=False,
            )
            probe = RuntimeValidationProbe(scenario_config, output_dir=output_path)
            cv2_module = run_kwargs.get("cv2_module")
            if cv2_module is None:
                cv2_module = _HeadlessCv2Proxy(load_cv2(), key_sequence=[ord("e")])
            per_run_kwargs = dict(run_kwargs)
            per_run_kwargs.setdefault("cv2_module", cv2_module)
            per_run_kwargs.setdefault("viewer_factory", _ValidationViewer)
            per_run_kwargs["validation_probe"] = probe

            try:
                run_fn(scenario_config, **per_run_kwargs)
            except Exception as exc:
                probe.record_exception(str(exc))
            report = probe.build_report()
            report_file = output_path / f"{scenario_id}-{mode}.json"
            report_file.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            reports.append(report)

    summary = ValidationSummaryReport(
        output_dir=str(output_path),
        total_runs=len(reports),
        pass_runs=sum(1 for report in reports if report.overall_status == "pass"),
        warn_runs=sum(1 for report in reports if report.overall_status == "warn"),
        fail_runs=sum(1 for report in reports if report.overall_status == "fail"),
        skipped_runs=sum(1 for report in reports if report.overall_status == "skipped"),
        reports=tuple(reports),
    )
    (output_path / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def explain_api_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))
