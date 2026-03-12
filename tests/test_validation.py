from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from obj_recog.config import AppConfig
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.scene_graph import GraphEdge, GraphNode, SceneGraphSnapshot
from obj_recog.simulation import SimulationScenarioState
from obj_recog.situation_explainer import ExplanationResult, ExplanationStatus
from obj_recog.types import DepthDiagnostics, Detection, FrameArtifacts, PanopticSegment
from obj_recog.validation import RuntimeValidationProbe, run_validation_suite


def _base_config(**overrides) -> AppConfig:
    values = dict(
        camera_index=0,
        width=32,
        height=24,
        device="cpu",
        conf_threshold=0.35,
        point_stride=4,
        max_points=1000,
        input_source="sim",
        scenario="studio_open_v1",
        segmentation_mode="panoptic",
        explanation_enabled=True,
        graph_enabled=True,
    )
    values.update(overrides)
    return AppConfig(**values)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=28.0, fy=28.0, cx=16.0, cy=12.0)


def _scene_graph_snapshot(frame_index: int) -> SceneGraphSnapshot:
    return SceneGraphSnapshot(
        frame_index=frame_index,
        camera_pose_world=np.eye(4, dtype=np.float32),
        nodes=(
            GraphNode(
                id="ego",
                type="ego",
                label="camera",
                state="visible",
                confidence=1.0,
                world_centroid=np.zeros(3, dtype=np.float32),
                last_seen_frame=frame_index,
                last_seen_direction="front",
                source_track_id=None,
            ),
            GraphNode(
                id="obj_target_1",
                type="object",
                label="target",
                state="visible",
                confidence=0.95,
                world_centroid=np.array([0.0, 0.0, 3.5], dtype=np.float32),
                last_seen_frame=frame_index,
                last_seen_direction="front",
                source_track_id=1,
            ),
        ),
        edges=(
            GraphEdge(
                source="ego",
                target="obj_target_1",
                relation="front",
                confidence=0.92,
                last_updated_frame=frame_index,
                distance_bucket="mid",
                source_kind="detection",
            ),
        ),
        visible_node_ids=("ego", "obj_target_1"),
        visible_edge_keys=(("ego", "obj_target_1", "front"),),
    )


def _artifacts(
    *,
    mesh_vertices_xyz: np.ndarray,
    mesh_triangles: np.ndarray,
    mesh_vertex_colors: np.ndarray,
    detections: list[Detection],
    segments: list[PanopticSegment],
    scene_graph_snapshot: SceneGraphSnapshot | None,
    segmentation_overlay_bgr: np.ndarray | None,
    calibration_source: str = "auto",
    mesh_z_span: float = 0.8,
) -> FrameArtifacts:
    frame_bgr = np.full((24, 32, 3), 32, dtype=np.uint8)
    depth_map = np.full((24, 32), 3.6, dtype=np.float32)
    intrinsics = _intrinsics()
    dense_map_points_xyz = (
        mesh_vertices_xyz.copy()
        if mesh_vertices_xyz.size > 0
        else np.array([[0.0, 0.0, 3.6]], dtype=np.float32)
    )
    dense_map_points_rgb = (
        mesh_vertex_colors.copy()
        if mesh_vertex_colors.size > 0
        else np.array([[0.3, 0.3, 0.3]], dtype=np.float32)
    )
    return FrameArtifacts(
        frame_bgr=frame_bgr,
        intrinsics=intrinsics,
        detections=list(detections),
        depth_map=depth_map,
        points_xyz=dense_map_points_xyz,
        points_rgb=dense_map_points_rgb,
        dense_map_points_xyz=dense_map_points_xyz,
        dense_map_points_rgb=dense_map_points_rgb,
        mesh_vertices_xyz=mesh_vertices_xyz,
        mesh_triangles=mesh_triangles,
        mesh_vertex_colors=mesh_vertex_colors,
        camera_pose_world=np.eye(4, dtype=np.float32),
        tracking_ok=True,
        is_keyframe=True,
        trajectory_xyz=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        segment_id=1,
        slam_tracking_state="TRACKING",
        keyframe_id=1,
        sparse_map_points_xyz=np.empty((0, 3), dtype=np.float32),
        loop_closure_applied=False,
        segmentation_overlay_bgr=(
            np.zeros_like(frame_bgr)
            if segmentation_overlay_bgr is None
            else segmentation_overlay_bgr
        ),
        segments=list(segments),
        depth_diagnostics=DepthDiagnostics(
            calibration_source=calibration_source,
            profile="balanced",
            raw_percentiles=(0.1, 0.5, 0.9),
            normalizer_low_high=(0.2, 0.8),
            normalized_distance_percentiles=(0.1, 0.5, 0.9),
            valid_depth_ratio=1.0,
            dense_z_span=max(float(mesh_z_span), 0.2),
            mesh_z_span=float(mesh_z_span),
            intrinsics_summary=(intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy),
            hint="ok",
        ),
        scene_graph_snapshot=scene_graph_snapshot,
        visible_graph_nodes=[] if scene_graph_snapshot is None else list(scene_graph_snapshot.visible_nodes),
        visible_graph_edges=[] if scene_graph_snapshot is None else list(scene_graph_snapshot.visible_edges),
    )


def _scenario_state(*, selfcal_converged: bool, render_backend: str = "analytic") -> SimulationScenarioState:
    return SimulationScenarioState(
        scene_id="studio_open_v1",
        difficulty_level=1,
        phase="VERIFY_VIEW",
        step_index=4,
        elapsed_sec=9.5,
        selfcal_converged=selfcal_converged,
        rig_x=0.0,
        rig_z=0.0,
        yaw_deg=0.0,
        visible_labels=("target",),
        active_goal=None,
        target_motion_state="static",
        render_backend=render_backend,
    )


def test_runtime_validation_probe_marks_passes_for_complete_pipeline() -> None:
    probe = RuntimeValidationProbe(_base_config())
    probe.on_start(explanation_api_available=True)
    mesh_vertices_xyz = np.array(
        [[0.0, 0.0, 3.2], [0.2, 0.0, 3.4], [0.0, 0.2, 3.6], [0.1, 0.1, 3.5]],
        dtype=np.float32,
    )
    mesh_triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    mesh_vertex_colors = np.full((4, 3), 0.6, dtype=np.float32)
    detection = Detection(
        xyxy=(8, 5, 26, 22),
        class_id=1,
        label="target",
        confidence=0.91,
        color=(40, 80, 225),
    )
    segment = PanopticSegment(
        segment_id=3,
        label_id=4,
        label="table",
        color_rgb=(1, 2, 3),
        mask=np.ones((24, 32), dtype=bool),
        area_pixels=768,
    )
    snapshot = _scene_graph_snapshot(frame_index=0)

    probe.record_frame(
        frame_index=0,
        frame_packet=type(
            "Packet",
            (),
            {
                "scenario_state": _scenario_state(selfcal_converged=True),
                "calibration_source": "auto",
            },
        )(),
        artifacts=_artifacts(
            mesh_vertices_xyz=mesh_vertices_xyz,
            mesh_triangles=mesh_triangles,
            mesh_vertex_colors=mesh_vertex_colors,
            detections=[detection],
            segments=[segment],
            scene_graph_snapshot=snapshot,
            segmentation_overlay_bgr=np.full((24, 32, 3), 25, dtype=np.uint8),
        ),
        explanation_status=ExplanationStatus.READY,
        explanation_result=ExplanationResult(
            text="현재 장면 설명\n핵심 객체: target\n공간 관계: front\n불확실성: 낮음",
            status=ExplanationStatus.READY,
            latency_ms=20.0,
            model="gpt-5-mini",
            error_message=None,
        ),
        viewer_active=True,
    )

    report = probe.build_report()

    assert report.subsystems["reconstruction"].status == "pass"
    assert report.subsystems["calibration"].status == "pass"
    assert report.subsystems["object_detection"].status == "pass"
    assert report.subsystems["segmentation"].status == "pass"
    assert report.subsystems["scene_graph"].status == "pass"
    assert report.subsystems["llm_explanation"].status == "pass"
    assert report.render_backend == "analytic"


def test_runtime_validation_probe_marks_failures_and_skips() -> None:
    probe = RuntimeValidationProbe(_base_config())
    probe.on_start(explanation_api_available=False)
    probe.record_frame(
        frame_index=0,
        frame_packet=type(
            "Packet",
            (),
            {
                "scenario_state": _scenario_state(selfcal_converged=False, render_backend="analytic"),
                "calibration_source": "disabled/approx",
            },
        )(),
        artifacts=_artifacts(
            mesh_vertices_xyz=np.empty((0, 3), dtype=np.float32),
            mesh_triangles=np.empty((0, 3), dtype=np.int32),
            mesh_vertex_colors=np.empty((0, 3), dtype=np.float32),
            detections=[],
            segments=[],
            scene_graph_snapshot=None,
            segmentation_overlay_bgr=None,
            calibration_source="disabled/approx",
            mesh_z_span=0.0,
        ),
        explanation_status=ExplanationStatus.DISABLED,
        explanation_result=ExplanationResult(
            text="",
            status=ExplanationStatus.DISABLED,
            latency_ms=None,
            model="gpt-5-mini",
            error_message=None,
        ),
        viewer_active=True,
    )

    report = probe.build_report()

    assert report.subsystems["reconstruction"].status == "fail"
    assert report.subsystems["calibration"].status == "fail"
    assert report.subsystems["object_detection"].status == "fail"
    assert report.subsystems["segmentation"].status == "fail"
    assert report.subsystems["scene_graph"].status == "fail"
    assert report.subsystems["llm_explanation"].status == "skipped"


def test_run_validation_suite_writes_per_scenario_reports_and_summary(tmp_path: Path) -> None:
    def fake_run_fn(config, *, validation_probe=None, **_kwargs) -> None:
        assert validation_probe is not None
        validation_probe.on_start(explanation_api_available=False)
        validation_probe.record_frame(
            frame_index=0,
            frame_packet=type(
                "Packet",
                (),
                {
                    "scenario_state": SimulationScenarioState(
                        scene_id=config.scenario,
                        difficulty_level=1,
                        phase="REPORT",
                        step_index=1,
                        elapsed_sec=10.0,
                        selfcal_converged=True,
                        rig_x=0.0,
                        rig_z=0.0,
                        yaw_deg=0.0,
                        visible_labels=("target",),
                        active_goal=None,
                        target_motion_state="static",
                        render_backend="analytic",
                    ),
                    "calibration_source": "auto",
                },
            )(),
            artifacts=_artifacts(
                mesh_vertices_xyz=np.array(
                    [[0.0, 0.0, 3.2], [0.2, 0.0, 3.4], [0.0, 0.2, 3.6], [0.1, 0.1, 3.5]],
                    dtype=np.float32,
                ),
                mesh_triangles=np.array([[0, 1, 2]], dtype=np.int32),
                mesh_vertex_colors=np.full((4, 3), 0.6, dtype=np.float32),
                detections=[
                    Detection(
                        xyxy=(8, 5, 26, 22),
                        class_id=1,
                        label="target",
                        confidence=0.91,
                        color=(40, 80, 225),
                    )
                ],
                segments=[
                    PanopticSegment(
                        segment_id=3,
                        label_id=4,
                        label="table",
                        color_rgb=(1, 2, 3),
                        mask=np.ones((24, 32), dtype=bool),
                        area_pixels=768,
                    )
                ],
                scene_graph_snapshot=_scene_graph_snapshot(frame_index=0),
                segmentation_overlay_bgr=np.full((24, 32, 3), 25, dtype=np.uint8),
            ),
            explanation_status=ExplanationStatus.DISABLED,
            explanation_result=ExplanationResult(
                text="",
                status=ExplanationStatus.DISABLED,
                latency_ms=None,
                model="gpt-5-mini",
                error_message=None,
            ),
            viewer_active=True,
        )

    summary = run_validation_suite(
        _base_config(sim_max_steps=60),
        output_dir=tmp_path,
        scenarios=("studio_open_v1", "office_clutter_v1"),
        perception_modes=("assisted",),
        run_fn=fake_run_fn,
    )

    assert summary.total_runs == 2
    assert (tmp_path / "studio_open_v1-assisted.json").is_file()
    assert (tmp_path / "office_clutter_v1-assisted.json").is_file()
    summary_path = tmp_path / "summary.json"
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_runs"] == 2
    assert payload["reports"][0]["scenario"] == "studio_open_v1"
