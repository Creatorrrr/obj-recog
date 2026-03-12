from __future__ import annotations

from collections import OrderedDict
import inspect
import os
import time
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from obj_recog.auto_calibration import close_calibration_window, ensure_runtime_calibration
from obj_recog.calibration import CalibrationResult, intrinsics_from_calibration, load_orbslam3_settings
from obj_recog.camera import CameraSession, list_available_cameras, open_camera, read_camera_frame
from obj_recog.config import AppConfig, parse_config, resolve_depth_profile, resolve_device
from obj_recog.depth import DepthEstimator
from obj_recog.detector import ObjectDetector
from obj_recog.frame_source import FramePacket, LiveCameraFrameSource
from obj_recog.mapping import LocalMapBuilder, TsdfMeshMapBuilder
from obj_recog.opencv_runtime import load_cv2
from obj_recog.reconstruct import depth_to_point_cloud, intrinsics_for_frame
from obj_recog.scene_graph import SceneGraphMemory
from obj_recog.segmenter import PanopticSegmenter, SegmentationWorker
from obj_recog.slam_bridge import OrbSlam3Bridge, SlamFrameResult
from obj_recog.simulation import SimulationRuntime
from obj_recog.situation_explainer import (
    ExplanationResult,
    ExplanationStatus,
    OpenAISituationExplainer,
    SituationExplanationWorker,
    build_explanation_snapshot,
)
from obj_recog.tracking import PoseTracker
from obj_recog.types import DepthDiagnostics, Detection, FrameArtifacts, SegmentationResult
from obj_recog.visualization import (
    Open3DMeshViewer,
    draw_detections,
    explanation_button_rect,
    point_in_rect,
    render_explanation_panel,
)


def _default_debug_log(message: str) -> None:
    print(f"[obj-recog] {message}", file=sys.stderr, flush=True)


def resize_for_inference(frame_bgr: np.ndarray, inference_width: int) -> tuple[np.ndarray, float, float]:
    height, width = frame_bgr.shape[:2]
    if width <= inference_width:
        return frame_bgr, 1.0, 1.0

    cv2 = load_cv2()

    scale = inference_width / float(width)
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_bgr, (inference_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized, width / float(inference_width), height / float(resized_height)


def _scale_detection(detection: Detection, scale_x: float, scale_y: float) -> Detection:
    x1, y1, x2, y2 = detection.xyxy
    return Detection(
        xyxy=(
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        ),
        class_id=detection.class_id,
        label=detection.label,
        confidence=detection.confidence,
        color=detection.color,
    )


def resize_for_slam(frame_bgr: np.ndarray, slam_width: int, slam_height: int, *, cv2_module=None) -> np.ndarray:
    cv2 = load_cv2(cv2_module)
    resized = cv2.resize(frame_bgr, (slam_width, slam_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _legacy_tracking_to_slam_result(tracking_result) -> SlamFrameResult:
    pose_world = getattr(tracking_result, "camera_pose_world", None)
    if pose_world is None:
        pose_world = getattr(tracking_result, "pose_world")
    return SlamFrameResult(
        tracking_state="TRACKING" if getattr(tracking_result, "tracking_ok", False) else "LOST",
        pose_world=np.asarray(pose_world, dtype=np.float32),
        keyframe_inserted=bool(getattr(tracking_result, "did_reset", False)),
        keyframe_id=None,
        optimized_keyframe_poses={},
        sparse_map_points_xyz=np.empty((0, 3), dtype=np.float32),
        loop_closure_applied=False,
    )


def _update_map_builder(
    map_builder,
    slam_result: SlamFrameResult,
    frame_index: int,
    frame_points_xyz: np.ndarray,
    frame_points_rgb: np.ndarray,
    frame_bgr: np.ndarray,
    depth_map: np.ndarray,
    intrinsics,
):
    try:
        return map_builder.update(
            slam_result=slam_result,
            frame_bgr=frame_bgr,
            depth_map=depth_map,
            intrinsics=intrinsics,
        )
    except TypeError:
        pass

    try:
        return map_builder.update(
            slam_result=slam_result,
            frame_points_xyz=frame_points_xyz,
            frame_points_rgb=frame_points_rgb,
        )
    except TypeError:
        return map_builder.update(
            frame_index=frame_index,
            pose_world=slam_result.pose_world,
            frame_points_xyz=frame_points_xyz,
            frame_points_rgb=frame_points_rgb,
            did_reset=slam_result.keyframe_inserted,
        )


def _build_map_builder(map_builder_factory, *candidate_kwargs_sets):
    try:
        signature = inspect.signature(map_builder_factory)
    except (TypeError, ValueError):
        signature = None

    if signature is None:
        return map_builder_factory(**candidate_kwargs_sets[0])

    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )

    for candidate_kwargs in candidate_kwargs_sets:
        if accepts_var_kwargs:
            return map_builder_factory(**candidate_kwargs)

        filtered_kwargs = {key: value for key, value in candidate_kwargs.items() if key in parameters}
        missing_required = [
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect._empty
            and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and name not in filtered_kwargs
        ]
        if not missing_required:
            return map_builder_factory(**filtered_kwargs)

    return map_builder_factory(**candidate_kwargs_sets[0])


class _FramePacketOnlyDetector:
    def detect(self, _frame_bgr):
        raise RuntimeError("runtime detector path disabled while sim ground-truth perception is enabled")


class _FramePacketOnlyDepthEstimator:
    def estimate(self, _frame_bgr):
        raise RuntimeError("runtime depth path disabled while sim ground-truth perception is enabled")


def _update_viewer(viewer, artifacts: FrameArtifacts) -> bool:
    try:
        return viewer.update(
            artifacts.mesh_vertices_xyz,
            artifacts.mesh_triangles,
            artifacts.mesh_vertex_colors,
            artifacts.scene_graph_snapshot,
        )
    except TypeError:
        return viewer.update(
            artifacts.mesh_vertices_xyz,
            artifacts.mesh_triangles,
            artifacts.mesh_vertex_colors,
        )


def process_frame(
    frame_bgr: np.ndarray,
    detector: ObjectDetector,
    depth_estimator: DepthEstimator,
    map_builder,
    config: AppConfig,
    frame_index: int,
    timestamp_sec: float | None,
    cached_detections: list[Detection],
    calibration: CalibrationResult | None = None,
    calibration_source: str = "disabled/approx",
    *,
    slam_bridge=None,
    tracker=None,
    frame_packet: FramePacket | None = None,
    prefer_frame_packet_ground_truth: bool = False,
    assist_frame_packet_ground_truth: bool = False,
    cv2_module=None,
) -> tuple[FrameArtifacts, list[Detection]]:
    cv2 = load_cv2(cv2_module)
    inference_frame, scale_x, scale_y = resize_for_inference(frame_bgr, config.inference_width)

    packet_detections = None if frame_packet is None else frame_packet.detections
    if prefer_frame_packet_ground_truth and packet_detections is not None:
        detections = list(packet_detections)
    elif frame_index % config.detection_interval == 0 or not cached_detections:
        detections = detector.detect(inference_frame)
        detections = [_scale_detection(detection, scale_x, scale_y) for detection in detections]
    else:
        detections = cached_detections
    if assist_frame_packet_ground_truth and not detections and packet_detections is not None:
        detections = list(packet_detections)

    packet_depth_map = None if frame_packet is None else frame_packet.depth_map
    depth_map = (
        packet_depth_map
        if (prefer_frame_packet_ground_truth or assist_frame_packet_ground_truth) and packet_depth_map is not None
        else depth_estimator.estimate(inference_frame)
    )
    if depth_map.shape != frame_bgr.shape[:2]:
        depth_map = cv2.resize(
            depth_map,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    if calibration is not None:
        intrinsics = intrinsics_from_calibration(
            calibration,
            target_width=frame_bgr.shape[1],
            target_height=frame_bgr.shape[0],
        )
    elif frame_packet is not None and frame_packet.intrinsics_gt is not None:
        intrinsics = frame_packet.intrinsics_gt
    else:
        intrinsics = intrinsics_for_frame(frame_bgr.shape[1], frame_bgr.shape[0])
    depth_profile = resolve_depth_profile(config.depth_profile)
    if getattr(map_builder, "requires_point_cloud", True):
        points_xyz, point_colors, _point_pixels = depth_to_point_cloud(
            frame_bgr=frame_bgr,
            depth_map=depth_map,
            intrinsics=intrinsics,
            stride=config.point_stride,
            max_points=config.max_points,
            max_depth=depth_profile.max_depth,
        )
        map_point_colors = point_colors
    else:
        points_xyz = np.empty((0, 3), dtype=np.float32)
        map_point_colors = np.empty((0, 3), dtype=np.float32)
    if prefer_frame_packet_ground_truth and frame_packet is not None and frame_packet.pose_world_gt is not None:
        slam_result = SlamFrameResult(
            tracking_state=str(frame_packet.tracking_state),
            pose_world=np.asarray(frame_packet.pose_world_gt, dtype=np.float32),
            keyframe_inserted=bool(frame_packet.keyframe_inserted),
            keyframe_id=frame_packet.keyframe_id,
            optimized_keyframe_poses={
                int(key): np.asarray(value, dtype=np.float32)
                for key, value in frame_packet.optimized_keyframe_poses.items()
            },
            sparse_map_points_xyz=np.asarray(frame_packet.sparse_map_points_xyz, dtype=np.float32).reshape(-1, 3),
            loop_closure_applied=bool(frame_packet.loop_closure_applied),
            tracked_feature_count=int(frame_packet.tracked_feature_count),
            median_reprojection_error=frame_packet.median_reprojection_error,
            keyframe_observations=list(frame_packet.keyframe_observations),
            map_points_changed=bool(frame_packet.map_points_changed),
        )
    elif slam_bridge is not None:
        if timestamp_sec is None:
            raise RuntimeError("SLAM bridge requires a real frame timestamp in seconds")
        slam_gray = resize_for_slam(frame_bgr, config.slam_width, config.slam_height, cv2_module=cv2)
        slam_result = slam_bridge.track(slam_gray, float(timestamp_sec))
    elif tracker is not None:
        tracking_result = tracker.update(
            frame_bgr=frame_bgr,
            depth_map=depth_map,
            intrinsics=intrinsics,
        )
        slam_result = _legacy_tracking_to_slam_result(tracking_result)
    else:
        raise RuntimeError("process_frame requires either slam_bridge or tracker")

    camera_pose_world = np.asarray(slam_result.pose_world, dtype=np.float32)
    map_update = _update_map_builder(
        map_builder,
        slam_result,
        frame_index,
        points_xyz,
        map_point_colors,
        frame_bgr,
        depth_map,
        intrinsics,
    )
    mesh_vertices_xyz = np.asarray(
        getattr(map_update, "mesh_vertices_xyz", getattr(map_update, "dense_map_points_xyz", np.empty((0, 3), dtype=np.float32))),
        dtype=np.float32,
    ).reshape(-1, 3)
    mesh_triangles = np.asarray(
        getattr(map_update, "mesh_triangles", np.empty((0, 3), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1, 3)
    mesh_vertex_colors = np.asarray(
        getattr(map_update, "mesh_vertex_colors", getattr(map_update, "dense_map_points_rgb", np.empty((0, 3), dtype=np.float32))),
        dtype=np.float32,
    ).reshape(-1, 3)
    dense_map_points_xyz = np.asarray(
        getattr(map_update, "dense_map_points_xyz", mesh_vertices_xyz),
        dtype=np.float32,
    ).reshape(-1, 3)
    dense_map_points_rgb = np.asarray(
        getattr(map_update, "dense_map_points_rgb", mesh_vertex_colors),
        dtype=np.float32,
    ).reshape(-1, 3)
    dense_z_span = _camera_space_z_span(dense_map_points_xyz, camera_pose_world)
    mesh_z_span = _camera_space_z_span(mesh_vertices_xyz, camera_pose_world)

    last_diagnostics_getter = getattr(depth_estimator, "last_diagnostics", None)
    depth_diagnostics = last_diagnostics_getter() if callable(last_diagnostics_getter) else None
    if depth_diagnostics is None:
        valid = np.isfinite(depth_map) & (depth_map > 0.05)
        if np.any(valid):
            values = depth_map[valid]
            raw_percentiles = (
                float(np.percentile(values, 5.0)),
                float(np.percentile(values, 50.0)),
                float(np.percentile(values, 95.0)),
            )
            normalized_distance_percentiles = (
                float(np.percentile(values, 10.0)),
                float(np.percentile(values, 50.0)),
                float(np.percentile(values, 90.0)),
            )
            normalizer_low_high = (
                float(np.percentile(values, depth_profile.low_percentile)),
                float(np.percentile(values, depth_profile.high_percentile)),
            )
            valid_depth_ratio = float(valid.mean())
        else:
            raw_percentiles = (0.0, 0.0, 0.0)
            normalized_distance_percentiles = (
                depth_profile.min_depth,
                depth_profile.min_depth,
                depth_profile.min_depth,
            )
            normalizer_low_high = (0.0, 0.0)
            valid_depth_ratio = 0.0
        depth_diagnostics = DepthDiagnostics(
            calibration_source=calibration_source,
            profile=depth_profile.name,
            raw_percentiles=raw_percentiles,
            normalizer_low_high=normalizer_low_high,
            normalized_distance_percentiles=normalized_distance_percentiles,
            valid_depth_ratio=valid_depth_ratio,
            dense_z_span=dense_z_span,
            mesh_z_span=mesh_z_span,
            intrinsics_summary=(
                float(intrinsics.fx),
                float(intrinsics.fy),
                float(intrinsics.cx),
                float(intrinsics.cy),
            ),
            hint=_depth_hint(
                calibration_source=calibration_source,
                profile=depth_profile.name,
                normalized_distance_percentiles=normalized_distance_percentiles,
                dense_z_span=dense_z_span,
                mesh_z_span=mesh_z_span,
            ),
        )
    else:
        depth_diagnostics = DepthDiagnostics(
            calibration_source=calibration_source,
            profile=depth_diagnostics.profile,
            raw_percentiles=depth_diagnostics.raw_percentiles,
            normalizer_low_high=depth_diagnostics.normalizer_low_high,
            normalized_distance_percentiles=depth_diagnostics.normalized_distance_percentiles,
            valid_depth_ratio=depth_diagnostics.valid_depth_ratio,
            dense_z_span=dense_z_span,
            mesh_z_span=mesh_z_span,
            intrinsics_summary=(
                float(intrinsics.fx),
                float(intrinsics.fy),
                float(intrinsics.cx),
                float(intrinsics.cy),
            ),
            hint=_depth_hint(
                calibration_source=calibration_source,
                profile=depth_diagnostics.profile,
                normalized_distance_percentiles=depth_diagnostics.normalized_distance_percentiles,
                dense_z_span=dense_z_span,
                mesh_z_span=mesh_z_span,
            ),
        )

    artifacts = FrameArtifacts(
        frame_bgr=frame_bgr,
        intrinsics=intrinsics,
        detections=list(detections),
        depth_map=depth_map,
        points_xyz=mesh_vertices_xyz,
        points_rgb=mesh_vertex_colors,
        dense_map_points_xyz=dense_map_points_xyz,
        dense_map_points_rgb=dense_map_points_rgb,
        mesh_vertices_xyz=mesh_vertices_xyz,
        mesh_triangles=mesh_triangles,
        mesh_vertex_colors=mesh_vertex_colors,
        camera_pose_world=camera_pose_world,
        tracking_ok=slam_result.tracking_ok,
        is_keyframe=bool(map_update.is_keyframe),
        trajectory_xyz=map_update.trajectory_xyz,
        segment_id=int(getattr(map_update, "segment_id", 0)),
        slam_tracking_state=slam_result.tracking_state,
        keyframe_id=getattr(map_update, "keyframe_id", slam_result.keyframe_id),
        sparse_map_points_xyz=getattr(map_update, "sparse_map_points_xyz", slam_result.sparse_map_points_xyz),
        loop_closure_applied=bool(getattr(map_update, "loop_closure_applied", slam_result.loop_closure_applied)),
        segmentation_overlay_bgr=np.zeros_like(frame_bgr),
        segments=[],
        depth_diagnostics=depth_diagnostics,
    )
    return artifacts, list(detections)

def window_is_visible(cv2_module, window_name: str) -> bool:
    try:
        return cv2_module.getWindowProperty(window_name, cv2_module.WND_PROP_VISIBLE) >= 1.0
    except Exception:
        return False


def _show_runtime_loading_preview(cv2_module, width: int, height: int, message: str) -> None:
    frame = np.zeros((max(height, 240), max(width, 320), 3), dtype=np.uint8)
    cv2_module.putText(
        frame,
        message,
        (24, max(48, frame.shape[0] // 2)),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 255),
        2,
        cv2_module.LINE_AA,
    )
    cv2_module.imshow("Object Recognition", frame)
    cv2_module.waitKey(1)


def _camera_space_z_span(points_xyz: np.ndarray, camera_pose_world: np.ndarray) -> float:
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points_xyz.size == 0:
        return 0.0
    pose_inv = np.linalg.inv(np.asarray(camera_pose_world, dtype=np.float32))
    homogeneous = np.concatenate(
        (points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float32)),
        axis=1,
    )
    camera_points = (pose_inv @ homogeneous.T).T[:, :3]
    camera_z = camera_points[:, 2]
    valid = np.isfinite(camera_z) & (camera_z > 1e-4)
    if not np.any(valid):
        return 0.0
    z_values = camera_z[valid]
    if z_values.size == 1:
        return float(z_values[0])
    return float(np.percentile(z_values, 90.0) - np.percentile(z_values, 10.0))


def _depth_hint(
    *,
    calibration_source: str,
    profile: str,
    normalized_distance_percentiles: tuple[float, float, float],
    dense_z_span: float,
    mesh_z_span: float,
) -> str:
    if calibration_source == "disabled/approx":
        return "approx intrinsics in use"
    if dense_z_span > 0.0 and mesh_z_span > 0.0 and mesh_z_span < (dense_z_span * 0.6) and profile == "fast":
        return "mesh simplification likely flattening"
    spread = float(normalized_distance_percentiles[2] - normalized_distance_percentiles[0])
    if spread < 1.8:
        return "depth normalization compression likely"
    return "monocular pseudo-depth scale limited"


def _normalize_calibration_source(raw_source: str | None) -> str:
    source = str(raw_source or "").strip().lower()
    if source == "explicit":
        return "explicit"
    if source in {"auto", "cache"}:
        return "auto"
    return "disabled/approx"


def _load_app_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:  # pragma: no cover - dependency is optional at import time.
        dotenv_path = Path.cwd() / ".env"
        if not dotenv_path.is_file():
            return
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)


def run(
    config: AppConfig,
    *,
    cv2_module=None,
    detector_factory=ObjectDetector,
    depth_estimator_factory=DepthEstimator,
    tracker_factory=PoseTracker,
    map_builder_factory=LocalMapBuilder,
    slam_bridge_factory=None,
    segmenter_factory=PanopticSegmenter,
    segmentation_worker_factory=SegmentationWorker,
    situation_explainer_factory=None,
    explanation_worker_factory=SituationExplanationWorker,
    viewer_factory=Open3DMeshViewer,
    open_camera_fn=open_camera,
    camera_lister=list_available_cameras,
    runtime_calibration_resolver=ensure_runtime_calibration,
    overlay_renderer=draw_detections,
    explanation_snapshot_builder=build_explanation_snapshot,
    explanation_panel_renderer=render_explanation_panel,
    scene_graph_memory_factory=SceneGraphMemory,
    frame_source_factory=None,
    time_source=time.perf_counter,
    debug_log=_default_debug_log,
    validation_probe=None,
) -> None:
    cv2 = load_cv2(cv2_module)
    _load_app_dotenv()
    effective_device = resolve_device(config.device)
    depth_profile = resolve_depth_profile(config.depth_profile)
    camera_session: CameraSession | None = None
    viewer = None
    slam_bridge = None
    segmentation_worker = None
    explanation_worker = None
    scene_graph_memory = None
    calibration = None
    runtime_settings_path = config.camera_calibration
    explanation_status = (
        ExplanationStatus.IDLE if config.explanation_enabled else None
    )
    explanation_result = ExplanationResult(
        text="",
        status=ExplanationStatus.IDLE,
        latency_ms=None,
        model=config.explanation_model,
        error_message=None,
    )
    explanation_snapshot_id = 0
    latest_explanation_request_id: int | None = None
    latest_explanation_timestamp = "-"
    explanation_refresh_status = "idle"
    explanation_auto_refresh_enabled = False
    last_explanation_request_time: float | None = None
    explanation_button_state = {
        "pending_toggle": False,
        "rect": None,
    }
    explanation_panel_state = {
        "scroll_offset": 0,
        "pending_scroll": 0,
        "up_rect": None,
        "down_rect": None,
    }
    explanation_mouse_callback_registered = False
    explanation_panel_mouse_callback_registered = False
    depth_debug_level = "basic"
    calibration_source = "disabled/approx"
    frame_source = None
    sim_perception_mode = getattr(config, "sim_perception_mode", "assisted")
    use_frame_packet_ground_truth = config.input_source == "sim" and sim_perception_mode == "ground_truth"
    assist_frame_packet_ground_truth = config.input_source == "sim" and sim_perception_mode == "assisted"

    def _default_sim_frame_source(current_config: AppConfig):
        report_path = Path.cwd() / "reports" / f"{current_config.scenario}-seed{current_config.sim_seed}.json"
        return SimulationRuntime(
            config=current_config,
            report_path=report_path,
            cv2_module=cv2,
        ).create_frame_source()

    def _live_frame_timestamp() -> float | None:
        if slam_time_origin is None:
            return None
        return max(0.0, time_source() - slam_time_origin)

    def _build_live_frame_source(current_session: CameraSession):
        if frame_source_factory is not None:
            return frame_source_factory(
                config,
                camera_session=current_session,
                cv2_module=cv2,
                time_source=_live_frame_timestamp,
                frame_reader=read_camera_frame,
            )
        return LiveCameraFrameSource(
            camera_session=current_session,
            time_source=_live_frame_timestamp,
            frame_reader=read_camera_frame,
        )

    def _handle_object_window_mouse(event, x, y, _flags, _param) -> None:
        if event != getattr(cv2, "EVENT_LBUTTONDOWN", None):
            return
        if point_in_rect(int(x), int(y), explanation_button_state.get("rect")):
            explanation_button_state["pending_toggle"] = True

    def _handle_explanation_window_mouse(event, x, y, _flags, _param) -> None:
        if event == getattr(cv2, "EVENT_LBUTTONDOWN", None):
            if point_in_rect(int(x), int(y), explanation_panel_state.get("up_rect")):
                explanation_panel_state["pending_scroll"] = int(explanation_panel_state["pending_scroll"]) - 1
            elif point_in_rect(int(x), int(y), explanation_panel_state.get("down_rect")):
                explanation_panel_state["pending_scroll"] = int(explanation_panel_state["pending_scroll"]) + 1
            return
        if event == getattr(cv2, "EVENT_MOUSEWHEEL", None):
            get_mouse_wheel_delta = getattr(cv2, "getMouseWheelDelta", None)
            try:
                delta = int(get_mouse_wheel_delta(_flags)) if callable(get_mouse_wheel_delta) else int(_flags)
            except Exception:
                delta = 0
            if delta == 0:
                return
            step_count = max(1, abs(int(delta)) // 120) if abs(int(delta)) >= 120 else 1
            direction = -1 if delta > 0 else 1
            explanation_panel_state["pending_scroll"] = int(explanation_panel_state["pending_scroll"]) + (
                direction * step_count
            )

    def _submit_explanation_request(
        artifacts: FrameArtifacts,
        *,
        requested_at: float | None = None,
    ) -> bool:
        nonlocal explanation_snapshot_id
        nonlocal latest_explanation_request_id
        nonlocal latest_explanation_timestamp
        nonlocal explanation_status
        nonlocal explanation_result
        nonlocal explanation_refresh_status
        nonlocal last_explanation_request_time
        if not config.explanation_enabled or explanation_worker is None:
            return False
        preserve_displayed_explanation = bool((explanation_result.text or "").strip()) and (
            explanation_status == ExplanationStatus.READY
        )
        request_timestamp_label = time.strftime("%H:%M:%S")
        explanation_snapshot_id += 1
        latest_explanation_request_id = explanation_snapshot_id
        last_explanation_request_time = (
            float(requested_at) if requested_at is not None else float(time_source())
        )
        explanation_refresh_status = "updating"
        if not preserve_displayed_explanation:
            latest_explanation_timestamp = request_timestamp_label
        snapshot = explanation_snapshot_builder(
            artifacts,
            snapshot_id=explanation_snapshot_id,
            max_detections=config.explanation_max_detections,
            max_graph_nodes=config.explanation_max_graph_nodes,
            max_graph_edges=config.explanation_max_graph_edges,
            timestamp_label=request_timestamp_label,
        )
        explanation_worker.submit(snapshot.snapshot_id, snapshot)
        if not preserve_displayed_explanation:
            explanation_status = ExplanationStatus.LOADING
            explanation_result = ExplanationResult(
                text="",
                status=ExplanationStatus.LOADING,
                latency_ms=None,
                model=config.explanation_model,
                error_message=None,
            )
            explanation_panel_state["scroll_offset"] = 0
            explanation_panel_state["pending_scroll"] = 0
        return True

    def _toggle_explanation_auto_refresh(
        artifacts: FrameArtifacts,
        *,
        toggled_at: float | None = None,
    ) -> None:
        nonlocal explanation_auto_refresh_enabled
        nonlocal explanation_status
        nonlocal explanation_result
        nonlocal latest_explanation_request_id
        nonlocal latest_explanation_timestamp
        nonlocal explanation_refresh_status
        nonlocal last_explanation_request_time
        if not config.explanation_enabled or explanation_worker is None:
            explanation_auto_refresh_enabled = False
            explanation_refresh_status = "idle"
            return
        explanation_auto_refresh_enabled = not explanation_auto_refresh_enabled
        if explanation_auto_refresh_enabled:
            _submit_explanation_request(artifacts, requested_at=toggled_at)
        else:
            latest_explanation_request_id = None
            last_explanation_request_time = None
            explanation_refresh_status = "idle"
            if explanation_status == ExplanationStatus.LOADING:
                latest_explanation_timestamp = "-"
                explanation_result = ExplanationResult(
                    text="",
                    status=ExplanationStatus.IDLE,
                    latency_ms=None,
                    model=config.explanation_model,
                    error_message=None,
                )
                explanation_status = ExplanationStatus.IDLE
                explanation_panel_state["scroll_offset"] = 0
                explanation_panel_state["pending_scroll"] = 0

    if config.list_cameras:
        for device in camera_lister():
            print(f"{device.index}: {device.name}")
        return

    use_slam_bridge = slam_bridge_factory is not None and config.input_source != "sim"
    if use_slam_bridge:
        if not config.slam_vocabulary:
            raise RuntimeError("ORB-SLAM3 requires --slam-vocabulary")
        if not Path(config.slam_vocabulary).is_file():
            raise RuntimeError(f"SLAM vocabulary file not found: {config.slam_vocabulary}")
        if config.camera_calibration and Path(config.camera_calibration).is_file():
            calibration = load_orbslam3_settings(config.camera_calibration)

    cached_detections: list[Detection] = []
    cached_segmentation: SegmentationResult | None = None
    segmentation_frame_metadata: OrderedDict[int, dict[str, object]] = OrderedDict()
    frame_index = 0
    last_frame_time = 0.0
    slam_time_origin = None
    object_window_has_been_visible = False

    try:
        if config.input_source == "sim":
            frame_source = (
                frame_source_factory(config, cv2_module=cv2, time_source=time_source)
                if frame_source_factory is not None
                else _default_sim_frame_source(config)
            )
        else:
            camera_session = open_camera_fn(
                config,
                cv2_module=cv2,
                preferred_name=config.camera_name,
            )
            if use_slam_bridge:
                runtime_calibration = runtime_calibration_resolver(
                    config,
                    camera_session,
                    cv2_module=cv2,
                    frame_reader=read_camera_frame,
                    slam_bridge_factory=slam_bridge_factory,
                    time_source=time_source,
                    debug_log=debug_log,
                )
                calibration = runtime_calibration.calibration
                runtime_settings_path = runtime_calibration.settings_path
                calibration_source = _normalize_calibration_source(getattr(runtime_calibration, "source", None))
                debug_log(
                    f"runtime calibration ready (source={runtime_calibration.source}, settings={runtime_settings_path})"
                )
                if getattr(runtime_calibration, "source", None) in {"auto", "cache"}:
                    close_calibration_window(cv2)
                    _show_runtime_loading_preview(
                        cv2,
                        int(config.width),
                        int(config.height),
                        "Calibration complete. Loading models...",
                    )
            frame_source = _build_live_frame_source(camera_session)
        if use_frame_packet_ground_truth:
            detector = _FramePacketOnlyDetector()
            depth_estimator = _FramePacketOnlyDepthEstimator()
        elif assist_frame_packet_ground_truth:
            debug_log("detector init start")
            detector = detector_factory(conf_threshold=config.conf_threshold, device=effective_device)
            debug_log("detector init done")
            depth_estimator = _FramePacketOnlyDepthEstimator()
        else:
            debug_log("detector init start")
            detector = detector_factory(conf_threshold=config.conf_threshold, device=effective_device)
            debug_log("detector init done")
            debug_log("depth init start")
            try:
                depth_estimator = depth_estimator_factory(
                    device=effective_device,
                    profile=depth_profile.name,
                    low_percentile=depth_profile.low_percentile,
                    high_percentile=depth_profile.high_percentile,
                    min_depth=depth_profile.min_depth,
                    max_depth=depth_profile.max_depth,
                    gamma=depth_profile.gamma,
                )
            except TypeError:
                depth_estimator = depth_estimator_factory(device=effective_device)
            debug_log("depth init done")
        if config.segmentation_mode != "off":
            try:
                debug_log("segmentation init start")
                segmentation_worker = segmentation_worker_factory(
                    segmenter=segmenter_factory(
                        device=effective_device,
                        input_size=config.segmentation_input_size,
                    )
                )
                debug_log("segmentation init done")
            except Exception as exc:
                debug_log(f"segmentation disabled ({exc})")
        if config.explanation_enabled:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                explanation_status = ExplanationStatus.DISABLED
            else:
                try:
                    debug_log("explanation init start")
                    explainer_factory = situation_explainer_factory or OpenAISituationExplainer
                    explainer = explainer_factory(
                        model=config.explanation_model,
                        timeout_sec=config.explanation_timeout_sec,
                        api_key=api_key,
                        cv2_module=cv2,
                    )
                    explanation_worker = explanation_worker_factory(explainer=explainer)
                    debug_log("explanation init done")
                except Exception as exc:
                    explanation_status = ExplanationStatus.ERROR
                    explanation_result = ExplanationResult(
                        text="",
                        status=ExplanationStatus.ERROR,
                        latency_ms=None,
                        model=config.explanation_model,
                        error_message=str(exc),
                    )
                    explanation_refresh_status = "failed"
                    debug_log(f"explanation disabled ({exc})")
        if validation_probe is not None and hasattr(validation_probe, "on_start"):
            validation_probe.on_start(
                explanation_api_available=bool(config.explanation_enabled and os.getenv("OPENAI_API_KEY"))
            )
        tracker = None
        if use_slam_bridge:
            slam_bridge = getattr(runtime_calibration, "promoted_bridge", getattr(runtime_calibration, "bridge", None))
            if slam_bridge is None:
                debug_log("creating runtime SLAM bridge")
                slam_bridge = slam_bridge_factory(
                    vocabulary_path=config.slam_vocabulary,
                    settings_path=runtime_settings_path,
                    frame_width=config.slam_width,
                    frame_height=config.slam_height,
                )
            else:
                debug_log("reusing promoted SLAM bridge from calibration")
            map_builder = _build_map_builder(
                map_builder_factory,
                dict(
                    window_keyframes=config.mapping_window_keyframes,
                    voxel_size=depth_profile.voxel_size,
                    max_mesh_triangles=depth_profile.max_mesh_triangles,
                    depth_trunc=depth_profile.max_depth,
                    depth_sampling_stride=depth_profile.depth_sampling_stride,
                ),
                dict(
                    translation_threshold=config.keyframe_translation,
                    rotation_threshold_deg=config.keyframe_rotation_deg,
                    frame_interval=12,
                    window_keyframes=config.mapping_window_keyframes,
                    voxel_size=config.map_voxel_size,
                    max_map_points=config.max_map_points,
                ),
            )
        else:
            tracker = tracker_factory(orb_features=config.orb_features)
            map_builder = _build_map_builder(
                map_builder_factory,
                dict(
                    translation_threshold=config.keyframe_translation,
                    rotation_threshold_deg=config.keyframe_rotation_deg,
                    frame_interval=12,
                    window_keyframes=config.mapping_window_keyframes,
                    voxel_size=config.map_voxel_size,
                    max_map_points=config.max_map_points,
                ),
                dict(
                    window_keyframes=config.mapping_window_keyframes,
                    voxel_size=depth_profile.voxel_size,
                    max_mesh_triangles=depth_profile.max_mesh_triangles,
                    depth_trunc=depth_profile.max_depth,
                    depth_sampling_stride=depth_profile.depth_sampling_stride,
                ),
            )
        if config.graph_enabled:
            scene_graph_memory = scene_graph_memory_factory(
                graph_max_visible_nodes=config.graph_max_visible_nodes,
                graph_relation_smoothing_frames=config.graph_relation_smoothing_frames,
                occlusion_ttl_frames=config.graph_occlusion_ttl_frames,
            )
        viewer = viewer_factory()
        debug_log("viewer init done")
        last_frame_time = time_source()
        if use_slam_bridge:
            slam_time_origin = time_source()

        while True:
            frame_packet = None if frame_source is None else frame_source.next_frame(timeout_sec=1.0)
            if frame_packet is None:
                restarted_stream = False
                if (
                    config.input_source != "sim"
                    and camera_session is not None
                    and camera_session.requested_name
                    and not camera_session.used_fallback
                ):
                    if frame_source is not None:
                        frame_source.close()
                    camera_session = open_camera_fn(
                        config,
                        cv2_module=cv2,
                        preferred_name=config.camera_name,
                        force_default=True,
                    )
                    if tracker is not None:
                        tracker.reset()
                    if slam_bridge is not None:
                        runtime_calibration = runtime_calibration_resolver(
                            config,
                            camera_session,
                            cv2_module=cv2,
                            frame_reader=read_camera_frame,
                            slam_bridge_factory=slam_bridge_factory,
                            time_source=time_source,
                        )
                        calibration = runtime_calibration.calibration
                        runtime_settings_path = runtime_calibration.settings_path
                        calibration_source = _normalize_calibration_source(
                            getattr(runtime_calibration, "source", None)
                        )
                        slam_bridge.close()
                        slam_bridge = getattr(runtime_calibration, "promoted_bridge", getattr(runtime_calibration, "bridge", None))
                        if slam_bridge is None:
                            slam_bridge = slam_bridge_factory(
                                vocabulary_path=config.slam_vocabulary,
                                settings_path=runtime_settings_path,
                                frame_width=config.slam_width,
                                frame_height=config.slam_height,
                            )
                        slam_time_origin = time_source()
                    frame_source = _build_live_frame_source(camera_session)
                    map_builder.reset()
                    restarted_stream = True
                cached_detections = []
                latest_explanation_request_id = None
                explanation_refresh_status = "idle"
                last_explanation_request_time = None
                explanation_button_state["pending_toggle"] = False
                if config.explanation_enabled:
                    explanation_result = ExplanationResult(
                        text="",
                        status=ExplanationStatus.IDLE,
                        latency_ms=None,
                        model=config.explanation_model,
                        error_message=None,
                    )
                    explanation_status = (
                        ExplanationStatus.DISABLED
                        if explanation_status == ExplanationStatus.DISABLED
                        else ExplanationStatus.IDLE
                    )
                if restarted_stream:
                    continue
                break
            frame_bgr = np.asarray(frame_packet.frame_bgr, dtype=np.uint8)
            frame_timestamp_sec = frame_packet.timestamp_sec
            artifacts, cached_detections = process_frame(
                frame_bgr=frame_bgr,
                detector=detector,
                depth_estimator=depth_estimator,
                slam_bridge=slam_bridge,
                tracker=tracker,
                map_builder=map_builder,
                config=config,
                frame_index=frame_index,
                timestamp_sec=frame_timestamp_sec,
                cached_detections=cached_detections,
                calibration=calibration,
                calibration_source=(
                    frame_packet.calibration_source
                    if frame_packet is not None and frame_packet.calibration_source is not None
                    else calibration_source
                ),
                frame_packet=frame_packet,
                prefer_frame_packet_ground_truth=use_frame_packet_ground_truth,
                assist_frame_packet_ground_truth=assist_frame_packet_ground_truth,
                cv2_module=cv2,
            )
            record_runtime_observation = getattr(frame_source, "record_runtime_observation", None)
            if frame_packet is not None and callable(record_runtime_observation):
                record_runtime_observation(frame_packet=frame_packet, artifacts=artifacts)
            if frame_index == 0:
                debug_log("first runtime frame processed")

            if segmentation_worker is not None:
                if frame_index % config.segmentation_interval == 0 and segmentation_worker.is_idle():
                    segmentation_worker.submit(frame_index, artifacts.frame_bgr)
                    segmentation_frame_metadata[frame_index] = {
                        "camera_pose_world": artifacts.camera_pose_world.copy(),
                        "intrinsics": artifacts.intrinsics,
                        "tracking_ok": bool(artifacts.slam_tracking_state in {"TRACKING", "RELOCALIZED"}),
                    }
                    while len(segmentation_frame_metadata) > 16:
                        segmentation_frame_metadata.popitem(last=False)
                try:
                    segmentation_poll_result = segmentation_worker.poll()
                except RuntimeError as exc:
                    debug_log(str(exc))
                    segmentation_worker.close()
                    segmentation_worker = None
                    segmentation_poll_result = None
                if segmentation_poll_result is not None:
                    segmentation_result_frame_index, segmentation_result = segmentation_poll_result
                    cached_segmentation = segmentation_result
                    metadata = segmentation_frame_metadata.pop(segmentation_result_frame_index, None)
                    if (
                        metadata is not None
                        and bool(metadata.get("tracking_ok", False))
                        and hasattr(map_builder, "ingest_segmentation_observation")
                    ):
                        map_builder.ingest_segmentation_observation(
                            frame_index=segmentation_result_frame_index,
                            camera_pose_world=np.asarray(metadata["camera_pose_world"], dtype=np.float32),
                            intrinsics=metadata["intrinsics"],
                            segment_id_map=segmentation_result.segment_id_map,
                            segments=list(segmentation_result.segments),
                        )
                        if hasattr(map_builder, "current_mesh_state"):
                            mesh_vertices_xyz, mesh_triangles, mesh_vertex_colors = map_builder.current_mesh_state()
                            artifacts.mesh_vertices_xyz = np.asarray(mesh_vertices_xyz, dtype=np.float32).reshape(-1, 3)
                            artifacts.mesh_triangles = np.asarray(mesh_triangles, dtype=np.int32).reshape(-1, 3)
                            artifacts.mesh_vertex_colors = np.asarray(mesh_vertex_colors, dtype=np.float32).reshape(-1, 3)
                            artifacts.points_xyz = artifacts.mesh_vertices_xyz
                            artifacts.points_rgb = artifacts.mesh_vertex_colors
                            artifacts.dense_map_points_xyz = artifacts.mesh_vertices_xyz
                            artifacts.dense_map_points_rgb = artifacts.mesh_vertex_colors
                            if artifacts.depth_diagnostics is not None:
                                mesh_z_span = _camera_space_z_span(
                                    artifacts.mesh_vertices_xyz,
                                    artifacts.camera_pose_world,
                                )
                                artifacts.depth_diagnostics = DepthDiagnostics(
                                    calibration_source=artifacts.depth_diagnostics.calibration_source,
                                    profile=artifacts.depth_diagnostics.profile,
                                    raw_percentiles=artifacts.depth_diagnostics.raw_percentiles,
                                    normalizer_low_high=artifacts.depth_diagnostics.normalizer_low_high,
                                    normalized_distance_percentiles=artifacts.depth_diagnostics.normalized_distance_percentiles,
                                    valid_depth_ratio=artifacts.depth_diagnostics.valid_depth_ratio,
                                    dense_z_span=artifacts.depth_diagnostics.dense_z_span,
                                    mesh_z_span=mesh_z_span,
                                    intrinsics_summary=artifacts.depth_diagnostics.intrinsics_summary,
                                    hint=_depth_hint(
                                        calibration_source=artifacts.depth_diagnostics.calibration_source,
                                        profile=artifacts.depth_diagnostics.profile,
                                        normalized_distance_percentiles=artifacts.depth_diagnostics.normalized_distance_percentiles,
                                        dense_z_span=artifacts.depth_diagnostics.dense_z_span,
                                        mesh_z_span=mesh_z_span,
                                    ),
                                )

            if cached_segmentation is not None:
                artifacts.segmentation_overlay_bgr = cached_segmentation.overlay_bgr
                artifacts.segments = list(cached_segmentation.segments)

            if scene_graph_memory is not None:
                if calibration is not None:
                    graph_intrinsics = intrinsics_from_calibration(
                        calibration,
                        target_width=artifacts.frame_bgr.shape[1],
                        target_height=artifacts.frame_bgr.shape[0],
                    )
                else:
                    graph_intrinsics = artifacts.intrinsics
                scene_graph_snapshot = scene_graph_memory.update(
                    frame_index=frame_index,
                    detections=artifacts.detections,
                    segments=artifacts.segments,
                    depth_map=artifacts.depth_map,
                    intrinsics=graph_intrinsics,
                    camera_pose_world=artifacts.camera_pose_world,
                    slam_tracking_state=artifacts.slam_tracking_state,
                )
                artifacts.scene_graph_snapshot = scene_graph_snapshot
                artifacts.visible_graph_nodes = list(scene_graph_snapshot.visible_nodes)
                artifacts.visible_graph_edges = list(scene_graph_snapshot.visible_edges)

            if explanation_worker is not None:
                explanation_poll_result = explanation_worker.poll()
                if explanation_poll_result is not None:
                    result_snapshot_id, result = explanation_poll_result
                    if latest_explanation_request_id is not None and result_snapshot_id == latest_explanation_request_id:
                        preserve_previous_explanation = bool((explanation_result.text or "").strip()) and (
                            explanation_status == ExplanationStatus.READY
                        )
                        received_valid_text = bool((result.text or "").strip())
                        if result.status == ExplanationStatus.READY and received_valid_text:
                            explanation_result = result
                            explanation_status = result.status
                            latest_explanation_timestamp = time.strftime("%H:%M:%S")
                            explanation_refresh_status = "idle"
                            explanation_panel_state["scroll_offset"] = 0
                            explanation_panel_state["pending_scroll"] = 0
                        elif preserve_previous_explanation:
                            explanation_refresh_status = "failed"
                        else:
                            explanation_result = result
                            explanation_status = result.status
                            explanation_refresh_status = "failed"
                            explanation_panel_state["scroll_offset"] = 0
                            explanation_panel_state["pending_scroll"] = 0

            now = time_source()
            fps = 1.0 / max(now - last_frame_time, 1e-6)
            last_frame_time = now

            overlay = overlay_renderer(
                artifacts.frame_bgr,
                artifacts.detections,
                fps,
                segmentation_overlay_bgr=artifacts.segmentation_overlay_bgr,
                segments=artifacts.segments,
                segmentation_alpha=config.segmentation_alpha,
                slam_tracking_state=artifacts.slam_tracking_state,
                keyframe_id=artifacts.keyframe_id,
                mesh_triangle_count=int(artifacts.mesh_triangles.shape[0]),
                mesh_vertex_count=int(artifacts.mesh_vertices_xyz.shape[0]),
                scene_graph_snapshot=artifacts.scene_graph_snapshot,
                visible_graph_nodes=artifacts.visible_graph_nodes,
                visible_graph_edges=artifacts.visible_graph_edges,
                explanation_status=str(explanation_status) if explanation_status is not None else None,
                explanation_auto_refresh_enabled=explanation_auto_refresh_enabled,
                depth_diagnostics=artifacts.depth_diagnostics,
                depth_debug_level=depth_debug_level,
                cv2_module=cv2,
            )
            cv2.imshow("Object Recognition", overlay)
            explanation_button_state["rect"] = (
                explanation_button_rect(
                    frame_width=overlay.shape[1],
                    frame_height=overlay.shape[0],
                )
                if config.explanation_enabled
                else None
            )
            if (
                config.explanation_enabled
                and not explanation_mouse_callback_registered
                and callable(getattr(cv2, "setMouseCallback", None))
            ):
                cv2.setMouseCallback("Object Recognition", _handle_object_window_mouse)
                explanation_mouse_callback_registered = True
            if config.explanation_enabled:
                if int(explanation_panel_state["pending_scroll"]) != 0:
                    explanation_panel_state["scroll_offset"] = max(
                        0,
                        int(explanation_panel_state["scroll_offset"]) + int(explanation_panel_state["pending_scroll"]),
                    )
                    explanation_panel_state["pending_scroll"] = 0
                try:
                    panel_render_result = explanation_panel_renderer(
                        status=str(explanation_status or ExplanationStatus.IDLE),
                        text=explanation_result.text or (explanation_result.error_message or ""),
                        model=explanation_result.model or config.explanation_model,
                        latency_ms=explanation_result.latency_ms,
                        timestamp_label=latest_explanation_timestamp,
                        refresh_status=explanation_refresh_status,
                        scroll_offset=int(explanation_panel_state["scroll_offset"]),
                        cv2_module=cv2,
                        return_metadata=True,
                    )
                except TypeError:
                    panel_render_result = explanation_panel_renderer(
                        status=str(explanation_status or ExplanationStatus.IDLE),
                        text=explanation_result.text or (explanation_result.error_message or ""),
                        model=explanation_result.model or config.explanation_model,
                        latency_ms=explanation_result.latency_ms,
                        timestamp_label=latest_explanation_timestamp,
                        refresh_status=explanation_refresh_status,
                        cv2_module=cv2,
                    )
                if isinstance(panel_render_result, tuple):
                    panel, panel_metadata = panel_render_result
                    explanation_panel_state["scroll_offset"] = int(
                        panel_metadata.get("scroll_offset", explanation_panel_state["scroll_offset"])
                    )
                    explanation_panel_state["up_rect"] = panel_metadata.get("up_rect")
                    explanation_panel_state["down_rect"] = panel_metadata.get("down_rect")
                else:
                    panel = panel_render_result
                    explanation_panel_state["up_rect"] = None
                    explanation_panel_state["down_rect"] = None
                cv2.imshow("Situation Explanation", panel)
                if (
                    not explanation_panel_mouse_callback_registered
                    and callable(getattr(cv2, "setMouseCallback", None))
                ):
                    cv2.setMouseCallback("Situation Explanation", _handle_explanation_window_mouse)
                    explanation_panel_mouse_callback_registered = True
            viewer_active = _update_viewer(viewer, artifacts)
            if validation_probe is not None and hasattr(validation_probe, "record_frame"):
                validation_probe.record_frame(
                    frame_index=frame_index,
                    frame_packet=frame_packet,
                    artifacts=artifacts,
                    explanation_status=explanation_status,
                    explanation_result=explanation_result,
                    viewer_active=viewer_active,
                )
            key = cv2.waitKey(1) & 0xFF
            window_visible = window_is_visible(cv2, "Object Recognition")
            object_window_has_been_visible = object_window_has_been_visible or window_visible
            window_closed = object_window_has_been_visible and not window_visible

            if key == ord("r"):
                if tracker is not None:
                    tracker.reset()
                if slam_bridge is not None:
                    slam_bridge.close()
                    slam_bridge = slam_bridge_factory(
                        vocabulary_path=config.slam_vocabulary,
                        settings_path=runtime_settings_path,
                        frame_width=config.slam_width,
                        frame_height=config.slam_height,
                    )
                    slam_time_origin = time_source()
                map_builder.reset()
                cached_detections = []
                cached_segmentation = None
                latest_explanation_request_id = None
                explanation_refresh_status = "idle"
                last_explanation_request_time = None
                if config.explanation_enabled:
                    explanation_result = ExplanationResult(
                        text="",
                        status=ExplanationStatus.IDLE,
                        latency_ms=None,
                        model=config.explanation_model,
                        error_message=None,
                    )
                    explanation_status = (
                        ExplanationStatus.DISABLED
                        if explanation_status == ExplanationStatus.DISABLED
                        else ExplanationStatus.IDLE
                    )
                explanation_panel_state["scroll_offset"] = 0
                explanation_panel_state["pending_scroll"] = 0
                explanation_panel_state["up_rect"] = None
                explanation_panel_state["down_rect"] = None
                explanation_button_state["pending_toggle"] = False
                frame_index += 1
                continue

            if key == ord("d"):
                depth_debug_level = {
                    "off": "basic",
                    "basic": "detailed",
                    "detailed": "off",
                }[depth_debug_level]

            if explanation_button_state["pending_toggle"]:
                explanation_button_state["pending_toggle"] = False
                _toggle_explanation_auto_refresh(artifacts, toggled_at=now)

            if key == ord("e"):
                _toggle_explanation_auto_refresh(artifacts, toggled_at=now)

            if key == ord("q") or not viewer_active or window_closed:
                break

            if (
                explanation_auto_refresh_enabled
                and explanation_worker is not None
                and explanation_worker.is_idle()
                and (
                    last_explanation_request_time is None
                    or (now - last_explanation_request_time) >= config.explanation_refresh_interval_sec
                )
            ):
                _submit_explanation_request(artifacts, requested_at=now)

            frame_index += 1
    finally:
        if validation_probe is not None and hasattr(validation_probe, "finish"):
            validation_probe.finish()
        if frame_source is not None:
            frame_source.close()
            if isinstance(frame_source, LiveCameraFrameSource):
                camera_session = None
        if camera_session is not None:
            camera_session.capture.release()
        if slam_bridge is not None:
            slam_bridge.close()
        if segmentation_worker is not None:
            segmentation_worker.close()
        if explanation_worker is not None:
            explanation_worker.close()
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()


def main() -> None:
    _load_app_dotenv()
    config = parse_config()
    if config.validate_all_scenarios:
        from obj_recog.validation import run_validation_suite

        summary = run_validation_suite(
            replace(config, validate_all_scenarios=False),
            output_dir=config.validation_output_dir,
            run_fn=run,
            slam_bridge_factory=OrbSlam3Bridge,
            map_builder_factory=TsdfMeshMapBuilder,
            situation_explainer_factory=OpenAISituationExplainer,
            explanation_worker_factory=SituationExplanationWorker,
        )
        print(
            f"validation summary: total={summary.total_runs} pass={summary.pass_runs} "
            f"warn={summary.warn_runs} fail={summary.fail_runs} skipped={summary.skipped_runs} "
            f"output={summary.output_dir}"
        )
        return
    run(
        config,
        slam_bridge_factory=OrbSlam3Bridge,
        map_builder_factory=TsdfMeshMapBuilder,
        situation_explainer_factory=OpenAISituationExplainer,
        explanation_worker_factory=SituationExplanationWorker,
    )


if __name__ == "__main__":
    main()
