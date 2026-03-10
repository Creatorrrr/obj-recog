from __future__ import annotations

from collections import OrderedDict
import time
from pathlib import Path
import sys

import numpy as np

from obj_recog.auto_calibration import close_calibration_window, ensure_runtime_calibration
from obj_recog.calibration import CalibrationResult, intrinsics_from_calibration, load_orbslam3_settings
from obj_recog.camera import CameraSession, list_available_cameras, open_camera, read_camera_frame
from obj_recog.config import AppConfig, parse_config, resolve_device
from obj_recog.depth import DepthEstimator
from obj_recog.detector import ObjectDetector
from obj_recog.mapping import LocalMapBuilder, TsdfMeshMapBuilder
from obj_recog.opencv_runtime import load_cv2
from obj_recog.reconstruct import depth_to_point_cloud, intrinsics_for_frame
from obj_recog.scene_graph import SceneGraphMemory
from obj_recog.segmenter import PanopticSegmenter, SegmentationWorker
from obj_recog.slam_bridge import OrbSlam3Bridge, SlamFrameResult
from obj_recog.tracking import PoseTracker
from obj_recog.types import Detection, FrameArtifacts, SegmentationResult
from obj_recog.visualization import (
    Open3DMeshViewer,
    draw_detections,
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
    *,
    slam_bridge=None,
    tracker=None,
    cv2_module=None,
) -> tuple[FrameArtifacts, list[Detection]]:
    cv2 = load_cv2(cv2_module)
    inference_frame, scale_x, scale_y = resize_for_inference(frame_bgr, config.inference_width)

    if frame_index % config.detection_interval == 0 or not cached_detections:
        detections = detector.detect(inference_frame)
        detections = [_scale_detection(detection, scale_x, scale_y) for detection in detections]
    else:
        detections = cached_detections

    depth_map = depth_estimator.estimate(inference_frame)
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
    else:
        intrinsics = intrinsics_for_frame(frame_bgr.shape[1], frame_bgr.shape[0])
    if getattr(map_builder, "requires_point_cloud", True):
        points_xyz, point_colors, _point_pixels = depth_to_point_cloud(
            frame_bgr=frame_bgr,
            depth_map=depth_map,
            intrinsics=intrinsics,
            stride=config.point_stride,
            max_points=config.max_points,
        )
        map_point_colors = point_colors
    else:
        points_xyz = np.empty((0, 3), dtype=np.float32)
        map_point_colors = np.empty((0, 3), dtype=np.float32)
    if slam_bridge is not None:
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

    artifacts = FrameArtifacts(
        frame_bgr=frame_bgr,
        intrinsics=intrinsics,
        detections=list(detections),
        depth_map=depth_map,
        points_xyz=mesh_vertices_xyz,
        points_rgb=mesh_vertex_colors,
        dense_map_points_xyz=np.asarray(
            getattr(map_update, "dense_map_points_xyz", mesh_vertices_xyz),
            dtype=np.float32,
        ).reshape(-1, 3),
        dense_map_points_rgb=np.asarray(
            getattr(map_update, "dense_map_points_rgb", mesh_vertex_colors),
            dtype=np.float32,
        ).reshape(-1, 3),
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
    viewer_factory=Open3DMeshViewer,
    open_camera_fn=open_camera,
    camera_lister=list_available_cameras,
    runtime_calibration_resolver=ensure_runtime_calibration,
    overlay_renderer=draw_detections,
    scene_graph_memory_factory=SceneGraphMemory,
    time_source=time.perf_counter,
    debug_log=_default_debug_log,
) -> None:
    cv2 = load_cv2(cv2_module)
    effective_device = resolve_device(config.device)
    camera_session: CameraSession | None = None
    viewer = None
    slam_bridge = None
    segmentation_worker = None
    scene_graph_memory = None
    calibration = None
    runtime_settings_path = config.camera_calibration

    if config.list_cameras:
        for device in camera_lister():
            print(f"{device.index}: {device.name}")
        return

    use_slam_bridge = slam_bridge_factory is not None
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
        debug_log("detector init start")
        detector = detector_factory(conf_threshold=config.conf_threshold, device=effective_device)
        debug_log("detector init done")
        debug_log("depth init start")
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
            map_builder = map_builder_factory(
                window_keyframes=config.mapping_window_keyframes,
                voxel_size=config.map_voxel_size,
                max_mesh_triangles=config.max_mesh_triangles,
            )
        else:
            tracker = tracker_factory(orb_features=config.orb_features)
            map_builder = map_builder_factory(
                translation_threshold=config.keyframe_translation,
                rotation_threshold_deg=config.keyframe_rotation_deg,
                frame_interval=12,
                window_keyframes=config.mapping_window_keyframes,
                voxel_size=config.map_voxel_size,
                max_map_points=config.max_map_points,
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
            ok, frame_bgr = read_camera_frame(camera_session.capture, timeout_sec=1.0)
            if not ok:
                if camera_session.requested_name and not camera_session.used_fallback:
                    camera_session.capture.release()
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
                    map_builder.reset()
                    cached_detections = []
                    continue
                break

            frame_timestamp_sec = None if slam_time_origin is None else max(0.0, time_source() - slam_time_origin)
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
                cv2_module=cv2,
            )
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
                    graph_intrinsics = intrinsics_for_frame(
                        artifacts.frame_bgr.shape[1],
                        artifacts.frame_bgr.shape[0],
                    )
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
            )
            cv2.imshow("Object Recognition", overlay)
            viewer_active = _update_viewer(viewer, artifacts)
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
                frame_index += 1
                continue

            if key == ord("q") or not viewer_active or window_closed:
                break

            frame_index += 1
    finally:
        if camera_session is not None:
            camera_session.capture.release()
        if slam_bridge is not None:
            slam_bridge.close()
        if segmentation_worker is not None:
            segmentation_worker.close()
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()


def main() -> None:
    run(
        parse_config(),
        slam_bridge_factory=OrbSlam3Bridge,
        map_builder_factory=TsdfMeshMapBuilder,
    )


if __name__ == "__main__":
    main()
