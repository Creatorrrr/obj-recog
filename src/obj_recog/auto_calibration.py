from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from obj_recog.calibration import CalibrationResult, load_orbslam3_settings, render_orbslam3_settings_yaml
from obj_recog.camera import CameraSession, read_camera_frame
from obj_recog.config import AppConfig
from obj_recog.opencv_runtime import load_cv2
from obj_recog.slam_bridge import KeyframeObservation, TRACKING_OK_STATES

ALGORITHM_VERSION = "slam-selfcal-v1"
CALIBRATION_WINDOW_NAME = "SLAM Self-Calibration"
_MAX_REFINEMENT_KEYFRAMES = 8
_MAX_REFINEMENT_OBSERVATIONS_PER_KEYFRAME = 160


def _default_debug_log(message: str) -> None:
    print(f"[obj-recog] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True, slots=True)
class AutoCalibrationCacheEntry:
    yaml_path: str
    metadata_path: str
    camera_fingerprint: str
    created_at: str
    algorithm_version: str
    validation_metrics: dict[str, float]
    stale: bool = False


@dataclass(frozen=True, slots=True)
class SlamBootstrapMetrics:
    tracking_ok_ratio: float
    valid_keyframes: int
    unique_points: int
    mean_track_length: float
    median_reprojection_error: float | None

    def as_dict(self) -> dict[str, float]:
        metrics = {
            "tracking_ok_ratio": float(self.tracking_ok_ratio),
            "valid_keyframes": float(self.valid_keyframes),
            "unique_points": float(self.unique_points),
            "mean_track_length": float(self.mean_track_length),
        }
        if self.median_reprojection_error is not None:
            metrics["median_reprojection_error"] = float(self.median_reprojection_error)
        return metrics


@dataclass(frozen=True, slots=True)
class CalibrationValidationResult:
    accepted: bool
    reason: str
    metrics: SlamBootstrapMetrics
    bridge: object | None = None


@dataclass(frozen=True, slots=True)
class WarmupCalibrationResult:
    calibration: CalibrationResult
    settings_path: str
    metrics: SlamBootstrapMetrics
    bridge: object | None
    warmup_restarted: bool


@dataclass(frozen=True, slots=True)
class RuntimeCalibrationState:
    source: str
    settings_path: str
    calibration: CalibrationResult
    cache_entry: AutoCalibrationCacheEntry | None
    warmup_restarted: bool
    promoted_bridge: object | None = None


def default_calibration_cache_dir() -> Path:
    return Path.home() / "Library" / "Caches" / "obj-recog" / "calibration"


def create_approximate_calibration(*, image_width: int, image_height: int) -> CalibrationResult:
    fx = 0.9 * float(image_width)
    fy = 0.9 * float(image_width)
    cx = float(image_width) / 2.0
    cy = float(image_height) / 2.0
    return CalibrationResult(
        camera_matrix=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32),
        distortion_coefficients=np.zeros((1, 5), dtype=np.float32),
        image_width=int(image_width),
        image_height=int(image_height),
        rms_error=0.0,
    )


def build_camera_fingerprint(camera_session: CameraSession, config: AppConfig) -> str:
    capture_width = int(round(float(getattr(camera_session.capture, "get", lambda _p: 0.0)(3) or 0.0))) or int(config.width)
    capture_height = int(round(float(getattr(camera_session.capture, "get", lambda _p: 0.0)(4) or 0.0))) or int(config.height)
    return "|".join(
        [
            camera_session.active_name,
            f"{capture_width}x{capture_height}",
            f"{config.slam_width}x{config.slam_height}",
            ALGORITHM_VERSION,
        ]
    )


def _fingerprint_slug(fingerprint: str) -> str:
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]


def _cache_paths(cache_dir: str | Path, fingerprint: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    slug = _fingerprint_slug(fingerprint)
    return root / f"{slug}.yaml", root / f"{slug}.json"


def _warmup_settings_path(cache_dir: str | Path, fingerprint: str, label: str) -> Path:
    root = Path(cache_dir)
    slug = _fingerprint_slug(fingerprint)
    return root / f"{slug}-{label}.yaml"


def _write_settings_yaml(path: str | Path, calibration: CalibrationResult, *, fps: float) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_orbslam3_settings_yaml(calibration, fps=fps), encoding="utf-8")
    return destination


def close_calibration_window(cv2_module=None) -> None:
    cv2 = load_cv2(cv2_module)
    destroy_window = getattr(cv2, "destroyWindow", None)
    if callable(destroy_window):
        try:
            destroy_window(CALIBRATION_WINDOW_NAME)
        except Exception:
            pass


def load_cached_calibration_entry(cache_dir: str | Path, fingerprint: str) -> AutoCalibrationCacheEntry | None:
    yaml_path, metadata_path = _cache_paths(cache_dir, fingerprint)
    if not yaml_path.is_file() or not metadata_path.is_file():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return AutoCalibrationCacheEntry(
        yaml_path=str(yaml_path),
        metadata_path=str(metadata_path),
        camera_fingerprint=str(metadata.get("camera_fingerprint", fingerprint)),
        created_at=str(metadata.get("created_at", "")),
        algorithm_version=str(metadata.get("algorithm_version", ALGORITHM_VERSION)),
        validation_metrics=dict(metadata.get("validation_metrics") or {}),
        stale=bool(metadata.get("stale", False)),
    )


def store_calibration_cache(
    *,
    cache_dir: str | Path,
    fingerprint: str,
    calibration: CalibrationResult,
    active_name: str,
    validation_metrics: dict[str, float],
    fps: float,
    created_at: str | None = None,
    stale: bool = False,
    stale_reason: str | None = None,
) -> AutoCalibrationCacheEntry:
    yaml_path, metadata_path = _cache_paths(cache_dir, fingerprint)
    _write_settings_yaml(yaml_path, calibration, fps=fps)
    metadata = {
        "camera_fingerprint": fingerprint,
        "active_name": active_name,
        "created_at": created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm_version": ALGORITHM_VERSION,
        "validation_metrics": {key: float(value) for key, value in validation_metrics.items()},
        "stale": bool(stale),
        "stale_reason": stale_reason,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return AutoCalibrationCacheEntry(
        yaml_path=str(yaml_path),
        metadata_path=str(metadata_path),
        camera_fingerprint=fingerprint,
        created_at=str(metadata["created_at"]),
        algorithm_version=ALGORITHM_VERSION,
        validation_metrics=dict(metadata["validation_metrics"]),
        stale=bool(stale),
    )


def mark_cached_calibration_stale(
    entry: AutoCalibrationCacheEntry,
    *,
    reason: str,
    metrics: SlamBootstrapMetrics,
) -> None:
    metadata_path = Path(entry.metadata_path)
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["stale"] = True
    metadata["stale_reason"] = reason
    metadata["validation_metrics"] = metrics.as_dict()
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def calibration_requires_restart(
    current: CalibrationResult,
    refined: CalibrationResult,
    *,
    threshold_ratio: float = 0.01,
) -> bool:
    current_fx = float(np.asarray(current.camera_matrix, dtype=np.float64)[0, 0])
    current_fy = float(np.asarray(current.camera_matrix, dtype=np.float64)[1, 1])
    refined_fx = float(np.asarray(refined.camera_matrix, dtype=np.float64)[0, 0])
    refined_fy = float(np.asarray(refined.camera_matrix, dtype=np.float64)[1, 1])
    delta_fx = abs(refined_fx - current_fx) / max(abs(current_fx), 1e-6)
    delta_fy = abs(refined_fy - current_fy) / max(abs(current_fy), 1e-6)
    return max(delta_fx, delta_fy) >= float(threshold_ratio)


def _project_points(
    points_world: np.ndarray,
    rotation_cw: np.ndarray,
    translation_cw: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    camera_points = (rotation_cw @ points_world.T).T + translation_cw.reshape(1, 3)
    z = camera_points[:, 2:3]
    valid = z[:, 0] > 1e-6
    projected = np.empty((points_world.shape[0], 2), dtype=np.float64)
    projected[:, 0] = fx * (camera_points[:, 0] / np.maximum(z[:, 0], 1e-6)) + cx
    projected[:, 1] = fy * (camera_points[:, 1] / np.maximum(z[:, 0], 1e-6)) + cy
    return projected, valid


def _group_keyframe_observations(
    keyframe_poses: dict[int, np.ndarray],
    keyframe_observations: list[KeyframeObservation],
) -> tuple[list[int], list[int], list[np.ndarray], np.ndarray, dict[int, list[tuple[int, np.ndarray]]]]:
    available_keyframes = sorted({int(obs.keyframe_id) for obs in keyframe_observations if int(obs.keyframe_id) in keyframe_poses})
    if len(available_keyframes) < 2:
        return [], [], [], np.empty((0, 3), dtype=np.float32), {}

    frame_index_by_id = {keyframe_id: index for index, keyframe_id in enumerate(available_keyframes)}
    point_xyz_by_id: dict[int, list[np.ndarray]] = {}
    observations_by_point: dict[int, list[tuple[int, np.ndarray]]] = {}
    for observation in keyframe_observations:
        if observation.keyframe_id not in frame_index_by_id:
            continue
        point_xyz_by_id.setdefault(observation.point_id, []).append(
            np.array([observation.x, observation.y, observation.z], dtype=np.float32)
        )
        observations_by_point.setdefault(observation.point_id, []).append(
            (frame_index_by_id[observation.keyframe_id], np.array([observation.u, observation.v], dtype=np.float32))
        )

    ordered_point_ids: list[int] = []
    ordered_points: list[np.ndarray] = []
    compact_observations: dict[int, list[tuple[int, np.ndarray]]] = {}
    for point_id in sorted(observations_by_point):
        dedup: dict[int, np.ndarray] = {}
        for frame_index, uv in observations_by_point[point_id]:
            dedup[int(frame_index)] = np.asarray(uv, dtype=np.float32)
        if len(dedup) < 2:
            continue
        ordered_point_ids.append(int(point_id))
        ordered_points.append(np.mean(np.asarray(point_xyz_by_id[point_id], dtype=np.float32), axis=0))
        compact_observations[len(ordered_point_ids) - 1] = [
            (frame_index, uv) for frame_index, uv in sorted(dedup.items(), key=lambda item: item[0])
        ]

    ordered_poses = [np.asarray(keyframe_poses[keyframe_id], dtype=np.float32) for keyframe_id in available_keyframes]
    return available_keyframes, ordered_point_ids, ordered_poses, np.asarray(ordered_points, dtype=np.float32), compact_observations


def refine_focal_lengths(
    *,
    initial_fx: float,
    initial_fy: float,
    cx: float,
    cy: float,
    keyframe_poses: dict[int, np.ndarray],
    keyframe_observations: list[KeyframeObservation],
    cv2_module=None,
    debug_log=_default_debug_log,
) -> tuple[float, float, dict[int, np.ndarray], dict[int, np.ndarray]]:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - depends on local install.
        raise RuntimeError("scipy is required for SLAM self-calibration refinement") from exc

    keyframe_ids, point_ids, ordered_poses, initial_points_world, observations = _group_keyframe_observations(
        keyframe_poses,
        keyframe_observations,
    )
    grouped_observation_count = sum(len(frame_observations) for frame_observations in observations.values())
    debug_log(
        "SLAM self-calibration grouped input "
        f"(keyframes={len(keyframe_ids)}, points={len(point_ids)}, observations={grouped_observation_count})"
    )
    if len(ordered_poses) < 2 or initial_points_world.size == 0:
        debug_log("SLAM self-calibration skipped solver due to insufficient grouped input")
        return float(initial_fx), float(initial_fy), dict(keyframe_poses), {}

    def _pose_world_to_camera_components(pose_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        world_to_camera = np.linalg.inv(np.asarray(pose_world, dtype=np.float64))
        return world_to_camera[:3, :3], world_to_camera[:3, 3]
    poses_cw = [_pose_world_to_camera_components(pose_world) for pose_world in ordered_poses]
    x0 = np.array([float(initial_fx), float(initial_fy)], dtype=np.float64)
    point_count = int(initial_points_world.shape[0])
    debug_log(
        "SLAM self-calibration solver start "
        f"(variables={x0.size}, keyframes={len(keyframe_ids)}, points={point_count}, observations={grouped_observation_count})"
    )
    width_estimate = max(cx * 2.0, 100.0)
    height_estimate = max(cy * 2.0, 100.0)
    lower_bounds = np.array([0.3 * width_estimate, 0.3 * height_estimate], dtype=np.float64)
    upper_bounds = np.array([2.5 * width_estimate, 2.5 * height_estimate], dtype=np.float64)

    def _residuals(vector: np.ndarray) -> np.ndarray:
        fx = float(vector[0])
        fy = float(vector[1])
        residuals: list[float] = []
        for point_id, frame_observations in observations.items():
            point_world = np.asarray(initial_points_world[int(point_id)], dtype=np.float64)
            for frame_index, observed_uv in frame_observations:
                rotation_cw, translation_cw = poses_cw[int(frame_index)]
                projected, valid = _project_points(
                    point_world.reshape(1, 3),
                    rotation_cw,
                    translation_cw,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                )
                if not bool(valid[0]):
                    residuals.extend([500.0, 500.0])
                    continue
                residuals.extend((projected[0] - np.asarray(observed_uv, dtype=np.float64)).tolist())
        return np.asarray(residuals, dtype=np.float64)

    optimized = least_squares(
        _residuals,
        x0,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        max_nfev=50,
    )
    debug_log(
        "SLAM self-calibration solver done "
        f"(status={optimized.status}, nfev={optimized.nfev}, cost={optimized.cost:.4f})"
    )
    optimized_vector = optimized.x
    refined_fx = float(optimized_vector[0])
    refined_fy = float(optimized_vector[1])
    refined_poses_world = {
        int(keyframe_id): np.asarray(keyframe_poses[keyframe_id], dtype=np.float32)
        for keyframe_id in keyframe_ids
    }
    refined_points_world = {
        point_id: np.asarray(initial_points_world[index], dtype=np.float32)
        for index, point_id in enumerate(point_ids)
    }
    return refined_fx, refined_fy, refined_poses_world, refined_points_world


def _sample_observations(
    observations: list[KeyframeObservation],
    *,
    sample_count: int,
) -> list[KeyframeObservation]:
    if len(observations) <= sample_count:
        return list(observations)
    indices = np.linspace(0, len(observations) - 1, num=sample_count, dtype=np.int32)
    return [observations[int(index)] for index in indices]


def _limit_observations_for_refinement(
    observations: list[KeyframeObservation],
) -> list[KeyframeObservation]:
    if not observations:
        return []
    recent_keyframes = sorted({int(observation.keyframe_id) for observation in observations})[-_MAX_REFINEMENT_KEYFRAMES:]
    allowed = set(recent_keyframes)
    limited: list[KeyframeObservation] = []
    for keyframe_id in recent_keyframes:
        grouped = [observation for observation in observations if observation.keyframe_id == keyframe_id and observation.keyframe_id in allowed]
        grouped.sort(key=lambda observation: (int(observation.point_id), float(observation.u), float(observation.v)))
        limited.extend(
            _sample_observations(
                grouped,
                sample_count=_MAX_REFINEMENT_OBSERVATIONS_PER_KEYFRAME,
            )
        )
    return limited


class SLAMBootstrapManager:
    def __init__(self, *, calibration: CalibrationResult) -> None:
        self.calibration = calibration
        self._total_frames = 0
        self._tracking_ok_frames = 0
        self._reprojection_errors: list[float] = []
        self._keyframe_poses: dict[int, np.ndarray] = {}
        self._observation_by_key: dict[tuple[int, int], KeyframeObservation] = {}
        self.latest_tracking_state = "INITIALIZING"
        self.latest_tracked_feature_count = 0

    def record(self, result) -> None:
        self._total_frames += 1
        self.latest_tracking_state = str(result.tracking_state)
        self.latest_tracked_feature_count = int(getattr(result, "tracked_feature_count", 0))
        if getattr(result, "tracking_ok", False):
            self._tracking_ok_frames += 1
        if getattr(result, "median_reprojection_error", None) is not None:
            self._reprojection_errors.append(float(result.median_reprojection_error))
        for keyframe_id, pose_world in getattr(result, "optimized_keyframe_poses", {}).items():
            self._keyframe_poses[int(keyframe_id)] = np.asarray(pose_world, dtype=np.float32)
        if result.keyframe_id is not None and result.keyframe_id not in self._keyframe_poses:
            self._keyframe_poses[int(result.keyframe_id)] = np.asarray(result.pose_world, dtype=np.float32)
        for observation in getattr(result, "keyframe_observations", []):
            self._observation_by_key[(int(observation.keyframe_id), int(observation.point_id))] = observation

    def observations(self) -> list[KeyframeObservation]:
        return list(self._observation_by_key.values())

    def metrics(self) -> SlamBootstrapMetrics:
        observations = self.observations()
        point_track_lengths: dict[int, set[int]] = {}
        for observation in observations:
            point_track_lengths.setdefault(int(observation.point_id), set()).add(int(observation.keyframe_id))
        mean_track_length = (
            float(np.mean([len(keyframes) for keyframes in point_track_lengths.values()]))
            if point_track_lengths
            else 0.0
        )
        median_error = (
            float(np.median(np.asarray(self._reprojection_errors, dtype=np.float64)))
            if self._reprojection_errors
            else None
        )
        return SlamBootstrapMetrics(
            tracking_ok_ratio=(
                float(self._tracking_ok_frames) / float(self._total_frames) if self._total_frames else 0.0
            ),
            valid_keyframes=len(self._keyframe_poses),
            unique_points=len(point_track_lengths),
            mean_track_length=mean_track_length,
            median_reprojection_error=median_error,
        )

    def meets_acceptance(self) -> bool:
        metrics = self.metrics()
        return (
            metrics.tracking_ok_ratio >= 0.8
            and metrics.valid_keyframes >= 6
            and metrics.unique_points >= 250
            and metrics.mean_track_length >= 3.0
            and metrics.median_reprojection_error is not None
            and metrics.median_reprojection_error <= 2.5
        )

    def refine_calibration(self, *, cv2_module=None, debug_log=_default_debug_log) -> CalibrationResult | None:
        observations = self.observations()
        if len(self._keyframe_poses) < 4 or len(observations) < 20:
            return None
        refinement_observations = _limit_observations_for_refinement(observations)
        debug_log(
            "SLAM self-calibration refinement input prepared "
            f"(raw_observations={len(observations)}, limited_observations={len(refinement_observations)}, "
            f"raw_keyframes={len(self._keyframe_poses)}, limited_keyframes={len({obs.keyframe_id for obs in refinement_observations})})"
        )
        matrix = np.asarray(self.calibration.camera_matrix, dtype=np.float32)
        refined_fx, refined_fy, _refined_poses, _refined_points = refine_focal_lengths(
            initial_fx=float(matrix[0, 0]),
            initial_fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
            keyframe_poses=self._keyframe_poses,
            keyframe_observations=refinement_observations,
            cv2_module=cv2_module,
            debug_log=debug_log,
        )
        refined_matrix = matrix.copy()
        refined_matrix[0, 0] = refined_fx
        refined_matrix[1, 1] = refined_fy
        rms_error = self.metrics().median_reprojection_error
        return CalibrationResult(
            camera_matrix=refined_matrix,
            distortion_coefficients=np.zeros((1, 5), dtype=np.float32),
            image_width=self.calibration.image_width,
            image_height=self.calibration.image_height,
            rms_error=float(rms_error or 0.0),
        )


def _resize_for_slam(frame_bgr: np.ndarray, config: AppConfig, cv2_module) -> np.ndarray:
    resized = cv2_module.resize(frame_bgr, (config.slam_width, config.slam_height), interpolation=cv2_module.INTER_AREA)
    return cv2_module.cvtColor(resized, cv2_module.COLOR_BGR2GRAY)


def _annotate_warmup_preview(
    frame_bgr: np.ndarray,
    *,
    tracking_state: str,
    tracked_features: int,
    metrics: SlamBootstrapMetrics,
    warmup_restarted: bool,
    cv2_module,
    stage_message: str | None = None,
) -> np.ndarray:
    preview = np.asarray(frame_bgr, dtype=np.uint8).copy()
    lines = [
        "SLAM self-calibration: move slowly left/right and forward/back",
        f"Tracking {tracking_state} | features {tracked_features}",
        f"Keyframes {metrics.valid_keyframes}/6 | points {metrics.unique_points}/250",
        f"Track ratio {metrics.tracking_ok_ratio:.2f} | mean track {metrics.mean_track_length:.2f}",
        f"Median reprojection {metrics.median_reprojection_error if metrics.median_reprojection_error is not None else -1:.2f}px",
        f"Refined restart {'done' if warmup_restarted else 'pending'} | q cancel | r reset",
    ]
    if stage_message:
        lines.append(stage_message)
    for index, line in enumerate(lines):
        cv2_module.putText(
            preview,
            line,
            (16, 32 + (index * 28)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
            cv2_module.LINE_AA,
        )
    return preview


def validate_cached_calibration(
    *,
    camera_session: CameraSession,
    config: AppConfig,
    calibration: CalibrationResult,
    settings_path: str,
    slam_bridge_factory,
    cv2_module=None,
    frame_reader=read_camera_frame,
    time_source=time.perf_counter,
    duration_sec: float = 1.5,
    debug_log=_default_debug_log,
) -> CalibrationValidationResult:
    cv2 = load_cv2(cv2_module)
    debug_log(f"cached calibration validation started ({settings_path})")
    bridge = slam_bridge_factory(
        vocabulary_path=config.slam_vocabulary,
        settings_path=settings_path,
        frame_width=config.slam_width,
        frame_height=config.slam_height,
    )
    manager = SLAMBootstrapManager(calibration=calibration)
    start = time_source()
    origin = start
    try:
        while time_source() - start < duration_sec:
            ok, frame_bgr = frame_reader(camera_session.capture, timeout_sec=0.5)
            if not ok or frame_bgr is None:
                break
            frame_gray = _resize_for_slam(frame_bgr, config, cv2)
            result = bridge.track(frame_gray, max(0.0, time_source() - origin))
            manager.record(result)
            preview = _annotate_warmup_preview(
                frame_bgr,
                tracking_state=manager.latest_tracking_state,
                tracked_features=manager.latest_tracked_feature_count,
                metrics=manager.metrics(),
                warmup_restarted=False,
                cv2_module=cv2,
            )
            cv2.imshow(CALIBRATION_WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                bridge.close()
                raise RuntimeError("SLAM self-calibration canceled")
            if manager.meets_acceptance():
                debug_log("cached calibration validation accepted")
                return CalibrationValidationResult(
                    accepted=True,
                    reason="ok",
                    metrics=manager.metrics(),
                    bridge=bridge,
                )
    except Exception:
        bridge.close()
        raise
    bridge.close()
    debug_log("cached calibration validation failed")
    return CalibrationValidationResult(
        accepted=False,
        reason="warmup validation failed",
        metrics=manager.metrics(),
        bridge=None,
    )


def run_slam_self_calibration(
    *,
    camera_session: CameraSession,
    config: AppConfig,
    calibration: CalibrationResult,
    settings_path: str,
    slam_bridge_factory,
    cv2_module=None,
    frame_reader=read_camera_frame,
    time_source=time.perf_counter,
    fps: float = 30.0,
    debug_log=_default_debug_log,
) -> WarmupCalibrationResult:
    cv2 = load_cv2(cv2_module)
    debug_log(f"SLAM warm-up started ({settings_path})")
    bridge = slam_bridge_factory(
        vocabulary_path=config.slam_vocabulary,
        settings_path=settings_path,
        frame_width=config.slam_width,
        frame_height=config.slam_height,
    )
    current_calibration = calibration
    current_settings_path = str(settings_path)
    warmup_restarted = False
    manager = SLAMBootstrapManager(calibration=current_calibration)
    origin = time_source()

    while True:
        ok, frame_bgr = frame_reader(camera_session.capture, timeout_sec=1.0)
        if not ok or frame_bgr is None:
            bridge.close()
            raise RuntimeError("failed to read frame during SLAM self-calibration")

        frame_gray = _resize_for_slam(frame_bgr, config, cv2)
        result = bridge.track(frame_gray, max(0.0, time_source() - origin))
        manager.record(result)
        metrics = manager.metrics()
        preview = _annotate_warmup_preview(
            frame_bgr,
            tracking_state=manager.latest_tracking_state,
            tracked_features=manager.latest_tracked_feature_count,
            metrics=metrics,
            warmup_restarted=warmup_restarted,
            cv2_module=cv2,
        )
        cv2.imshow(CALIBRATION_WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            bridge.close()
            raise RuntimeError("SLAM self-calibration canceled")
        if key == ord("r"):
            bridge.close()
            bridge = slam_bridge_factory(
                vocabulary_path=config.slam_vocabulary,
                settings_path=current_settings_path,
                frame_width=config.slam_width,
                frame_height=config.slam_height,
            )
            manager = SLAMBootstrapManager(calibration=current_calibration)
            origin = time_source()
            continue
        if not manager.meets_acceptance():
            continue

        debug_log(
            "SLAM warm-up acceptance reached "
            f"(keyframes={metrics.valid_keyframes}, points={metrics.unique_points}, reprojection={metrics.median_reprojection_error})"
        )
        preview = _annotate_warmup_preview(
            frame_bgr,
            tracking_state=manager.latest_tracking_state,
            tracked_features=manager.latest_tracked_feature_count,
            metrics=metrics,
            warmup_restarted=warmup_restarted,
            cv2_module=cv2,
            stage_message="Acceptance reached. Refining focal lengths...",
        )
        cv2.imshow(CALIBRATION_WINDOW_NAME, preview)
        cv2.waitKey(1)
        debug_log(
            "SLAM self-calibration refinement started "
            f"(observations={len(manager.observations())}, warmup_restarted={warmup_restarted})"
        )
        refine_start = time_source()
        refined_calibration = manager.refine_calibration(cv2_module=cv2, debug_log=debug_log) or current_calibration
        refine_duration = max(0.0, time_source() - refine_start)
        debug_log(
            "SLAM self-calibration refinement finished "
            f"(duration={refine_duration:.2f}s, fx={float(refined_calibration.camera_matrix[0, 0]):.2f}, fy={float(refined_calibration.camera_matrix[1, 1]):.2f})"
        )
        if calibration_requires_restart(current_calibration, refined_calibration) and not warmup_restarted:
            debug_log("SLAM self-calibration restart required after refinement")
            refined_settings_path = _warmup_settings_path(
                Path(current_settings_path).parent,
                Path(current_settings_path).stem,
                "refined",
            )
            _write_settings_yaml(refined_settings_path, refined_calibration, fps=fps)
            bridge.close()
            bridge = slam_bridge_factory(
                vocabulary_path=config.slam_vocabulary,
                settings_path=str(refined_settings_path),
                frame_width=config.slam_width,
                frame_height=config.slam_height,
            )
            current_calibration = refined_calibration
            current_settings_path = str(refined_settings_path)
            warmup_restarted = True
            manager = SLAMBootstrapManager(calibration=current_calibration)
            origin = time_source()
            debug_log(f"SLAM warm-up restarted with refined settings ({current_settings_path})")
            continue

        debug_log(f"SLAM warm-up complete ({current_settings_path})")
        return WarmupCalibrationResult(
            calibration=refined_calibration,
            settings_path=current_settings_path,
            metrics=metrics,
            bridge=bridge,
            warmup_restarted=warmup_restarted,
        )


def run_auto_calibration(**kwargs) -> WarmupCalibrationResult:
    return run_slam_self_calibration(**kwargs)


def ensure_runtime_calibration(
    config: AppConfig,
    camera_session: CameraSession,
    *,
    cv2_module=None,
    frame_reader=read_camera_frame,
    slam_bridge_factory,
    time_source=time.perf_counter,
    validator_runner=validate_cached_calibration,
    warmup_runner=run_slam_self_calibration,
    debug_log=_default_debug_log,
) -> RuntimeCalibrationState:
    explicit_output_path: Path | None = None
    if config.camera_calibration:
        path = Path(config.camera_calibration)
        if path.is_file():
            calibration = load_orbslam3_settings(path)
            return RuntimeCalibrationState(
                source="explicit",
                settings_path=str(path),
                calibration=calibration,
                cache_entry=None,
                warmup_restarted=False,
                promoted_bridge=None,
            )
        explicit_output_path = path

    cache_dir = Path(config.calibration_cache_dir) if config.calibration_cache_dir else default_calibration_cache_dir()
    fingerprint = build_camera_fingerprint(camera_session, config)
    if config.disable_slam_calibration:
        approx_calibration = create_approximate_calibration(
            image_width=config.slam_width,
            image_height=config.slam_height,
        )
        disabled_settings_path = explicit_output_path or _warmup_settings_path(cache_dir, fingerprint, "disabled")
        _write_settings_yaml(disabled_settings_path, approx_calibration, fps=30.0)
        debug_log(f"runtime calibration using approximate settings with self-calibration disabled ({disabled_settings_path})")
        return RuntimeCalibrationState(
            source="disabled",
            settings_path=str(disabled_settings_path),
            calibration=approx_calibration,
            cache_entry=None,
            warmup_restarted=False,
            promoted_bridge=None,
        )

    if explicit_output_path is None and not config.recalibrate:
        cached_entry = load_cached_calibration_entry(cache_dir, fingerprint)
        if cached_entry is not None and not cached_entry.stale:
            cached_calibration = load_orbslam3_settings(cached_entry.yaml_path)
            validation = validator_runner(
                camera_session=camera_session,
                config=config,
                calibration=cached_calibration,
                settings_path=cached_entry.yaml_path,
                slam_bridge_factory=slam_bridge_factory,
                cv2_module=cv2_module,
                frame_reader=frame_reader,
                time_source=time_source,
                debug_log=debug_log,
            )
            if validation.accepted:
                debug_log(f"runtime calibration reusing cached settings ({cached_entry.yaml_path})")
                return RuntimeCalibrationState(
                    source="cache",
                    settings_path=cached_entry.yaml_path,
                    calibration=cached_calibration,
                    cache_entry=cached_entry,
                    warmup_restarted=False,
                    promoted_bridge=validation.bridge,
                )
            mark_cached_calibration_stale(cached_entry, reason=validation.reason, metrics=validation.metrics)

    approx_calibration = create_approximate_calibration(image_width=config.slam_width, image_height=config.slam_height)
    approx_settings_path = _warmup_settings_path(cache_dir, fingerprint, "approx")
    _write_settings_yaml(approx_settings_path, approx_calibration, fps=30.0)
    warmup = warmup_runner(
        camera_session=camera_session,
        config=config,
        calibration=approx_calibration,
        settings_path=str(approx_settings_path),
        slam_bridge_factory=slam_bridge_factory,
        cv2_module=cv2_module,
        frame_reader=frame_reader,
        time_source=time_source,
        fps=30.0,
        debug_log=debug_log,
    )
    if explicit_output_path is not None:
        _write_settings_yaml(explicit_output_path, warmup.calibration, fps=30.0)
        return RuntimeCalibrationState(
            source="auto",
            settings_path=str(explicit_output_path),
            calibration=warmup.calibration,
            cache_entry=None,
            warmup_restarted=warmup.warmup_restarted,
            promoted_bridge=warmup.bridge,
        )
    cache_entry = store_calibration_cache(
        cache_dir=cache_dir,
        fingerprint=fingerprint,
        calibration=warmup.calibration,
        active_name=camera_session.active_name,
        validation_metrics=warmup.metrics.as_dict(),
        fps=30.0,
    )
    return RuntimeCalibrationState(
        source="auto",
        settings_path=cache_entry.yaml_path,
        calibration=warmup.calibration,
        cache_entry=cache_entry,
        warmup_restarted=warmup.warmup_restarted,
        promoted_bridge=warmup.bridge,
    )
