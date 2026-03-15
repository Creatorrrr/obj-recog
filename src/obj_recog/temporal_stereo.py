from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from obj_recog.opencv_runtime import load_cv2
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.types import Detection, TemporalStereoDiagnostics


_DEFAULT_DYNAMIC_LABELS = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "dog",
        "cat",
        "bird",
    }
)


@dataclass(frozen=True, slots=True)
class TemporalStereoResult:
    stereo_depth_map: np.ndarray
    fused_depth_map: np.ndarray
    diagnostics: TemporalStereoDiagnostics


@dataclass(frozen=True, slots=True)
class _StereoDepthComputation:
    stereo_depth_map: np.ndarray
    valid_mask: np.ndarray
    coverage_ratio: float
    median_disparity_px: float


@dataclass(frozen=True, slots=True)
class _TemporalStereoKeyframe:
    keyframe_id: int
    frame_bgr: np.ndarray
    frame_gray: np.ndarray
    pose_world: np.ndarray
    intrinsics: CameraIntrinsics


def align_midas_depth_to_stereo(
    *,
    midas_depth_map: np.ndarray,
    stereo_depth_map: np.ndarray,
    valid_mask: np.ndarray,
    min_samples: int,
    max_fit_rmse: float,
) -> tuple[np.ndarray, int, float | None, bool]:
    midas_depth = np.asarray(midas_depth_map, dtype=np.float32)
    stereo_depth = np.asarray(stereo_depth_map, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(midas_depth) & np.isfinite(stereo_depth)
    mask &= midas_depth > 0.05
    mask &= stereo_depth > 0.05
    sample_count = int(mask.sum())
    if sample_count < int(min_samples):
        return midas_depth.copy(), sample_count, None, False

    midas_values = np.log(np.clip(midas_depth[mask].astype(np.float64), 1e-6, None))
    stereo_values = np.log(np.clip(stereo_depth[mask].astype(np.float64), 1e-6, None))
    design = np.stack((midas_values, np.ones_like(midas_values)), axis=1)
    coefficients, *_ = np.linalg.lstsq(design, stereo_values, rcond=None)
    predicted = design @ coefficients
    rmse = float(np.sqrt(np.mean(np.square(predicted - stereo_values))))
    if not np.isfinite(rmse) or rmse > float(max_fit_rmse):
        return midas_depth.copy(), sample_count, rmse, False

    full_predicted = np.exp(
        (coefficients[0] * np.log(np.clip(midas_depth.astype(np.float64), 1e-6, None))) + coefficients[1]
    )
    aligned = np.asarray(full_predicted, dtype=np.float32)
    return aligned, sample_count, rmse, True


class TemporalStereoDepthEstimator:
    def __init__(
        self,
        *,
        max_keyframes: int = 4,
        max_rotation_deg: float = 15.0,
        stereo_width: int = 384,
        min_coverage_ratio: float = 0.08,
        min_median_disparity_px: float = 1.0,
        min_fit_samples: int = 500,
        max_fit_rmse: float = 0.35,
        dynamic_labels: set[str] | frozenset[str] | None = None,
        cv2_module=None,
    ) -> None:
        self._max_keyframes = int(max_keyframes)
        self._max_rotation_deg = float(max_rotation_deg)
        self._stereo_width = int(max(32, stereo_width))
        self._min_coverage_ratio = float(min_coverage_ratio)
        self._min_median_disparity_px = float(min_median_disparity_px)
        self._min_fit_samples = int(min_fit_samples)
        self._max_fit_rmse = float(max_fit_rmse)
        self._dynamic_labels = {
            str(item).strip().casefold()
            for item in (dynamic_labels if dynamic_labels is not None else _DEFAULT_DYNAMIC_LABELS)
            if str(item).strip()
        }
        self._cv2_module = cv2_module
        self._keyframes: deque[_TemporalStereoKeyframe] = deque(maxlen=self._max_keyframes)

    @property
    def cached_keyframe_ids(self) -> list[int]:
        return [item.keyframe_id for item in self._keyframes]

    def update_keyframe_cache(
        self,
        *,
        keyframe_id: int | None,
        frame_bgr: np.ndarray,
        frame_gray: np.ndarray,
        pose_world: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> None:
        if keyframe_id is None:
            return
        self._keyframes.append(
            _TemporalStereoKeyframe(
                keyframe_id=int(keyframe_id),
                frame_bgr=np.asarray(frame_bgr, dtype=np.uint8).copy(),
                frame_gray=np.asarray(frame_gray, dtype=np.uint8).copy(),
                pose_world=np.asarray(pose_world, dtype=np.float32).copy(),
                intrinsics=intrinsics,
            )
        )

    def estimate(
        self,
        *,
        frame_bgr: np.ndarray,
        provisional_depth_map: np.ndarray,
        slam_result,
        intrinsics: CameraIntrinsics,
        detections: list[Detection],
    ) -> TemporalStereoResult:
        provisional_depth = np.asarray(provisional_depth_map, dtype=np.float32)
        fallback = self._fallback_result(
            provisional_depth,
            reason="no_reference_keyframe",
        )
        reference = self._select_reference_keyframe(np.asarray(slam_result.pose_world, dtype=np.float32))
        if reference is None:
            return fallback

        stereo_result = self._compute_stereo_depth(
            frame_bgr=np.asarray(frame_bgr, dtype=np.uint8),
            reference=reference,
            current_pose_world=np.asarray(slam_result.pose_world, dtype=np.float32),
            intrinsics=intrinsics,
        )
        if stereo_result is None:
            return self._fallback_result(
                provisional_depth,
                reason="stereo_compute_failed",
                reference_keyframe_id=reference.keyframe_id,
            )
        if stereo_result.coverage_ratio < self._min_coverage_ratio:
            return self._fallback_result(
                provisional_depth,
                reason="low_stereo_coverage",
                reference_keyframe_id=reference.keyframe_id,
                coverage_ratio=stereo_result.coverage_ratio,
                median_disparity_px=stereo_result.median_disparity_px,
            )
        if stereo_result.median_disparity_px < self._min_median_disparity_px:
            return self._fallback_result(
                provisional_depth,
                reason="low_median_disparity",
                reference_keyframe_id=reference.keyframe_id,
                coverage_ratio=stereo_result.coverage_ratio,
                median_disparity_px=stereo_result.median_disparity_px,
            )

        valid_mask = np.asarray(stereo_result.valid_mask, dtype=bool)
        valid_mask &= ~self._dynamic_detection_mask(
            frame_shape=provisional_depth.shape,
            detections=detections,
        )
        aligned_midas, sample_count, fit_rmse, fit_applied = align_midas_depth_to_stereo(
            midas_depth_map=provisional_depth,
            stereo_depth_map=stereo_result.stereo_depth_map,
            valid_mask=valid_mask,
            min_samples=self._min_fit_samples,
            max_fit_rmse=self._max_fit_rmse,
        )
        if not fit_applied:
            return self._fallback_result(
                provisional_depth,
                reason=(
                    "insufficient_overlap_samples"
                    if sample_count < self._min_fit_samples
                    else "fit_rmse_too_high"
                ),
                reference_keyframe_id=reference.keyframe_id,
                coverage_ratio=stereo_result.coverage_ratio,
                median_disparity_px=stereo_result.median_disparity_px,
                fit_sample_count=sample_count,
                fit_rmse=fit_rmse,
            )

        fused_depth = aligned_midas.copy()
        fused_depth[valid_mask] = stereo_result.stereo_depth_map[valid_mask]
        return TemporalStereoResult(
            stereo_depth_map=np.asarray(stereo_result.stereo_depth_map, dtype=np.float32),
            fused_depth_map=fused_depth.astype(np.float32, copy=False),
            diagnostics=TemporalStereoDiagnostics(
                enabled=True,
                applied=True,
                reference_keyframe_id=reference.keyframe_id,
                coverage_ratio=float(stereo_result.coverage_ratio),
                median_disparity_px=float(stereo_result.median_disparity_px),
                fit_sample_count=int(sample_count),
                fit_rmse=(None if fit_rmse is None else float(fit_rmse)),
                fallback_reason=None,
            ),
        )

    def _fallback_result(
        self,
        provisional_depth: np.ndarray,
        *,
        reason: str,
        reference_keyframe_id: int | None = None,
        coverage_ratio: float = 0.0,
        median_disparity_px: float = 0.0,
        fit_sample_count: int = 0,
        fit_rmse: float | None = None,
    ) -> TemporalStereoResult:
        depth = np.asarray(provisional_depth, dtype=np.float32)
        return TemporalStereoResult(
            stereo_depth_map=np.zeros_like(depth, dtype=np.float32),
            fused_depth_map=depth.copy(),
            diagnostics=TemporalStereoDiagnostics(
                enabled=True,
                applied=False,
                reference_keyframe_id=reference_keyframe_id,
                coverage_ratio=float(coverage_ratio),
                median_disparity_px=float(median_disparity_px),
                fit_sample_count=int(fit_sample_count),
                fit_rmse=(None if fit_rmse is None else float(fit_rmse)),
                fallback_reason=str(reason),
            ),
        )

    def _select_reference_keyframe(self, current_pose_world: np.ndarray) -> _TemporalStereoKeyframe | None:
        current_pose = np.asarray(current_pose_world, dtype=np.float32)
        for candidate in reversed(self._keyframes):
            rotation_delta = self._rotation_delta_deg(current_pose, candidate.pose_world)
            if rotation_delta <= self._max_rotation_deg:
                return candidate
        return None

    @staticmethod
    def _rotation_delta_deg(current_pose_world: np.ndarray, reference_pose_world: np.ndarray) -> float:
        relative_rotation = reference_pose_world[:3, :3].T @ current_pose_world[:3, :3]
        trace = float(np.trace(relative_rotation))
        cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def _dynamic_detection_mask(
        self,
        *,
        frame_shape: tuple[int, int],
        detections: list[Detection],
    ) -> np.ndarray:
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        for detection in detections:
            if str(detection.label).casefold() not in self._dynamic_labels:
                continue
            x1, y1, x2, y2 = detection.xyxy
            x1 = max(0, min(width, int(x1)))
            x2 = max(0, min(width, int(x2) + 1))
            y1 = max(0, min(height, int(y1)))
            y2 = max(0, min(height, int(y2) + 1))
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = True
        return mask

    def _compute_stereo_depth(
        self,
        *,
        frame_bgr: np.ndarray,
        reference: _TemporalStereoKeyframe,
        current_pose_world: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> _StereoDepthComputation | None:
        cv2 = load_cv2(self._cv2_module)
        current_bgr = np.asarray(frame_bgr, dtype=np.uint8)
        reference_bgr = np.asarray(reference.frame_bgr, dtype=np.uint8)
        if current_bgr.shape[:2] != reference_bgr.shape[:2]:
            return None

        current_small, reference_small, intrinsics_small = self._resize_pair_for_stereo(
            current_bgr,
            reference_bgr,
            intrinsics,
            cv2,
        )
        current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference_small, cv2.COLOR_BGR2GRAY)

        relative_transform = np.linalg.inv(np.asarray(reference.pose_world, dtype=np.float64)) @ np.asarray(
            current_pose_world,
            dtype=np.float64,
        )
        rotation = relative_transform[:3, :3]
        translation = relative_transform[:3, 3]
        translation_norm = float(np.linalg.norm(translation))
        if translation_norm < 1e-6:
            return None

        camera_matrix = np.array(
            [
                [intrinsics_small.fx, 0.0, intrinsics_small.cx],
                [0.0, intrinsics_small.fy, intrinsics_small.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        zero_dist = np.zeros((5, 1), dtype=np.float64)
        image_size = (int(current_small.shape[1]), int(current_small.shape[0]))
        rectify_result = cv2.stereoRectify(
            camera_matrix,
            zero_dist,
            camera_matrix,
            zero_dist,
            image_size,
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        r1, r2, p1, _p2, q, *_rest = rectify_result
        map1_x, map1_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            zero_dist,
            r1,
            p1,
            image_size,
            cv2.CV_32FC1,
        )
        map2_x, map2_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            zero_dist,
            r2,
            p1,
            image_size,
            cv2.CV_32FC1,
        )
        rectified_current = cv2.remap(current_gray, map1_x, map1_y, cv2.INTER_LINEAR)
        rectified_reference = cv2.remap(reference_gray, map2_x, map2_y, cv2.INTER_LINEAR)

        matcher = self._build_matcher(cv2, image_width=image_size[0])
        disparity_left = matcher.compute(rectified_current, rectified_reference).astype(np.float32) / 16.0
        disparity_right = matcher.compute(rectified_reference, rectified_current).astype(np.float32) / 16.0
        if hasattr(cv2, "filterSpeckles"):
            cv2.filterSpeckles(disparity_left, 0.0, 128, 1.0)
            cv2.filterSpeckles(disparity_right, 0.0, 128, 1.0)

        consistency_mask = self._left_right_consistency_mask(disparity_left, disparity_right)
        valid_mask = consistency_mask & np.isfinite(disparity_left) & (disparity_left > 0.0)
        coverage_ratio = float(valid_mask.mean())
        if not np.any(valid_mask):
            return _StereoDepthComputation(
                stereo_depth_map=np.zeros(current_bgr.shape[:2], dtype=np.float32),
                valid_mask=np.zeros(current_bgr.shape[:2], dtype=bool),
                coverage_ratio=coverage_ratio,
                median_disparity_px=0.0,
            )

        points_rectified = cv2.reprojectImageTo3D(disparity_left, q)
        points_current = (np.asarray(r1, dtype=np.float32).T @ points_rectified.reshape(-1, 3).T).T
        stereo_depth_map = self._rasterize_depth(
            points_current_xyz=points_current,
            valid_mask=valid_mask.reshape(-1),
            intrinsics=intrinsics_small,
            image_shape=current_small.shape[:2],
        )
        stereo_depth_map = cv2.resize(
            stereo_depth_map,
            (current_bgr.shape[1], current_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)
        full_valid_mask = np.isfinite(stereo_depth_map) & (stereo_depth_map > 0.05)
        valid_disparities = disparity_left[valid_mask]
        return _StereoDepthComputation(
            stereo_depth_map=stereo_depth_map,
            valid_mask=full_valid_mask,
            coverage_ratio=float(full_valid_mask.mean()),
            median_disparity_px=float(np.median(valid_disparities.astype(np.float64))),
        )

    def _resize_pair_for_stereo(
        self,
        current_bgr: np.ndarray,
        reference_bgr: np.ndarray,
        intrinsics: CameraIntrinsics,
        cv2_module,
    ) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        height, width = current_bgr.shape[:2]
        target_width = min(width, self._stereo_width)
        if target_width == width:
            return current_bgr, reference_bgr, intrinsics
        scale = target_width / float(width)
        target_height = max(1, int(round(height * scale)))
        current_small = cv2_module.resize(current_bgr, (target_width, target_height), interpolation=cv2_module.INTER_AREA)
        reference_small = cv2_module.resize(
            reference_bgr,
            (target_width, target_height),
            interpolation=cv2_module.INTER_AREA,
        )
        return (
            current_small,
            reference_small,
            CameraIntrinsics(
                fx=float(intrinsics.fx) * scale,
                fy=float(intrinsics.fy) * scale,
                cx=float(intrinsics.cx) * scale,
                cy=float(intrinsics.cy) * scale,
            ),
        )

    @staticmethod
    def _build_matcher(cv2_module, *, image_width: int):
        num_disparities = int(max(16, ((image_width // 8) + 15) // 16 * 16))
        block_size = 5
        return cv2_module.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=8,
            speckleWindowSize=64,
            speckleRange=2,
            preFilterCap=31,
            mode=getattr(cv2_module, "STEREO_SGBM_MODE_SGBM_3WAY", 0),
        )

    @staticmethod
    def _left_right_consistency_mask(disparity_left: np.ndarray, disparity_right: np.ndarray) -> np.ndarray:
        disparity_left = np.asarray(disparity_left, dtype=np.float32)
        disparity_right = np.asarray(disparity_right, dtype=np.float32)
        height, width = disparity_left.shape
        xs = np.arange(width, dtype=np.float32)
        xs = np.broadcast_to(xs.reshape(1, width), (height, width))
        right_x = np.rint(xs - disparity_left).astype(np.int32)
        valid = (right_x >= 0) & (right_x < width) & np.isfinite(disparity_left) & np.isfinite(disparity_right)
        sampled_right = np.zeros_like(disparity_left, dtype=np.float32)
        valid_positions = np.where(valid)
        sampled_right[valid_positions] = disparity_right[valid_positions[0], right_x[valid_positions]]
        return valid & (np.abs(disparity_left + sampled_right) <= 1.0)

    @staticmethod
    def _rasterize_depth(
        *,
        points_current_xyz: np.ndarray,
        valid_mask: np.ndarray,
        intrinsics: CameraIntrinsics,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        height, width = image_shape
        if points_current_xyz.size == 0:
            return np.zeros((height, width), dtype=np.float32)
        points = np.asarray(points_current_xyz, dtype=np.float32).reshape(-1, 3)
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        points = points[valid]
        if points.size == 0:
            return np.zeros((height, width), dtype=np.float32)
        z = points[:, 2]
        valid = np.isfinite(z) & (z > 0.05)
        points = points[valid]
        z = z[valid]
        if points.size == 0:
            return np.zeros((height, width), dtype=np.float32)

        u = np.rint((points[:, 0] * float(intrinsics.fx) / z) + float(intrinsics.cx)).astype(np.int32)
        v = np.rint((points[:, 1] * float(intrinsics.fy) / z) + float(intrinsics.cy)).astype(np.int32)
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u = u[valid]
        v = v[valid]
        z = z[valid]
        if z.size == 0:
            return np.zeros((height, width), dtype=np.float32)

        flat_indices = (v.astype(np.int64) * int(width)) + u.astype(np.int64)
        order = np.argsort(z)
        sorted_indices = flat_indices[order]
        sorted_depth = z[order]
        unique_indices, first_positions = np.unique(sorted_indices, return_index=True)
        depth_map = np.zeros((height * width,), dtype=np.float32)
        depth_map[unique_indices] = sorted_depth[first_positions].astype(np.float32)
        return depth_map.reshape((height, width))
