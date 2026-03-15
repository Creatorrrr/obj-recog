from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.slam_bridge import SlamFrameResult
from obj_recog.temporal_stereo import TemporalStereoDepthEstimator, align_midas_depth_to_stereo
from obj_recog.types import Detection


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0)


def _slam_result(*, rotation_deg: float = 0.0) -> SlamFrameResult:
    radians = np.deg2rad(rotation_deg)
    pose_world = np.eye(4, dtype=np.float32)
    pose_world[:3, :3] = np.array(
        [
            [np.cos(radians), 0.0, np.sin(radians)],
            [0.0, 1.0, 0.0],
            [-np.sin(radians), 0.0, np.cos(radians)],
        ],
        dtype=np.float32,
    )
    return SlamFrameResult(
        tracking_state="TRACKING",
        pose_world=pose_world,
        keyframe_inserted=False,
        keyframe_id=None,
        optimized_keyframe_poses={},
        sparse_map_points_xyz=np.empty((0, 3), dtype=np.float32),
        loop_closure_applied=False,
    )


def test_update_keyframe_cache_keeps_latest_four_entries() -> None:
    estimator = TemporalStereoDepthEstimator()

    for keyframe_id in range(1, 7):
        estimator.update_keyframe_cache(
            keyframe_id=keyframe_id,
            frame_bgr=np.full((8, 8, 3), keyframe_id, dtype=np.uint8),
            frame_gray=np.full((8, 8), keyframe_id, dtype=np.uint8),
            pose_world=np.eye(4, dtype=np.float32),
            intrinsics=_intrinsics(),
        )

    assert estimator.cached_keyframe_ids == [3, 4, 5, 6]


def test_align_midas_depth_to_stereo_applies_log_affine_fit() -> None:
    midas_depth = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    stereo_depth = midas_depth * 2.0
    valid_mask = np.ones_like(midas_depth, dtype=bool)

    aligned_depth, fit_sample_count, fit_rmse, applied = align_midas_depth_to_stereo(
        midas_depth_map=midas_depth,
        stereo_depth_map=stereo_depth,
        valid_mask=valid_mask,
        min_samples=4,
        max_fit_rmse=0.2,
    )

    assert applied is True
    assert fit_sample_count == 6
    assert fit_rmse is not None and fit_rmse < 1e-4
    assert np.allclose(aligned_depth, stereo_depth, atol=1e-3)


def test_estimate_falls_back_to_midas_when_no_reference_keyframe_is_available() -> None:
    estimator = TemporalStereoDepthEstimator()
    provisional_depth = np.full((8, 8), 1.5, dtype=np.float32)

    result = estimator.estimate(
        frame_bgr=np.full((8, 8, 3), 127, dtype=np.uint8),
        provisional_depth_map=provisional_depth,
        slam_result=_slam_result(),
        intrinsics=_intrinsics(),
        detections=[],
    )

    assert np.array_equal(result.fused_depth_map, provisional_depth)
    assert result.diagnostics.applied is False
    assert result.diagnostics.fallback_reason == "no_reference_keyframe"


def test_estimate_fuses_stereo_depth_when_matching_succeeds(monkeypatch) -> None:
    estimator = TemporalStereoDepthEstimator(min_fit_samples=8)
    estimator.update_keyframe_cache(
        keyframe_id=7,
        frame_bgr=np.full((8, 8, 3), 90, dtype=np.uint8),
        frame_gray=np.full((8, 8), 90, dtype=np.uint8),
        pose_world=np.eye(4, dtype=np.float32),
        intrinsics=_intrinsics(),
    )
    provisional_depth = np.full((8, 8), 2.0, dtype=np.float32)
    stereo_depth = np.zeros((8, 8), dtype=np.float32)
    stereo_depth[:, :4] = 4.0
    valid_mask = stereo_depth > 0.0

    monkeypatch.setattr(
        estimator,
        "_compute_stereo_depth",
        lambda **_kwargs: SimpleNamespace(
            stereo_depth_map=stereo_depth,
            valid_mask=valid_mask,
            coverage_ratio=0.5,
            median_disparity_px=3.0,
        ),
    )

    result = estimator.estimate(
        frame_bgr=np.full((8, 8, 3), 127, dtype=np.uint8),
        provisional_depth_map=provisional_depth,
        slam_result=_slam_result(),
        intrinsics=_intrinsics(),
        detections=[
            Detection(
                xyxy=(6, 0, 7, 7),
                class_id=0,
                label="person",
                confidence=0.9,
                color=(255, 0, 0),
            )
        ],
    )

    assert result.diagnostics.applied is True
    assert result.diagnostics.reference_keyframe_id == 7
    assert np.allclose(result.fused_depth_map[:, :4], 4.0)
    assert np.allclose(result.fused_depth_map[:, 4:6], 4.0, atol=1e-3)


def test_compute_stereo_depth_filters_raw_sgbm_disparity_before_float_conversion(monkeypatch) -> None:
    class _FakeMatcher:
        def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.full(left.shape, 32, dtype=np.int16)

    class _FakeCv2:
        COLOR_BGR2GRAY = 0
        CALIB_ZERO_DISPARITY = 0
        CV_32FC1 = 5
        INTER_LINEAR = 1

        def __init__(self) -> None:
            self.filter_speckles_dtypes: list[np.dtype] = []

        def cvtColor(self, image: np.ndarray, _code: int) -> np.ndarray:
            return np.asarray(image[..., 0], dtype=np.uint8)

        def stereoRectify(self, *_args, **_kwargs):
            identity = np.eye(3, dtype=np.float64)
            projection = np.array(
                [
                    [8.0, 0.0, 4.0, 0.0],
                    [0.0, 8.0, 4.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )
            q = np.eye(4, dtype=np.float64)
            return identity, identity, projection, projection, q, None, None, None, None

        def initUndistortRectifyMap(self, *_args, **_kwargs):
            map_x = np.zeros((8, 8), dtype=np.float32)
            map_y = np.zeros((8, 8), dtype=np.float32)
            return map_x, map_y

        def remap(self, image: np.ndarray, _map_x: np.ndarray, _map_y: np.ndarray, _interpolation: int) -> np.ndarray:
            return image

        def filterSpeckles(self, image: np.ndarray, _new_value: float, _max_speckle_size: int, _max_diff: float) -> None:
            self.filter_speckles_dtypes.append(image.dtype)
            assert image.dtype == np.int16

        def reprojectImageTo3D(self, disparity: np.ndarray, _q: np.ndarray) -> np.ndarray:
            points = np.zeros(disparity.shape + (3,), dtype=np.float32)
            points[..., 2] = np.where(disparity > 0.0, 2.0, 0.0)
            return points

        def resize(self, image: np.ndarray, _size: tuple[int, int], interpolation: int) -> np.ndarray:
            assert interpolation == self.INTER_LINEAR
            return image

    fake_cv2 = _FakeCv2()
    estimator = TemporalStereoDepthEstimator(cv2_module=fake_cv2)
    estimator.update_keyframe_cache(
        keyframe_id=5,
        frame_bgr=np.full((8, 8, 3), 90, dtype=np.uint8),
        frame_gray=np.full((8, 8), 90, dtype=np.uint8),
        pose_world=np.eye(4, dtype=np.float32),
        intrinsics=_intrinsics(),
    )
    current_pose_world = np.eye(4, dtype=np.float32)
    current_pose_world[0, 3] = 0.1
    monkeypatch.setattr(estimator, "_build_matcher", lambda *_args, **_kwargs: _FakeMatcher())

    result = estimator._compute_stereo_depth(
        frame_bgr=np.full((8, 8, 3), 127, dtype=np.uint8),
        reference=estimator._keyframes[0],
        current_pose_world=current_pose_world,
        intrinsics=_intrinsics(),
    )

    assert result is not None
    assert fake_cv2.filter_speckles_dtypes == [np.dtype(np.int16), np.dtype(np.int16)]
