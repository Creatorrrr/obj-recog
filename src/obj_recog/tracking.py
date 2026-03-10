from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from obj_recog.opencv_runtime import load_cv2
from obj_recog.reconstruct import CameraIntrinsics, back_project_pixels


def _load_cv2(cv2_module=None):
    return load_cv2(cv2_module)


def _camera_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _invert_camera_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = -(rotation.T @ translation.reshape(3))
    return pose


@dataclass(slots=True)
class TrackingResult:
    pose_world: np.ndarray
    tracking_ok: bool
    did_reset: bool
    inlier_count: int = 0
    reprojection_error: float = float("inf")

    @property
    def camera_pose_world(self) -> np.ndarray:
        return self.pose_world


def estimate_pose_from_correspondences(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: CameraIntrinsics,
    previous_pose_world: np.ndarray,
    *,
    min_correspondences: int = 60,
    min_inliers: int = 35,
    reprojection_error_threshold: float = 3.0,
    cv2_module=None,
) -> TrackingResult:
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    previous_pose_world = np.asarray(previous_pose_world, dtype=np.float32)

    if object_points.shape[0] != image_points.shape[0] or object_points.shape[0] < min_correspondences:
        return TrackingResult(
            pose_world=previous_pose_world.copy(),
            tracking_ok=False,
            did_reset=False,
            inlier_count=0,
        )

    cv2 = _load_cv2(cv2_module)
    camera_matrix = _camera_matrix(intrinsics)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        None,
        iterationsCount=200,
        reprojectionError=reprojection_error_threshold,
        confidence=0.99,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        return TrackingResult(
            pose_world=previous_pose_world.copy(),
            tracking_ok=False,
            did_reset=False,
            inlier_count=0,
        )

    inlier_indices = inliers.reshape(-1)
    success, rvec, tvec = cv2.solvePnP(
        object_points[inlier_indices],
        image_points[inlier_indices],
        camera_matrix,
        None,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return TrackingResult(
            pose_world=previous_pose_world.copy(),
            tracking_ok=False,
            did_reset=False,
            inlier_count=0,
        )

    rotation, _ = cv2.Rodrigues(rvec)
    reprojection, _ = cv2.projectPoints(
        object_points[inlier_indices],
        rvec,
        tvec,
        camera_matrix,
        None,
    )
    reprojection = reprojection.reshape(-1, 2)
    reprojection_error = float(
        np.linalg.norm(reprojection - image_points[inlier_indices], axis=1).mean()
    )
    if reprojection_error > reprojection_error_threshold:
        return TrackingResult(
            pose_world=previous_pose_world.copy(),
            tracking_ok=False,
            did_reset=False,
            inlier_count=0,
        )

    current_pose_from_previous = _invert_camera_transform(
        rotation.astype(np.float32),
        tvec.astype(np.float32),
    )
    pose_world = previous_pose_world @ current_pose_from_previous
    return TrackingResult(
        pose_world=pose_world.astype(np.float32, copy=False),
        tracking_ok=True,
        did_reset=False,
        inlier_count=int(inlier_indices.size),
        reprojection_error=reprojection_error,
    )


class PoseTracker:
    def __init__(
        self,
        *,
        orb_features: int = 1200,
        ratio_test: float = 0.75,
        min_correspondences: int = 60,
        min_inliers: int = 35,
        reprojection_error_threshold: float = 3.0,
        cv2_module=None,
    ) -> None:
        self._cv2 = _load_cv2(cv2_module)
        self._orb = self._cv2.ORB_create(nfeatures=orb_features)
        self._matcher = self._cv2.BFMatcher(self._cv2.NORM_HAMMING, crossCheck=False)
        self._ratio_test = ratio_test
        self._min_correspondences = min_correspondences
        self._min_inliers = min_inliers
        self._reprojection_error_threshold = reprojection_error_threshold
        self.reset()

    def reset(self) -> None:
        self._previous_keypoints = None
        self._previous_descriptors = None
        self._previous_depth_map = None
        self._previous_pose_world = np.eye(4, dtype=np.float32)

    def _store_reference(
        self,
        keypoints,
        descriptors,
        depth_map: np.ndarray,
        pose_world: np.ndarray,
    ) -> None:
        self._previous_keypoints = keypoints
        self._previous_descriptors = descriptors
        self._previous_depth_map = np.asarray(depth_map, dtype=np.float32)
        self._previous_pose_world = np.asarray(pose_world, dtype=np.float32)

    def update(
        self,
        frame_bgr: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> TrackingResult:
        gray = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)

        if self._previous_descriptors is None or self._previous_keypoints is None:
            pose_world = np.eye(4, dtype=np.float32)
            self._store_reference(keypoints, descriptors, depth_map, pose_world)
            return TrackingResult(
                pose_world=pose_world,
                tracking_ok=True,
                did_reset=True,
                inlier_count=0,
                reprojection_error=0.0,
            )

        if descriptors is None or self._previous_descriptors is None:
            pose_world = np.eye(4, dtype=np.float32)
            self._store_reference(keypoints, descriptors, depth_map, pose_world)
            return TrackingResult(
                pose_world=pose_world,
                tracking_ok=False,
                did_reset=True,
            )

        matches = self._matcher.knnMatch(self._previous_descriptors, descriptors, k=2)
        object_points: list[list[float]] = []
        image_points: list[tuple[float, float]] = []

        previous_depth_map = np.asarray(self._previous_depth_map, dtype=np.float32)
        height, width = previous_depth_map.shape

        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            first, second = match_pair
            if first.distance >= self._ratio_test * second.distance:
                continue

            prev_keypoint = self._previous_keypoints[first.queryIdx].pt
            curr_keypoint = keypoints[first.trainIdx].pt
            px = int(round(prev_keypoint[0]))
            py = int(round(prev_keypoint[1]))
            if px < 0 or py < 0 or px >= width or py >= height:
                continue

            depth_value = float(previous_depth_map[py, px])
            if not np.isfinite(depth_value) or depth_value <= 0.0:
                continue

            object_point = back_project_pixels(
                np.array([[prev_keypoint[0], prev_keypoint[1]]], dtype=np.float32),
                np.array([depth_value], dtype=np.float32),
                intrinsics,
            )[0]
            object_points.append(object_point.tolist())
            image_points.append((float(curr_keypoint[0]), float(curr_keypoint[1])))

        result = estimate_pose_from_correspondences(
            object_points=np.asarray(object_points, dtype=np.float32),
            image_points=np.asarray(image_points, dtype=np.float32),
            intrinsics=intrinsics,
            previous_pose_world=self._previous_pose_world,
            min_correspondences=self._min_correspondences,
            min_inliers=self._min_inliers,
            reprojection_error_threshold=self._reprojection_error_threshold,
            cv2_module=self._cv2,
        )

        if result.tracking_ok:
            self._store_reference(keypoints, descriptors, depth_map, result.pose_world)
            return result

        pose_world = np.eye(4, dtype=np.float32)
        self._store_reference(keypoints, descriptors, depth_map, pose_world)
        return TrackingResult(
            pose_world=pose_world,
            tracking_ok=False,
            did_reset=True,
            inlier_count=result.inlier_count,
            reprojection_error=result.reprojection_error,
        )
