from __future__ import annotations

import numpy as np

from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.tracking import estimate_pose_from_correspondences


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array(
        [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]],
        dtype=np.float32,
    )


def _project(points_xyz: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    x = (points_xyz[:, 0] * intrinsics.fx / points_xyz[:, 2]) + intrinsics.cx
    y = (points_xyz[:, 1] * intrinsics.fy / points_xyz[:, 2]) + intrinsics.cy
    return np.stack((x, y), axis=1).astype(np.float32)


def test_estimate_pose_from_correspondences_returns_pose_for_valid_matches() -> None:
    intrinsics = CameraIntrinsics(fx=420.0, fy=420.0, cx=320.0, cy=240.0)
    rng = np.random.default_rng(7)
    object_points = np.column_stack(
        (
            rng.uniform(-0.3, 0.3, 80),
            rng.uniform(-0.2, 0.2, 80),
            rng.uniform(1.2, 2.2, 80),
        )
    ).astype(np.float32)
    rotation = _rotation_y(5.0)
    translation = np.array([0.08, -0.02, 0.04], dtype=np.float32)
    current_points = (rotation @ object_points.T).T + translation
    image_points = _project(current_points, intrinsics)

    result = estimate_pose_from_correspondences(
        object_points=object_points,
        image_points=image_points,
        intrinsics=intrinsics,
        previous_pose_world=np.eye(4, dtype=np.float32),
    )

    expected_pose = np.eye(4, dtype=np.float32)
    expected_pose[:3, :3] = rotation.T
    expected_pose[:3, 3] = -(rotation.T @ translation)

    assert result.tracking_ok is True
    assert result.inlier_count >= 35
    assert np.allclose(result.pose_world[:3, :3], expected_pose[:3, :3], atol=1e-2)
    assert np.allclose(result.pose_world[:3, 3], expected_pose[:3, 3], atol=1e-2)


def test_estimate_pose_from_correspondences_rejects_small_match_sets() -> None:
    intrinsics = CameraIntrinsics(fx=420.0, fy=420.0, cx=320.0, cy=240.0)
    object_points = np.ones((10, 3), dtype=np.float32)
    image_points = np.ones((10, 2), dtype=np.float32)

    result = estimate_pose_from_correspondences(
        object_points=object_points,
        image_points=image_points,
        intrinsics=intrinsics,
        previous_pose_world=np.eye(4, dtype=np.float32),
    )

    assert result.tracking_ok is False
    assert result.inlier_count == 0
