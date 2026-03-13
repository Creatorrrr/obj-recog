from __future__ import annotations

import numpy as np

from obj_recog.reconstruct import (
    CameraIntrinsics,
    depth_to_point_cloud,
    depth_to_point_cloud_torch,
    transform_points,
)


def test_depth_to_point_cloud_returns_expected_shapes() -> None:
    frame_bgr = np.full((4, 4, 3), 120, dtype=np.uint8)
    depth_map = np.linspace(0.3, 1.2, 16, dtype=np.float32).reshape(4, 4)
    intrinsics = CameraIntrinsics(fx=3.6, fy=3.6, cx=2.0, cy=2.0)

    points_xyz, points_rgb, point_pixels = depth_to_point_cloud(
        frame_bgr=frame_bgr,
        depth_map=depth_map,
        intrinsics=intrinsics,
        stride=1,
        max_points=32,
    )

    assert points_xyz.shape == (16, 3)
    assert points_rgb.shape == (16, 3)
    assert point_pixels.shape == (16, 2)
    assert points_xyz.dtype == np.float32
    assert points_rgb.dtype == np.float32
    assert np.all((points_rgb >= 0.0) & (points_rgb <= 1.0))


def test_depth_to_point_cloud_applies_stride_and_max_points() -> None:
    frame_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    depth_map = np.ones((8, 8), dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=7.2, fy=7.2, cx=4.0, cy=4.0)

    points_xyz, _, point_pixels = depth_to_point_cloud(
        frame_bgr=frame_bgr,
        depth_map=depth_map,
        intrinsics=intrinsics,
        stride=2,
        max_points=6,
    )

    assert points_xyz.shape == (6, 3)
    assert point_pixels.shape == (6, 2)


def test_depth_to_point_cloud_clips_depth_range() -> None:
    frame_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    depth_map = np.array([[0.1, 0.5], [2.5, 8.5]], dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=1.8, fy=1.8, cx=1.0, cy=1.0)

    points_xyz, _, _ = depth_to_point_cloud(
        frame_bgr=frame_bgr,
        depth_map=depth_map,
        intrinsics=intrinsics,
        stride=1,
        max_points=8,
        min_depth=0.3,
        max_depth=6.0,
    )

    assert np.isclose(points_xyz[:, 2].min(), 0.3)
    assert np.isclose(points_xyz[:, 2].max(), 6.0)


def test_transform_points_applies_camera_pose_world() -> None:
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
    pose_world = np.eye(4, dtype=np.float32)
    pose_world[:3, 3] = np.array([1.0, 0.5, 0.0], dtype=np.float32)

    transformed = transform_points(points, pose_world)

    assert np.allclose(
        transformed,
        np.array([[1.0, 0.5, 1.0], [1.1, 0.5, 1.0]], dtype=np.float32),
    )


def test_depth_to_point_cloud_torch_matches_numpy_shapes() -> None:
    frame_bgr = np.full((4, 4, 3), 90, dtype=np.uint8)
    depth_map = np.linspace(0.3, 1.2, 16, dtype=np.float32).reshape(4, 4)
    intrinsics = CameraIntrinsics(fx=3.6, fy=3.6, cx=2.0, cy=2.0)

    points_xyz, points_rgb, point_pixels = depth_to_point_cloud_torch(
        frame_bgr=frame_bgr,
        depth_map=depth_map,
        intrinsics=intrinsics,
        stride=1,
        max_points=32,
    )

    assert points_xyz.shape == (16, 3)
    assert points_rgb.shape == (16, 3)
    assert point_pixels.shape == (16, 2)
