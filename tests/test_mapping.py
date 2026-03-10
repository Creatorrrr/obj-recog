from __future__ import annotations

import numpy as np

from obj_recog.mapping import LocalMapBuilder


def _pose(tx: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]],
        dtype=np.float32,
    )
    pose[:3, 3] = np.array([tx, 0.0, 0.0], dtype=np.float32)
    return pose


def test_local_map_builder_seeds_first_keyframe_in_world_space() -> None:
    builder = LocalMapBuilder(
        translation_threshold=0.12,
        rotation_threshold_deg=8.0,
        frame_interval=12,
        window_keyframes=20,
        voxel_size=0.001,
        max_map_points=150_000,
    )
    points = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.2]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    update = builder.update(
        frame_index=0,
        pose_world=_pose(),
        frame_points_xyz=points,
        frame_points_rgb=colors,
        did_reset=True,
    )

    assert update.is_keyframe is True
    assert np.allclose(update.map_points_xyz, points)
    assert update.trajectory_xyz.shape == (1, 3)


def test_local_map_builder_limits_window_and_point_budget() -> None:
    builder = LocalMapBuilder(
        translation_threshold=0.05,
        rotation_threshold_deg=1.0,
        frame_interval=1,
        window_keyframes=2,
        voxel_size=0.001,
        max_map_points=5,
    )
    points = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.4, 0.0, 1.0]], dtype=np.float32)
    colors = np.ones((3, 3), dtype=np.float32)

    builder.update(0, _pose(0.0), points, colors, did_reset=True)
    builder.update(1, _pose(0.2), points, colors, did_reset=False)
    update = builder.update(2, _pose(0.4), points, colors, did_reset=False)

    assert builder.keyframe_count == 2
    assert update.map_points_xyz.shape[0] <= 5
    assert update.map_points_xyz[:, 0].min() >= 0.19


def test_local_map_builder_resets_segment_when_tracking_restarts() -> None:
    builder = LocalMapBuilder(
        translation_threshold=0.12,
        rotation_threshold_deg=8.0,
        frame_interval=12,
        window_keyframes=20,
        voxel_size=0.001,
        max_map_points=150_000,
    )
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    colors = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    builder.update(0, _pose(0.0), points, colors, did_reset=True)
    builder.update(1, _pose(0.2), points, colors, did_reset=False)
    update = builder.update(2, _pose(0.0), points, colors, did_reset=True)

    assert update.segment_id == 2
    assert update.trajectory_xyz.shape == (1, 3)
