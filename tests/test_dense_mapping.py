from __future__ import annotations

import numpy as np

from obj_recog.mapping import DenseKeyframeMapBuilder
from obj_recog.slam_bridge import SlamFrameResult


def _pose(tx: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = tx
    return pose


def _slam_result(
    *,
    tracking_state: str = "TRACKING",
    pose_world: np.ndarray | None = None,
    keyframe_inserted: bool = False,
    keyframe_id: int | None = None,
    optimized_keyframe_poses: dict[int, np.ndarray] | None = None,
    loop_closure_applied: bool = False,
) -> SlamFrameResult:
    return SlamFrameResult(
        tracking_state=tracking_state,
        pose_world=np.eye(4, dtype=np.float32) if pose_world is None else pose_world,
        keyframe_inserted=keyframe_inserted,
        keyframe_id=keyframe_id,
        optimized_keyframe_poses=optimized_keyframe_poses or {},
        sparse_map_points_xyz=np.empty((0, 3), dtype=np.float32),
        loop_closure_applied=loop_closure_applied,
    )


def test_dense_keyframe_map_builder_seeds_and_aggregates_inserted_keyframes() -> None:
    builder = DenseKeyframeMapBuilder(
        window_keyframes=30,
        voxel_size=0.0,
        max_map_points=200_000,
    )
    local_points = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=np.float32)
    colors = np.ones((2, 3), dtype=np.float32)

    update = builder.update(
        slam_result=_slam_result(
            pose_world=_pose(0.1),
            keyframe_inserted=True,
            keyframe_id=11,
            optimized_keyframe_poses={11: _pose(0.1)},
        ),
        frame_points_xyz=local_points,
        frame_points_rgb=colors,
    )

    assert update.is_keyframe is True
    assert update.keyframe_id == 11
    assert np.allclose(update.dense_map_points_xyz[:, 0], np.array([0.1, 0.3], dtype=np.float32))


def test_dense_keyframe_map_builder_reprojects_cached_clouds_after_loop_closure() -> None:
    builder = DenseKeyframeMapBuilder(
        window_keyframes=30,
        voxel_size=0.0,
        max_map_points=200_000,
    )
    local_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    colors = np.ones((1, 3), dtype=np.float32)

    builder.update(
        slam_result=_slam_result(
            pose_world=_pose(1.0),
            keyframe_inserted=True,
            keyframe_id=5,
            optimized_keyframe_poses={5: _pose(1.0)},
        ),
        frame_points_xyz=local_points,
        frame_points_rgb=colors,
    )
    update = builder.update(
        slam_result=_slam_result(
            pose_world=_pose(0.3),
            optimized_keyframe_poses={5: _pose(0.3)},
            loop_closure_applied=True,
        ),
        frame_points_xyz=local_points,
        frame_points_rgb=colors,
    )

    assert update.loop_closure_applied is True
    assert np.allclose(update.dense_map_points_xyz[:, 0], np.array([0.3], dtype=np.float32))


def test_dense_keyframe_map_builder_skips_append_when_tracking_is_lost() -> None:
    builder = DenseKeyframeMapBuilder(
        window_keyframes=30,
        voxel_size=0.0,
        max_map_points=200_000,
    )
    local_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    colors = np.ones((1, 3), dtype=np.float32)

    builder.update(
        slam_result=_slam_result(
            pose_world=_pose(0.0),
            keyframe_inserted=True,
            keyframe_id=2,
            optimized_keyframe_poses={2: _pose(0.0)},
        ),
        frame_points_xyz=local_points,
        frame_points_rgb=colors,
    )
    update = builder.update(
        slam_result=_slam_result(
            tracking_state="LOST",
            pose_world=_pose(0.5),
            keyframe_inserted=True,
            keyframe_id=3,
            optimized_keyframe_poses={2: _pose(0.0)},
        ),
        frame_points_xyz=local_points,
        frame_points_rgb=colors,
    )

    assert update.is_keyframe is False
    assert update.keyframe_id is None
    assert update.dense_map_points_xyz.shape == (1, 3)
