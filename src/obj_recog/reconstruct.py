from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def intrinsics_for_frame(width: int, height: int) -> CameraIntrinsics:
    focal = 0.9 * float(width)
    return CameraIntrinsics(
        fx=focal,
        fy=focal,
        cx=float(width) / 2.0,
        cy=float(height) / 2.0,
    )


def depth_to_point_cloud(
    frame_bgr: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    stride: int,
    max_points: int,
    min_depth: float = 0.3,
    max_depth: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[:2] != depth_map.shape:
        raise ValueError("frame and depth map must share the same height and width")

    ys, xs = np.mgrid[0 : depth_map.shape[0] : stride, 0 : depth_map.shape[1] : stride]
    sampled_depth = depth_map[::stride, ::stride].astype(np.float32, copy=False)
    sampled_depth = np.clip(sampled_depth, min_depth, max_depth)

    xs_flat = xs.reshape(-1).astype(np.float32, copy=False)
    ys_flat = ys.reshape(-1).astype(np.float32, copy=False)
    depth_flat = sampled_depth.reshape(-1)

    if depth_flat.size > max_points:
        keep = np.linspace(0, depth_flat.size - 1, max_points, dtype=np.int32)
        xs_flat = xs_flat[keep]
        ys_flat = ys_flat[keep]
        depth_flat = depth_flat[keep]

    x = (xs_flat - intrinsics.cx) * depth_flat / intrinsics.fx
    y = (ys_flat - intrinsics.cy) * depth_flat / intrinsics.fy
    points_xyz = np.stack((x, y, depth_flat), axis=1).astype(np.float32, copy=False)

    pixel_x = xs_flat.astype(np.int32, copy=False)
    pixel_y = ys_flat.astype(np.int32, copy=False)
    point_pixels = np.stack((pixel_x, pixel_y), axis=1)

    sampled_bgr = frame_bgr[pixel_y, pixel_x]
    points_rgb = sampled_bgr[:, ::-1].astype(np.float32) / 255.0
    return points_xyz, points_rgb, point_pixels


def back_project_pixels(
    pixel_xy: np.ndarray,
    depth_values: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    pixel_xy = np.asarray(pixel_xy, dtype=np.float32)
    depth_values = np.asarray(depth_values, dtype=np.float32).reshape(-1)
    x = (pixel_xy[:, 0] - intrinsics.cx) * depth_values / intrinsics.fx
    y = (pixel_xy[:, 1] - intrinsics.cy) * depth_values / intrinsics.fy
    return np.stack((x, y, depth_values), axis=1).astype(np.float32, copy=False)


def transform_points(points_xyz: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points_xyz.size == 0:
        return points_xyz.copy()

    homogeneous = np.concatenate(
        (
            points_xyz,
            np.ones((points_xyz.shape[0], 1), dtype=np.float32),
        ),
        axis=1,
    )
    transformed = (np.asarray(pose_world, dtype=np.float32) @ homogeneous.T).T[:, :3]
    return transformed.astype(np.float32, copy=False)
