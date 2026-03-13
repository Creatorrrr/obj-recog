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
    max_depth: float = 6.0,
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


def depth_to_point_cloud_torch(
    frame_bgr,
    depth_map,
    intrinsics: CameraIntrinsics,
    stride: int,
    max_points: int,
    *,
    min_depth: float = 0.3,
    max_depth: float = 6.0,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError:
        return depth_to_point_cloud(
            frame_bgr=np.asarray(frame_bgr),
            depth_map=np.asarray(depth_map),
            intrinsics=intrinsics,
            stride=stride,
            max_points=max_points,
            min_depth=min_depth,
            max_depth=max_depth,
        )

    frame_tensor = torch.as_tensor(frame_bgr)
    depth_tensor = torch.as_tensor(depth_map)
    target_device = device or (
        str(depth_tensor.device)
        if isinstance(depth_tensor, torch.Tensor)
        else str(frame_tensor.device)
    )
    frame_tensor = frame_tensor.to(device=target_device)
    depth_tensor = depth_tensor.to(device=target_device, dtype=torch.float32)
    if frame_tensor.ndim != 3 or tuple(frame_tensor.shape[:2]) != tuple(depth_tensor.shape):
        raise ValueError("frame and depth map must share the same height and width")

    ys, xs = torch.meshgrid(
        torch.arange(0, depth_tensor.shape[0], stride, device=target_device, dtype=torch.float32),
        torch.arange(0, depth_tensor.shape[1], stride, device=target_device, dtype=torch.float32),
        indexing="ij",
    )
    sampled_depth = torch.clamp(depth_tensor[::stride, ::stride], min=min_depth, max=max_depth)
    xs_flat = xs.reshape(-1)
    ys_flat = ys.reshape(-1)
    depth_flat = sampled_depth.reshape(-1)
    if int(depth_flat.numel()) > int(max_points):
        keep = torch.linspace(
            0,
            int(depth_flat.numel()) - 1,
            steps=int(max_points),
            device=target_device,
            dtype=torch.int64,
        )
        xs_flat = xs_flat.index_select(0, keep)
        ys_flat = ys_flat.index_select(0, keep)
        depth_flat = depth_flat.index_select(0, keep)

    x = (xs_flat - float(intrinsics.cx)) * depth_flat / float(intrinsics.fx)
    y = (ys_flat - float(intrinsics.cy)) * depth_flat / float(intrinsics.fy)
    points_xyz = torch.stack((x, y, depth_flat), dim=1).to(dtype=torch.float32)

    pixel_x = xs_flat.to(dtype=torch.int64)
    pixel_y = ys_flat.to(dtype=torch.int64)
    sampled_bgr = frame_tensor.index_select(0, pixel_y).gather(
        1,
        pixel_x.view(-1, 1, 1).expand(-1, 1, int(frame_tensor.shape[2])),
    ).squeeze(1)
    points_rgb = sampled_bgr[:, [2, 1, 0]].to(dtype=torch.float32) / 255.0
    point_pixels = torch.stack((pixel_x, pixel_y), dim=1)
    return (
        points_xyz.detach().cpu().numpy().astype(np.float32, copy=False),
        points_rgb.detach().cpu().numpy().astype(np.float32, copy=False),
        point_pixels.detach().cpu().numpy().astype(np.int32, copy=False),
    )


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
