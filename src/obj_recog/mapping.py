from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np

from obj_recog.reconstruct import CameraIntrinsics, transform_points
from obj_recog.slam_bridge import SlamFrameResult


@dataclass(slots=True)
class DenseMapUpdate:
    dense_map_points_xyz: np.ndarray
    dense_map_points_rgb: np.ndarray
    trajectory_xyz: np.ndarray
    is_keyframe: bool
    keyframe_id: int | None
    sparse_map_points_xyz: np.ndarray
    loop_closure_applied: bool
    segment_id: int

    @property
    def map_points_xyz(self) -> np.ndarray:
        return self.dense_map_points_xyz

    @property
    def map_points_rgb(self) -> np.ndarray:
        return self.dense_map_points_rgb


@dataclass(slots=True)
class MeshMapUpdate:
    mesh_vertices_xyz: np.ndarray
    mesh_triangles: np.ndarray
    mesh_vertex_colors: np.ndarray
    is_keyframe: bool
    keyframe_id: int | None
    trajectory_xyz: np.ndarray
    segment_id: int
    loop_closure_applied: bool

    @property
    def dense_map_points_xyz(self) -> np.ndarray:
        return self.mesh_vertices_xyz

    @property
    def dense_map_points_rgb(self) -> np.ndarray:
        return self.mesh_vertex_colors


@dataclass(slots=True)
class _TsdfKeyframe:
    keyframe_id: int
    frame_bgr: np.ndarray
    depth_map: np.ndarray
    intrinsics: CameraIntrinsics


@dataclass(slots=True)
class _DenseKeyframe:
    keyframe_id: int
    points_local_xyz: np.ndarray
    points_rgb: np.ndarray


class DenseKeyframeMapBuilder:
    def __init__(
        self,
        *,
        window_keyframes: int,
        voxel_size: float,
        max_map_points: int,
    ) -> None:
        self._window_keyframes = int(window_keyframes)
        self._voxel_size = float(voxel_size)
        self._max_map_points = int(max_map_points)
        self._segment_id = 0
        self.reset()

    def reset(self) -> None:
        self._segment_id += 1
        self._keyframes: OrderedDict[int, _DenseKeyframe] = OrderedDict()
        self._optimized_poses: dict[int, np.ndarray] = {}
        self._trajectory: list[np.ndarray] = []

    def _trim_window(self) -> bool:
        trimmed = False
        while len(self._keyframes) > self._window_keyframes:
            keyframe_id, _ = self._keyframes.popitem(last=False)
            self._optimized_poses.pop(keyframe_id, None)
            trimmed = True
        return trimmed

    def _voxel_downsample(
        self,
        points_xyz: np.ndarray,
        points_rgb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._voxel_size <= 0.0 or points_xyz.shape[0] == 0:
            return points_xyz, points_rgb

        voxel_keys = np.floor(points_xyz / self._voxel_size).astype(np.int64)
        _, inverse, counts = np.unique(voxel_keys, axis=0, return_inverse=True, return_counts=True)
        down_points = np.zeros((counts.shape[0], 3), dtype=np.float32)
        down_colors = np.zeros((counts.shape[0], 3), dtype=np.float32)
        np.add.at(down_points, inverse, points_xyz)
        np.add.at(down_colors, inverse, points_rgb)
        down_points /= counts[:, None]
        down_colors /= counts[:, None]
        return down_points, down_colors

    def _aggregate_dense_map(self) -> tuple[np.ndarray, np.ndarray]:
        world_points: list[np.ndarray] = []
        world_colors: list[np.ndarray] = []
        for keyframe_id, keyframe in self._keyframes.items():
            pose_world = self._optimized_poses.get(keyframe_id)
            if pose_world is None:
                continue
            world_points.append(transform_points(keyframe.points_local_xyz, pose_world))
            world_colors.append(keyframe.points_rgb)

        if not world_points:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )

        dense_points = np.concatenate(world_points, axis=0)
        dense_colors = np.concatenate(world_colors, axis=0)
        dense_points, dense_colors = self._voxel_downsample(dense_points, dense_colors)
        if dense_points.shape[0] > self._max_map_points:
            keep = np.linspace(0, dense_points.shape[0] - 1, self._max_map_points, dtype=np.int32)
            dense_points = dense_points[keep]
            dense_colors = dense_colors[keep]
        return (
            dense_points.astype(np.float32, copy=False),
            dense_colors.astype(np.float32, copy=False),
        )

    def update(
        self,
        *,
        slam_result: SlamFrameResult,
        frame_points_xyz: np.ndarray,
        frame_points_rgb: np.ndarray,
    ) -> DenseMapUpdate:
        pose_world = np.asarray(slam_result.pose_world, dtype=np.float32)
        self._trajectory.append(pose_world[:3, 3].astype(np.float32, copy=True))
        for keyframe_id, keyframe_pose in slam_result.optimized_keyframe_poses.items():
            self._optimized_poses[int(keyframe_id)] = np.asarray(keyframe_pose, dtype=np.float32)

        inserted_keyframe_id: int | None = None
        if slam_result.tracking_ok and slam_result.keyframe_inserted and slam_result.keyframe_id is not None:
            inserted_keyframe_id = int(slam_result.keyframe_id)
            self._keyframes[inserted_keyframe_id] = _DenseKeyframe(
                keyframe_id=inserted_keyframe_id,
                points_local_xyz=np.asarray(frame_points_xyz, dtype=np.float32).copy(),
                points_rgb=np.asarray(frame_points_rgb, dtype=np.float32).copy(),
            )
            self._optimized_poses.setdefault(inserted_keyframe_id, pose_world.copy())
            self._trim_window()

        dense_points, dense_colors = self._aggregate_dense_map()
        trajectory_xyz = (
            np.vstack(self._trajectory).astype(np.float32, copy=False)
            if self._trajectory
            else np.empty((0, 3), dtype=np.float32)
        )
        return DenseMapUpdate(
            dense_map_points_xyz=dense_points,
            dense_map_points_rgb=dense_colors,
            trajectory_xyz=trajectory_xyz,
            is_keyframe=inserted_keyframe_id is not None,
            keyframe_id=inserted_keyframe_id,
            sparse_map_points_xyz=np.asarray(slam_result.sparse_map_points_xyz, dtype=np.float32).reshape(-1, 3),
            loop_closure_applied=bool(slam_result.loop_closure_applied),
            segment_id=self._segment_id,
        )


class TsdfMeshMapBuilder:
    requires_point_cloud = False

    def __init__(
        self,
        *,
        window_keyframes: int,
        voxel_size: float,
        max_mesh_triangles: int,
        sdf_trunc_multiplier: float = 4.0,
        depth_scale: float = 1000.0,
        depth_trunc: float = 6.0,
        volume_unit_resolution: int = 16,
        depth_sampling_stride: int = 6,
        o3d_module=None,
    ) -> None:
        if o3d_module is None:
            try:
                import open3d as o3d
            except ImportError as exc:  # pragma: no cover - depends on local install.
                raise RuntimeError("open3d is required for TSDF mesh fusion") from exc
        else:
            o3d = o3d_module

        self._o3d = o3d
        self._window_keyframes = int(window_keyframes)
        self._voxel_size = float(voxel_size)
        self._max_mesh_triangles = int(max_mesh_triangles)
        self._sdf_trunc = float(max(sdf_trunc_multiplier * voxel_size, voxel_size))
        self._depth_scale = float(depth_scale)
        self._depth_trunc = float(depth_trunc)
        self._volume_unit_resolution = int(volume_unit_resolution)
        self._depth_sampling_stride = int(depth_sampling_stride)
        self._segment_id = 0
        self.reset()

    def _create_volume(self):
        return self._o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self._voxel_size,
            sdf_trunc=self._sdf_trunc,
            color_type=self._o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            volume_unit_resolution=self._volume_unit_resolution,
            depth_sampling_stride=self._depth_sampling_stride,
        )

    def reset(self) -> None:
        self._segment_id += 1
        self._keyframes: OrderedDict[int, _TsdfKeyframe] = OrderedDict()
        self._optimized_poses: dict[int, np.ndarray] = {}
        self._trajectory: list[np.ndarray] = []
        self._volume = self._create_volume()
        self._cached_vertices = np.empty((0, 3), dtype=np.float32)
        self._cached_triangles = np.empty((0, 3), dtype=np.int32)
        self._cached_vertex_colors = np.empty((0, 3), dtype=np.float32)

    def _trim_window(self) -> bool:
        trimmed = False
        while len(self._keyframes) > self._window_keyframes:
            keyframe_id, _ = self._keyframes.popitem(last=False)
            self._optimized_poses.pop(keyframe_id, None)
            trimmed = True
        return trimmed

    def _to_rgbd(self, frame_bgr: np.ndarray, depth_map: np.ndarray):
        color_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1].astype(np.uint8, copy=False))
        depth_scaled = np.clip(depth_map, 0.0, self._depth_trunc).astype(np.float32, copy=False)
        depth_uint16 = np.ascontiguousarray(
            np.round(depth_scaled * self._depth_scale).astype(np.uint16, copy=False)
        )
        color = self._o3d.geometry.Image(color_rgb)
        depth = self._o3d.geometry.Image(depth_uint16)
        return self._o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=self._depth_scale,
            depth_trunc=self._depth_trunc,
            convert_rgb_to_intensity=False,
        )

    def _to_intrinsic(self, intrinsics: CameraIntrinsics, width: int, height: int):
        return self._o3d.camera.PinholeCameraIntrinsic(
            int(width),
            int(height),
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(intrinsics.cx),
            float(intrinsics.cy),
        )

    def _extract_mesh_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mesh = self._volume.extract_triangle_mesh()
        triangle_count = len(np.asarray(mesh.triangles))
        if self._max_mesh_triangles > 0 and triangle_count > self._max_mesh_triangles:
            mesh = mesh.simplify_quadric_decimation(self._max_mesh_triangles)
        compute_normals = getattr(mesh, "compute_vertex_normals", None)
        if callable(compute_normals):
            compute_normals()
        vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
        triangles = np.asarray(mesh.triangles, dtype=np.int32).reshape(-1, 3)
        vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float32).reshape(-1, 3)
        if vertex_colors.shape[0] != vertices.shape[0]:
            vertex_colors = np.zeros((vertices.shape[0], 3), dtype=np.float32)
        return vertices, triangles, vertex_colors

    def _integrate_keyframe(self, keyframe: _TsdfKeyframe, pose_world: np.ndarray) -> None:
        rgbd = self._to_rgbd(keyframe.frame_bgr, keyframe.depth_map)
        intrinsic = self._to_intrinsic(
            keyframe.intrinsics,
            keyframe.frame_bgr.shape[1],
            keyframe.frame_bgr.shape[0],
        )
        extrinsic = np.linalg.inv(np.asarray(pose_world, dtype=np.float64))
        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def _rebuild_volume(self) -> None:
        self._volume = self._create_volume()
        for keyframe_id, keyframe in self._keyframes.items():
            pose_world = self._optimized_poses.get(keyframe_id)
            if pose_world is None:
                continue
            self._integrate_keyframe(keyframe, pose_world)
        (
            self._cached_vertices,
            self._cached_triangles,
            self._cached_vertex_colors,
        ) = self._extract_mesh_arrays()

    def update(
        self,
        *,
        slam_result: SlamFrameResult,
        frame_bgr: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> MeshMapUpdate:
        pose_world = np.asarray(slam_result.pose_world, dtype=np.float32)
        self._trajectory.append(pose_world[:3, 3].astype(np.float32, copy=True))
        for keyframe_id, keyframe_pose in slam_result.optimized_keyframe_poses.items():
            self._optimized_poses[int(keyframe_id)] = np.asarray(keyframe_pose, dtype=np.float32)

        inserted_keyframe_id: int | None = None
        if slam_result.tracking_ok and slam_result.keyframe_inserted and slam_result.keyframe_id is not None:
            inserted_keyframe_id = int(slam_result.keyframe_id)
            inserted_keyframe = _TsdfKeyframe(
                keyframe_id=inserted_keyframe_id,
                frame_bgr=np.asarray(frame_bgr, dtype=np.uint8).copy(),
                depth_map=np.asarray(depth_map, dtype=np.float32).copy(),
                intrinsics=intrinsics,
            )
            self._keyframes[inserted_keyframe_id] = inserted_keyframe
            self._optimized_poses.setdefault(inserted_keyframe_id, pose_world.copy())
            trimmed = self._trim_window()
            if trimmed or slam_result.loop_closure_applied:
                self._rebuild_volume()
            else:
                current_pose = self._optimized_poses.get(inserted_keyframe_id, pose_world)
                self._integrate_keyframe(inserted_keyframe, current_pose)
                (
                    self._cached_vertices,
                    self._cached_triangles,
                    self._cached_vertex_colors,
                ) = self._extract_mesh_arrays()
        elif slam_result.loop_closure_applied:
            self._rebuild_volume()

        trajectory_xyz = (
            np.vstack(self._trajectory).astype(np.float32, copy=False)
            if self._trajectory
            else np.empty((0, 3), dtype=np.float32)
        )
        return MeshMapUpdate(
            mesh_vertices_xyz=self._cached_vertices,
            mesh_triangles=self._cached_triangles,
            mesh_vertex_colors=self._cached_vertex_colors,
            is_keyframe=inserted_keyframe_id is not None,
            keyframe_id=inserted_keyframe_id,
            trajectory_xyz=trajectory_xyz,
            segment_id=self._segment_id,
            loop_closure_applied=bool(slam_result.loop_closure_applied),
        )


@dataclass(slots=True)
class _Keyframe:
    frame_index: int
    pose_world: np.ndarray
    points_xyz: np.ndarray
    points_rgb: np.ndarray


@dataclass(slots=True)
class MapUpdate:
    map_points_xyz: np.ndarray
    map_points_rgb: np.ndarray
    trajectory_xyz: np.ndarray
    is_keyframe: bool
    segment_id: int


class LocalMapBuilder:
    def __init__(
        self,
        *,
        translation_threshold: float,
        rotation_threshold_deg: float,
        frame_interval: int,
        window_keyframes: int,
        voxel_size: float,
        max_map_points: int,
    ) -> None:
        self._translation_threshold = translation_threshold
        self._rotation_threshold_deg = rotation_threshold_deg
        self._frame_interval = frame_interval
        self._voxel_size = voxel_size
        self._max_map_points = max_map_points
        self._window_keyframes = window_keyframes
        self._segment_id = 0
        self._pending_new_segment = True
        self._keyframes: deque[_Keyframe] = deque(maxlen=window_keyframes)
        self._trajectory: list[np.ndarray] = []

    @property
    def keyframe_count(self) -> int:
        return len(self._keyframes)

    def reset(self) -> None:
        self._pending_new_segment = True
        self._keyframes.clear()
        self._trajectory = []

    def _start_new_segment(self) -> None:
        self._segment_id += 1
        self._pending_new_segment = False
        self._keyframes.clear()
        self._trajectory = []

    def _rotation_delta_deg(self, current_pose: np.ndarray, reference_pose: np.ndarray) -> float:
        relative_rotation = reference_pose[:3, :3].T @ current_pose[:3, :3]
        trace = float(np.trace(relative_rotation))
        cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def _should_add_keyframe(self, frame_index: int, pose_world: np.ndarray) -> bool:
        if not self._keyframes:
            return True

        last_keyframe = self._keyframes[-1]
        translation = float(np.linalg.norm(pose_world[:3, 3] - last_keyframe.pose_world[:3, 3]))
        rotation = self._rotation_delta_deg(pose_world, last_keyframe.pose_world)
        frame_gap = frame_index - last_keyframe.frame_index
        return (
            translation > self._translation_threshold
            or rotation > self._rotation_threshold_deg
            or frame_gap >= self._frame_interval
        )

    def _append_keyframe(
        self,
        frame_index: int,
        pose_world: np.ndarray,
        frame_points_xyz: np.ndarray,
        frame_points_rgb: np.ndarray,
    ) -> None:
        world_points = transform_points(frame_points_xyz, pose_world)
        self._keyframes.append(
            _Keyframe(
                frame_index=frame_index,
                pose_world=np.asarray(pose_world, dtype=np.float32).copy(),
                points_xyz=world_points.astype(np.float32, copy=False),
                points_rgb=np.asarray(frame_points_rgb, dtype=np.float32).copy(),
            )
        )

    def _aggregate_points(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._keyframes:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )

        all_points = np.concatenate([keyframe.points_xyz for keyframe in self._keyframes], axis=0)
        all_colors = np.concatenate([keyframe.points_rgb for keyframe in self._keyframes], axis=0)
        all_points, all_colors = self._voxel_downsample(all_points, all_colors)

        if all_points.shape[0] > self._max_map_points:
            keep = np.linspace(0, all_points.shape[0] - 1, self._max_map_points, dtype=np.int32)
            all_points = all_points[keep]
            all_colors = all_colors[keep]

        return (
            all_points.astype(np.float32, copy=False),
            all_colors.astype(np.float32, copy=False),
        )

    def _voxel_downsample(
        self,
        points_xyz: np.ndarray,
        points_rgb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._voxel_size <= 0.0 or points_xyz.shape[0] == 0:
            return points_xyz, points_rgb

        voxel_keys = np.floor(points_xyz / self._voxel_size).astype(np.int64)
        _, inverse, counts = np.unique(voxel_keys, axis=0, return_inverse=True, return_counts=True)

        down_points = np.zeros((counts.shape[0], 3), dtype=np.float32)
        down_colors = np.zeros((counts.shape[0], 3), dtype=np.float32)
        np.add.at(down_points, inverse, points_xyz)
        np.add.at(down_colors, inverse, points_rgb)
        down_points /= counts[:, None]
        down_colors /= counts[:, None]
        return down_points, down_colors

    def update(
        self,
        frame_index: int,
        pose_world: np.ndarray,
        frame_points_xyz: np.ndarray,
        frame_points_rgb: np.ndarray,
        did_reset: bool,
    ) -> MapUpdate:
        pose_world = np.asarray(pose_world, dtype=np.float32)
        if did_reset or self._pending_new_segment or not self._keyframes:
            self._start_new_segment()
            is_keyframe = True
        else:
            is_keyframe = self._should_add_keyframe(frame_index, pose_world)

        self._trajectory.append(pose_world[:3, 3].astype(np.float32, copy=True))
        if is_keyframe:
            self._append_keyframe(frame_index, pose_world, frame_points_xyz, frame_points_rgb)

        map_points_xyz, map_points_rgb = self._aggregate_points()
        trajectory_xyz = np.vstack(self._trajectory).astype(np.float32, copy=False)
        return MapUpdate(
            map_points_xyz=map_points_xyz,
            map_points_rgb=map_points_rgb,
            trajectory_xyz=trajectory_xyz,
            is_keyframe=is_keyframe,
            segment_id=self._segment_id,
        )
