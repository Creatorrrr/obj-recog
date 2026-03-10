from __future__ import annotations

import numpy as np
import pytest

from obj_recog.mapping import TsdfMeshMapBuilder
from obj_recog.reconstruct import CameraIntrinsics
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


class _FakeIntrinsic:
    def __init__(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> None:
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


class _FakeImage:
    def __init__(self, data) -> None:
        self.data = np.asarray(data)


class _FakeRGBDImage:
    @staticmethod
    def create_from_color_and_depth(color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=True):
        return {
            "color": color,
            "depth": depth,
            "depth_scale": depth_scale,
            "depth_trunc": depth_trunc,
            "convert_rgb_to_intensity": convert_rgb_to_intensity,
        }


class _FakeMesh:
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, vertex_colors: np.ndarray) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int32)
        self.vertex_colors = np.asarray(vertex_colors, dtype=np.float64)
        self.decimation_targets: list[int] = []
        self.normals_computed = False

    def simplify_quadric_decimation(self, target_number_of_triangles: int, maximum_error=float("inf"), boundary_weight: float = 1.0):
        self.decimation_targets.append(target_number_of_triangles)
        keep = min(target_number_of_triangles, self.triangles.shape[0])
        return _FakeMesh(self.vertices, self.triangles[:keep], self.vertex_colors)

    def compute_vertex_normals(self) -> None:
        self.normals_computed = True


class _FakeVolume:
    def __init__(
        self,
        mesh: _FakeMesh,
        *,
        voxel_length: float,
        sdf_trunc: float,
        volume_unit_resolution: int,
        depth_sampling_stride: int,
    ) -> None:
        self.mesh = mesh
        self.integrations: list[tuple[object, object, np.ndarray]] = []
        self.reset_calls = 0
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        self.volume_unit_resolution = volume_unit_resolution
        self.depth_sampling_stride = depth_sampling_stride

    def integrate(self, image, intrinsic, extrinsic) -> None:
        self.integrations.append((image, intrinsic, np.asarray(extrinsic, dtype=np.float64)))

    def extract_triangle_mesh(self):
        return _FakeMesh(self.mesh.vertices, self.mesh.triangles, self.mesh.vertex_colors)

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeO3D:
    class pipelines:
        class integration:
            class TSDFVolumeColorType:
                RGB8 = "rgb8"

            created_volumes: list[_FakeVolume] = []

            @classmethod
            def ScalableTSDFVolume(
                cls,
                voxel_length: float,
                sdf_trunc: float,
                color_type,
                volume_unit_resolution: int = 16,
                depth_sampling_stride: int = 4,
            ):
                volume = _FakeVolume(
                    _FakeMesh(
                        vertices=np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.2, 1.0]], dtype=np.float64),
                        triangles=np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32),
                        vertex_colors=np.array([[1.0, 0.0, 0.0], [0.8, 0.8, 0.8], [0.0, 1.0, 0.0]], dtype=np.float64),
                    ),
                    voxel_length=voxel_length,
                    sdf_trunc=sdf_trunc,
                    volume_unit_resolution=volume_unit_resolution,
                    depth_sampling_stride=depth_sampling_stride,
                )
                cls.created_volumes.append(volume)
                return volume

    class camera:
        PinholeCameraIntrinsic = _FakeIntrinsic

    class geometry:
        Image = _FakeImage
        RGBDImage = _FakeRGBDImage


def test_tsdf_mesh_map_builder_integrates_keyframes_and_decimates_mesh() -> None:
    _FakeO3D.pipelines.integration.created_volumes.clear()
    builder = TsdfMeshMapBuilder(
        window_keyframes=30,
        voxel_size=0.03,
        max_mesh_triangles=1,
        o3d_module=_FakeO3D,
    )

    update = builder.update(
        slam_result=_slam_result(
            pose_world=_pose(1.0),
            keyframe_inserted=True,
            keyframe_id=11,
            optimized_keyframe_poses={11: _pose(1.0)},
        ),
        frame_bgr=np.full((4, 4, 3), 64, dtype=np.uint8),
        depth_map=np.full((4, 4), 1.5, dtype=np.float32),
        intrinsics=CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0),
    )

    volume = _FakeO3D.pipelines.integration.created_volumes[-1]
    assert len(volume.integrations) == 1
    assert volume.integrations[0][2][0, 3] == -1.0
    assert volume.depth_sampling_stride == 6
    assert update.is_keyframe is True
    assert update.keyframe_id == 11
    assert update.mesh_vertices_xyz.shape == (3, 3)
    assert update.mesh_triangles.shape == (1, 3)
    assert update.mesh_vertex_colors.shape == (3, 3)


def test_tsdf_mesh_map_builder_rebuilds_cached_keyframes_after_map_update() -> None:
    _FakeO3D.pipelines.integration.created_volumes.clear()
    builder = TsdfMeshMapBuilder(
        window_keyframes=30,
        voxel_size=0.03,
        max_mesh_triangles=4,
        o3d_module=_FakeO3D,
    )
    frame = np.full((4, 4, 3), 127, dtype=np.uint8)
    depth = np.full((4, 4), 1.0, dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0)

    builder.update(
        slam_result=_slam_result(
            pose_world=_pose(1.0),
            keyframe_inserted=True,
            keyframe_id=5,
            optimized_keyframe_poses={5: _pose(1.0)},
        ),
        frame_bgr=frame,
        depth_map=depth,
        intrinsics=intrinsics,
    )
    update = builder.update(
        slam_result=_slam_result(
            pose_world=_pose(0.3),
            optimized_keyframe_poses={5: _pose(0.3)},
            loop_closure_applied=True,
        ),
        frame_bgr=frame,
        depth_map=depth,
        intrinsics=intrinsics,
    )

    assert len(_FakeO3D.pipelines.integration.created_volumes) == 2
    rebuilt_volume = _FakeO3D.pipelines.integration.created_volumes[-1]
    assert len(rebuilt_volume.integrations) == 1
    assert rebuilt_volume.integrations[0][2][0, 3] == pytest.approx(-0.3)
    assert update.loop_closure_applied is True


def test_tsdf_mesh_map_builder_skips_append_when_tracking_is_lost() -> None:
    _FakeO3D.pipelines.integration.created_volumes.clear()
    builder = TsdfMeshMapBuilder(
        window_keyframes=30,
        voxel_size=0.03,
        max_mesh_triangles=4,
        o3d_module=_FakeO3D,
    )
    frame = np.full((4, 4, 3), 127, dtype=np.uint8)
    depth = np.full((4, 4), 1.0, dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0)

    builder.update(
        slam_result=_slam_result(
            pose_world=_pose(0.0),
            keyframe_inserted=True,
            keyframe_id=2,
            optimized_keyframe_poses={2: _pose(0.0)},
        ),
        frame_bgr=frame,
        depth_map=depth,
        intrinsics=intrinsics,
    )
    update = builder.update(
        slam_result=_slam_result(
            tracking_state="LOST",
            pose_world=_pose(0.5),
            keyframe_inserted=True,
            keyframe_id=3,
            optimized_keyframe_poses={2: _pose(0.0)},
        ),
        frame_bgr=frame,
        depth_map=depth,
        intrinsics=intrinsics,
    )

    assert len(_FakeO3D.pipelines.integration.created_volumes) == 1
    assert update.is_keyframe is False
    assert update.keyframe_id is None
    assert update.mesh_triangles.shape == (2, 3)
