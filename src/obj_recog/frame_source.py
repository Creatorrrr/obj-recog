from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from obj_recog.camera import CameraSession, read_camera_frame
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.slam_bridge import KeyframeObservation
from obj_recog.types import Detection


@dataclass(slots=True)
class FramePacket:
    frame_bgr: np.ndarray
    timestamp_sec: float | None
    depth_map: np.ndarray | None = None
    pose_world_gt: np.ndarray | None = None
    intrinsics_gt: CameraIntrinsics | None = None
    detections: list[Detection] | None = None
    scenario_state: Any | None = None
    tracking_state: str = "TRACKING"
    keyframe_inserted: bool = False
    keyframe_id: int | None = None
    optimized_keyframe_poses: dict[int, np.ndarray] = field(default_factory=dict)
    sparse_map_points_xyz: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    loop_closure_applied: bool = False
    tracked_feature_count: int = 0
    median_reprojection_error: float | None = None
    keyframe_observations: list[KeyframeObservation] = field(default_factory=list)
    map_points_changed: bool = False
    calibration_source: str | None = None


class FrameSource(Protocol):
    def next_frame(self, *, timeout_sec: float | None = 1.0) -> FramePacket | None:
        ...

    def close(self) -> None:
        ...


class LiveCameraFrameSource:
    def __init__(
        self,
        *,
        camera_session: CameraSession,
        time_source: Callable[[], float | None],
        frame_reader=read_camera_frame,
    ) -> None:
        self.camera_session = camera_session
        self._time_source = time_source
        self._frame_reader = frame_reader

    def next_frame(self, *, timeout_sec: float | None = 1.0) -> FramePacket | None:
        ok, frame_bgr = self._frame_reader(self.camera_session.capture, timeout_sec=timeout_sec)
        if not ok or frame_bgr is None:
            return None
        timestamp_sec = self._time_source()
        return FramePacket(
            frame_bgr=np.asarray(frame_bgr, dtype=np.uint8).copy(),
            timestamp_sec=None if timestamp_sec is None else float(timestamp_sec),
        )

    def close(self) -> None:
        self.camera_session.capture.release()
