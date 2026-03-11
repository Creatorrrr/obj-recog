from __future__ import annotations

import argparse
from dataclasses import dataclass
import os

try:
    import torch as _torch
except ImportError:  # pragma: no cover - exercised only without torch installed.
    class _MPSBackend:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Backends:
        mps = _MPSBackend()

    class _TorchStub:
        backends = _Backends()

    torch = _TorchStub()
else:
    torch = _torch


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera_index: int
    width: int
    height: int
    device: str
    conf_threshold: float
    point_stride: int
    max_points: int
    detection_interval: int = 2
    inference_width: int = 640
    orb_features: int = 1200
    keyframe_translation: float = 0.12
    keyframe_rotation_deg: float = 8.0
    mapping_window_keyframes: int = 12
    map_voxel_size: float = 0.05
    max_map_points: int = 200_000
    max_mesh_triangles: int = 10_000
    segmentation_mode: str = "panoptic"
    segmentation_alpha: float = 0.35
    segmentation_interval: int = 6
    segmentation_input_size: int = 512
    camera_name: str | None = None
    list_cameras: bool = False
    camera_calibration: str | None = None
    slam_vocabulary: str | None = None
    slam_width: int = 640
    slam_height: int = 360
    recalibrate: bool = False
    disable_slam_calibration: bool = False
    calibration_cache_dir: str | None = None
    graph_enabled: bool = True
    graph_max_visible_nodes: int = 64
    graph_relation_smoothing_frames: int = 15
    graph_occlusion_ttl_frames: int = 15
    explanation_enabled: bool = True
    explanation_model: str = "gpt-5-mini"
    explanation_timeout_sec: float = 8.0
    explanation_refresh_interval_sec: float = 10.0
    explanation_max_detections: int = 12
    explanation_max_graph_nodes: int = 20
    explanation_max_graph_edges: int = 20
    depth_profile: str = "balanced"


@dataclass(frozen=True, slots=True)
class DepthProfileSettings:
    name: str
    low_percentile: float
    high_percentile: float
    gamma: float
    min_depth: float
    max_depth: float
    voxel_size: float
    max_mesh_triangles: int
    depth_sampling_stride: int


DEFAULT_DETECTION_INTERVAL = 2
DEFAULT_INFERENCE_WIDTH = 640
DEFAULT_ORB_FEATURES = 1200
DEFAULT_KEYFRAME_TRANSLATION = 0.12
DEFAULT_KEYFRAME_ROTATION_DEG = 8.0
DEFAULT_MAPPING_WINDOW_KEYFRAMES = 12
DEFAULT_MAP_VOXEL_SIZE = 0.05
DEFAULT_MAX_MAP_POINTS = 200_000
DEFAULT_MAX_MESH_TRIANGLES = 10_000
DEFAULT_SEGMENTATION_MODE = "panoptic"
DEFAULT_SEGMENTATION_ALPHA = 0.35
DEFAULT_SEGMENTATION_INTERVAL = 6
DEFAULT_SEGMENTATION_INPUT_SIZE = 512
DEFAULT_SLAM_WIDTH = 640
DEFAULT_SLAM_HEIGHT = 360
DEFAULT_EXPLANATION_MODEL = "gpt-5-mini"
DEFAULT_EXPLANATION_TIMEOUT_SEC = 8.0
DEFAULT_EXPLANATION_REFRESH_INTERVAL_SEC = 10.0
DEFAULT_EXPLANATION_MAX_DETECTIONS = 12
DEFAULT_EXPLANATION_MAX_GRAPH_NODES = 20
DEFAULT_EXPLANATION_MAX_GRAPH_EDGES = 20
DEFAULT_DEPTH_PROFILE = "balanced"

DEPTH_PROFILE_SETTINGS: dict[str, DepthProfileSettings] = {
    "fast": DepthProfileSettings(
        name="fast",
        low_percentile=5.0,
        high_percentile=95.0,
        gamma=1.0,
        min_depth=0.3,
        max_depth=6.0,
        voxel_size=0.05,
        max_mesh_triangles=10_000,
        depth_sampling_stride=6,
    ),
    "balanced": DepthProfileSettings(
        name="balanced",
        low_percentile=2.0,
        high_percentile=98.0,
        gamma=0.82,
        min_depth=0.3,
        max_depth=6.0,
        voxel_size=0.045,
        max_mesh_triangles=14_000,
        depth_sampling_stride=5,
    ),
    "depthy": DepthProfileSettings(
        name="depthy",
        low_percentile=1.0,
        high_percentile=99.0,
        gamma=0.70,
        min_depth=0.3,
        max_depth=6.0,
        voxel_size=0.04,
        max_mesh_triangles=18_000,
        depth_sampling_stride=4,
    ),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _confidence(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("confidence threshold must be between 0 and 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time object recognition and monocular 3D reconstruction demo"
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-name", type=str, default=None)
    parser.add_argument("--width", type=_positive_int, default=1280)
    parser.add_argument("--height", type=_positive_int, default=720)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--conf-threshold", type=_confidence, default=0.35)
    parser.add_argument("--point-stride", type=_positive_int, default=4)
    parser.add_argument("--max-points", type=_positive_int, default=60_000)
    parser.add_argument("--segmentation-mode", choices=("panoptic", "off"), default=DEFAULT_SEGMENTATION_MODE)
    parser.add_argument("--segmentation-alpha", type=_confidence, default=DEFAULT_SEGMENTATION_ALPHA)
    parser.add_argument("--segmentation-interval", type=_positive_int, default=DEFAULT_SEGMENTATION_INTERVAL)
    parser.add_argument("--depth-profile", choices=tuple(DEPTH_PROFILE_SETTINGS), default=DEFAULT_DEPTH_PROFILE)
    parser.add_argument("--explanation-mode", choices=("on", "off"), default="on")
    parser.add_argument("--explanation-model", type=str, default=DEFAULT_EXPLANATION_MODEL)
    parser.add_argument("--camera-calibration", type=str, default=os.getenv("CAMERA_CALIBRATION"))
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--disable-slam-calibration", action="store_true")
    parser.add_argument("--calibration-cache-dir", type=str, default=None)
    parser.add_argument("--slam-vocabulary", type=str, default=None)
    parser.add_argument("--slam-width", type=_positive_int, default=DEFAULT_SLAM_WIDTH)
    parser.add_argument("--slam-height", type=_positive_int, default=DEFAULT_SLAM_HEIGHT)
    parser.add_argument("--list-cameras", action="store_true")
    return parser


def parse_config(argv: list[str] | None = None) -> AppConfig:
    args = build_parser().parse_args(argv)
    explanation_refresh_interval_raw = os.getenv("EXPLANATION_REFRESH_INTERVAL_SEC")
    explanation_refresh_interval_sec = (
        DEFAULT_EXPLANATION_REFRESH_INTERVAL_SEC
        if explanation_refresh_interval_raw is None
        else _positive_float(explanation_refresh_interval_raw)
    )
    return AppConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        device=args.device,
        conf_threshold=args.conf_threshold,
        point_stride=args.point_stride,
        max_points=args.max_points,
        segmentation_mode=args.segmentation_mode,
        segmentation_alpha=args.segmentation_alpha,
        segmentation_interval=args.segmentation_interval,
        segmentation_input_size=DEFAULT_SEGMENTATION_INPUT_SIZE,
        camera_name=args.camera_name,
        camera_calibration=args.camera_calibration,
        recalibrate=args.recalibrate,
        disable_slam_calibration=args.disable_slam_calibration,
        calibration_cache_dir=args.calibration_cache_dir,
        slam_vocabulary=args.slam_vocabulary,
        slam_width=args.slam_width,
        slam_height=args.slam_height,
        list_cameras=args.list_cameras,
        detection_interval=DEFAULT_DETECTION_INTERVAL,
        inference_width=DEFAULT_INFERENCE_WIDTH,
        orb_features=DEFAULT_ORB_FEATURES,
        keyframe_translation=DEFAULT_KEYFRAME_TRANSLATION,
        keyframe_rotation_deg=DEFAULT_KEYFRAME_ROTATION_DEG,
        mapping_window_keyframes=DEFAULT_MAPPING_WINDOW_KEYFRAMES,
        map_voxel_size=DEFAULT_MAP_VOXEL_SIZE,
        max_map_points=DEFAULT_MAX_MAP_POINTS,
        max_mesh_triangles=DEFAULT_MAX_MESH_TRIANGLES,
        explanation_enabled=args.explanation_mode == "on",
        explanation_model=args.explanation_model,
        explanation_timeout_sec=DEFAULT_EXPLANATION_TIMEOUT_SEC,
        explanation_refresh_interval_sec=explanation_refresh_interval_sec,
        explanation_max_detections=DEFAULT_EXPLANATION_MAX_DETECTIONS,
        explanation_max_graph_nodes=DEFAULT_EXPLANATION_MAX_GRAPH_NODES,
        explanation_max_graph_edges=DEFAULT_EXPLANATION_MAX_GRAPH_EDGES,
        depth_profile=args.depth_profile,
    )


def resolve_depth_profile(profile_name: str) -> DepthProfileSettings:
    try:
        return DEPTH_PROFILE_SETTINGS[str(profile_name)]
    except KeyError as exc:
        raise ValueError(f"unknown depth profile: {profile_name}") from exc


def resolve_device(requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"

    has_mps = hasattr(torch, "backends") and hasattr(torch.backends, "mps")
    mps_available = has_mps and torch.backends.mps.is_available()

    if requested_device == "mps":
        return "mps" if mps_available else "cpu"

    if mps_available:
        return "mps"
    return "cpu"
