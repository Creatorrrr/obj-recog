from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

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
    input_source: str = "live"
    scenario: str = "studio_open_v1"
    sim_seed: int = 0
    sim_max_steps: int = 600
    sim_profile: str = "lightweight"
    eval_budget_sec: float = 20.0
    sim_camera_fps: float = 10.0
    sim_camera_fov_deg: float = 72.0
    sim_camera_near: float = 0.2
    sim_camera_far: float = 8.0
    sim_depth_noise_std: float = 0.02
    sim_motion_blur: float = 0.1
    sim_enable_distortion: bool = False
    sim_yaw_rate_limit_deg: float = 45.0
    sim_linear_velocity_limit: float = 0.5
    sim_goal_selector: str = "heuristic"
    sim_goal_model: str = "gpt-5-mini"
    sim_goal_timeout_sec: float = 4.0
    sim_external_manifest: str | None = None
    sim_perception_mode: str = "assisted"
    render_profile: str = "fast"
    asset_cache_dir: str = str(Path.home() / ".cache" / "obj-recog" / "assets")
    asset_quality: str = "low"
    blender_exec: str | None = None
    scenario_preview_shots: bool = False
    validate_all_scenarios: bool = False
    validation_output_dir: str | None = None


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


def _default_slam_vocabulary() -> str | None:
    bundled_vocabulary = Path(__file__).resolve().parents[2] / "third_party" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"
    if bundled_vocabulary.is_file():
        return str(bundled_vocabulary)
    return None


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
DEFAULT_INPUT_SOURCE = "live"
SIM_SCENARIO_CHOICES = (
    "studio_open_v1",
    "office_clutter_v1",
    "lab_corridor_v1",
    "showroom_occlusion_v1",
    "office_crossflow_v1",
    "warehouse_moving_target_v1",
)
DEFAULT_SCENARIO = "studio_open_v1"
DEFAULT_SIM_SEED = 0
DEFAULT_SIM_MAX_STEPS = 600
DEFAULT_SIM_PROFILE = "lightweight"
DEFAULT_EVAL_BUDGET_SEC = 20.0
DEFAULT_SIM_CAMERA_FPS = 10.0
DEFAULT_SIM_CAMERA_FOV_DEG = 72.0
DEFAULT_SIM_CAMERA_NEAR = 0.2
DEFAULT_SIM_CAMERA_FAR = 8.0
DEFAULT_SIM_DEPTH_NOISE_STD = 0.02
DEFAULT_SIM_MOTION_BLUR = 0.1
DEFAULT_SIM_YAW_RATE_LIMIT_DEG = 45.0
DEFAULT_SIM_LINEAR_VELOCITY_LIMIT = 0.5
DEFAULT_SIM_GOAL_SELECTOR = "heuristic"
DEFAULT_SIM_GOAL_MODEL = "gpt-5-mini"
DEFAULT_SIM_GOAL_TIMEOUT_SEC = 4.0
DEFAULT_SIM_PERCEPTION_MODE = "assisted"
DEFAULT_RENDER_PROFILE = "fast"
DEFAULT_ASSET_CACHE_DIR = str(Path.home() / ".cache" / "obj-recog" / "assets")
DEFAULT_ASSET_QUALITY = "low"

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
    parser.add_argument("--input-source", choices=("live", "sim"), default=DEFAULT_INPUT_SOURCE)
    parser.add_argument("--scenario", choices=SIM_SCENARIO_CHOICES, default=DEFAULT_SCENARIO)
    parser.add_argument("--sim-seed", type=int, default=DEFAULT_SIM_SEED)
    parser.add_argument("--sim-max-steps", type=_positive_int, default=DEFAULT_SIM_MAX_STEPS)
    parser.add_argument("--sim-profile", choices=("lightweight", "external"), default=DEFAULT_SIM_PROFILE)
    parser.add_argument("--eval-budget-sec", type=_positive_float, default=DEFAULT_EVAL_BUDGET_SEC)
    parser.add_argument("--sim-camera-fps", type=_positive_float, default=DEFAULT_SIM_CAMERA_FPS)
    parser.add_argument("--sim-camera-fov-deg", type=_positive_float, default=DEFAULT_SIM_CAMERA_FOV_DEG)
    parser.add_argument("--sim-camera-near", type=_positive_float, default=DEFAULT_SIM_CAMERA_NEAR)
    parser.add_argument("--sim-camera-far", type=_positive_float, default=DEFAULT_SIM_CAMERA_FAR)
    parser.add_argument("--sim-depth-noise-std", type=float, default=DEFAULT_SIM_DEPTH_NOISE_STD)
    parser.add_argument("--sim-motion-blur", type=_confidence, default=DEFAULT_SIM_MOTION_BLUR)
    parser.add_argument("--sim-enable-distortion", action="store_true")
    parser.add_argument("--sim-yaw-rate-limit-deg", type=_positive_float, default=DEFAULT_SIM_YAW_RATE_LIMIT_DEG)
    parser.add_argument("--sim-linear-velocity-limit", type=_positive_float, default=DEFAULT_SIM_LINEAR_VELOCITY_LIMIT)
    parser.add_argument("--sim-goal-selector", choices=("heuristic", "llm"), default=DEFAULT_SIM_GOAL_SELECTOR)
    parser.add_argument("--sim-goal-model", type=str, default=DEFAULT_SIM_GOAL_MODEL)
    parser.add_argument("--sim-goal-timeout-sec", type=_positive_float, default=DEFAULT_SIM_GOAL_TIMEOUT_SEC)
    parser.add_argument("--sim-external-manifest", type=str, default=None)
    parser.add_argument(
        "--sim-perception-mode",
        choices=("runtime", "ground_truth", "assisted"),
        default=DEFAULT_SIM_PERCEPTION_MODE,
    )
    parser.add_argument("--render-profile", choices=("fast", "photoreal"), default=DEFAULT_RENDER_PROFILE)
    parser.add_argument("--asset-cache-dir", type=str, default=DEFAULT_ASSET_CACHE_DIR)
    parser.add_argument("--asset-quality", choices=("low", "high"), default=DEFAULT_ASSET_QUALITY)
    parser.add_argument("--blender-exec", type=str, default=None)
    parser.add_argument("--scenario-preview-shots", action="store_true")
    parser.add_argument("--validate-all-scenarios", action="store_true")
    parser.add_argument("--validation-output-dir", type=str, default=None)
    parser.add_argument("--explanation-mode", choices=("on", "off"), default="on")
    parser.add_argument("--explanation-model", type=str, default=DEFAULT_EXPLANATION_MODEL)
    parser.add_argument("--camera-calibration", type=str, default=os.getenv("CAMERA_CALIBRATION"))
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--disable-slam-calibration", action="store_true")
    parser.add_argument("--calibration-cache-dir", type=str, default=None)
    parser.add_argument("--slam-vocabulary", type=str, default=_default_slam_vocabulary())
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
        input_source=args.input_source,
        scenario=args.scenario,
        sim_seed=args.sim_seed,
        sim_max_steps=args.sim_max_steps,
        sim_profile=args.sim_profile,
        eval_budget_sec=args.eval_budget_sec,
        sim_camera_fps=args.sim_camera_fps,
        sim_camera_fov_deg=args.sim_camera_fov_deg,
        sim_camera_near=args.sim_camera_near,
        sim_camera_far=args.sim_camera_far,
        sim_depth_noise_std=max(0.0, float(args.sim_depth_noise_std)),
        sim_motion_blur=args.sim_motion_blur,
        sim_enable_distortion=bool(args.sim_enable_distortion),
        sim_yaw_rate_limit_deg=args.sim_yaw_rate_limit_deg,
        sim_linear_velocity_limit=args.sim_linear_velocity_limit,
        sim_goal_selector=args.sim_goal_selector,
        sim_goal_model=args.sim_goal_model,
        sim_goal_timeout_sec=args.sim_goal_timeout_sec,
        sim_external_manifest=args.sim_external_manifest,
        sim_perception_mode=args.sim_perception_mode,
        render_profile=args.render_profile,
        asset_cache_dir=args.asset_cache_dir,
        asset_quality=args.asset_quality,
        blender_exec=args.blender_exec,
        scenario_preview_shots=bool(args.scenario_preview_shots),
        validate_all_scenarios=bool(args.validate_all_scenarios),
        validation_output_dir=args.validation_output_dir,
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
