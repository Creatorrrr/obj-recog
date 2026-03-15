from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import tarfile

from obj_recog.runtime_accel import detect_runtime_capabilities

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
    precision: str = "auto"
    detector_backend: str = "torch"
    depth_backend: str = "torch"
    segmentation_backend: str = "torch"
    opencv_cuda: str = "auto"
    detection_interval: int = 2
    inference_width: int = 640
    orb_features: int = 1200
    keyframe_translation: float = 0.12
    keyframe_rotation_deg: float = 8.0
    mapping_window_keyframes: int = 24
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
    scenario: str = "living_room_navigation_v1"
    sim_seed: int = 0
    sim_max_steps: int = 600
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
    sim_planner_model: str = "gpt-5-mini"
    sim_planner_timeout_sec: float = 8.0
    sim_replan_interval_sec: float = 4.0
    sim_selfcal_max_sec: float = 6.0
    sim_action_batch_size: int = 6
    sim_headless: bool = False
    sim_open3d_view: bool = False
    sim_interface_mode: str = "rgb_only"
    sim_render_backend: str = "software"
    blender_exec: str | None = None
    unity_player_path: str | None = None
    unity_host: str = "127.0.0.1"
    unity_port: int = 8765
    episode_output_dir: str | None = None
    sim_goal_selector: str = "llm"
    sim_goal_model: str = "gpt-5-mini"
    sim_goal_timeout_sec: float = 8.0
    sim_external_manifest: str | None = None
    sim_perception_mode: str = "runtime"
    render_profile: str = "photoreal"
    asset_cache_dir: str = str(Path.home() / ".cache" / "obj-recog" / "assets")
    asset_quality: str = "low"
    sim_profile: str = "living_room"
    scenario_preview_shots: bool = False
    validate_all_scenarios: bool = False
    validation_output_dir: str | None = None
    temporal_stereo: str = "on"


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
    bundled_vocabulary = _bundled_slam_vocabulary_path()
    if bundled_vocabulary.is_file() or _bundled_slam_vocabulary_archive().is_file():
        return str(bundled_vocabulary)
    return None


def prepare_slam_vocabulary(vocabulary_path: str | None) -> str | None:
    if vocabulary_path is None:
        return None

    candidate_path = Path(vocabulary_path)
    if candidate_path.is_file():
        return str(candidate_path)

    archive_path = (
        candidate_path
        if candidate_path.name.endswith(".tar.gz")
        else candidate_path.with_name(f"{candidate_path.name}.tar.gz")
    )
    if not archive_path.is_file():
        return str(candidate_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        member = next(
            (
                item
                for item in archive.getmembers()
                if item.isfile() and Path(item.name).name == candidate_path.name
            ),
            None,
        )
        if member is None:
            raise RuntimeError(
                f"SLAM vocabulary archive does not contain {candidate_path.name}: {archive_path}"
            )
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"failed to read SLAM vocabulary archive member: {archive_path}")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        with extracted, candidate_path.open("wb") as destination:
            destination.write(extracted.read())

    return str(candidate_path)


DEFAULT_DETECTION_INTERVAL = 2
DEFAULT_INFERENCE_WIDTH = 640
DEFAULT_ORB_FEATURES = 1200
DEFAULT_KEYFRAME_TRANSLATION = 0.12
DEFAULT_KEYFRAME_ROTATION_DEG = 8.0
DEFAULT_MAPPING_WINDOW_KEYFRAMES = 24
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
SIM_SCENARIO_CHOICES = ("living_room_navigation_v1",)
DEFAULT_SCENARIO = "living_room_navigation_v1"
DEFAULT_SIM_SEED = 0
DEFAULT_SIM_MAX_STEPS = 600
DEFAULT_EVAL_BUDGET_SEC = 20.0
DEFAULT_SIM_CAMERA_FPS = 10.0
DEFAULT_SIM_CAMERA_FOV_DEG = 72.0
DEFAULT_SIM_CAMERA_NEAR = 0.2
DEFAULT_SIM_CAMERA_FAR = 8.0
DEFAULT_SIM_DEPTH_NOISE_STD = 0.02
DEFAULT_SIM_MOTION_BLUR = 0.1
DEFAULT_SIM_YAW_RATE_LIMIT_DEG = 45.0
DEFAULT_SIM_LINEAR_VELOCITY_LIMIT = 0.5
DEFAULT_SIM_PLANNER_MODEL = "gpt-5-mini"
DEFAULT_SIM_PLANNER_TIMEOUT_SEC = 8.0
DEFAULT_SIM_REPLAN_INTERVAL_SEC = 4.0
DEFAULT_SIM_SELFCAL_MAX_SEC = 6.0
DEFAULT_SIM_ACTION_BATCH_SIZE = 6
DEFAULT_SIM_OPEN3D_VIEW = "off"
DEFAULT_SIM_INTERFACE_MODE = "rgb_only"
DEFAULT_UNITY_HOST = "127.0.0.1"
DEFAULT_UNITY_PORT = 8765
DEFAULT_PRECISION = "auto"
DEFAULT_DETECTOR_BACKEND = "tensorrt" if os.name == "nt" else "torch"
DEFAULT_DEPTH_BACKEND = "torch"
DEFAULT_SEGMENTATION_BACKEND = "torch"
DEFAULT_OPENCV_CUDA = "auto"
DEFAULT_SIM_RENDER_BACKEND = "software"

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
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default=DEFAULT_PRECISION)
    parser.add_argument(
        "--detector-backend",
        choices=("torch", "tensorrt"),
        default=DEFAULT_DETECTOR_BACKEND,
    )
    parser.add_argument("--depth-backend", choices=("torch",), default=DEFAULT_DEPTH_BACKEND)
    parser.add_argument(
        "--segmentation-backend",
        choices=("torch",),
        default=DEFAULT_SEGMENTATION_BACKEND,
    )
    parser.add_argument("--opencv-cuda", choices=("auto", "on", "off"), default=DEFAULT_OPENCV_CUDA)
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
    parser.add_argument("--sim-planner-model", type=str, default=DEFAULT_SIM_PLANNER_MODEL)
    parser.add_argument("--sim-planner-timeout-sec", type=_positive_float, default=DEFAULT_SIM_PLANNER_TIMEOUT_SEC)
    parser.add_argument("--sim-replan-interval-sec", type=_positive_float, default=DEFAULT_SIM_REPLAN_INTERVAL_SEC)
    parser.add_argument("--sim-selfcal-max-sec", type=_positive_float, default=DEFAULT_SIM_SELFCAL_MAX_SEC)
    parser.add_argument("--sim-action-batch-size", type=_positive_int, default=DEFAULT_SIM_ACTION_BATCH_SIZE)
    parser.add_argument("--sim-headless", action="store_true")
    parser.add_argument("--sim-open3d-view", choices=("on", "off"), default=DEFAULT_SIM_OPEN3D_VIEW)
    parser.add_argument("--sim-interface-mode", choices=("rgb_only",), default=DEFAULT_SIM_INTERFACE_MODE)
    parser.add_argument(
        "--sim-render-backend",
        choices=("software", "blender-gpu"),
        default=DEFAULT_SIM_RENDER_BACKEND,
    )
    parser.add_argument("--blender-exec", type=str, default=None)
    parser.add_argument("--unity-player-path", type=str, default=None)
    parser.add_argument("--unity-host", type=str, default=DEFAULT_UNITY_HOST)
    parser.add_argument("--unity-port", type=_positive_int, default=DEFAULT_UNITY_PORT)
    parser.add_argument("--explanation-mode", choices=("on", "off"), default="on")
    parser.add_argument("--explanation-model", type=str, default=DEFAULT_EXPLANATION_MODEL)
    parser.add_argument("--temporal-stereo", choices=("on", "off"), default="on")
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
    camera_calibration = args.camera_calibration
    if camera_calibration is None and args.input_source == "sim":
        bundled_calibration = _default_sim_camera_calibration()
        if bundled_calibration is not None:
            camera_calibration = bundled_calibration
    return AppConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        device=args.device,
        precision=args.precision,
        detector_backend=args.detector_backend,
        depth_backend=args.depth_backend,
        segmentation_backend=args.segmentation_backend,
        opencv_cuda=args.opencv_cuda,
        conf_threshold=args.conf_threshold,
        point_stride=args.point_stride,
        max_points=args.max_points,
        segmentation_mode=args.segmentation_mode,
        segmentation_alpha=args.segmentation_alpha,
        segmentation_interval=args.segmentation_interval,
        segmentation_input_size=DEFAULT_SEGMENTATION_INPUT_SIZE,
        camera_name=args.camera_name,
        camera_calibration=camera_calibration,
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
        temporal_stereo=args.temporal_stereo,
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
        sim_planner_model=args.sim_planner_model,
        sim_planner_timeout_sec=args.sim_planner_timeout_sec,
        sim_replan_interval_sec=args.sim_replan_interval_sec,
        sim_selfcal_max_sec=args.sim_selfcal_max_sec,
        sim_action_batch_size=args.sim_action_batch_size,
        sim_headless=bool(args.sim_headless),
        sim_open3d_view=(args.sim_open3d_view == "on"),
        sim_interface_mode=args.sim_interface_mode,
        sim_render_backend=args.sim_render_backend,
        blender_exec=args.blender_exec,
        unity_player_path=args.unity_player_path,
        unity_host=args.unity_host,
        unity_port=args.unity_port,
        sim_goal_selector="llm",
        sim_goal_model=args.sim_planner_model,
        sim_goal_timeout_sec=args.sim_planner_timeout_sec,
        sim_perception_mode="runtime",
        render_profile="photoreal",
        sim_profile="living_room",
    )


def _default_sim_camera_calibration() -> str | None:
    bundled_calibration = Path(__file__).resolve().parents[2] / "calibration" / "calibration.yaml"
    if bundled_calibration.is_file():
        return str(bundled_calibration)
    return None


def _bundled_slam_vocabulary_path() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"


def _bundled_slam_vocabulary_archive() -> Path:
    bundled_vocabulary = _bundled_slam_vocabulary_path()
    return bundled_vocabulary.with_name(f"{bundled_vocabulary.name}.tar.gz")


def resolve_depth_profile(profile_name: str) -> DepthProfileSettings:
    try:
        return DEPTH_PROFILE_SETTINGS[str(profile_name)]
    except KeyError as exc:
        raise ValueError(f"unknown depth profile: {profile_name}") from exc


def resolve_device(requested_device: str) -> str:
    capabilities = detect_runtime_capabilities(torch_module=torch)
    if requested_device == "cpu":
        return "cpu"

    cuda_available = bool(capabilities.cuda_available)
    mps_available = bool(capabilities.mps_available)

    if requested_device == "cuda":
        if cuda_available:
            return "cuda"
        raise RuntimeError("CUDA requested but the current torch runtime does not expose a CUDA device")

    if requested_device == "mps":
        return "mps" if mps_available else "cpu"

    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"
