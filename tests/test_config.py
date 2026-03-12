from __future__ import annotations

import pytest

from obj_recog.config import build_parser, parse_config, resolve_device


_SCENARIO_IDS = (
    "studio_open_v1",
    "office_clutter_v1",
    "lab_corridor_v1",
    "showroom_occlusion_v1",
    "office_crossflow_v1",
    "warehouse_moving_target_v1",
)


def test_parse_config_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAMERA_CALIBRATION", raising=False)
    config = parse_config([])

    assert config.camera_index == 0
    assert config.width == 1280
    assert config.height == 720
    assert config.device == "auto"
    assert config.conf_threshold == pytest.approx(0.35)
    assert config.point_stride == 4
    assert config.max_points == 60_000
    assert config.camera_name is None
    assert config.list_cameras is False
    assert config.detection_interval == 2
    assert config.inference_width == 640
    assert config.orb_features == 1200
    assert config.keyframe_translation == pytest.approx(0.12)
    assert config.keyframe_rotation_deg == pytest.approx(8.0)
    assert config.mapping_window_keyframes == 12
    assert config.map_voxel_size == pytest.approx(0.05)
    assert config.max_map_points == 200_000
    assert config.max_mesh_triangles == 10_000
    assert config.segmentation_mode == "panoptic"
    assert config.segmentation_alpha == pytest.approx(0.35)
    assert config.segmentation_interval == 6
    assert config.segmentation_input_size == 512
    assert config.camera_calibration is None
    assert config.slam_vocabulary is None
    assert config.slam_width == 640
    assert config.slam_height == 360
    assert config.input_source == "live"
    assert config.scenario == "studio_open_v1"
    assert config.sim_seed == 0
    assert config.sim_max_steps == 600
    assert config.sim_profile == "lightweight"
    assert config.eval_budget_sec == pytest.approx(20.0)
    assert config.sim_camera_fps == pytest.approx(10.0)
    assert config.sim_camera_fov_deg == pytest.approx(72.0)
    assert config.sim_camera_near == pytest.approx(0.2)
    assert config.sim_camera_far == pytest.approx(8.0)
    assert config.sim_depth_noise_std == pytest.approx(0.02)
    assert config.sim_motion_blur == pytest.approx(0.1)
    assert config.sim_enable_distortion is False
    assert config.sim_yaw_rate_limit_deg == pytest.approx(45.0)
    assert config.sim_linear_velocity_limit == pytest.approx(0.5)
    assert config.sim_goal_selector == "heuristic"
    assert config.sim_goal_model == "gpt-5-mini"
    assert config.sim_goal_timeout_sec == pytest.approx(4.0)
    assert config.sim_external_manifest is None
    assert config.sim_perception_mode == "assisted"
    assert config.validate_all_scenarios is False
    assert config.validation_output_dir is None
    assert config.recalibrate is False
    assert config.disable_slam_calibration is False
    assert config.calibration_cache_dir is None
    assert config.explanation_enabled is True
    assert config.explanation_model == "gpt-5-mini"
    assert config.explanation_timeout_sec == pytest.approx(8.0)
    assert config.explanation_refresh_interval_sec == pytest.approx(10.0)
    assert config.explanation_max_detections == 12
    assert config.explanation_max_graph_nodes == 20
    assert config.explanation_max_graph_edges == 20
    assert config.depth_profile == "balanced"


def test_parse_config_accepts_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAMERA_CALIBRATION", raising=False)
    config = parse_config(
        [
            "--camera-index",
            "2",
            "--camera-name",
            "iPhone",
            "--width",
            "800",
            "--height",
            "600",
            "--device",
            "cpu",
            "--conf-threshold",
            "0.55",
            "--point-stride",
            "8",
            "--max-points",
            "5000",
            "--camera-calibration",
            "/tmp/camera.yaml",
            "--input-source",
            "sim",
            "--scenario",
            "warehouse_moving_target_v1",
            "--sim-seed",
            "17",
            "--sim-max-steps",
            "240",
            "--sim-profile",
            "external",
            "--eval-budget-sec",
            "18.5",
            "--sim-camera-fps",
            "12.5",
            "--sim-camera-fov-deg",
            "80.0",
            "--sim-camera-near",
            "0.15",
            "--sim-camera-far",
            "10.0",
            "--sim-depth-noise-std",
            "0.05",
            "--sim-motion-blur",
            "0.25",
            "--sim-enable-distortion",
            "--sim-yaw-rate-limit-deg",
            "60.0",
            "--sim-linear-velocity-limit",
            "0.8",
            "--sim-goal-selector",
            "llm",
            "--sim-goal-model",
            "gpt-4.1-mini",
            "--sim-goal-timeout-sec",
            "6.5",
            "--sim-external-manifest",
            "/tmp/blender-manifest.json",
            "--sim-perception-mode",
            "ground_truth",
            "--validate-all-scenarios",
            "--validation-output-dir",
            "/tmp/validation",
            "--segmentation-mode",
            "off",
            "--segmentation-alpha",
            "0.5",
            "--segmentation-interval",
            "9",
            "--slam-vocabulary",
            "/tmp/ORBvoc.txt",
            "--slam-width",
            "512",
            "--slam-height",
            "288",
            "--depth-profile",
            "depthy",
            "--recalibrate",
            "--disable-slam-calibration",
            "--calibration-cache-dir",
            "/tmp/calibration-cache",
            "--explanation-mode",
            "off",
            "--explanation-model",
            "gpt-4.1",
        ]
    )

    assert config.camera_index == 2
    assert config.camera_name == "iPhone"
    assert config.width == 800
    assert config.height == 600
    assert config.device == "cpu"
    assert config.conf_threshold == pytest.approx(0.55)
    assert config.point_stride == 8
    assert config.max_points == 5000
    assert config.camera_calibration == "/tmp/camera.yaml"
    assert config.input_source == "sim"
    assert config.scenario == "warehouse_moving_target_v1"
    assert config.sim_seed == 17
    assert config.sim_max_steps == 240
    assert config.sim_profile == "external"
    assert config.eval_budget_sec == pytest.approx(18.5)
    assert config.sim_camera_fps == pytest.approx(12.5)
    assert config.sim_camera_fov_deg == pytest.approx(80.0)
    assert config.sim_camera_near == pytest.approx(0.15)
    assert config.sim_camera_far == pytest.approx(10.0)
    assert config.sim_depth_noise_std == pytest.approx(0.05)
    assert config.sim_motion_blur == pytest.approx(0.25)
    assert config.sim_enable_distortion is True
    assert config.sim_yaw_rate_limit_deg == pytest.approx(60.0)
    assert config.sim_linear_velocity_limit == pytest.approx(0.8)
    assert config.sim_goal_selector == "llm"
    assert config.sim_goal_model == "gpt-4.1-mini"
    assert config.sim_goal_timeout_sec == pytest.approx(6.5)
    assert config.sim_external_manifest == "/tmp/blender-manifest.json"
    assert config.sim_perception_mode == "ground_truth"
    assert config.validate_all_scenarios is True
    assert config.validation_output_dir == "/tmp/validation"
    assert config.segmentation_mode == "off"
    assert config.segmentation_alpha == pytest.approx(0.5)
    assert config.segmentation_interval == 9
    assert config.segmentation_input_size == 512
    assert config.slam_vocabulary == "/tmp/ORBvoc.txt"
    assert config.slam_width == 512
    assert config.slam_height == 288
    assert config.depth_profile == "depthy"
    assert config.recalibrate is True
    assert config.disable_slam_calibration is True
    assert config.calibration_cache_dir == "/tmp/calibration-cache"
    assert config.explanation_enabled is False
    assert config.explanation_model == "gpt-4.1"
    assert config.list_cameras is False
    assert config.orb_features == 1200
    assert config.max_map_points == 200_000
    assert config.max_mesh_triangles == 10_000


def test_parse_config_uses_camera_calibration_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAMERA_CALIBRATION", "/tmp/from-env.yaml")

    config = parse_config([])

    assert config.camera_calibration == "/tmp/from-env.yaml"


def test_parse_config_prefers_cli_camera_calibration_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAMERA_CALIBRATION", "/tmp/from-env.yaml")

    config = parse_config(["--camera-calibration", "/tmp/from-cli.yaml"])

    assert config.camera_calibration == "/tmp/from-cli.yaml"


def test_parse_config_uses_explanation_refresh_interval_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPLANATION_REFRESH_INTERVAL_SEC", "3.5")

    config = parse_config([])

    assert config.explanation_refresh_interval_sec == pytest.approx(3.5)


def test_parse_config_accepts_assisted_sim_perception_mode() -> None:
    config = parse_config(["--sim-perception-mode", "assisted"])

    assert config.sim_perception_mode == "assisted"


def test_parse_config_accepts_validation_flags() -> None:
    config = parse_config(
        [
            "--validate-all-scenarios",
            "--validation-output-dir",
            "/tmp/validation",
        ]
    )

    assert config.validate_all_scenarios is True
    assert config.validation_output_dir == "/tmp/validation"


@pytest.mark.parametrize("scenario_name", _SCENARIO_IDS)
def test_parse_config_accepts_all_supported_scenarios(scenario_name: str) -> None:
    config = parse_config(["--scenario", scenario_name])

    assert config.scenario == scenario_name


def test_parser_rejects_invalid_choices() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "cuda"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--segmentation-mode", "semantic"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--depth-profile", "extreme"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--input-source", "dataset"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--scenario", "warehouse"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--sim-profile", "hybrid"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--sim-goal-selector", "manual"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--sim-perception-mode", "hybrid"])


def test_resolve_device_prefers_available_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("obj_recog.config.torch.backends.mps.is_available", lambda: True)

    assert resolve_device("auto") == "mps"


def test_resolve_device_falls_back_to_cpu_when_mps_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("obj_recog.config.torch.backends.mps.is_available", lambda: False)

    assert resolve_device("auto") == "cpu"
