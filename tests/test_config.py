from __future__ import annotations

from pathlib import Path
import tarfile

import pytest

from obj_recog.config import AppConfig, build_parser, parse_config, prepare_slam_vocabulary, resolve_device


def test_parse_config_uses_living_room_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAMERA_CALIBRATION", raising=False)
    config = parse_config([])
    bundled_vocabulary = (
        Path(__file__).resolve().parents[1] / "third_party" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"
    )
    bundled_vocabulary_archive = bundled_vocabulary.with_name(f"{bundled_vocabulary.name}.tar.gz")
    expected_vocabulary = (
        str(bundled_vocabulary)
        if bundled_vocabulary.is_file() or bundled_vocabulary_archive.is_file()
        else None
    )

    assert config.input_source == "live"
    assert config.device == "auto"
    assert config.scenario == "living_room_navigation_v1"
    assert config.sim_planner_model == "gpt-5-mini"
    assert config.sim_planner_timeout_sec == pytest.approx(8.0)
    assert config.sim_replan_interval_sec == pytest.approx(4.0)
    assert config.sim_selfcal_max_sec == pytest.approx(6.0)
    assert config.sim_action_batch_size == 6
    assert config.sim_camera_fps == pytest.approx(24.0)
    assert config.sim_headless is False
    assert config.sim_open3d_view is False
    assert config.segmentation_interval == 2
    assert config.reconstruction_viewer_mode == "auto"
    assert config.sim_interface_mode == "rgb_only"
    assert config.sim_render_backend == "software"
    assert config.blender_exec is None
    assert config.unity_player_path is None
    assert config.unity_host == "127.0.0.1"
    assert config.unity_port == 8765
    assert config.camera_calibration is None
    assert config.slam_vocabulary == expected_vocabulary
    assert config.explanation_enabled is True
    assert config.temporal_stereo == "on"


def test_parse_config_accepts_living_room_sim_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAMERA_CALIBRATION", raising=False)

    config = parse_config(
        [
            "--input-source",
            "sim",
            "--scenario",
            "living_room_navigation_v1",
            "--sim-planner-model",
            "gpt-5.4",
            "--sim-planner-timeout-sec",
            "12.5",
            "--sim-replan-interval-sec",
            "2.5",
            "--sim-selfcal-max-sec",
            "9.0",
            "--sim-action-batch-size",
            "4",
            "--sim-headless",
            "--unity-player-path",
            "C:/UnityBuild/obj-recog.exe",
            "--unity-host",
            "127.0.0.2",
            "--unity-port",
            "9001",
            "--camera-calibration",
            "/tmp/camera.yaml",
            "--temporal-stereo",
            "off",
            "--reconstruction-viewer-mode",
            "direct",
        ]
    )

    assert config.input_source == "sim"
    assert config.scenario == "living_room_navigation_v1"
    assert config.sim_planner_model == "gpt-5.4"
    assert config.sim_planner_timeout_sec == pytest.approx(12.5)
    assert config.sim_replan_interval_sec == pytest.approx(2.5)
    assert config.sim_selfcal_max_sec == pytest.approx(9.0)
    assert config.sim_action_batch_size == 4
    assert config.sim_headless is True
    assert config.sim_open3d_view is False
    assert config.reconstruction_viewer_mode == "direct"
    assert config.sim_interface_mode == "rgb_only"
    assert config.sim_render_backend == "software"
    assert config.unity_player_path == "C:/UnityBuild/obj-recog.exe"
    assert config.unity_host == "127.0.0.2"
    assert config.unity_port == 9001
    assert config.camera_calibration == "/tmp/camera.yaml"
    assert config.temporal_stereo == "off"


def test_parse_config_defaults_sim_calibration_from_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAMERA_CALIBRATION", raising=False)

    config = parse_config(
        [
            "--input-source",
            "sim",
        ]
    )

    assert config.input_source == "sim"
    assert config.scenario == "living_room_navigation_v1"
    assert config.camera_calibration is not None
    assert Path(config.camera_calibration).is_file()


def test_prepare_slam_vocabulary_extracts_bundled_archive(tmp_path: Path) -> None:
    target_path = tmp_path / "ORBvoc.txt"
    archive_path = target_path.with_name(f"{target_path.name}.tar.gz")
    source_payload = b"vocabulary-stub"

    source_file = tmp_path / "source.txt"
    source_file.write_bytes(source_payload)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_file, arcname="ORBvoc.txt")
    source_file.unlink()

    resolved_path = prepare_slam_vocabulary(str(target_path))

    assert resolved_path == str(target_path)
    assert target_path.read_bytes() == source_payload


def test_parse_config_allows_disabling_open3d_view() -> None:
    config = parse_config(["--sim-open3d-view", "off"])

    assert config.sim_open3d_view is False


def test_parse_config_accepts_reconstruction_viewer_mode_overrides() -> None:
    worker_config = parse_config(["--reconstruction-viewer-mode", "worker"])
    direct_config = parse_config(["--reconstruction-viewer-mode", "direct"])

    assert worker_config.reconstruction_viewer_mode == "worker"
    assert direct_config.reconstruction_viewer_mode == "direct"


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


def test_parser_rejects_removed_legacy_sim_flags() -> None:
    parser = build_parser()

    for option in (
        "--sim-profile",
        "--sim-external-manifest",
        "--sim-perception-mode",
        "--render-profile",
        "--asset-cache-dir",
        "--asset-quality",
        "--scenario-preview-shots",
        "--validate-all-scenarios",
        "--validation-output-dir",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([option, "x"] if option.endswith("-dir") or option.endswith("manifest") or option.endswith("profile") or option.endswith("quality") else [option])


def test_parser_rejects_invalid_choices() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "metal"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--precision", "fp8"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--detector-backend", "onnx"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--segmentation-mode", "semantic"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--depth-profile", "extreme"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--input-source", "dataset"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--scenario", "warehouse"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--sim-open3d-view", "sideways"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--temporal-stereo", "auto"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--reconstruction-viewer-mode", "foreground"])


def test_resolve_device_prefers_available_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("obj_recog.config.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("obj_recog.config.torch.backends.mps.is_available", lambda: True)

    assert resolve_device("auto") == "mps"


def test_resolve_device_falls_back_to_cpu_when_mps_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("obj_recog.config.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("obj_recog.config.torch.backends.mps.is_available", lambda: False)

    assert resolve_device("auto") == "cpu"


def test_resolve_device_prefers_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("obj_recog.config.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("obj_recog.config.torch.backends.mps.is_available", lambda: False)

    assert resolve_device("auto") == "cuda"


def test_resolve_device_raises_when_cuda_is_requested_without_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("obj_recog.config.torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda")


def test_app_config_direct_defaults_keep_sim_open3d_enabled() -> None:
    config = AppConfig(
        camera_index=0,
        width=64,
        height=48,
        device="cpu",
        conf_threshold=0.35,
        point_stride=2,
        max_points=1000,
    )

    assert config.scenario == "living_room_navigation_v1"
    assert config.sim_camera_fps == pytest.approx(24.0)
    assert config.sim_open3d_view is False
    assert config.segmentation_interval == 2
    assert config.reconstruction_viewer_mode == "auto"
