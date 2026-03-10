from __future__ import annotations

import pytest

from obj_recog.config import build_parser, parse_config, resolve_device


def test_parse_config_uses_expected_defaults() -> None:
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
    assert config.camera_calibration is None
    assert config.slam_vocabulary is None
    assert config.slam_width == 640
    assert config.slam_height == 360
    assert config.recalibrate is False
    assert config.calibration_cache_dir is None


def test_parse_config_accepts_custom_values() -> None:
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
            "--slam-vocabulary",
            "/tmp/ORBvoc.txt",
            "--slam-width",
            "512",
            "--slam-height",
            "288",
            "--recalibrate",
            "--calibration-cache-dir",
            "/tmp/calibration-cache",
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
    assert config.slam_vocabulary == "/tmp/ORBvoc.txt"
    assert config.slam_width == 512
    assert config.slam_height == 288
    assert config.recalibrate is True
    assert config.calibration_cache_dir == "/tmp/calibration-cache"
    assert config.list_cameras is False
    assert config.orb_features == 1200
    assert config.max_map_points == 200_000
    assert config.max_mesh_triangles == 10_000


def test_parser_rejects_invalid_choices() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "cuda"])


def test_resolve_device_prefers_available_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("obj_recog.config.torch.backends.mps.is_available", lambda: True)

    assert resolve_device("auto") == "mps"


def test_resolve_device_falls_back_to_cpu_when_mps_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("obj_recog.config.torch.backends.mps.is_available", lambda: False)

    assert resolve_device("auto") == "cpu"
