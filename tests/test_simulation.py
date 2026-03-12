from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from obj_recog.blender_worker import BlenderFrameResponse
from obj_recog.config import AppConfig
from obj_recog.frame_source import FramePacket
from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.sim_assets import build_scenario_asset_manifest
from obj_recog.simulation import (
    BlenderRealtimeFrameSource,
    CameraRigSpec,
    SCENARIO_SPECS,
    ExternalManifestFrameSource,
    HeuristicGoalSelector,
    NavigationGoal,
    OpenAINavigationGoalSelector,
    SimulationFrameSource,
    SimulationRuntime,
    _VisibleObject,
)
from obj_recog.types import Detection, FrameArtifacts


_SCENARIO_IDS = (
    "studio_open_v1",
    "office_clutter_v1",
    "lab_corridor_v1",
    "showroom_occlusion_v1",
    "office_crossflow_v1",
    "warehouse_moving_target_v1",
)


def _config(**overrides: object) -> AppConfig:
    values = dict(
        camera_index=0,
        width=96,
        height=72,
        device="cpu",
        conf_threshold=0.35,
        point_stride=2,
        max_points=4096,
        input_source="sim",
        scenario="studio_open_v1",
        sim_seed=7,
        sim_max_steps=96,
        sim_camera_fps=4.0,
        eval_budget_sec=12.0,
        sim_goal_selector="heuristic",
        sim_goal_model="gpt-5-mini",
        sim_goal_timeout_sec=4.0,
        sim_external_manifest=None,
    )
    values.update(overrides)
    return AppConfig(**values)


class _FakePhotorealRealtimeSource:
    backend_name = "blender-realtime"

    def __init__(self) -> None:
        self._emitted = False
        self.closed = False

    def next_frame(self, *, timeout_sec: float | None = 1.0) -> FramePacket | None:
        _ = timeout_sec
        if self._emitted:
            return None
        self._emitted = True
        return FramePacket(
            frame_bgr=np.full((4, 6, 3), 121, dtype=np.uint8),
            timestamp_sec=0.25,
            depth_map=np.full((4, 6), 2.25, dtype=np.float32),
            pose_world_gt=np.eye(4, dtype=np.float32),
            intrinsics_gt=CameraIntrinsics(fx=10.0, fy=10.0, cx=3.0, cy=2.0),
            detections=[
                Detection(
                    xyxy=(1, 1, 5, 3),
                    class_id=1,
                    label="backpack",
                    confidence=0.95,
                    color=(0, 255, 0),
                )
            ],
        )

    def close(self) -> None:
        self.closed = True


class _FakeBlenderWorker:
    def __init__(self, response: BlenderFrameResponse) -> None:
        self.response = response
        self.requests = []
        self.closed = False

    def start(self) -> None:
        return None

    def request_frame(self, request, *, timeout_sec: float | None = None) -> BlenderFrameResponse:
        self.requests.append((request, timeout_sec))
        return self.response

    def close(self) -> None:
        self.closed = True


class _FakeBlenderWorker:
    def __init__(self, response) -> None:
        self._response = response
        self.started = False
        self.closed = False
        self.requests = []

    def start(self) -> None:
        self.started = True

    def request_frame(self, request, *, timeout_sec: float | None = None):
        self.requests.append((request, timeout_sec))
        return self._response

    def close(self) -> None:
        self.closed = True


def test_scenario_registry_contains_expected_specs() -> None:
    assert tuple(SCENARIO_SPECS) == _SCENARIO_IDS
    assert SCENARIO_SPECS["studio_open_v1"].difficulty_level == 1
    assert len(SCENARIO_SPECS["studio_open_v1"].dynamic_actors) == 0
    assert len(SCENARIO_SPECS["office_crossflow_v1"].dynamic_actors) == 2
    assert len(SCENARIO_SPECS["warehouse_moving_target_v1"].dynamic_actors) == 4


def test_blender_realtime_frame_source_emits_valid_packet_for_studio_open_v1(tmp_path: Path) -> None:
    rgb = np.full((5, 7, 3), 129, dtype=np.uint8)
    depth = np.full((5, 7), 2.75, dtype=np.float32)
    rgb_path = tmp_path / "rgb.npy"
    depth_path = tmp_path / "depth.npy"
    np.save(rgb_path, rgb)
    np.save(depth_path, depth)
    worker = _FakeBlenderWorker(
        response={
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "semantic_mask_path": str(tmp_path / "semantic.png"),
            "instance_mask_path": str(tmp_path / "instance.png"),
            "pose_world_gt": np.eye(4, dtype=np.float32).tolist(),
            "intrinsics_gt": {"fx": 12.0, "fy": 13.0, "cx": 3.5, "cy": 2.5},
            "render_time_ms": 16.4,
            "worker_state": "ready",
        }
    )
    config = _config(render_profile="photoreal", blender_exec="/Applications/Blender.app/Contents/MacOS/Blender")
    source = BlenderRealtimeFrameSource(
        config=config,
        scenario=SCENARIO_SPECS["studio_open_v1"],
        camera_rig=CameraRigSpec.from_config(config),
        asset_manifest=build_scenario_asset_manifest(
            SCENARIO_SPECS["studio_open_v1"],
            seed=7,
            cache_dir=tmp_path,
            quality="low",
        ),
        report_path=tmp_path / "photoreal-report.json",
        worker_client=worker,
    )

    packet = source.next_frame(timeout_sec=0.75)

    assert packet is not None
    assert packet.frame_bgr.shape == (5, 7, 3)
    assert packet.depth_map is not None
    assert packet.depth_map.shape == (5, 7)
    np.testing.assert_allclose(packet.pose_world_gt, np.eye(4, dtype=np.float32))
    assert packet.intrinsics_gt == CameraIntrinsics(fx=12.0, fy=13.0, cx=3.5, cy=2.5)
    assert packet.scenario_state.render_backend == "blender-realtime"
    assert packet.scenario_state.render_profile == "photoreal"
    assert packet.scenario_state.scene_id == "studio_open_v1"
    assert packet.scenario_state.semantic_mask_path == str(tmp_path / "semantic.png")
    assert packet.scenario_state.instance_mask_path == str(tmp_path / "instance.png")
    assert packet.scenario_state.worker_state == "ready"
    assert packet.scenario_state.render_time_ms == pytest.approx(16.4)
    assert worker.started is True
    assert len(worker.requests) == 1
    assert worker.requests[0][1] == pytest.approx(0.75)
    source.close()
    assert worker.closed is True


def test_simulation_runtime_routes_photoreal_to_blender_realtime_frame_source(tmp_path: Path) -> None:
    rgb = np.full((4, 6, 3), 117, dtype=np.uint8)
    depth = np.full((4, 6), 2.1, dtype=np.float32)
    rgb_path = tmp_path / "runtime-rgb.npy"
    depth_path = tmp_path / "runtime-depth.npy"
    np.save(rgb_path, rgb)
    np.save(depth_path, depth)
    worker = _FakeBlenderWorker(
        response={
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "semantic_mask_path": str(tmp_path / "runtime-semantic.png"),
            "instance_mask_path": str(tmp_path / "runtime-instance.png"),
            "pose_world_gt": np.eye(4, dtype=np.float32).tolist(),
            "intrinsics_gt": {"fx": 10.0, "fy": 10.0, "cx": 3.0, "cy": 2.0},
            "render_time_ms": 12.0,
            "worker_state": "ready",
        }
    )
    runtime = SimulationRuntime(
        config=_config(
            render_profile="photoreal",
            blender_exec="/Applications/Blender.app/Contents/MacOS/Blender",
        ),
        report_path=tmp_path / "runtime-report.json",
        blender_worker_client_factory=lambda **_kwargs: worker,
    )

    source = runtime.create_frame_source()
    packet = source.next_frame()

    assert packet is not None
    assert packet.scenario_state.render_backend == "blender-realtime"
    assert packet.scenario_state.scene_id == "studio_open_v1"
    assert packet.scenario_state.semantic_mask_path == str(tmp_path / "runtime-semantic.png")
    assert packet.scenario_state.instance_mask_path == str(tmp_path / "runtime-instance.png")


@pytest.mark.parametrize("scenario_name", _SCENARIO_IDS)
def test_simulation_frame_source_is_deterministic_for_same_seed(
    tmp_path: Path,
    scenario_name: str,
) -> None:
    config = _config(scenario=scenario_name)

    source_a = SimulationFrameSource(
        config=config,
        report_path=tmp_path / "a.json",
        goal_selector=HeuristicGoalSelector(),
    )
    source_b = SimulationFrameSource(
        config=config,
        report_path=tmp_path / "b.json",
        goal_selector=HeuristicGoalSelector(),
    )

    packets_a = [source_a.next_frame() for _ in range(3)]
    packets_b = [source_b.next_frame() for _ in range(3)]

    assert [packet.timestamp_sec for packet in packets_a] == [packet.timestamp_sec for packet in packets_b]
    assert all(np.allclose(a.pose_world_gt, b.pose_world_gt) for a, b in zip(packets_a, packets_b))
    assert all(np.array_equal(a.frame_bgr, b.frame_bgr) for a, b in zip(packets_a, packets_b))


def test_simulation_frame_source_emits_scenario_metadata_in_state_and_report(tmp_path: Path) -> None:
    report_path = tmp_path / "metadata-report.json"
    source = SimulationFrameSource(
        config=_config(scenario="office_crossflow_v1", sim_seed=31),
        report_path=report_path,
        goal_selector=HeuristicGoalSelector(),
    )

    packet = source.next_frame()
    assert packet is not None
    assert packet.scenario_state.scene_id == "office_crossflow_v1"
    assert packet.scenario_state.difficulty_level == 5
    assert packet.scenario_state.target_motion_state == "static"

    source.close()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["scenario_family"] == "office"
    assert report["difficulty_level"] == 5
    assert report["dynamic_actor_count"] == 2
    assert report["target_class"] == "laptop"


def test_simulation_frame_source_emits_render_profile_and_asset_manifest_metadata(tmp_path: Path) -> None:
    report_path = tmp_path / "render-profile-report.json"
    source = SimulationFrameSource(
        config=_config(scenario="studio_open_v1", sim_seed=29),
        report_path=report_path,
        goal_selector=HeuristicGoalSelector(),
    )

    packet = source.next_frame()
    assert packet is not None
    assert packet.scenario_state.render_profile == "fast"
    assert packet.scenario_state.semantic_target_class == "backpack"
    assert packet.scenario_state.asset_manifest_id

    source.close()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["render_profile"] == "fast"
    assert report["asset_manifest_id"]
    assert report["target_class"] == "backpack"


def test_build_scenario_asset_manifest_matches_runtime_target_class() -> None:
    manifest = build_scenario_asset_manifest(
        SCENARIO_SPECS["warehouse_moving_target_v1"],
        seed=7,
        cache_dir=Path("/tmp/obj-recog-assets-test"),
        quality="low",
    )

    assert manifest.semantic_target_class == "suitcase"
    assert any(item.target_role and item.semantic_class == "suitcase" for item in manifest.placements)


def test_fast_sim_renderer_produces_textured_target_patch(tmp_path: Path) -> None:
    source = SimulationFrameSource(
        config=_config(scenario="studio_open_v1", sim_seed=7, sim_max_steps=32),
        report_path=tmp_path / "rendered.json",
        goal_selector=HeuristicGoalSelector(),
    )

    target_crop = None
    while True:
        packet = source.next_frame()
        assert packet is not None
        target_detection = next((item for item in (packet.detections or []) if item.label == "backpack"), None)
        if target_detection is None:
            continue
        x1, y1, x2, y2 = target_detection.xyxy
        target_crop = packet.frame_bgr[y1:y2, x1:x2]
        break

    assert target_crop is not None
    unique_colors = np.unique(target_crop.reshape(-1, 3), axis=0)
    assert unique_colors.shape[0] >= 12


def test_simulation_frame_source_runs_closed_loop_and_writes_report(tmp_path: Path) -> None:
    report_path = tmp_path / "mission-report.json"
    source = SimulationFrameSource(
        config=_config(sim_seed=11),
        report_path=report_path,
        goal_selector=HeuristicGoalSelector(),
    )

    phases: list[str] = []
    detection_counts: list[int] = []
    while True:
        packet = source.next_frame()
        if packet is None:
            break
        phases.append(packet.scenario_state.phase)
        detection_counts.append(len(packet.detections or []))

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert phases[0] == "BOOTSTRAP_SELF_CAL"
    assert "EXPLORE" in phases
    assert "NAVIGATE_TO_VIEW" in phases
    assert "VERIFY_VIEW" in phases
    assert phases[-1] == "REPORT"
    assert any(count > 0 for count in detection_counts)
    assert report["selfcal_converged"] is True
    assert report["mission_success"] is True
    assert report["failure_reason"] is None
    assert report["time_to_first_valid_view"] <= 12.0
    assert report["tracking_uptime"] == 1.0
    assert report["fallback_count"] == 0
    assert report["llm_goal_accept_rate"] == 0.0


def test_showroom_occlusion_scenario_hides_target_during_occlusion_interval(tmp_path: Path) -> None:
    source = SimulationFrameSource(
        config=_config(scenario="showroom_occlusion_v1", sim_seed=37, sim_max_steps=64),
        report_path=tmp_path / "occlusion-report.json",
        goal_selector=HeuristicGoalSelector(),
    )

    target_seen = False
    target_hidden_after_seen = False
    while True:
        packet = source.next_frame()
        if packet is None:
            break
        labels = tuple(packet.scenario_state.visible_labels)
        if packet.timestamp_sec >= 8.0 and "backpack" in labels:
            target_seen = True
        elif packet.timestamp_sec >= 8.0 and target_seen and "backpack" not in labels:
            target_hidden_after_seen = True
            break

    assert target_hidden_after_seen is True


def test_warehouse_moving_target_scenario_updates_target_pose_over_time(tmp_path: Path) -> None:
    source = SimulationFrameSource(
        config=_config(scenario="warehouse_moving_target_v1", sim_seed=41, sim_max_steps=32),
        report_path=tmp_path / "moving-target-report.json",
        goal_selector=HeuristicGoalSelector(),
    )

    centers: list[tuple[float, float, float]] = []
    for _ in range(6):
        packet = source.next_frame()
        assert packet is not None
        target = next(item for item in source._active_scene_objects if item.target_role or item.label == "suitcase")
        centers.append(target.center_world)

    assert len({(round(center[0], 3), round(center[2], 3)) for center in centers}) > 1
    assert packet.scenario_state.target_motion_state == "moving"


class _InvalidLlmGoalSelector:
    def select_goal(self, *, visible_objects, target_label: str):
        _ = (visible_objects, target_label)
        return NavigationGoal(
            target_label="wrong-target",
            desired_bearing=float("nan"),
            desired_distance_band=(4.0, 2.0),
            reason="bad output",
            confidence=0.1,
            source="llm",
        )


def test_simulation_frame_source_falls_back_when_primary_goal_is_invalid(tmp_path: Path) -> None:
    report_path = tmp_path / "fallback-report.json"
    source = SimulationFrameSource(
        config=_config(sim_seed=13),
        report_path=report_path,
        goal_selector=_InvalidLlmGoalSelector(),
    )

    while source.next_frame() is not None:
        pass

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["mission_success"] is True
    assert report["fallback_count"] >= 1
    assert report["llm_goal_accept_rate"] == 0.0


def test_simulation_frame_source_marks_selfcal_failure_when_refinement_is_bad(tmp_path: Path) -> None:
    report_path = tmp_path / "selfcal-failure.json"

    def _bad_refine_intrinsics(**kwargs):
        return kwargs["initial_fx"] * 3.0, kwargs["initial_fy"] * 3.0, {}, {}

    source = SimulationFrameSource(
        config=_config(sim_seed=19),
        report_path=report_path,
        goal_selector=HeuristicGoalSelector(),
        refine_intrinsics=_bad_refine_intrinsics,
    )

    while source.next_frame() is not None:
        pass

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["selfcal_converged"] is False


def test_simulation_frame_source_emits_abort_report_on_close(tmp_path: Path) -> None:
    report_path = tmp_path / "abort-report.json"
    source = SimulationFrameSource(
        config=_config(sim_seed=23),
        report_path=report_path,
        goal_selector=HeuristicGoalSelector(),
    )

    assert source.next_frame() is not None

    source.close()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["mission_success"] is False
    assert report["failure_reason"] == "aborted"


class _FakeResponsesClient:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, object]] = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": self.output_text})()


def _runtime_artifacts(packet, *, pose_world: np.ndarray, tracking_state: str = "TRACKING") -> FrameArtifacts:
    frame_bgr = np.asarray(packet.frame_bgr, dtype=np.uint8)
    depth_map = np.full(frame_bgr.shape[:2], 2.5, dtype=np.float32)
    return FrameArtifacts(
        frame_bgr=frame_bgr,
        intrinsics=packet.intrinsics_gt or CameraIntrinsics(fx=40.0, fy=40.0, cx=frame_bgr.shape[1] / 2.0, cy=frame_bgr.shape[0] / 2.0),
        detections=[
            Detection(
                xyxy=(10, 10, max(12, frame_bgr.shape[1] - 10), max(12, frame_bgr.shape[0] - 10)),
                class_id=1,
                label="target",
                confidence=0.95,
                color=(0, 255, 0),
            )
        ],
        depth_map=depth_map,
        points_xyz=np.empty((0, 3), dtype=np.float32),
        points_rgb=np.empty((0, 3), dtype=np.float32),
        dense_map_points_xyz=np.empty((0, 3), dtype=np.float32),
        dense_map_points_rgb=np.empty((0, 3), dtype=np.float32),
        mesh_vertices_xyz=np.empty((0, 3), dtype=np.float32),
        mesh_triangles=np.empty((0, 3), dtype=np.int32),
        mesh_vertex_colors=np.empty((0, 3), dtype=np.float32),
        camera_pose_world=np.asarray(pose_world, dtype=np.float32),
        tracking_ok=tracking_state in {"TRACKING", "RELOCALIZED"},
        is_keyframe=False,
        trajectory_xyz=np.empty((0, 3), dtype=np.float32),
        segment_id=0,
        slam_tracking_state=tracking_state,
        keyframe_id=None,
        sparse_map_points_xyz=np.empty((0, 3), dtype=np.float32),
        loop_closure_applied=False,
        segmentation_overlay_bgr=frame_bgr.copy(),
        segments=[],
    )


class _FakeRenderer:
    def __init__(self) -> None:
        self.calls = 0

    def render(self, *, pose_world, intrinsics, scene_objects, previous_frame_bgr):
        _ = (pose_world, intrinsics, scene_objects, previous_frame_bgr)
        self.calls += 1
        frame = np.full((12, 16, 3), 77, dtype=np.uint8)
        depth_map = np.full((12, 16), 3.0, dtype=np.float32)
        return frame, depth_map, [
            _VisibleObject(
                label="target",
                bbox=(2, 2, 12, 10),
                depth_m=3.0,
                bearing_rad=0.0,
                area_pixels=80,
                color_bgr=(0, 255, 0),
            )
        ]


def test_openai_navigation_goal_selector_parses_structured_goal() -> None:
    client = _FakeResponsesClient(
        json.dumps(
            {
                "target_label": "target",
                "desired_bearing": -0.2,
                "desired_distance_band": [2.8, 3.4],
                "reason": "Shift slightly left and approach.",
                "confidence": 0.81,
            }
        )
    )
    selector = OpenAINavigationGoalSelector(
        model="gpt-5-mini",
        timeout_sec=4.0,
        client=client,
    )

    goal = selector.select_goal(
        visible_objects=[
            _VisibleObject(
                label="target",
                bbox=(10, 10, 30, 28),
                depth_m=4.2,
                bearing_rad=0.15,
                area_pixels=360,
                color_bgr=(0, 255, 0),
            )
        ],
        target_label="target",
    )

    assert goal is not None
    assert goal.source == "llm"
    assert goal.target_label == "target"
    assert goal.desired_distance_band == pytest.approx((2.8, 3.4))
    assert client.calls[0]["model"] == "gpt-5-mini"
    assert client.calls[0]["text"]["format"]["type"] == "json_schema"
    assert client.calls[0]["text"]["format"]["strict"] is True


def test_openai_navigation_goal_selector_returns_none_without_client_or_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    selector = OpenAINavigationGoalSelector(
        model="gpt-5-mini",
        timeout_sec=4.0,
    )

    goal = selector.select_goal(
        visible_objects=[],
        target_label="target",
    )

    assert goal is None


def test_external_manifest_frame_source_reads_frame_packets(tmp_path: Path) -> None:
    rgb_path = tmp_path / "frame.npy"
    depth_path = tmp_path / "depth.npy"
    np.save(rgb_path, np.full((4, 6, 3), 123, dtype=np.uint8))
    np.save(depth_path, np.full((4, 6), 2.25, dtype=np.float32))
    manifest_path = tmp_path / "blender-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "rgb_path": rgb_path.name,
                        "depth_path": depth_path.name,
                        "timestamp_sec": 0.25,
                        "pose_world_gt": np.eye(4, dtype=np.float32).tolist(),
                        "intrinsics_gt": {"fx": 10.0, "fy": 10.0, "cx": 3.0, "cy": 2.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    source = ExternalManifestFrameSource(manifest_path=manifest_path)
    packet = source.next_frame()

    assert packet is not None
    assert packet.timestamp_sec == pytest.approx(0.25)
    assert packet.frame_bgr.shape == (4, 6, 3)
    assert packet.depth_map is not None and packet.depth_map.shape == (4, 6)


def test_simulation_runtime_external_profile_still_uses_simulation_runtime(tmp_path: Path) -> None:
    rgb_path = tmp_path / "frame.npy"
    depth_path = tmp_path / "depth.npy"
    np.save(rgb_path, np.full((4, 6, 3), 123, dtype=np.uint8))
    np.save(depth_path, np.full((4, 6), 2.25, dtype=np.float32))
    manifest_path = tmp_path / "blender-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "rgb_path": rgb_path.name,
                        "depth_path": depth_path.name,
                        "timestamp_sec": 0.25,
                        "pose_world_gt": np.eye(4, dtype=np.float32).tolist(),
                        "intrinsics_gt": {"fx": 10.0, "fy": 10.0, "cx": 3.0, "cy": 2.0},
                        "detections": [
                            {
                                "xyxy": [1, 1, 5, 3],
                                "class_id": 1,
                                "label": "target",
                                "confidence": 0.95,
                                "color": [0, 255, 0],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runtime = SimulationRuntime(
        config=_config(sim_profile="external", sim_external_manifest=str(manifest_path)),
        report_path=tmp_path / "report.json",
    )

    source = runtime.create_frame_source()

    assert isinstance(source, SimulationFrameSource)
    assert source.next_frame() is not None
    assert source.next_frame() is None
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["render_backend"] == "external-manifest"
    assert report["failure_reason"] == "external_stream_end"


def test_simulation_runtime_photoreal_profile_uses_external_manifest_bundle(tmp_path: Path) -> None:
    rgb_path = tmp_path / "frame.npy"
    depth_path = tmp_path / "depth.npy"
    np.save(rgb_path, np.full((4, 6, 3), 123, dtype=np.uint8))
    np.save(depth_path, np.full((4, 6), 2.25, dtype=np.float32))
    manifest_path = tmp_path / "photoreal-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "rgb_path": rgb_path.name,
                        "depth_path": depth_path.name,
                        "timestamp_sec": 0.25,
                        "pose_world_gt": np.eye(4, dtype=np.float32).tolist(),
                        "intrinsics_gt": {"fx": 10.0, "fy": 10.0, "cx": 3.0, "cy": 2.0},
                        "detections": [
                            {
                                "xyxy": [1, 1, 5, 3],
                                "class_id": 1,
                                "label": "backpack",
                                "confidence": 0.95,
                                "color": [0, 255, 0],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runtime = SimulationRuntime(
        config=_config(
            render_profile="photoreal",
            sim_external_manifest=str(manifest_path),
        ),
        report_path=tmp_path / "report.json",
    )

    source = runtime.create_frame_source()
    packet = source.next_frame()

    assert packet is not None
    assert packet.scenario_state.render_profile == "photoreal"
    assert packet.scenario_state.semantic_target_class == "backpack"


def test_simulation_runtime_photoreal_profile_requires_blender_exec_or_render_bundle_manifest(
    tmp_path: Path,
) -> None:
    runtime = SimulationRuntime(
        config=_config(render_profile="photoreal", sim_external_manifest=None),
        report_path=tmp_path / "report.json",
    )

    with pytest.raises(
        RuntimeError,
        match="render_profile=photoreal requires --blender-exec or --sim-external-manifest",
    ):
        runtime.create_frame_source()


def test_simulation_runtime_photoreal_profile_uses_realtime_factory_without_manifest(tmp_path: Path) -> None:
    rgb_path = tmp_path / "frame.npy"
    depth_path = tmp_path / "depth.npy"
    np.save(rgb_path, np.full((4, 6, 3), 121, dtype=np.uint8))
    np.save(depth_path, np.full((4, 6), 2.0, dtype=np.float32))
    factory_calls: list[tuple[str, str | None]] = []

    def _factory(*, config: AppConfig, scenario, camera_rig, asset_manifest, report_path):
        _ = report_path
        factory_calls.append((config.render_profile, config.blender_exec))
        return BlenderRealtimeFrameSource(
            config=config,
            scenario=scenario,
            camera_rig=camera_rig,
            asset_manifest=asset_manifest,
            worker_client=_FakeBlenderWorker(
                BlenderFrameResponse(
                    rgb_path=str(rgb_path),
                    depth_path=str(depth_path),
                    semantic_mask_path=str(tmp_path / "semantic.png"),
                    instance_mask_path=str(tmp_path / "instance.png"),
                    pose_world_gt=np.eye(4, dtype=np.float32),
                    intrinsics_gt={"fx": 10.0, "fy": 10.0, "cx": 3.0, "cy": 2.0},
                    render_time_ms=12.5,
                    worker_state="ready",
                )
            ),
        )

    runtime = SimulationRuntime(
        config=_config(
            render_profile="photoreal",
            blender_exec="/Applications/Blender.app/Contents/MacOS/Blender",
            sim_external_manifest=None,
        ),
        report_path=tmp_path / "report.json",
        photoreal_frame_source_factory=_factory,
    )

    source = runtime.create_frame_source()
    packet = source.next_frame()

    assert packet is not None
    assert factory_calls == [("photoreal", "/Applications/Blender.app/Contents/MacOS/Blender")]
    assert packet.scenario_state.render_profile == "photoreal"
    assert packet.scenario_state.render_backend == "blender-realtime"


def test_blender_realtime_frame_source_emits_frame_packet_for_studio_scenario(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.npy"
    depth_path = tmp_path / "depth.npy"
    semantic_path = tmp_path / "semantic.png"
    instance_path = tmp_path / "instance.png"
    np.save(rgb_path, np.full((5, 7, 3), 88, dtype=np.uint8))
    np.save(depth_path, np.full((5, 7), 1.75, dtype=np.float32))
    semantic_path.write_bytes(b"semantic")
    instance_path.write_bytes(b"instance")
    config = _config(
        render_profile="photoreal",
        blender_exec="/Applications/Blender.app/Contents/MacOS/Blender",
    )
    scenario = SCENARIO_SPECS["studio_open_v1"]
    asset_manifest = build_scenario_asset_manifest(
        scenario,
        seed=int(config.sim_seed),
        cache_dir=tmp_path / "assets",
        quality="low",
    )
    fake_worker = _FakeBlenderWorker(
        BlenderFrameResponse(
            rgb_path=str(rgb_path),
            depth_path=str(depth_path),
            semantic_mask_path=str(semantic_path),
            instance_mask_path=str(instance_path),
            pose_world_gt=np.eye(4, dtype=np.float32),
            intrinsics_gt={"fx": 12.0, "fy": 12.0, "cx": 3.5, "cy": 2.5},
            render_time_ms=15.0,
            worker_state="ready",
        )
    )

    source = BlenderRealtimeFrameSource(
        config=config,
        scenario=scenario,
        camera_rig=CameraRigSpec.from_config(config),
        asset_manifest=asset_manifest,
        worker_client=fake_worker,
    )

    packet = source.next_frame(timeout_sec=0.25)

    assert packet is not None
    assert packet.timestamp_sec == pytest.approx(0.0)
    assert packet.frame_bgr.shape == (5, 7, 3)
    assert packet.depth_map is not None
    assert packet.pose_world_gt is not None
    assert packet.intrinsics_gt is not None
    assert packet.intrinsics_gt.fx == pytest.approx(12.0)
    assert packet.scenario_state.scene_id == "studio_open_v1"
    assert packet.scenario_state.render_backend == "blender-realtime"
    assert packet.scenario_state.render_profile == "photoreal"
    assert packet.scenario_state.semantic_mask_path == str(semantic_path)
    assert packet.scenario_state.instance_mask_path == str(instance_path)
    assert packet.scenario_state.worker_state == "ready"
    assert packet.scenario_state.render_time_ms == pytest.approx(15.0)
    assert packet.calibration_source == "blender-ground-truth"
    assert len(fake_worker.requests) == 1
    request, timeout_sec = fake_worker.requests[0]
    assert request.scenario_id == "studio_open_v1"
    assert timeout_sec == pytest.approx(0.25)
    assert fake_worker.closed is False

    source.close()
    assert fake_worker.closed is True


def test_simulation_runtime_llm_selector_falls_back_when_api_key_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runtime = SimulationRuntime(
        config=_config(sim_goal_selector="llm"),
        report_path=tmp_path / "llm-fallback-report.json",
    )

    source = runtime.create_frame_source()
    while source.next_frame() is not None:
        pass

    report = json.loads((tmp_path / "llm-fallback-report.json").read_text(encoding="utf-8"))
    assert report["mission_success"] is True
    assert report["fallback_count"] >= 1
    assert report["llm_goal_accept_rate"] == 0.0


def test_simulation_frame_source_records_runtime_pose_error_into_report(tmp_path: Path) -> None:
    report_path = tmp_path / "pose-error-report.json"
    source = SimulationFrameSource(
        config=_config(sim_seed=29),
        report_path=report_path,
        goal_selector=HeuristicGoalSelector(),
    )

    packet = source.next_frame()
    assert packet is not None
    estimated_pose = np.asarray(packet.pose_world_gt, dtype=np.float32).copy()
    estimated_pose[0, 3] += 0.5
    source.record_runtime_observation(
        frame_packet=packet,
        artifacts=_runtime_artifacts(packet, pose_world=estimated_pose),
    )

    source.close()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["pose_error_vs_gt"] == pytest.approx(0.5)


def test_simulation_frame_source_uses_injected_renderer_backend(tmp_path: Path) -> None:
    renderer = _FakeRenderer()
    source = SimulationFrameSource(
        config=_config(width=16, height=12),
        report_path=tmp_path / "renderer-report.json",
        goal_selector=HeuristicGoalSelector(),
        renderer=renderer,
    )

    packet = source.next_frame()

    assert packet is not None
    assert renderer.calls == 1
    assert packet.frame_bgr.shape == (12, 16, 3)
    assert np.all(packet.depth_map == pytest.approx(3.0))
