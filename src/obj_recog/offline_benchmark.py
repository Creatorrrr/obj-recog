from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GroundTruthEpisodeRecord:
    scenario_id: str
    success: bool
    goal_distance_m: float | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class OfflineBenchmarkReport:
    scenario_id: str
    runtime_report_path: str
    ground_truth_path: str
    success_matches_ground_truth: bool | None
    runtime_success: bool | None
    ground_truth_success: bool
    notes: str = ""


def write_ground_truth_episode_record(path: str | Path, record: GroundTruthEpisodeRecord) -> None:
    Path(path).write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")


def load_ground_truth_episode_record(path: str | Path) -> GroundTruthEpisodeRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GroundTruthEpisodeRecord(
        scenario_id=str(payload["scenario_id"]),
        success=bool(payload["success"]),
        goal_distance_m=(
            None if payload.get("goal_distance_m") is None else float(payload.get("goal_distance_m"))
        ),
        notes=str(payload.get("notes", "")),
    )


def evaluate_runtime_report(
    *,
    runtime_report_path: str | Path,
    ground_truth_path: str | Path,
) -> OfflineBenchmarkReport:
    runtime_payload = json.loads(Path(runtime_report_path).read_text(encoding="utf-8"))
    ground_truth = load_ground_truth_episode_record(ground_truth_path)
    runtime_success_raw = runtime_payload.get("success")
    runtime_success = None if runtime_success_raw is None else bool(runtime_success_raw)
    return OfflineBenchmarkReport(
        scenario_id=str(runtime_payload.get("scenario_id", ground_truth.scenario_id)),
        runtime_report_path=str(Path(runtime_report_path)),
        ground_truth_path=str(Path(ground_truth_path)),
        success_matches_ground_truth=(
            None if runtime_success is None else bool(runtime_success == ground_truth.success)
        ),
        runtime_success=runtime_success,
        ground_truth_success=ground_truth.success,
        notes=str(runtime_payload.get("notes", "")),
    )
