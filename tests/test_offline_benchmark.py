from __future__ import annotations

import json

from obj_recog.offline_benchmark import (
    GroundTruthEpisodeRecord,
    evaluate_runtime_report,
    load_ground_truth_episode_record,
    write_ground_truth_episode_record,
)


def test_offline_benchmark_reads_ground_truth_and_compares_runtime_report(tmp_path) -> None:
    ground_truth_path = tmp_path / "ground_truth.json"
    runtime_report_path = tmp_path / "episode_report.json"

    write_ground_truth_episode_record(
        ground_truth_path,
        GroundTruthEpisodeRecord(
            scenario_id="living_room_navigation_v1",
            success=True,
            goal_distance_m=0.18,
        ),
    )
    runtime_report_path.write_text(
        json.dumps(
            {
                "scenario_id": "living_room_navigation_v1",
                "success": None,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_ground_truth_episode_record(ground_truth_path)
    report = evaluate_runtime_report(
        runtime_report_path=runtime_report_path,
        ground_truth_path=ground_truth_path,
    )

    assert loaded.success is True
    assert report.runtime_success is None
    assert report.success_matches_ground_truth is None
    assert report.ground_truth_success is True
