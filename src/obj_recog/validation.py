from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class EpisodeArtifactWriter:
    output_dir: Path

    def write_json(self, filename: str, payload: dict[str, object]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def run_validation_suite(*_args, **_kwargs):
    raise RuntimeError(
        "Legacy multi-scenario validation has been retired. "
        "Use the living-room episode artifacts under reports/sim/living_room_navigation_v1 instead."
    )
