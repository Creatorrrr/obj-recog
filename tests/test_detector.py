from __future__ import annotations

import sys
import types

from obj_recog.detector import ObjectDetector


def test_object_detector_uses_yolo26n_by_default(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeYOLO:
        def __init__(self, model_name: str) -> None:
            captured["model_name"] = model_name

    fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    ObjectDetector(conf_threshold=0.35, device="cpu")

    assert captured["model_name"] == "yolo26n.pt"
