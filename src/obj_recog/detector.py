from __future__ import annotations

from obj_recog.types import Detection


_PALETTE = (
    (255, 99, 71),
    (64, 196, 255),
    (255, 179, 71),
    (144, 238, 144),
    (255, 105, 180),
    (186, 85, 211),
    (255, 215, 0),
    (72, 209, 204),
)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


class ObjectDetector:
    def __init__(self, conf_threshold: float, device: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on local install.
            raise RuntimeError("ultralytics is required for object detection") from exc

        self._model = YOLO("models/yolo26n.pt")
        self._conf_threshold = conf_threshold
        self._device = device

    def detect(self, frame_bgr) -> list[Detection]:
        result = self._model.predict(
            source=frame_bgr,
            conf=self._conf_threshold,
            device=self._device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        names = result.names
        for box in boxes:
            cls_id = int(box.cls.item())
            xyxy = tuple(int(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    xyxy=xyxy,
                    class_id=cls_id,
                    label=str(names.get(cls_id, cls_id)),
                    confidence=float(box.conf.item()),
                    color=color_for_class(cls_id),
                )
            )
        return detections
