from __future__ import annotations

from pathlib import Path

from obj_recog.types import Detection
from obj_recog.runtime_accel import device_is_cuda, resolve_precision


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
    def __init__(
        self,
        conf_threshold: float,
        device: str,
        *,
        backend: str = "torch",
        precision: str = "auto",
        input_size: int = 640,
        debug_log=None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on local install.
            raise RuntimeError("ultralytics is required for object detection") from exc

        self._debug_log = debug_log or (lambda _message: None)
        self._conf_threshold = conf_threshold
        self._device = device
        self._backend = str(backend)
        self._precision = resolve_precision(precision, device)
        self._input_size = int(input_size)
        self._model_path = Path("models") / "yolo26n.pt"
        model_source = self._model_path.as_posix()
        if self._backend == "tensorrt":
            model_source = self._prepare_tensorrt_model(YOLO, self._model_path)
        self._model = YOLO(model_source)

    def _prepare_tensorrt_model(self, yolo_cls, model_path: Path) -> str:
        if not device_is_cuda(self._device):
            self._debug_log("TensorRT detector backend requested without CUDA; falling back to torch backend")
            self._backend = "torch"
            return model_path.as_posix()

        engine_path = model_path.with_suffix(f".{self._input_size}.{self._precision}.engine")
        if engine_path.is_file():
            return str(engine_path)

        exporter = yolo_cls(str(model_path))
        export = getattr(exporter, "export", None)
        if not callable(export):
            self._debug_log("Ultralytics build does not expose TensorRT export; falling back to torch backend")
            self._backend = "torch"
            return model_path.as_posix()

        try:
            exported_path = export(
                format="engine",
                imgsz=self._input_size,
                half=(self._precision == "fp16"),
                device=0,
            )
        except Exception as exc:
            self._debug_log(f"TensorRT export failed ({exc}); falling back to torch backend")
            self._backend = "torch"
            return model_path.as_posix()

        if exported_path:
            return str(exported_path)
        if engine_path.is_file():
            return str(engine_path)
        self._backend = "torch"
        return model_path.as_posix()

    def detect(self, frame_bgr) -> list[Detection]:
        result = self._model.predict(
            source=frame_bgr,
            conf=self._conf_threshold,
            device=(0 if device_is_cuda(self._device) and self._backend == "tensorrt" else self._device),
            imgsz=self._input_size,
            half=bool(device_is_cuda(self._device) and self._precision == "fp16" and self._backend == "torch"),
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
