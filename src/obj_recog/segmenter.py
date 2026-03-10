from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from obj_recog.types import PanopticSegment, SegmentationResult


_SEGMENTATION_PALETTE = (
    (255, 99, 71),
    (64, 196, 255),
    (255, 179, 71),
    (144, 238, 144),
    (255, 105, 180),
    (186, 85, 211),
    (255, 215, 0),
    (72, 209, 204),
    (100, 149, 237),
    (255, 160, 122),
    (152, 251, 152),
    (240, 128, 128),
)


def color_for_label(label_id: int) -> tuple[int, int, int]:
    return _SEGMENTATION_PALETTE[label_id % len(_SEGMENTATION_PALETTE)]


class PanopticSegmenter:
    def __init__(
        self,
        *,
        device: str,
        input_size: int,
        model_id: str = "facebook/mask2former-swin-tiny-coco-panoptic",
        min_area_ratio: float = 0.005,
        processor=None,
        model=None,
    ) -> None:
        self._device = device
        self._input_size = int(input_size)
        self._min_area_ratio = float(min_area_ratio)

        if processor is None or model is None:
            try:
                from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            except ImportError as exc:  # pragma: no cover - depends on local install.
                raise RuntimeError("transformers is required for panoptic segmentation") from exc

            processor = AutoImageProcessor.from_pretrained(model_id)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)

        self._processor = processor
        self._model = model.to(device)
        self._model.eval()
        self._id2label = dict(getattr(getattr(self._model, "config", None), "id2label", {}) or {})

    def _resize_for_inference(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        height, width = frame_bgr.shape[:2]
        long_edge = max(height, width)
        if long_edge <= self._input_size:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        scale = self._input_size / float(long_edge)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized_bgr = cv2.resize(
            frame_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    def segment(self, frame_bgr: np.ndarray) -> SegmentationResult:
        import torch

        frame_bgr = np.asarray(frame_bgr, dtype=np.uint8)
        frame_height, frame_width = frame_bgr.shape[:2]
        input_rgb = self._resize_for_inference(frame_bgr)
        inputs = self._processor(images=input_rgb, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)
        else:
            inputs = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        with torch.no_grad():
            outputs = self._model(**inputs)

        processed = self._processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(frame_height, frame_width)],
        )[0]
        segmentation = processed["segmentation"]
        if hasattr(segmentation, "detach"):
            segmentation = segmentation.detach().cpu().numpy()
        segmentation = np.asarray(segmentation, dtype=np.int32)

        overlay_bgr = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        segment_id_map = np.full((frame_height, frame_width), -1, dtype=np.int32)
        min_area_pixels = max(1, int(round(frame_height * frame_width * self._min_area_ratio)))
        segments: list[PanopticSegment] = []

        for segment_info in processed.get("segments_info", []):
            segment_id = int(segment_info["id"])
            label_id = int(segment_info["label_id"])
            mask = segmentation == segment_id
            area_pixels = int(mask.sum())
            if area_pixels < min_area_pixels:
                continue

            color_rgb = color_for_label(label_id)
            overlay_bgr[mask] = np.asarray(color_rgb[::-1], dtype=np.uint8)
            segment_id_map[mask] = segment_id
            segments.append(
                PanopticSegment(
                    segment_id=segment_id,
                    label_id=label_id,
                    label=str(self._id2label.get(label_id, label_id)),
                    color_rgb=color_rgb,
                    mask=mask,
                    area_pixels=area_pixels,
                )
            )

        return SegmentationResult(
            overlay_bgr=overlay_bgr,
            segment_id_map=segment_id_map,
            segments=segments,
        )


@dataclass(slots=True)
class _WorkerState:
    pending_frame_index: int | None = None
    pending_frame_bgr: np.ndarray | None = None
    latest_result: tuple[int, SegmentationResult] | None = None
    idle: bool = True
    error: Exception | None = None


class SegmentationWorker:
    def __init__(self, *, segmenter: PanopticSegmenter) -> None:
        self._segmenter = segmenter
        self._state = _WorkerState()
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="segmentation-worker", daemon=True)
        self._thread.start()

    def is_idle(self) -> bool:
        with self._lock:
            return self._state.idle

    def submit(self, frame_index: int, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._state.pending_frame_index = int(frame_index)
            self._state.pending_frame_bgr = np.asarray(frame_bgr, dtype=np.uint8).copy()
            self._state.idle = False
        self._event.set()

    def poll(self) -> tuple[int, SegmentationResult] | None:
        with self._lock:
            if self._state.error is not None:
                error = self._state.error
                self._state.error = None
                raise RuntimeError(f"segmentation worker failed: {error}") from error
            result = self._state.latest_result
            self._state.latest_result = None
            return result

    def _run(self) -> None:
        while True:
            self._event.wait()
            self._event.clear()
            if self._stop:
                return

            with self._lock:
                frame_index = self._state.pending_frame_index
                self._state.pending_frame_index = None
                frame_bgr = self._state.pending_frame_bgr
                self._state.pending_frame_bgr = None

            if frame_bgr is None or frame_index is None:
                continue

            try:
                result = self._segmenter.segment(frame_bgr)
            except Exception as exc:  # pragma: no cover - depends on runtime model behavior.
                with self._lock:
                    self._state.error = exc
                    self._state.idle = True
                continue

            with self._lock:
                self._state.latest_result = (int(frame_index), result)
                if self._state.pending_frame_bgr is None:
                    self._state.idle = True
                else:
                    self._event.set()

    def close(self) -> None:
        self._stop = True
        self._event.set()
        self._thread.join(timeout=1.0)
