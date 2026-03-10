from __future__ import annotations

import numpy as np
import torch

from obj_recog.segmenter import PanopticSegmenter, color_for_label


class _FakeInputs(dict):
    def to(self, device: str):
        return self


class _FakeProcessor:
    def __call__(self, images, return_tensors: str):
        assert return_tensors == "pt"
        return _FakeInputs(pixel_values=torch.zeros((1, 3, 4, 4), dtype=torch.float32))

    def post_process_panoptic_segmentation(self, outputs, target_sizes):
        assert target_sizes == [(4, 4)]
        return [
            {
                "segmentation": torch.tensor(
                    [
                        [1, 1, 2, 2],
                        [1, 1, 2, 2],
                        [1, 1, 2, 2],
                        [1, 1, 2, 3],
                    ],
                    dtype=torch.int64,
                ),
                "segments_info": [
                    {"id": 1, "label_id": 4},
                    {"id": 2, "label_id": 7},
                    {"id": 3, "label_id": 9},
                ],
            }
        ]


class _FakeConfig:
    id2label = {4: "chair", 7: "wall", 9: "plant"}


class _FakeModel:
    config = _FakeConfig()

    def to(self, device: str):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        return object()


class _StrideCheckingProcessor(_FakeProcessor):
    def __call__(self, images, return_tensors: str):
        assert all(stride >= 0 for stride in images.strides)
        return super().__call__(images, return_tensors)


def test_panoptic_segmenter_converts_result_and_filters_tiny_segments() -> None:
    frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    segmenter = PanopticSegmenter(
        device="cpu",
        input_size=512,
        processor=_FakeProcessor(),
        model=_FakeModel(),
        min_area_ratio=0.1,
    )

    result = segmenter.segment(frame_bgr)

    assert result.overlay_bgr.shape == frame_bgr.shape
    assert [segment.label for segment in result.segments] == ["chair", "wall"]
    assert [segment.area_pixels for segment in result.segments] == [8, 7]
    assert np.all(result.overlay_bgr[0, 0] == np.array(color_for_label(4)[::-1], dtype=np.uint8))
    assert np.all(result.overlay_bgr[0, 3] == np.array(color_for_label(7)[::-1], dtype=np.uint8))
    assert np.all(result.overlay_bgr[3, 3] == np.array([0, 0, 0], dtype=np.uint8))


def test_color_for_label_is_stable() -> None:
    assert color_for_label(5) == color_for_label(5)
    assert color_for_label(5) != color_for_label(6)


def test_panoptic_segmenter_provides_positive_stride_rgb_input() -> None:
    frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    segmenter = PanopticSegmenter(
        device="cpu",
        input_size=512,
        processor=_StrideCheckingProcessor(),
        model=_FakeModel(),
        min_area_ratio=0.1,
    )

    result = segmenter.segment(frame_bgr)

    assert result.overlay_bgr.shape == frame_bgr.shape
