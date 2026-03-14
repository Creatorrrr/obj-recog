from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from obj_recog.visualization import _find_unicode_font_path, render_multiline_unicode_text


def test_find_unicode_font_path_resolves_existing_font_when_available() -> None:
    font_path = _find_unicode_font_path()
    if font_path is None:
        pytest.skip("No Unicode-capable font found on this machine")

    assert Path(font_path).is_file()


def test_render_multiline_unicode_text_renders_korean_when_font_is_available() -> None:
    font_path = _find_unicode_font_path()
    if font_path is None:
        pytest.skip("No Unicode-capable font found on this machine")

    canvas = np.zeros((80, 220, 3), dtype=np.uint8)
    rendered = render_multiline_unicode_text(
        canvas,
        ["현재 화면 설명", "TV가 정면에 보입니다."],
        origin=(8, 10),
        line_height=24,
        color=(255, 255, 255),
        font_size=20,
    )

    assert rendered.shape == canvas.shape
    assert np.any(rendered != canvas)
