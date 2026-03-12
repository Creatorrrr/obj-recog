from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from obj_recog.opencv_runtime import load_cv2
from obj_recog.scene_graph import SceneGraphSnapshot
from obj_recog.situation_explainer import ExplanationStatus, wrap_explanation_text
from obj_recog.types import DepthDiagnostics, Detection, PanopticSegment


def _display_points_for_view(points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points_xyz.size == 0:
        return points_xyz.copy()
    transformed = points_xyz.copy()
    transformed[:, 1] *= -1.0
    transformed[:, 2] *= -1.0
    return transformed


def _measure_text(cv2, text: str, font: int, scale: float, thickness: int) -> tuple[int, int]:
    get_text_size = getattr(cv2, "getTextSize", None)
    if callable(get_text_size):
        (width, height), _baseline = get_text_size(text, font, scale, thickness)
        return int(width), int(height)
    return max(1, int(round(len(text) * 7 * scale))), max(1, int(round(14 * scale)))


def _draw_rectangle(cv2, canvas: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color, thickness: int) -> None:
    rectangle = getattr(cv2, "rectangle", None)
    if callable(rectangle):
        rectangle(canvas, pt1, pt2, color, thickness)


def _find_unicode_font_path() -> str | None:
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


def render_multiline_unicode_text(
    canvas: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int],
    line_height: int,
    color: tuple[int, int, int],
    font_size: int = 16,
    image_module=None,
    draw_module=None,
    font_module=None,
) -> np.ndarray:
    if not lines:
        return canvas

    try:
        if image_module is None or draw_module is None or font_module is None:
            from PIL import Image as pil_image
            from PIL import ImageDraw as pil_image_draw
            from PIL import ImageFont as pil_image_font
        else:
            pil_image = image_module
            pil_image_draw = draw_module
            pil_image_font = font_module
    except ImportError:
        return canvas

    font_path = _find_unicode_font_path()
    if font_path is None:
        return canvas

    try:
        font = pil_image_font.truetype(font_path, font_size)
    except Exception:
        return canvas

    image = pil_image.fromarray(canvas[:, :, ::-1].copy())
    draw = pil_image_draw.Draw(image)
    x, y = origin
    rgb_color = (int(color[2]), int(color[1]), int(color[0]))
    for index, line in enumerate(lines):
        draw.text((int(x), int(y + index * line_height)), str(line), fill=rgb_color, font=font)
    rendered = np.asarray(image, dtype=np.uint8)
    return rendered[:, :, ::-1].copy()


def explanation_button_rect(
    *,
    frame_width: int,
    frame_height: int,
    button_width: int = 108,
    button_height: int = 30,
    margin: int = 12,
) -> tuple[int, int, int, int]:
    x2 = max(margin, int(frame_width) - margin)
    y2 = max(margin, int(frame_height) - margin)
    x1 = max(0, x2 - int(button_width))
    y1 = max(0, y2 - int(button_height))
    return x1, y1, x2, y2


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int] | None) -> bool:
    if rect is None:
        return False
    x1, y1, x2, y2 = rect
    return x1 <= int(x) <= x2 and y1 <= int(y) <= y2


def _draw_explanation_button(
    cv2,
    canvas: np.ndarray,
    *,
    status: str | None,
    auto_refresh_enabled: bool = False,
) -> None:
    if status is None:
        return

    x1, y1, x2, y2 = explanation_button_rect(
        frame_width=canvas.shape[1],
        frame_height=canvas.shape[0],
    )
    if str(status) == ExplanationStatus.DISABLED:
        fill = (60, 60, 60)
        border = (120, 120, 120)
        text_color = (190, 190, 190)
    elif str(status) == ExplanationStatus.LOADING:
        fill = (30, 95, 180)
        border = (70, 140, 235)
        text_color = (255, 255, 255)
    elif str(status) == ExplanationStatus.READY:
        fill = (40, 120, 60)
        border = (90, 185, 110)
        text_color = (255, 255, 255)
    elif str(status) == ExplanationStatus.ERROR:
        fill = (45, 45, 155)
        border = (90, 90, 225)
        text_color = (255, 255, 255)
    else:
        fill = (90, 65, 20)
        border = (165, 130, 60)
        text_color = (255, 255, 255)

    _draw_rectangle(cv2, canvas, (x1, y1), (x2, y2), fill, -1)
    _draw_rectangle(cv2, canvas, (x1, y1), (x2, y2), border, 1)

    label = "Explain ON" if bool(auto_refresh_enabled) else "Explain OFF"
    text_width, text_height = _measure_text(
        cv2,
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )
    text_x = max(x1 + 8, x1 + ((x2 - x1 - text_width) // 2))
    text_y = y1 + ((y2 - y1 + text_height) // 2)
    cv2.putText(
        canvas,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
        cv2.LINE_AA,
    )


def explanation_panel_scroll_button_rects(
    *,
    panel_width: int,
    panel_height: int,
    margin: int = 16,
    button_width: int = 44,
    button_height: int = 34,
    top_y: int = 78,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    x2 = max(margin, int(panel_width) - margin)
    x1 = max(0, x2 - int(button_width))
    up_rect = (x1, int(top_y), x2, int(top_y) + int(button_height))
    down_rect = (
        x1,
        max(int(top_y) + int(button_height) + margin, int(panel_height) - margin - int(button_height)),
        x2,
        max(int(top_y) + int(button_height) + margin, int(panel_height) - margin - int(button_height))
        + int(button_height),
    )
    return up_rect, down_rect


def _draw_panel_scroll_button(cv2, canvas: np.ndarray, rect: tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = rect
    _draw_rectangle(cv2, canvas, (x1, y1), (x2, y2), (45, 55, 75), -1)
    _draw_rectangle(cv2, canvas, (x1, y1), (x2, y2), (100, 120, 165), 1)
    text_width, text_height = _measure_text(cv2, label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    text_x = max(x1 + 6, x1 + ((x2 - x1 - text_width) // 2))
    text_y = y1 + ((y2 - y1 + text_height) // 2)
    cv2.putText(
        canvas,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (235, 240, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_segmentation_legend(cv2, canvas: np.ndarray, segments: list[PanopticSegment]) -> None:
    if not segments:
        return

    unique_segments: dict[int, PanopticSegment] = {}
    for segment in sorted(segments, key=lambda item: int(item.area_pixels), reverse=True):
        unique_segments.setdefault(int(segment.label_id), segment)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_thickness = 1
    chip_size = 10
    row_height = 18
    top_padding = 12
    right_padding = 12
    text_gap = 6
    max_entries = 6

    entries = list(unique_segments.values())[:max_entries]
    for index, segment in enumerate(entries):
        label = str(segment.label)
        text_width, text_height = _measure_text(cv2, label, font, font_scale, text_thickness)
        total_width = chip_size + text_gap + text_width
        x_right = canvas.shape[1] - right_padding
        x_left = max(0, x_right - total_width)
        y_top = top_padding + (index * row_height)
        y_baseline = y_top + max(text_height, chip_size)
        chip_y = y_top + max(0, (text_height - chip_size) // 2)
        chip_color_bgr = tuple(int(channel) for channel in segment.color_rgb[::-1])

        _draw_rectangle(
            cv2,
            canvas,
            (x_left, chip_y),
            (x_left + chip_size, chip_y + chip_size),
            chip_color_bgr,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x_left + chip_size + text_gap, y_baseline),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )


def _draw_scene_graph_summary(
    cv2,
    canvas: np.ndarray,
    *,
    scene_graph_snapshot: SceneGraphSnapshot | None,
    visible_graph_nodes,
    visible_graph_edges,
) -> None:
    if scene_graph_snapshot is None:
        return

    edges = list(visible_graph_edges or scene_graph_snapshot.visible_edges)
    nodes = {node.id: node for node in (visible_graph_nodes or scene_graph_snapshot.visible_nodes)}
    ego_edges = [edge for edge in edges if edge.source == "ego" and edge.target in nodes]
    if not ego_edges:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    text_thickness = 1
    row_height = 16
    top_padding = 52
    right_padding = 12
    max_entries = 6
    entries = sorted(
        ego_edges,
        key=lambda edge: (-float(edge.confidence), edge.relation, nodes[edge.target].label),
    )[:max_entries]

    for index, edge in enumerate(entries):
        node = nodes[edge.target]
        suffix = f" ({edge.distance_bucket})" if edge.distance_bucket else ""
        text = f"{edge.relation} {node.label}{suffix}"
        text_width, text_height = _measure_text(cv2, text, font, font_scale, text_thickness)
        x = max(0, canvas.shape[1] - right_padding - text_width)
        y = top_padding + (index * row_height) + text_height
        cv2.putText(
            canvas,
            text,
            (x, y),
            font,
            font_scale,
            (220, 240, 255),
            text_thickness,
            cv2.LINE_AA,
        )


def _draw_runtime_status(
    cv2,
    canvas: np.ndarray,
    *,
    slam_tracking_state: str | None,
    keyframe_id: int | None,
    mesh_triangle_count: int | None,
    mesh_vertex_count: int | None,
    graph_node_count: int | None = None,
    graph_edge_count: int | None = None,
    localized_node_count: int | None = None,
    explanation_status: str | None = None,
    depth_diagnostics: DepthDiagnostics | None = None,
    depth_debug_level: str | None = None,
) -> None:
    if (
        slam_tracking_state is None
        and keyframe_id is None
        and mesh_triangle_count is None
        and mesh_vertex_count is None
        and graph_node_count is None
        and graph_edge_count is None
        and localized_node_count is None
        and explanation_status is None
        and depth_diagnostics is None
    ):
        return

    lines = [
        f"SLAM {slam_tracking_state or '-'}",
        f"KF {keyframe_id if keyframe_id is not None else '-'}",
        f"Mesh {int(mesh_triangle_count or 0)}t / {int(mesh_vertex_count or 0)}v",
    ]
    if graph_node_count is not None or graph_edge_count is not None or localized_node_count is not None:
        lines.extend(
            [
                f"Graph nodes {int(graph_node_count or 0)}",
                f"Graph edges {int(graph_edge_count or 0)}",
                f"Localized {int(localized_node_count or 0)}",
            ]
        )
    if explanation_status is not None:
        lines.append(f"Explain: {explanation_status}")
    if depth_diagnostics is not None and depth_debug_level in {"basic", "detailed"}:
        p10, p50, p90 = depth_diagnostics.normalized_distance_percentiles
        lines.extend(
            [
                f"Calib {depth_diagnostics.calibration_source}",
                f"Depth {depth_diagnostics.profile}",
                f"Dist {p10:.2f}/{p50:.2f}/{p90:.2f}m",
                f"Mesh z {depth_diagnostics.mesh_z_span:.2f}m",
                f"Hint {depth_diagnostics.hint}",
            ]
        )
        if depth_debug_level == "detailed":
            p05, raw_p50, p95 = depth_diagnostics.raw_percentiles
            norm_low, norm_high = depth_diagnostics.normalizer_low_high
            fx, fy, _cx, _cy = depth_diagnostics.intrinsics_summary
            lines.extend(
                [
                    f"Raw {p05:.2f}/{raw_p50:.2f}/{p95:.2f}",
                    f"Norm {norm_low:.2f}/{norm_high:.2f}",
                    f"Dense/Mesh {depth_diagnostics.dense_z_span:.2f}/{depth_diagnostics.mesh_z_span:.2f}m",
                    f"Valid {depth_diagnostics.valid_depth_ratio * 100.0:.1f}%",
                    f"fx/fy {fx:.1f}/{fy:.1f}",
                ]
            )
    x = 12
    y = max(20, canvas.shape[0] - 12 - ((len(lines) - 1) * 16))
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + (index * 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    fps: float,
    *,
    segmentation_overlay_bgr: np.ndarray | None = None,
    segments: list[PanopticSegment] | None = None,
    segmentation_alpha: float = 0.35,
    tracking_ok: bool | None = None,
    is_keyframe: bool | None = None,
    segment_id: int | None = None,
    camera_name: str | None = None,
    camera_fallback_active: bool = False,
    slam_tracking_state: str | None = None,
    keyframe_id: int | None = None,
    mesh_triangle_count: int | None = None,
    mesh_vertex_count: int | None = None,
    loop_closure_applied: bool = False,
    scene_graph_snapshot: SceneGraphSnapshot | None = None,
    visible_graph_nodes=None,
    visible_graph_edges=None,
    explanation_status: str | None = None,
    explanation_auto_refresh_enabled: bool = False,
    depth_diagnostics: DepthDiagnostics | None = None,
    depth_debug_level: str | None = None,
    cv2_module=None,
) -> np.ndarray:
    cv2 = load_cv2(cv2_module)

    canvas = frame_bgr.copy()
    graph_nodes = list(visible_graph_nodes or scene_graph_snapshot.visible_nodes) if scene_graph_snapshot is not None else []
    graph_edges = list(visible_graph_edges or scene_graph_snapshot.visible_edges) if scene_graph_snapshot is not None else []
    graph_node_count = None
    graph_edge_count = None
    localized_node_count = None
    runtime_status_enabled = any(
        value is not None
        for value in (slam_tracking_state, keyframe_id, mesh_triangle_count, mesh_vertex_count)
    )
    if scene_graph_snapshot is not None and runtime_status_enabled:
        graph_node_count = sum(1 for node in graph_nodes if node.id != "ego")
        graph_edge_count = len(graph_edges)
        localized_node_count = sum(
            1 for node in graph_nodes if node.id != "ego" and node.world_centroid is not None
        )
    if (
        segmentation_overlay_bgr is not None
        and segmentation_overlay_bgr.shape == canvas.shape
        and segmentation_alpha > 0.0
    ):
        overlay = np.asarray(segmentation_overlay_bgr, dtype=np.uint8)
        active_mask = np.any(overlay != 0, axis=2)
        if np.any(active_mask):
            blended = canvas.astype(np.float32, copy=True)
            overlay_float = overlay.astype(np.float32, copy=False)
            blended[active_mask] = (
                blended[active_mask] * (1.0 - segmentation_alpha)
                + overlay_float[active_mask] * segmentation_alpha
            )
            canvas = blended.astype(np.uint8)

    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        color_bgr = tuple(int(channel) for channel in detection.color[::-1])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_bgr,
            2,
            cv2.LINE_AA,
        )

    _draw_runtime_status(
        cv2,
        canvas,
        slam_tracking_state=slam_tracking_state,
        keyframe_id=keyframe_id,
        mesh_triangle_count=mesh_triangle_count,
        mesh_vertex_count=mesh_vertex_count,
        graph_node_count=graph_node_count,
        graph_edge_count=graph_edge_count,
        localized_node_count=localized_node_count,
        explanation_status=explanation_status,
        depth_diagnostics=depth_diagnostics,
        depth_debug_level=depth_debug_level,
    )

    if segments:
        _draw_segmentation_legend(cv2, canvas, list(segments))
    _draw_scene_graph_summary(
        cv2,
        canvas,
        scene_graph_snapshot=scene_graph_snapshot,
        visible_graph_nodes=graph_nodes,
        visible_graph_edges=graph_edges,
    )
    _draw_explanation_button(
        cv2,
        canvas,
        status=explanation_status,
        auto_refresh_enabled=explanation_auto_refresh_enabled,
    )
    return canvas


def render_explanation_panel(
    *,
    status: str,
    text: str,
    model: str,
    latency_ms: float | None,
    timestamp_label: str,
    refresh_status: str = "idle",
    width: int = 960,
    height: int = 360,
    scroll_offset: int = 0,
    cv2_module=None,
    unicode_text_renderer=render_multiline_unicode_text,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    cv2 = load_cv2(cv2_module)

    canvas = np.zeros((max(320, height), max(720, width), 3), dtype=np.uint8)
    header_lines = [
        f"Status: {str(status).upper()} | {timestamp_label}",
        f"Refresh: {str(refresh_status)}",
        f"Model: {model}",
        f"Latency: {'-' if latency_ms is None else int(round(latency_ms))}ms",
    ]
    reserved_scroll_width = 84
    wrap_width = max(56, int((canvas.shape[1] - reserved_scroll_width) / 13))
    body_lines = wrap_explanation_text(text, width=wrap_width, max_lines=None)
    if not body_lines:
        if str(status) == ExplanationStatus.DISABLED:
            body_lines = ["OPENAI_API_KEY가 없어 상황 설명 기능이 비활성화되었습니다."]
        elif str(status) == ExplanationStatus.LOADING:
            body_lines = ["현재 장면 설명을 생성하는 중입니다."]
        elif str(status) == ExplanationStatus.CAPTURING:
            body_lines = ["현재 프레임 스냅샷을 고정하는 중입니다."]
        elif str(status) == ExplanationStatus.ERROR:
            body_lines = ["설명 생성 중 오류가 발생했습니다."]
        elif str(status) == ExplanationStatus.READY:
            body_lines = ["모델이 비어 있는 설명을 반환했습니다. 다시 시도해 주세요."]
        else:
            body_lines = ["우하단 Explain 버튼을 클릭하거나 e 키를 눌러 현재 상황 설명을 요청하세요."]

    y = 28
    for line in header_lines:
        cv2.putText(
            canvas,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 230, 255),
            1,
            cv2.LINE_AA,
        )
        y += 20
    y += 12

    body_left = 16
    body_top = y - 12
    line_height = 22
    body_bottom = canvas.shape[0] - 20
    visible_line_count = max(4, int((body_bottom - body_top) // line_height))
    max_scroll_offset = max(0, len(body_lines) - visible_line_count)
    used_scroll_offset = min(max(0, int(scroll_offset)), max_scroll_offset)
    visible_lines = body_lines[used_scroll_offset : used_scroll_offset + visible_line_count]

    up_rect, down_rect = explanation_panel_scroll_button_rects(
        panel_width=canvas.shape[1],
        panel_height=canvas.shape[0],
    )
    if max_scroll_offset > 0:
        _draw_panel_scroll_button(cv2, canvas, up_rect, "UP")
        _draw_panel_scroll_button(cv2, canvas, down_rect, "DN")
        scroll_text = (
            f"Lines {used_scroll_offset + 1}-{used_scroll_offset + len(visible_lines)}/{len(body_lines)}"
        )
        cv2.putText(
            canvas,
            scroll_text,
            (16, min(canvas.shape[0] - 10, body_bottom + 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (170, 185, 210),
            1,
            cv2.LINE_AA,
        )
    else:
        up_rect = None
        down_rect = None

    body_origin = (body_left, body_top)
    rendered = unicode_text_renderer(
        canvas,
        visible_lines,
        origin=body_origin,
        line_height=line_height,
        color=(245, 245, 245),
    )
    if rendered is canvas:
        for line in visible_lines:
            cv2.putText(
                canvas,
                line,
                (body_left, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            y += line_height
        if return_metadata:
            return canvas, {
                "scroll_offset": used_scroll_offset,
                "max_scroll_offset": max_scroll_offset,
                "visible_line_count": visible_line_count,
                "up_rect": up_rect,
                "down_rect": down_rect,
            }
        return canvas
    if return_metadata:
        return rendered, {
            "scroll_offset": used_scroll_offset,
            "max_scroll_offset": max_scroll_offset,
            "visible_line_count": visible_line_count,
            "up_rect": up_rect,
            "down_rect": down_rect,
        }
    return rendered


def highlight_detected_points(
    point_pixels: np.ndarray,
    point_colors: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    highlighted = point_colors.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        inside = (
            (point_pixels[:, 0] >= x1)
            & (point_pixels[:, 0] <= x2)
            & (point_pixels[:, 1] >= y1)
            & (point_pixels[:, 1] <= y2)
        )
        if np.any(inside):
            highlighted[inside] = np.asarray(detection.color, dtype=np.float32) / 255.0
    return highlighted


def render_environment_model_panel(
    scenario_state,
    *,
    panel_width: int = 520,
    panel_height: int = 420,
    cv2_module=None,
) -> np.ndarray:
    cv2 = load_cv2(cv2_module)
    canvas = np.full((panel_height, panel_width, 3), (18, 20, 24), dtype=np.uint8)
    objects = tuple(getattr(scenario_state, "environment_objects", ()) or ())
    _draw_rectangle(cv2, canvas, (18, 18), (panel_width - 18, panel_height - 18), (46, 54, 70), 1)
    cv2.putText(
        canvas,
        "Environment Model",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 240, 250),
        1,
        cv2.LINE_AA,
    )
    if not objects:
        cv2.putText(
            canvas,
            "No environment objects",
            (24, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (170, 176, 188),
            1,
            cv2.LINE_AA,
        )
        return canvas

    scale = min(panel_width, panel_height) * 0.065
    center_x = int(panel_width * 0.5)
    center_y = int(panel_height * 0.64)
    _draw_isometric_floor_grid(cv2, canvas, center_x=center_x, center_y=center_y, scale=scale)
    for obj in sorted(objects, key=lambda item: float(item["center_world"][1]) + float(item["center_world"][2])):
        _draw_isometric_object(cv2, canvas, obj, center_x=center_x, center_y=center_y, scale=scale)
    _draw_isometric_camera(cv2, canvas, scenario_state, center_x=center_x, center_y=center_y, scale=scale)
    return canvas


def _iso_project(x: float, y: float, z: float, *, center_x: int, center_y: int, scale: float) -> tuple[int, int]:
    iso_x = center_x + ((x - z) * scale)
    iso_y = center_y + (((x + z) * scale * 0.5) - (y * scale * 1.35))
    return int(round(iso_x)), int(round(iso_y))


def _draw_isometric_floor_grid(cv2, canvas: np.ndarray, *, center_x: int, center_y: int, scale: float) -> None:
    line = getattr(cv2, "line", None)
    if not callable(line):
        return
    grid_color = (42, 50, 62)
    for offset in range(-6, 7):
        start = _iso_project(-6.0, 0.0, float(offset), center_x=center_x, center_y=center_y, scale=scale)
        end = _iso_project(6.0, 0.0, float(offset), center_x=center_x, center_y=center_y, scale=scale)
        line(canvas, start, end, grid_color, 1)
        start = _iso_project(float(offset), 0.0, -6.0, center_x=center_x, center_y=center_y, scale=scale)
        end = _iso_project(float(offset), 0.0, 6.0, center_x=center_x, center_y=center_y, scale=scale)
        line(canvas, start, end, grid_color, 1)


def _shade_color(color_bgr: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, round(channel * factor)))) for channel in color_bgr)


def _draw_isometric_object(
    cv2,
    canvas: np.ndarray,
    obj: dict[str, object],
    *,
    center_x: int,
    center_y: int,
    scale: float,
) -> None:
    fill_poly = getattr(cv2, "fillConvexPoly", None)
    polylines = getattr(cv2, "polylines", None)
    line = getattr(cv2, "line", None)
    center_world = tuple(float(value) for value in obj["center_world"])
    size_xyz = tuple(float(value) for value in obj["size_xyz"])
    half_x = size_xyz[0] * 0.5
    half_y = size_xyz[1] * 0.5
    half_z = size_xyz[2] * 0.5
    color_bgr = tuple(int(value) for value in obj["color_bgr"])
    if bool(obj.get("target_role")):
        color_bgr = (40, 210, 245)
    elif bool(obj.get("visible")):
        color_bgr = _shade_color(color_bgr, 1.15)

    x, y, z = center_world
    top = np.array(
        [
            _iso_project(x - half_x, y + half_y, z - half_z, center_x=center_x, center_y=center_y, scale=scale),
            _iso_project(x + half_x, y + half_y, z - half_z, center_x=center_x, center_y=center_y, scale=scale),
            _iso_project(x + half_x, y + half_y, z + half_z, center_x=center_x, center_y=center_y, scale=scale),
            _iso_project(x - half_x, y + half_y, z + half_z, center_x=center_x, center_y=center_y, scale=scale),
        ],
        dtype=np.int32,
    )
    right = np.array(
        [
            _iso_project(x + half_x, y + half_y, z - half_z, center_x=center_x, center_y=center_y, scale=scale),
            _iso_project(x + half_x, y - half_y, z - half_z, center_x=center_x, center_y=center_y, scale=scale),
            _iso_project(x + half_x, y - half_y, z + half_z, center_x=center_x, center_y=center_y, scale=scale),
            _iso_project(x + half_x, y + half_y, z + half_z, center_x=center_x, center_y=center_y, scale=scale),
        ],
        dtype=np.int32,
    )
    left = np.array(
        [
            _iso_project(x - half_x, y + half_y, z + half_z, center_x=center_x, center_y=center_y, scale=scale),
            _iso_project(x - half_x, y - half_y, z + half_z, center_x=center_x, center_y=center_y, scale=scale),
            _iso_project(x + half_x, y - half_y, z + half_z, center_x=center_x, center_y=center_y, scale=scale),
            _iso_project(x + half_x, y + half_y, z + half_z, center_x=center_x, center_y=center_y, scale=scale),
        ],
        dtype=np.int32,
    )
    if callable(fill_poly):
        fill_poly(canvas, top, _shade_color(color_bgr, 1.1))
        fill_poly(canvas, left, _shade_color(color_bgr, 0.88))
        fill_poly(canvas, right, _shade_color(color_bgr, 0.72))
    elif callable(line):
        x1, y1 = top.min(axis=0)
        x2, y2 = top.max(axis=0)
        _draw_rectangle(cv2, canvas, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, -1)
    if callable(polylines):
        polylines(canvas, [top, right, left], True, (16, 18, 22), 1)
    cv2.putText(
        canvas,
        str(obj["label"]),
        (int(top[0][0]) - 8, int(top[0][1]) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (235, 240, 250),
        1,
        cv2.LINE_AA,
    )


def _draw_isometric_camera(cv2, canvas: np.ndarray, scenario_state, *, center_x: int, center_y: int, scale: float) -> None:
    line = getattr(cv2, "line", None)
    if not callable(line):
        return
    rig_x = float(getattr(scenario_state, "rig_x", 0.0))
    rig_z = float(getattr(scenario_state, "rig_z", 0.0))
    yaw_rad = math.radians(float(getattr(scenario_state, "yaw_deg", 0.0)))
    camera_point = _iso_project(rig_x, 0.0, rig_z, center_x=center_x, center_y=center_y, scale=scale)
    forward = _iso_project(
        rig_x + (math.sin(yaw_rad) * 0.8),
        0.0,
        rig_z + (math.cos(yaw_rad) * 0.8),
        center_x=center_x,
        center_y=center_y,
        scale=scale,
    )
    left = _iso_project(
        rig_x + (math.sin(yaw_rad - 0.5) * 0.55),
        0.0,
        rig_z + (math.cos(yaw_rad - 0.5) * 0.55),
        center_x=center_x,
        center_y=center_y,
        scale=scale,
    )
    right = _iso_project(
        rig_x + (math.sin(yaw_rad + 0.5) * 0.55),
        0.0,
        rig_z + (math.cos(yaw_rad + 0.5) * 0.55),
        center_x=center_x,
        center_y=center_y,
        scale=scale,
    )
    line(canvas, camera_point, forward, (255, 255, 255), 2)
    line(canvas, camera_point, left, (180, 210, 255), 1)
    line(canvas, camera_point, right, (180, 210, 255), 1)


class Open3DMeshViewer:
    def __init__(self, window_name: str = "3D Reconstruction", o3d_module=None) -> None:
        if o3d_module is None:
            try:
                import open3d as o3d
            except ImportError as exc:  # pragma: no cover - depends on local install.
                raise RuntimeError("open3d is required to render the 3D mesh") from exc
        else:
            o3d = o3d_module

        self._o3d = o3d
        self._vis = o3d.visualization.Visualizer()
        window_created = self._vis.create_window(window_name=window_name, width=960, height=720)
        if not window_created:
            raise RuntimeError("failed to create Open3D window")
        self._mesh = o3d.geometry.TriangleMesh()
        self._graph_nodes = o3d.geometry.PointCloud()
        self._graph_edges = o3d.geometry.LineSet()
        self._ego_lines = o3d.geometry.LineSet()
        self._has_fitted_view = False
        self._vis.add_geometry(self._mesh)
        self._vis.add_geometry(self._graph_nodes)
        self._vis.add_geometry(self._graph_edges)
        self._vis.add_geometry(self._ego_lines)
        self._vis.poll_events()
        self._vis.update_renderer()

        render_option = self._vis.get_render_option()
        if render_option is not None:
            render_option.background_color = np.array([0.05, 0.05, 0.08], dtype=np.float64)
            show_back_face = getattr(render_option, "mesh_show_back_face", None)
            if show_back_face is not None:
                render_option.mesh_show_back_face = True

    def update(
        self,
        mesh_vertices_xyz: np.ndarray,
        mesh_triangles: np.ndarray,
        mesh_vertex_colors: np.ndarray,
        scene_graph_snapshot: SceneGraphSnapshot | None = None,
        *_unused,
    ) -> bool:
        display_vertices_xyz = _display_points_for_view(mesh_vertices_xyz)
        self._mesh.vertices = self._o3d.utility.Vector3dVector(
            display_vertices_xyz.astype(np.float64, copy=False)
        )
        self._mesh.triangles = self._o3d.utility.Vector3iVector(
            mesh_triangles.astype(np.int32, copy=False)
        )
        self._mesh.vertex_colors = self._o3d.utility.Vector3dVector(
            mesh_vertex_colors.astype(np.float64, copy=False)
        )
        compute_normals = getattr(self._mesh, "compute_vertex_normals", None)
        if callable(compute_normals):
            compute_normals()
        self._vis.update_geometry(self._mesh)
        self._update_scene_graph(scene_graph_snapshot)
        if not self._has_fitted_view and display_vertices_xyz.size > 0 and mesh_triangles.size > 0:
            self._vis.reset_view_point(True)
            self._has_fitted_view = True
        is_active = self._vis.poll_events()
        self._vis.update_renderer()
        return bool(is_active)

    def _update_scene_graph(self, scene_graph_snapshot: SceneGraphSnapshot | None) -> None:
        empty_points = np.empty((0, 3), dtype=np.float64)
        empty_lines = np.empty((0, 2), dtype=np.int32)
        if scene_graph_snapshot is None:
            self._graph_nodes.points = self._o3d.utility.Vector3dVector(empty_points)
            self._graph_nodes.colors = self._o3d.utility.Vector3dVector(empty_points)
            self._graph_edges.points = self._o3d.utility.Vector3dVector(empty_points)
            self._graph_edges.lines = self._o3d.utility.Vector2iVector(empty_lines)
            self._graph_edges.colors = self._o3d.utility.Vector3dVector(empty_points)
            self._ego_lines.points = self._o3d.utility.Vector3dVector(empty_points)
            self._ego_lines.lines = self._o3d.utility.Vector2iVector(empty_lines)
            self._ego_lines.colors = self._o3d.utility.Vector3dVector(empty_points)
            self._vis.update_geometry(self._graph_nodes)
            self._vis.update_geometry(self._graph_edges)
            self._vis.update_geometry(self._ego_lines)
            return

        nodes = {node.id: node for node in scene_graph_snapshot.visible_nodes}
        graph_points: list[np.ndarray] = []
        graph_colors: list[np.ndarray] = []
        for node in scene_graph_snapshot.visible_nodes:
            if node.id == "ego" or node.world_centroid is None:
                continue
            graph_points.append(np.asarray(node.world_centroid, dtype=np.float64))
            if node.type == "object":
                graph_colors.append(np.array([1.0, 0.6, 0.2], dtype=np.float64))
            else:
                graph_colors.append(np.array([0.7, 0.7, 0.7], dtype=np.float64))
        node_points_array = np.asarray(graph_points, dtype=np.float64).reshape(-1, 3)
        node_colors_array = np.asarray(graph_colors, dtype=np.float64).reshape(-1, 3)
        node_points_array = _display_points_for_view(node_points_array).astype(np.float64, copy=False)
        self._graph_nodes.points = self._o3d.utility.Vector3dVector(node_points_array)
        self._graph_nodes.colors = self._o3d.utility.Vector3dVector(node_colors_array)

        line_points: list[np.ndarray] = []
        line_indices: list[list[int]] = []
        line_colors: list[np.ndarray] = []
        point_index_by_node: dict[str, int] = {}

        def _point_index(node_id: str, xyz: np.ndarray) -> int:
            if node_id in point_index_by_node:
                return point_index_by_node[node_id]
            point_index_by_node[node_id] = len(line_points)
            line_points.append(np.asarray(xyz, dtype=np.float64))
            return point_index_by_node[node_id]

        relation_colors = {
            "on": np.array([0.2, 0.9, 0.2], dtype=np.float64),
            "inside": np.array([0.3, 0.8, 0.3], dtype=np.float64),
            "inside_region": np.array([0.3, 0.8, 0.3], dtype=np.float64),
            "attached_to": np.array([0.3, 0.8, 0.3], dtype=np.float64),
            "near": np.array([1.0, 0.55, 0.1], dtype=np.float64),
        }
        for edge in scene_graph_snapshot.visible_edges:
            source_xyz = (
                scene_graph_snapshot.camera_pose_world[:3, 3]
                if edge.source == "ego"
                else nodes.get(edge.source, None).world_centroid if edge.source in nodes else None
            )
            target_xyz = nodes.get(edge.target, None).world_centroid if edge.target in nodes else None
            if source_xyz is None or target_xyz is None:
                continue
            start_index = _point_index(edge.source, np.asarray(source_xyz, dtype=np.float64))
            end_index = _point_index(edge.target, np.asarray(target_xyz, dtype=np.float64))
            line_indices.append([start_index, end_index])
            line_colors.append(
                relation_colors.get(edge.relation, np.array([0.1, 0.9, 0.9], dtype=np.float64))
            )
        line_points_array = _display_points_for_view(
            np.asarray(line_points, dtype=np.float64).reshape(-1, 3)
        ).astype(np.float64, copy=False)
        self._graph_edges.points = self._o3d.utility.Vector3dVector(line_points_array)
        self._graph_edges.lines = self._o3d.utility.Vector2iVector(
            np.asarray(line_indices, dtype=np.int32).reshape(-1, 2)
        )
        self._graph_edges.colors = self._o3d.utility.Vector3dVector(
            np.asarray(line_colors, dtype=np.float64).reshape(-1, 3)
        )

        ego_points, ego_lines, ego_colors = _ego_frustum_geometry(scene_graph_snapshot.camera_pose_world)
        self._ego_lines.points = self._o3d.utility.Vector3dVector(ego_points)
        self._ego_lines.lines = self._o3d.utility.Vector2iVector(ego_lines)
        self._ego_lines.colors = self._o3d.utility.Vector3dVector(ego_colors)
        self._vis.update_geometry(self._graph_nodes)
        self._vis.update_geometry(self._graph_edges)
        self._vis.update_geometry(self._ego_lines)

    def close(self) -> None:
        self._vis.destroy_window()


Open3DPointCloudViewer = Open3DMeshViewer


def _ego_frustum_geometry(camera_pose_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.06, -0.04, 0.12],
            [0.06, -0.04, 0.12],
            [0.06, 0.04, 0.12],
            [-0.06, 0.04, 0.12],
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate(
        (local_points, np.ones((local_points.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    world_points = (np.asarray(camera_pose_world, dtype=np.float64) @ homogeneous.T).T[:, :3]
    world_points = _display_points_for_view(world_points).astype(np.float64, copy=False)
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    colors = np.tile(np.array([[0.2, 0.9, 0.9]], dtype=np.float64), (lines.shape[0], 1))
    return world_points, lines, colors
