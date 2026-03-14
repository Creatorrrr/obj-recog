from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from obj_recog.opencv_runtime import load_cv2
from obj_recog.scene_graph import SceneGraphSnapshot
from obj_recog.sim_scene import build_scene_mesh_components
from obj_recog.situation_explainer import ExplanationStatus, wrap_explanation_text
from obj_recog.types import DepthDiagnostics, Detection, PanopticSegment, PerceptionDiagnostics

_RUNTIME_WINDOW_MARGIN_X = 32
_RUNTIME_WINDOW_MARGIN_Y = 48
_RUNTIME_WINDOW_GAP = 32
_WINDOWS_FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"


def _display_points_for_view(points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points_xyz.size == 0:
        return points_xyz.copy()
    transformed = points_xyz.copy()
    transformed[:, 1] *= -1.0
    transformed[:, 2] *= -1.0
    return transformed


def _display_points_for_environment_view(points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points_xyz.size == 0:
        return points_xyz.copy()
    transformed = points_xyz.copy()
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
        _WINDOWS_FONT_DIR / "malgun.ttf",
        _WINDOWS_FONT_DIR / "malgunbd.ttf",
        _WINDOWS_FONT_DIR / "malgunsl.ttf",
        _WINDOWS_FONT_DIR / "gulim.ttc",
        _WINDOWS_FONT_DIR / "batang.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return str(candidate_path)
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


def runtime_window_position(
    window_name: str,
    *,
    primary_width: int = 640,
    primary_height: int = 360,
) -> tuple[int, int]:
    main_width = max(320, int(primary_width))
    main_height = max(240, int(primary_height))
    left_x = _RUNTIME_WINDOW_MARGIN_X
    top_y = _RUNTIME_WINDOW_MARGIN_Y
    right_x = left_x + main_width + _RUNTIME_WINDOW_GAP
    bottom_y = top_y + main_height + _RUNTIME_WINDOW_GAP
    positions = {
        "Object Recognition": (left_x, top_y),
        "Environment Model": (right_x, top_y),
        "Situation Explanation": (left_x, bottom_y),
        "3D Reconstruction": (right_x, bottom_y),
    }
    return positions.get(str(window_name), (left_x, top_y))


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


def explanation_panel_scrollbar_geometry(
    *,
    panel_width: int,
    body_top: int,
    body_bottom: int,
    total_line_count: int,
    visible_line_count: int,
    scroll_offset: int,
    margin: int = 16,
    track_width: int = 14,
    min_thumb_height: int = 44,
) -> dict[str, tuple[int, int, int, int]] | None:
    max_scroll_offset = max(0, int(total_line_count) - int(visible_line_count))
    if max_scroll_offset <= 0:
        return None

    track_x2 = max(margin, int(panel_width) - margin)
    track_x1 = max(0, track_x2 - int(track_width))
    track_y1 = int(body_top)
    track_y2 = max(track_y1 + int(min_thumb_height), int(body_bottom))
    track_height = max(1, track_y2 - track_y1)
    thumb_height = max(
        int(min_thumb_height),
        int(round(track_height * (float(visible_line_count) / max(1.0, float(total_line_count))))),
    )
    thumb_height = min(track_height, thumb_height)
    max_thumb_offset = max(0, track_height - thumb_height)
    thumb_offset = (
        0
        if max_scroll_offset <= 0
        else int(round((float(scroll_offset) / float(max_scroll_offset)) * float(max_thumb_offset)))
    )
    thumb_y1 = track_y1 + thumb_offset
    thumb_y2 = min(track_y2, thumb_y1 + thumb_height)

    return {
        "track_rect": (track_x1, track_y1, track_x2, track_y2),
        "thumb_rect": (track_x1, thumb_y1, track_x2, thumb_y2),
        "up_rect": (track_x1, track_y1, track_x2, thumb_y1),
        "down_rect": (track_x1, thumb_y2, track_x2, track_y2),
    }


def _draw_panel_scrollbar(
    cv2,
    canvas: np.ndarray,
    *,
    track_rect: tuple[int, int, int, int],
    thumb_rect: tuple[int, int, int, int],
) -> None:
    track_x1, track_y1, track_x2, track_y2 = track_rect
    thumb_x1, thumb_y1, thumb_x2, thumb_y2 = thumb_rect
    _draw_rectangle(cv2, canvas, (track_x1, track_y1), (track_x2, track_y2), (35, 42, 56), -1)
    _draw_rectangle(cv2, canvas, (track_x1, track_y1), (track_x2, track_y2), (78, 92, 120), 1)
    _draw_rectangle(cv2, canvas, (thumb_x1, thumb_y1), (thumb_x2, thumb_y2), (105, 140, 205), -1)
    _draw_rectangle(cv2, canvas, (thumb_x1, thumb_y1), (thumb_x2, thumb_y2), (190, 215, 255), 1)


def _apply_open3d_view_control(
    control,
    *,
    lookat: np.ndarray,
    front: np.ndarray,
    up: np.ndarray,
    zoom: float,
) -> None:
    if control is None:
        return
    set_lookat = getattr(control, "set_lookat", None)
    set_front = getattr(control, "set_front", None)
    set_up = getattr(control, "set_up", None)
    set_zoom = getattr(control, "set_zoom", None)
    if callable(set_lookat):
        set_lookat(np.asarray(lookat, dtype=np.float64))
    if callable(set_front):
        set_front(np.asarray(front, dtype=np.float64))
    if callable(set_up):
        set_up(np.asarray(up, dtype=np.float64))
    if callable(set_zoom):
        set_zoom(float(zoom))


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
    perception_diagnostics: PerceptionDiagnostics | None = None,
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
        and perception_diagnostics is None
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
    if perception_diagnostics is not None:
        lines.extend(
            [
                f"Perception {perception_diagnostics.perception_mode}",
                f"Detect {perception_diagnostics.detection_source}",
                f"Depth/Pose {perception_diagnostics.depth_source}/{perception_diagnostics.pose_source}",
                (
                    "Benchmark "
                    f"{'valid' if perception_diagnostics.benchmark_valid else 'invalid'} "
                    f"GT-visible {'yes' if perception_diagnostics.gt_target_visible else 'no'}"
                ),
            ]
        )
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
    perception_diagnostics: PerceptionDiagnostics | None = None,
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
        perception_diagnostics=perception_diagnostics,
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
    height: int = 480,
    scroll_offset: int = 0,
    cv2_module=None,
    unicode_text_renderer=render_multiline_unicode_text,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    cv2 = load_cv2(cv2_module)

    canvas = np.zeros((max(420, height), max(720, width), 3), dtype=np.uint8)
    header_lines = [
        f"Status: {str(status).upper()} | {timestamp_label}",
        f"Refresh: {str(refresh_status)}",
        f"Model: {model}",
        f"Latency: {'-' if latency_ms is None else int(round(latency_ms))}ms",
    ]
    reserved_scroll_width = 54
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
    footer_y = canvas.shape[0] - 14
    body_bottom = canvas.shape[0] - 42
    visible_line_count = max(4, int((body_bottom - body_top) // line_height))
    max_scroll_offset = max(0, len(body_lines) - visible_line_count)
    used_scroll_offset = min(max(0, int(scroll_offset)), max_scroll_offset)
    visible_lines = body_lines[used_scroll_offset : used_scroll_offset + visible_line_count]

    scrollbar_geometry = explanation_panel_scrollbar_geometry(
        panel_width=canvas.shape[1],
        body_top=body_top,
        body_bottom=body_bottom,
        total_line_count=len(body_lines),
        visible_line_count=visible_line_count,
        scroll_offset=used_scroll_offset,
    )
    if max_scroll_offset > 0:
        assert scrollbar_geometry is not None
        _draw_panel_scrollbar(
            cv2,
            canvas,
            track_rect=scrollbar_geometry["track_rect"],
            thumb_rect=scrollbar_geometry["thumb_rect"],
        )
        scroll_text = f"Wheel: scroll | Lines {used_scroll_offset + 1}-{used_scroll_offset + len(visible_lines)}/{len(body_lines)}"
        cv2.putText(
            canvas,
            scroll_text,
            (16, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (170, 185, 210),
            1,
            cv2.LINE_AA,
        )
        up_rect = scrollbar_geometry["up_rect"]
        down_rect = scrollbar_geometry["down_rect"]
    else:
        up_rect = None
        down_rect = None
        scrollbar_geometry = None

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
                "scrollbar_rect": None if scrollbar_geometry is None else scrollbar_geometry["track_rect"],
            }
        return canvas
    if return_metadata:
        return rendered, {
            "scroll_offset": used_scroll_offset,
            "max_scroll_offset": max_scroll_offset,
            "visible_line_count": visible_line_count,
            "up_rect": up_rect,
            "down_rect": down_rect,
            "scrollbar_rect": None if scrollbar_geometry is None else scrollbar_geometry["track_rect"],
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


def _combine_triangle_meshes(
    meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not meshes:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
        )
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    vertex_offset = 0
    for mesh_vertices, mesh_triangles, mesh_colors in meshes:
        current_vertices = np.asarray(mesh_vertices, dtype=np.float64).reshape(-1, 3)
        current_triangles = np.asarray(mesh_triangles, dtype=np.int32).reshape(-1, 3)
        current_colors = np.asarray(mesh_colors, dtype=np.float64).reshape(-1, 3)
        vertices.append(current_vertices)
        triangles.append(current_triangles + int(vertex_offset))
        colors.append(current_colors)
        vertex_offset += int(current_vertices.shape[0])
    return (
        np.vstack(vertices),
        np.vstack(triangles),
        np.vstack(colors),
    )


def _box_mesh_arrays(size_xyz: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = (max(float(value), 1e-3) * 0.5 for value in size_xyz)
    vertices = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return vertices, triangles


def _sphere_mesh_arrays(
    size_xyz: tuple[float, float, float],
    *,
    segments: int = 14,
    rings: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    rx, ry, rz = (max(float(value), 1e-3) * 0.5 for value in size_xyz)
    vertices: list[list[float]] = [[0.0, ry, 0.0]]
    triangles: list[list[int]] = []
    for ring_index in range(1, rings):
        phi = math.pi * (ring_index / float(rings))
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        for segment_index in range(segments):
            theta = (2.0 * math.pi * segment_index) / float(segments)
            vertices.append(
                [
                    rx * sin_phi * math.cos(theta),
                    ry * cos_phi,
                    rz * sin_phi * math.sin(theta),
                ]
            )
    south_index = len(vertices)
    vertices.append([0.0, -ry, 0.0])
    if rings > 1:
        for segment_index in range(segments):
            next_index = (segment_index + 1) % segments
            triangles.append([0, 1 + next_index, 1 + segment_index])
        for ring_index in range(rings - 2):
            ring_start = 1 + (ring_index * segments)
            next_ring_start = ring_start + segments
            for segment_index in range(segments):
                next_index = (segment_index + 1) % segments
                a = ring_start + segment_index
                b = ring_start + next_index
                c = next_ring_start + segment_index
                d = next_ring_start + next_index
                triangles.append([a, c, b])
                triangles.append([b, c, d])
        last_ring_start = 1 + ((rings - 2) * segments)
        for segment_index in range(segments):
            next_index = (segment_index + 1) % segments
            triangles.append([south_index, last_ring_start + segment_index, last_ring_start + next_index])
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32).reshape(-1, 3)


def _rotation_matrix_xyz(rotation_xyz_deg: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = (math.radians(float(value)) for value in rotation_xyz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _rotation_matrix_y(yaw_deg: float) -> np.ndarray:
    yaw_rad = math.radians(float(yaw_deg))
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _primitive_mesh_arrays(
    primitive_type: str,
    dimensions_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if str(primitive_type) == "UV_SPHERE":
        return _sphere_mesh_arrays(dimensions_xyz)
    return _box_mesh_arrays(dimensions_xyz)


def _colored_mesh(
    vertices_xyz: np.ndarray,
    triangles: np.ndarray,
    *,
    color_rgb: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex_count = int(np.asarray(vertices_xyz).reshape(-1, 3).shape[0])
    colors = np.tile(np.asarray(color_rgb, dtype=np.float64).reshape(1, 3), (vertex_count, 1))
    return (
        np.asarray(vertices_xyz, dtype=np.float64).reshape(-1, 3),
        np.asarray(triangles, dtype=np.int32).reshape(-1, 3),
        colors,
    )


def _translated_colored_primitive_mesh(
    primitive_type: str,
    dimensions_xyz: tuple[float, float, float],
    center_xyz: tuple[float, float, float],
    *,
    color_rgb: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_xyz, triangles = _primitive_mesh_arrays(primitive_type, dimensions_xyz)
    translated_vertices = vertices_xyz + np.asarray(center_xyz, dtype=np.float64).reshape(1, 3)
    return _colored_mesh(translated_vertices, triangles, color_rgb=color_rgb)


def _object_mesh_arrays(
    obj: dict[str, object],
    *,
    o3d_module=None,
    preview_mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preview_mesh_path = obj.get("preview_mesh_path")
    vertices_xyz = np.empty((0, 3), dtype=np.float64)
    triangles = np.empty((0, 3), dtype=np.int32)
    if preview_mesh_path and o3d_module is not None and hasattr(o3d_module, "io"):
        cache_key = str(preview_mesh_path)
        cached = None if preview_mesh_cache is None else preview_mesh_cache.get(cache_key)
        if cached is None:
            try:
                mesh = o3d_module.io.read_triangle_mesh(str(preview_mesh_path))
                raw_vertices = np.asarray(getattr(getattr(mesh, "vertices", None), "data", getattr(mesh, "vertices", np.empty((0, 3)))), dtype=np.float64).reshape(-1, 3)
                raw_triangles = np.asarray(getattr(getattr(mesh, "triangles", None), "data", getattr(mesh, "triangles", np.empty((0, 3)))), dtype=np.int32).reshape(-1, 3)
            except Exception:
                raw_vertices = np.empty((0, 3), dtype=np.float64)
                raw_triangles = np.empty((0, 3), dtype=np.int32)
            cached = (raw_vertices, raw_triangles)
            if preview_mesh_cache is not None:
                preview_mesh_cache[cache_key] = cached
        vertices_xyz, triangles = cached
    if vertices_xyz.size == 0:
        fallback_vertices, fallback_triangles = _box_mesh_arrays(tuple(float(value) for value in obj["size_xyz"]))
        color_bgr = tuple(int(value) for value in obj["color_bgr"])
        return _colored_mesh(
            fallback_vertices + np.asarray(obj["center_world"], dtype=np.float64),
            fallback_triangles,
            color_rgb=(color_bgr[2] / 255.0, color_bgr[1] / 255.0, color_bgr[0] / 255.0),
        )

    bounds_min = vertices_xyz.min(axis=0)
    bounds_max = vertices_xyz.max(axis=0)
    extents = np.maximum(bounds_max - bounds_min, 1e-3)
    centered_vertices = vertices_xyz - ((bounds_min + bounds_max) * 0.5)
    target_size = np.asarray(obj["size_xyz"], dtype=np.float64).reshape(3)
    scaled_vertices = centered_vertices * (target_size / extents)
    world_rotation = _rotation_matrix_y(float(obj.get("yaw_deg", 0.0)))
    world_vertices = (scaled_vertices @ world_rotation.T) + np.asarray(obj["center_world"], dtype=np.float64)
    color_bgr = tuple(int(value) for value in obj["color_bgr"])
    if bool(obj.get("target_role")):
        color_rgb = (245.0 / 255.0, 210.0 / 255.0, 40.0 / 255.0)
    elif bool(obj.get("visible")):
        color_rgb = (
            min(1.0, (color_bgr[2] / 255.0) * 1.15),
            min(1.0, (color_bgr[1] / 255.0) * 1.15),
            min(1.0, (color_bgr[0] / 255.0) * 1.15),
        )
    else:
        color_rgb = (color_bgr[2] / 255.0, color_bgr[1] / 255.0, color_bgr[0] / 255.0)
    return _colored_mesh(world_vertices, triangles, color_rgb=color_rgb)


def _room_mesh_arrays(
    *,
    room_width_m: float,
    room_depth_m: float,
    room_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wall_thickness = 0.05
    beam_thickness = 0.06
    room_meshes = [
        _colored_mesh(
            _box_mesh_arrays((room_width_m, wall_thickness, room_depth_m))[0]
            + np.asarray((0.0, -wall_thickness * 0.5, room_depth_m * 0.5), dtype=np.float64),
            _box_mesh_arrays((room_width_m, wall_thickness, room_depth_m))[1],
            color_rgb=(0.38, 0.37, 0.35),
        ),
        _colored_mesh(
            _box_mesh_arrays((room_width_m, room_height_m, wall_thickness))[0]
            + np.asarray((0.0, room_height_m * 0.5, room_depth_m), dtype=np.float64),
            _box_mesh_arrays((room_width_m, room_height_m, wall_thickness))[1],
            color_rgb=(0.56, 0.55, 0.58),
        ),
        _colored_mesh(
            _box_mesh_arrays((wall_thickness, room_height_m, room_depth_m))[0]
            + np.asarray((-room_width_m * 0.5, room_height_m * 0.5, room_depth_m * 0.5), dtype=np.float64),
            _box_mesh_arrays((wall_thickness, room_height_m, room_depth_m))[1],
            color_rgb=(0.50, 0.49, 0.47),
        ),
        _colored_mesh(
            _box_mesh_arrays((wall_thickness, room_height_m, room_depth_m))[0]
            + np.asarray((room_width_m * 0.5, room_height_m * 0.5, room_depth_m * 0.5), dtype=np.float64),
            _box_mesh_arrays((wall_thickness, room_height_m, room_depth_m))[1],
            color_rgb=(0.50, 0.49, 0.47),
        ),
        _colored_mesh(
            _box_mesh_arrays((room_width_m, beam_thickness, beam_thickness))[0]
            + np.asarray((0.0, room_height_m, 0.0), dtype=np.float64),
            _box_mesh_arrays((room_width_m, beam_thickness, beam_thickness))[1],
            color_rgb=(0.62, 0.63, 0.66),
        ),
        _colored_mesh(
            _box_mesh_arrays((room_width_m, beam_thickness, beam_thickness))[0]
            + np.asarray((0.0, room_height_m, room_depth_m), dtype=np.float64),
            _box_mesh_arrays((room_width_m, beam_thickness, beam_thickness))[1],
            color_rgb=(0.62, 0.63, 0.66),
        ),
        _colored_mesh(
            _box_mesh_arrays((beam_thickness, beam_thickness, room_depth_m))[0]
            + np.asarray((-room_width_m * 0.5, room_height_m, room_depth_m * 0.5), dtype=np.float64),
            _box_mesh_arrays((beam_thickness, beam_thickness, room_depth_m))[1],
            color_rgb=(0.62, 0.63, 0.66),
        ),
        _colored_mesh(
            _box_mesh_arrays((beam_thickness, beam_thickness, room_depth_m))[0]
            + np.asarray((room_width_m * 0.5, room_height_m, room_depth_m * 0.5), dtype=np.float64),
            _box_mesh_arrays((beam_thickness, beam_thickness, room_depth_m))[1],
            color_rgb=(0.62, 0.63, 0.66),
        ),
    ]
    return _combine_triangle_meshes(room_meshes)


def _outdoor_backdrop_meshes_for_scene(scene_spec) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    room_width, room_height, room_depth = (
        float(scene_spec.room_size_xyz[0]),
        float(scene_spec.room_size_xyz[1]),
        float(scene_spec.room_size_xyz[2]),
    )
    half_depth = room_depth * 0.5
    backdrop_z = half_depth + 0.72
    sky_width = room_width * 1.35
    sky_depth = 0.04

    backdrop_meshes = [
        _translated_colored_primitive_mesh(
            "BOX",
            (sky_width, room_height * 0.85, sky_depth),
            (0.0, room_height * 0.98, backdrop_z),
            color_rgb=(0.62, 0.80, 0.96),
        ),
        _translated_colored_primitive_mesh(
            "BOX",
            (sky_width, room_height * 0.45, sky_depth),
            (0.0, room_height * 0.48, backdrop_z - 0.01),
            color_rgb=(0.83, 0.92, 0.99),
        ),
        _translated_colored_primitive_mesh(
            "BOX",
            (sky_width, 0.95, sky_depth),
            (0.0, 0.18, backdrop_z - 0.02),
            color_rgb=(0.32, 0.58, 0.28),
        ),
    ]

    tree_layout = (
        (-2.25, 1.55, 1.25, 1.10, 0.28),
        (-0.65, 1.72, 1.45, 1.25, 0.30),
        (1.35, 1.62, 1.30, 1.15, 0.28),
        (2.55, 1.48, 1.10, 1.00, 0.26),
    )
    for tree_x, canopy_y, canopy_w, canopy_h, canopy_d in tree_layout:
        backdrop_meshes.append(
            _translated_colored_primitive_mesh(
                "UV_SPHERE",
                (canopy_w, canopy_h, canopy_d),
                (tree_x, canopy_y, backdrop_z - 0.08),
                color_rgb=(0.22, 0.42, 0.18),
            )
        )
        backdrop_meshes.append(
            _translated_colored_primitive_mesh(
                "BOX",
                (0.14, canopy_y - 0.28, 0.10),
                (tree_x, (canopy_y - 0.28) * 0.5, backdrop_z - 0.08),
                color_rgb=(0.38, 0.24, 0.12),
            )
        )
    return backdrop_meshes


def _camera_marker_lines(
    *,
    rig_x: float,
    rig_y: float,
    rig_z: float,
    yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.45],
            [-0.18, 0.10, 0.25],
            [0.18, 0.10, 0.25],
            [-0.18, -0.10, 0.25],
            [0.18, -0.10, 0.25],
        ],
        dtype=np.float64,
    )
    rotation = _rotation_matrix_y(float(yaw_deg))
    world_points = (local_points @ rotation.T) + np.asarray((rig_x, rig_y, rig_z), dtype=np.float64)
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [2, 3],
            [3, 5],
            [5, 4],
            [4, 2],
        ],
        dtype=np.int32,
    )
    colors = np.tile(np.asarray([[0.82, 0.90, 1.0]], dtype=np.float64), (lines.shape[0], 1))
    return world_points, lines, colors


class Open3DEnvironmentViewer:
    def __init__(
        self,
        window_name: str = "Environment Model",
        o3d_module=None,
        layout_primary_width: int = 640,
        layout_primary_height: int = 360,
    ) -> None:
        if o3d_module is None:
            try:
                import open3d as o3d
            except ImportError as exc:  # pragma: no cover - depends on local install.
                raise RuntimeError("open3d is required to render the environment model") from exc
        else:
            o3d = o3d_module

        self._o3d = o3d
        self._vis = o3d.visualization.Visualizer()
        window_left, window_top = runtime_window_position(
            window_name,
            primary_width=layout_primary_width,
            primary_height=layout_primary_height,
        )
        try:
            window_created = self._vis.create_window(
                window_name=window_name,
                width=960,
                height=720,
                left=window_left,
                top=window_top,
            )
        except TypeError:
            window_created = self._vis.create_window(window_name=window_name, width=960, height=720)
        if not window_created:
            raise RuntimeError("failed to create Open3D environment window")
        self._room_mesh = o3d.geometry.TriangleMesh()
        self._object_mesh = o3d.geometry.TriangleMesh()
        self._camera_lines = o3d.geometry.LineSet()
        self._preview_mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._has_fitted_view = False
        self._vis.add_geometry(self._room_mesh)
        self._vis.add_geometry(self._object_mesh)
        self._vis.add_geometry(self._camera_lines)
        self._vis.poll_events()
        self._vis.update_renderer()
        render_option = self._vis.get_render_option()
        if render_option is not None:
            render_option.background_color = np.array([0.05, 0.05, 0.08], dtype=np.float64)
            show_back_face = getattr(render_option, "mesh_show_back_face", None)
            if show_back_face is not None:
                render_option.mesh_show_back_face = True

    def update(self, scenario_state, *_unused, **_unused_kwargs) -> bool:
        scene_spec = getattr(scenario_state, "scene_spec", None)
        robot_pose = getattr(scenario_state, "robot_pose", None)
        if scene_spec is not None:
            scene_components = build_scene_mesh_components(scene_spec)
            room_component_meshes = [
                (component.vertices_xyz, component.triangles, component.vertex_colors)
                for component in scene_components
                if component.semantic_label in {"floor", "wall", "window_frame", "ceiling"}
            ]
            object_component_meshes = [
                (component.vertices_xyz, component.triangles, component.vertex_colors)
                for component in scene_components
                if component.semantic_label not in {"floor", "wall", "window_frame", "ceiling", "glass"}
            ]
            backdrop_meshes = [] if getattr(scene_spec, "blend_file_path", None) else _outdoor_backdrop_meshes_for_scene(scene_spec)
            room_vertices, room_triangles, room_colors = _combine_triangle_meshes(room_component_meshes + backdrop_meshes)
            object_vertices, object_triangles, object_colors = _combine_triangle_meshes(object_component_meshes)
            rig_x = float(getattr(robot_pose, "x", 0.0))
            rig_y = float(getattr(robot_pose, "y", 1.25))
            rig_z = float(getattr(robot_pose, "z", 0.0))
            yaw_deg = float(getattr(robot_pose, "yaw_deg", 0.0))
        else:
            room_width_m = float(getattr(scenario_state, "room_width_m", 0.0))
            room_depth_m = float(getattr(scenario_state, "room_depth_m", 0.0))
            room_height_m = float(getattr(scenario_state, "room_height_m", 0.0))
            room_vertices, room_triangles, room_colors = _room_mesh_arrays(
                room_width_m=max(room_width_m, 1.0),
                room_depth_m=max(room_depth_m, 1.0),
                room_height_m=max(room_height_m, 1.0),
            )
            object_meshes = [
                _object_mesh_arrays(
                    dict(obj),
                    o3d_module=self._o3d,
                    preview_mesh_cache=self._preview_mesh_cache,
                )
                for obj in tuple(getattr(scenario_state, "environment_objects", ()) or ())
            ]
            object_vertices, object_triangles, object_colors = _combine_triangle_meshes(object_meshes)
            rig_x = float(getattr(scenario_state, "rig_x", 0.0))
            rig_y = float(getattr(scenario_state, "rig_y", 1.6))
            rig_z = float(getattr(scenario_state, "rig_z", 0.0))
            yaw_deg = float(getattr(scenario_state, "yaw_deg", 0.0))

        self._room_mesh.vertices = self._o3d.utility.Vector3dVector(
            _display_points_for_environment_view(room_vertices)
        )
        self._room_mesh.triangles = self._o3d.utility.Vector3iVector(room_triangles)
        self._room_mesh.vertex_colors = self._o3d.utility.Vector3dVector(room_colors)
        compute_room_normals = getattr(self._room_mesh, "compute_vertex_normals", None)
        if callable(compute_room_normals):
            compute_room_normals()
        self._vis.update_geometry(self._room_mesh)

        self._object_mesh.vertices = self._o3d.utility.Vector3dVector(
            _display_points_for_environment_view(object_vertices)
        )
        self._object_mesh.triangles = self._o3d.utility.Vector3iVector(object_triangles)
        self._object_mesh.vertex_colors = self._o3d.utility.Vector3dVector(object_colors)
        compute_object_normals = getattr(self._object_mesh, "compute_vertex_normals", None)
        if callable(compute_object_normals):
            compute_object_normals()
        self._vis.update_geometry(self._object_mesh)

        camera_points, camera_lines, camera_colors = _camera_marker_lines(
            rig_x=rig_x,
            rig_y=rig_y,
            rig_z=rig_z,
            yaw_deg=yaw_deg,
        )
        self._camera_lines.points = self._o3d.utility.Vector3dVector(
            _display_points_for_environment_view(camera_points)
        )
        self._camera_lines.lines = self._o3d.utility.Vector2iVector(camera_lines)
        self._camera_lines.colors = self._o3d.utility.Vector3dVector(camera_colors)
        self._vis.update_geometry(self._camera_lines)

        if not self._has_fitted_view and (room_vertices.size > 0 or object_vertices.size > 0):
            self._vis.reset_view_point(True)
            get_view_control = getattr(self._vis, "get_view_control", None)
            if callable(get_view_control):
                control = get_view_control()
                _apply_open3d_view_control(
                    control,
                    lookat=np.array([0.0, 1.1, 0.0], dtype=np.float64),
                    front=np.array([0.52, -0.26, -0.81], dtype=np.float64),
                    up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                    zoom=0.72,
                )
            self._has_fitted_view = True
        is_active = self._vis.poll_events()
        self._vis.update_renderer()
        return bool(is_active)

    def close(self) -> None:
        destroy_window = getattr(self._vis, "destroy_window", None)
        if callable(destroy_window):
            destroy_window()


class Open3DMeshViewer:
    def __init__(
        self,
        window_name: str = "3D Reconstruction",
        o3d_module=None,
        layout_primary_width: int = 640,
        layout_primary_height: int = 360,
    ) -> None:
        if o3d_module is None:
            try:
                import open3d as o3d
            except ImportError as exc:  # pragma: no cover - depends on local install.
                raise RuntimeError("open3d is required to render the 3D mesh") from exc
        else:
            o3d = o3d_module

        self._o3d = o3d
        self._vis = o3d.visualization.Visualizer()
        window_left, window_top = runtime_window_position(
            window_name,
            primary_width=layout_primary_width,
            primary_height=layout_primary_height,
        )
        try:
            window_created = self._vis.create_window(
                window_name=window_name,
                width=960,
                height=720,
                left=window_left,
                top=window_top,
            )
        except TypeError:
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
            get_view_control = getattr(self._vis, "get_view_control", None)
            if callable(get_view_control):
                control = get_view_control()
                bounds_min = display_vertices_xyz.min(axis=0)
                bounds_max = display_vertices_xyz.max(axis=0)
                mesh_center_display = ((bounds_min + bounds_max) * 0.5).astype(np.float64, copy=False)
                _apply_open3d_view_control(
                    control,
                    lookat=mesh_center_display,
                    front=np.array([-0.58, 0.48, 0.66], dtype=np.float64),
                    up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                    zoom=0.68,
                )
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
