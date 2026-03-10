from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re

import networkx as nx
import numpy as np

from obj_recog.reconstruct import CameraIntrinsics, back_project_pixels, transform_points
from obj_recog.slam_bridge import TRACKING_OK_STATES
from obj_recog.types import Detection, PanopticSegment


_DYNAMIC_RELATION_SOURCE = "detection"
_STRUCTURAL_LABEL_MAP = {
    "wall": "wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "table": "table",
    "desk": "table",
    "shelf": "shelf",
    "cabinet": "cabinet",
    "sofa": "sofa",
    "couch": "sofa",
    "bed": "bed",
    "window": "window-like",
    "door": "window-like",
}


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    type: str
    label: str
    state: str
    confidence: float
    world_centroid: np.ndarray | None
    last_seen_frame: int
    last_seen_direction: str | None
    source_track_id: int | None


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    confidence: float
    last_updated_frame: int
    distance_bucket: str | None
    source_kind: str


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    node_id: str
    label: str
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    color_rgb: tuple[int, int, int]
    world_centroid: np.ndarray | None
    state: str
    source_track_id: int


@dataclass(frozen=True, slots=True)
class TrackedSegment:
    node_id: str
    label: str
    mask: np.ndarray
    confidence: float
    world_centroid: np.ndarray | None
    state: str
    source_track_id: int


@dataclass(frozen=True, slots=True)
class SceneGraphSnapshot:
    frame_index: int
    camera_pose_world: np.ndarray
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    visible_node_ids: tuple[str, ...]
    visible_edge_keys: tuple[tuple[str, str, str], ...]

    def node(self, node_id: str) -> GraphNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    @property
    def visible_nodes(self) -> tuple[GraphNode, ...]:
        visible = set(self.visible_node_ids)
        return tuple(node for node in self.nodes if node.id in visible)

    @property
    def visible_edges(self) -> tuple[GraphEdge, ...]:
        visible = set(self.visible_edge_keys)
        return tuple(
            edge for edge in self.edges if (edge.source, edge.target, edge.relation) in visible
        )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "item"


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = max(area_a + area_b - intersection, 1.0)
    return intersection / union


def _camera_point(world_centroid: np.ndarray, camera_pose_world: np.ndarray) -> np.ndarray:
    world_centroid = np.asarray(world_centroid, dtype=np.float32).reshape(3)
    pose_inv = np.linalg.inv(np.asarray(camera_pose_world, dtype=np.float32))
    homogeneous = np.concatenate((world_centroid, np.array([1.0], dtype=np.float32)))
    camera = pose_inv @ homogeneous
    return camera[:3].astype(np.float32, copy=False)


def bucket_ego_direction(world_centroid: np.ndarray, camera_pose_world: np.ndarray) -> str:
    camera_xyz = _camera_point(world_centroid, camera_pose_world)
    x = float(camera_xyz[0])
    z = float(camera_xyz[2])
    if z <= 0.0:
        return "behind"

    angle = math.atan2(x, z)
    if angle <= -0.75:
        return "left"
    if angle <= -0.20:
        return "front-left"
    if angle < 0.20:
        return "front"
    if angle < 0.75:
        return "front-right"
    return "right"


def bucket_distance(world_centroid: np.ndarray, camera_pose_world: np.ndarray) -> str:
    camera_xyz = _camera_point(world_centroid, camera_pose_world)
    distance = float(np.linalg.norm(camera_xyz))
    if distance <= 1.5:
        return "near"
    if distance <= 3.5:
        return "mid"
    return "far"


def _sample_box_pixels(bbox_xyxy: tuple[int, int, int, int], *, stride: int = 4) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    xs = np.arange(x1, max(x1 + 1, x2), stride, dtype=np.int32)
    ys = np.arange(y1, max(y1 + 1, y2), stride, dtype=np.int32)
    if xs.size == 0 or ys.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=1)


def _sample_mask_pixels(mask: np.ndarray, *, max_samples: int = 512) -> np.ndarray:
    pixel_yx = np.argwhere(mask)
    if pixel_yx.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    if pixel_yx.shape[0] > max_samples:
        keep = np.linspace(0, pixel_yx.shape[0] - 1, max_samples, dtype=np.int32)
        pixel_yx = pixel_yx[keep]
    return np.stack((pixel_yx[:, 1], pixel_yx[:, 0]), axis=1).astype(np.int32, copy=False)


def _world_centroid_from_pixels(
    pixel_xy: np.ndarray,
    *,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_pose_world: np.ndarray,
) -> np.ndarray | None:
    if pixel_xy.size == 0:
        return None
    pixel_xy = np.asarray(pixel_xy, dtype=np.int32).reshape(-1, 2)
    valid = (
        (pixel_xy[:, 0] >= 0)
        & (pixel_xy[:, 0] < depth_map.shape[1])
        & (pixel_xy[:, 1] >= 0)
        & (pixel_xy[:, 1] < depth_map.shape[0])
    )
    pixel_xy = pixel_xy[valid]
    if pixel_xy.size == 0:
        return None
    depth_values = depth_map[pixel_xy[:, 1], pixel_xy[:, 0]]
    depth_values = np.asarray(depth_values, dtype=np.float32)
    valid_depth = np.isfinite(depth_values) & (depth_values > 0.05) & (depth_values <= 6.0)
    if int(valid_depth.sum()) < 4:
        return None
    pixel_xy = pixel_xy[valid_depth]
    depth_values = depth_values[valid_depth]
    local_xyz = back_project_pixels(pixel_xy, depth_values, intrinsics)
    world_xyz = transform_points(local_xyz, camera_pose_world)
    if world_xyz.shape[0] == 0:
        return None
    return np.median(world_xyz, axis=0).astype(np.float32, copy=False)


def _detection_world_centroid(
    detection: Detection,
    *,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_pose_world: np.ndarray,
) -> np.ndarray | None:
    pixel_xy = _sample_box_pixels(detection.xyxy)
    return _world_centroid_from_pixels(
        pixel_xy,
        depth_map=depth_map,
        intrinsics=intrinsics,
        camera_pose_world=camera_pose_world,
    )


def _node_camera_distance(node: GraphNode, camera_pose_world: np.ndarray) -> float:
    if node.world_centroid is None:
        return float("inf")
    camera_xyz = _camera_point(node.world_centroid, camera_pose_world)
    return float(np.linalg.norm(camera_xyz))


def _visible_node_priority(node: GraphNode, camera_pose_world: np.ndarray) -> tuple[float, ...]:
    return (
        -float(node.last_seen_frame),
        _node_camera_distance(node, camera_pose_world),
        -float(node.confidence),
        0.0 if node.type == "object" else 1.0,
    )


def _segment_world_centroid(
    segment: PanopticSegment,
    *,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_pose_world: np.ndarray,
) -> np.ndarray | None:
    pixel_xy = _sample_mask_pixels(segment.mask)
    return _world_centroid_from_pixels(
        pixel_xy,
        depth_map=depth_map,
        intrinsics=intrinsics,
        camera_pose_world=camera_pose_world,
    )


def _coarse_structural_label(raw_label: str) -> str | None:
    normalized = raw_label.lower().strip()
    for needle, coarse in _STRUCTURAL_LABEL_MAP.items():
        if needle in normalized:
            return coarse
    return None


class ObjectInstanceTracker:
    def __init__(self, *, iou_threshold: float = 0.15, world_distance_threshold: float = 1.0) -> None:
        self._iou_threshold = float(iou_threshold)
        self._world_distance_threshold = float(world_distance_threshold)
        self._tracks: dict[str, TrackedDetection] = {}
        self._next_track_id = 1
        self._frame_counter = 0

    def update(
        self,
        detections: list[Detection],
        *,
        camera_pose_world: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
        frame_index: int | None = None,
        allow_localization: bool = True,
    ) -> list[TrackedDetection]:
        if frame_index is None:
            self._frame_counter += 1
            frame_index = self._frame_counter

        matched: set[str] = set()
        tracked: list[TrackedDetection] = []
        for detection in detections:
            world_centroid = None
            if allow_localization:
                world_centroid = _detection_world_centroid(
                    detection,
                    depth_map=depth_map,
                    intrinsics=intrinsics,
                    camera_pose_world=camera_pose_world,
                )
            best_track_id = None
            best_score = -1.0
            for existing in self._tracks.values():
                if existing.label != detection.label or existing.node_id in matched:
                    continue
                iou = _bbox_iou(existing.bbox_xyxy, detection.xyxy)
                distance_score = 0.0
                if world_centroid is not None and existing.world_centroid is not None:
                    distance = float(np.linalg.norm(world_centroid - existing.world_centroid))
                    if distance <= self._world_distance_threshold:
                        distance_score = 1.0 - (distance / max(self._world_distance_threshold, 1e-6))
                if iou < self._iou_threshold and distance_score <= 0.0:
                    continue
                score = (iou * 2.0) + distance_score
                if score > best_score:
                    best_score = score
                    best_track_id = existing.node_id

            if best_track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                node_id = f"obj_{_slugify(detection.label)}_{track_id}"
            else:
                node_id = best_track_id
                track_id = self._tracks[node_id].source_track_id

            tracked_detection = TrackedDetection(
                node_id=node_id,
                label=detection.label,
                bbox_xyxy=detection.xyxy,
                confidence=float(detection.confidence),
                color_rgb=detection.color,
                world_centroid=world_centroid,
                state="visible",
                source_track_id=track_id,
            )
            self._tracks[node_id] = tracked_detection
            matched.add(node_id)
            tracked.append(tracked_detection)
        return tracked


class SceneGraphMemory:
    def __init__(
        self,
        *,
        graph_max_visible_nodes: int = 32,
        graph_relation_smoothing_frames: int = 15,
        occlusion_ttl_frames: int = 15,
        object_tracker: ObjectInstanceTracker | None = None,
    ) -> None:
        self._graph = nx.MultiDiGraph()
        self._graph_max_visible_nodes = int(graph_max_visible_nodes)
        self._graph_relation_smoothing_frames = int(graph_relation_smoothing_frames)
        self._occlusion_ttl_frames = int(occlusion_ttl_frames)
        self._object_tracker = object_tracker or ObjectInstanceTracker()
        self._segment_counter = 1
        self._segment_nodes: dict[str, TrackedSegment] = {}
        self._edge_confidence: dict[tuple[str, str, str], float] = {}
        self._ensure_ego_node()

    def _ensure_ego_node(self) -> None:
        ego = GraphNode(
            id="ego",
            type="ego",
            label="camera",
            state="visible",
            confidence=1.0,
            world_centroid=np.zeros(3, dtype=np.float32),
            last_seen_frame=0,
            last_seen_direction="front",
            source_track_id=None,
        )
        self._graph.add_node("ego", data=ego)

    def _get_node(self, node_id: str) -> GraphNode:
        return self._graph.nodes[node_id]["data"]

    def _set_node(self, node: GraphNode) -> None:
        self._graph.add_node(node.id, data=node)

    def _match_segment_node(self, label: str, world_centroid: np.ndarray | None) -> str:
        best_node_id = None
        best_distance = float("inf")
        for node_id, existing in self._segment_nodes.items():
            if existing.label != label:
                continue
            if world_centroid is None or existing.world_centroid is None:
                if best_node_id is None:
                    best_node_id = node_id
                continue
            distance = float(np.linalg.norm(world_centroid - existing.world_centroid))
            if distance < best_distance and distance <= 1.5:
                best_distance = distance
                best_node_id = node_id
        if best_node_id is None:
            best_node_id = f"seg_{_slugify(label)}_{self._segment_counter}"
            self._segment_counter += 1
        return best_node_id

    def _track_segments(
        self,
        segments: list[PanopticSegment],
        *,
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
        camera_pose_world: np.ndarray,
        allow_localization: bool,
    ) -> list[TrackedSegment]:
        tracked: list[TrackedSegment] = []
        for segment in segments:
            label = _coarse_structural_label(segment.label)
            if label is None:
                continue
            world_centroid = None
            if allow_localization:
                world_centroid = _segment_world_centroid(
                    segment,
                    depth_map=depth_map,
                    intrinsics=intrinsics,
                    camera_pose_world=camera_pose_world,
                )
            node_id = self._match_segment_node(label, world_centroid)
            previous = self._segment_nodes.get(node_id)
            track_id = previous.source_track_id if previous is not None else self._segment_counter + 1000
            tracked_segment = TrackedSegment(
                node_id=node_id,
                label=label,
                mask=segment.mask,
                confidence=1.0,
                world_centroid=world_centroid,
                state="visible",
                source_track_id=int(track_id),
            )
            self._segment_nodes[node_id] = tracked_segment
            tracked.append(tracked_segment)
        return tracked

    def _update_states(self, *, frame_index: int, visible_node_ids: set[str]) -> None:
        for node_id, attrs in list(self._graph.nodes.items()):
            if node_id == "ego":
                continue
            node: GraphNode = attrs["data"]
            if node_id in visible_node_ids:
                continue
            age = frame_index - int(node.last_seen_frame)
            state = "occluded" if age <= self._occlusion_ttl_frames else "lost"
            self._set_node(replace(node, state=state))

    def _update_node_from_tracked(
        self,
        tracked,
        *,
        frame_index: int,
        node_type: str,
        last_seen_direction: str | None,
    ) -> None:
        previous = self._graph.nodes.get(tracked.node_id, {}).get("data")
        previous_confidence = 0.0 if previous is None else float(previous.confidence)
        confidence = (previous_confidence * 0.6) + (float(tracked.confidence) * 0.4)
        world_centroid = tracked.world_centroid
        if world_centroid is None and previous is not None:
            world_centroid = previous.world_centroid
        resolved_direction = last_seen_direction
        if resolved_direction is None and previous is not None:
            resolved_direction = previous.last_seen_direction
        node = GraphNode(
            id=tracked.node_id,
            type=node_type,
            label=tracked.label,
            state="visible",
            confidence=confidence,
            world_centroid=world_centroid,
            last_seen_frame=frame_index,
            last_seen_direction=resolved_direction,
            source_track_id=tracked.source_track_id,
        )
        self._set_node(node)

    def _add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        *,
        frame_index: int,
        distance_bucket: str | None = None,
        source_kind: str = _DYNAMIC_RELATION_SOURCE,
        base_confidence: float = 0.9,
    ) -> None:
        key = (source, target, relation)
        confidence = (self._edge_confidence.get(key, base_confidence) * 0.6) + (base_confidence * 0.4)
        self._edge_confidence[key] = confidence
        edge = GraphEdge(
            source=source,
            target=target,
            relation=relation,
            confidence=confidence,
            last_updated_frame=frame_index,
            distance_bucket=distance_bucket,
            source_kind=source_kind,
        )
        self._graph.add_edge(source, target, key=relation, data=edge)

    def _rebuild_edges(
        self,
        *,
        frame_index: int,
        camera_pose_world: np.ndarray,
        tracked_objects: list[TrackedDetection],
        tracked_segments: list[TrackedSegment],
    ) -> None:
        self._graph.remove_edges_from(list(self._graph.edges(keys=True)))
        visible_objects = [item for item in tracked_objects if item.world_centroid is not None]
        visible_segments = [item for item in tracked_segments if item.world_centroid is not None]

        for item in (*visible_objects, *visible_segments):
            direction = bucket_ego_direction(item.world_centroid, camera_pose_world)
            distance_bucket = bucket_distance(item.world_centroid, camera_pose_world)
            self._add_edge(
                "ego",
                item.node_id,
                direction,
                frame_index=frame_index,
                distance_bucket=distance_bucket,
            )

        for source in visible_objects:
            for target in visible_objects:
                if source.node_id == target.node_id:
                    continue
                source_node = self._get_node(source.node_id)
                target_node = self._get_node(target.node_id)
                distance = float(np.linalg.norm(source_node.world_centroid - target_node.world_centroid))
                if distance <= 1.25:
                    self._add_edge(
                        source.node_id,
                        target.node_id,
                        "near",
                        frame_index=frame_index,
                        distance_bucket=bucket_distance(source_node.world_centroid, camera_pose_world),
                        base_confidence=0.75,
                    )
                if _bbox_inside(source.bbox_xyxy, target.bbox_xyxy):
                    self._add_edge(source.node_id, target.node_id, "inside", frame_index=frame_index, base_confidence=0.85)
                if _bbox_on(source.bbox_xyxy, target.bbox_xyxy):
                    self._add_edge(source.node_id, target.node_id, "on", frame_index=frame_index, base_confidence=0.85)

        for obj in tracked_objects:
            for seg in tracked_segments:
                obj_node = self._get_node(obj.node_id)
                if obj_node.world_centroid is None:
                    continue
                if _bottom_center_inside_mask(obj.bbox_xyxy, seg.mask):
                    self._add_edge(obj.node_id, seg.node_id, "on", frame_index=frame_index, base_confidence=0.8)
                if _center_inside_mask(obj.bbox_xyxy, seg.mask):
                    self._add_edge(obj.node_id, seg.node_id, "inside_region", frame_index=frame_index, base_confidence=0.7)
                if seg.label in {"wall", "window-like"} and _touches_mask(obj.bbox_xyxy, seg.mask):
                    self._add_edge(obj.node_id, seg.node_id, "attached_to", frame_index=frame_index, base_confidence=0.72)

    def update(
        self,
        *,
        frame_index: int,
        detections: list[Detection],
        segments: list[PanopticSegment],
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
        camera_pose_world: np.ndarray,
        slam_tracking_state: str,
    ) -> SceneGraphSnapshot:
        tracking_ok = slam_tracking_state in TRACKING_OK_STATES
        tracked_objects = self._object_tracker.update(
            detections,
            camera_pose_world=camera_pose_world,
            depth_map=depth_map,
            intrinsics=intrinsics,
            frame_index=frame_index,
            allow_localization=tracking_ok,
        )
        tracked_segments = self._track_segments(
            segments,
            depth_map=depth_map,
            intrinsics=intrinsics,
            camera_pose_world=camera_pose_world,
            allow_localization=tracking_ok,
        )

        visible_node_ids = {"ego"}
        for tracked in tracked_objects:
            direction = None
            if tracking_ok and tracked.world_centroid is not None:
                direction = bucket_ego_direction(tracked.world_centroid, camera_pose_world)
            self._update_node_from_tracked(
                tracked,
                frame_index=frame_index,
                node_type="object",
                last_seen_direction=direction,
            )
            visible_node_ids.add(tracked.node_id)

        for tracked in tracked_segments:
            direction = None
            if tracking_ok and tracked.world_centroid is not None:
                direction = bucket_ego_direction(tracked.world_centroid, camera_pose_world)
            self._update_node_from_tracked(
                tracked,
                frame_index=frame_index,
                node_type="segment",
                last_seen_direction=direction,
            )
            visible_node_ids.add(tracked.node_id)

        self._update_states(frame_index=frame_index, visible_node_ids=visible_node_ids)
        if tracking_ok:
            self._rebuild_edges(
                frame_index=frame_index,
                camera_pose_world=camera_pose_world,
                tracked_objects=tracked_objects,
                tracked_segments=tracked_segments,
            )
        else:
            self._graph.remove_edges_from(list(self._graph.edges(keys=True)))

        return self.snapshot(camera_pose_world=camera_pose_world, frame_index=frame_index)

    def snapshot(self, *, camera_pose_world: np.ndarray, frame_index: int) -> SceneGraphSnapshot:
        nodes = tuple(
            sorted(
                (attrs["data"] for _node_id, attrs in self._graph.nodes.items()),
                key=lambda node: (node.type != "ego", node.label, node.id),
            )
        )
        edges = tuple(
            sorted(
                (attrs["data"] for _source, _target, _key, attrs in self._graph.edges(keys=True, data=True)),
                key=lambda edge: (edge.source, edge.target, edge.relation),
            )
        )
        visible_node_candidates = [node for node in nodes if node.state == "visible"]
        ego_nodes = [node for node in visible_node_candidates if node.id == "ego"]
        non_ego_nodes = [node for node in visible_node_candidates if node.id != "ego"]
        prioritized_non_ego = sorted(
            non_ego_nodes,
            key=lambda node: (
                *_visible_node_priority(node, camera_pose_world),
                node.label,
                node.id,
            ),
        )
        visible_nodes = [node.id for node in ego_nodes]
        remaining_slots = max(0, self._graph_max_visible_nodes - len(visible_nodes))
        visible_nodes.extend(node.id for node in prioritized_non_ego[:remaining_slots])
        visible_node_id_set = set(visible_nodes)
        visible_edges = [
            (edge.source, edge.target, edge.relation)
            for edge in edges
            if edge.source in visible_node_id_set and edge.target in visible_node_id_set
        ]
        return SceneGraphSnapshot(
            frame_index=frame_index,
            camera_pose_world=np.asarray(camera_pose_world, dtype=np.float32).copy(),
            nodes=nodes,
            edges=edges,
            visible_node_ids=tuple(visible_nodes),
            visible_edge_keys=tuple(visible_edges),
        )


class SceneGraphQueryService:
    def __init__(self, snapshot: SceneGraphSnapshot) -> None:
        self._snapshot = snapshot

    def objects_right_of_ego(self) -> list[GraphNode]:
        right_relations = {"right", "front-right"}
        target_ids = {
            edge.target
            for edge in self._snapshot.visible_edges
            if edge.source == "ego" and edge.relation in right_relations
        }
        return [
            node
            for node in self._snapshot.visible_nodes
            if node.type == "object" and node.id in target_ids
        ]

    def last_seen_direction(self, label: str) -> str | None:
        matches = [node for node in self._snapshot.nodes if node.label == label]
        if not matches:
            return None
        latest = max(matches, key=lambda node: node.last_seen_frame)
        return latest.last_seen_direction

    def objects_on(self, label: str) -> list[GraphNode]:
        target_ids = {node.id for node in self._snapshot.nodes if node.label == label}
        source_ids = {
            edge.source
            for edge in self._snapshot.edges
            if edge.relation == "on" and edge.target in target_ids
        }
        return [node for node in self._snapshot.nodes if node.id in source_ids]


def _bbox_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _bbox_on(source: tuple[int, int, int, int], target: tuple[int, int, int, int]) -> bool:
    sx1, sy1, sx2, sy2 = source
    tx1, ty1, tx2, ty2 = target
    source_center_x = (sx1 + sx2) / 2.0
    source_bottom = sy2
    target_height = max(1.0, float(ty2 - ty1))
    within_width = tx1 <= source_center_x <= tx2
    vertical_band = (ty1 - (0.15 * target_height)) <= source_bottom <= (ty1 + (0.35 * target_height))
    return bool(within_width and vertical_band and source_bottom <= ty2)


def _center_inside_mask(bbox_xyxy: tuple[int, int, int, int], mask: np.ndarray) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    center_x = int(round((x1 + x2) / 2.0))
    center_y = int(round((y1 + y2) / 2.0))
    center_x = int(np.clip(center_x, 0, mask.shape[1] - 1))
    center_y = int(np.clip(center_y, 0, mask.shape[0] - 1))
    return bool(mask[center_y, center_x])


def _bottom_center_inside_mask(bbox_xyxy: tuple[int, int, int, int], mask: np.ndarray) -> bool:
    x1, _y1, x2, y2 = bbox_xyxy
    center_x = int(round((x1 + x2) / 2.0))
    bottom_y = int(np.clip(y2 - 1, 0, mask.shape[0] - 1))
    center_x = int(np.clip(center_x, 0, mask.shape[1] - 1))
    return bool(mask[bottom_y, center_x])


def _touches_mask(bbox_xyxy: tuple[int, int, int, int], mask: np.ndarray) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(np.clip(x1, 0, mask.shape[1] - 1))
    x2 = int(np.clip(x2 - 1, 0, mask.shape[1] - 1))
    y1 = int(np.clip(y1, 0, mask.shape[0] - 1))
    y2 = int(np.clip(y2 - 1, 0, mask.shape[0] - 1))
    vertical_left = mask[y1 : y2 + 1, x1]
    vertical_right = mask[y1 : y2 + 1, x2]
    horizontal_top = mask[y1, x1 : x2 + 1]
    horizontal_bottom = mask[y2, x1 : x2 + 1]
    return bool(
        np.any(vertical_left)
        or np.any(vertical_right)
        or np.any(horizontal_top)
        or np.any(horizontal_bottom)
    )
