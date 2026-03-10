from __future__ import annotations

import numpy as np

from obj_recog.reconstruct import CameraIntrinsics
from obj_recog.types import Detection, PanopticSegment


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=120.0, fy=120.0, cx=60.0, cy=60.0)


def _pose_world(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return pose


def _depth_map(value: float = 2.0) -> np.ndarray:
    return np.full((120, 120), value, dtype=np.float32)


def _mask(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros((120, 120), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_object_instance_tracker_keeps_same_node_for_matching_detection() -> None:
    from obj_recog.scene_graph import ObjectInstanceTracker

    tracker = ObjectInstanceTracker()
    depth_map = _depth_map()
    intrinsics = _intrinsics()

    detections_a = [
        Detection(
            xyxy=(30, 30, 68, 90),
            class_id=56,
            label="chair",
            confidence=0.93,
            color=(255, 0, 0),
        )
    ]
    detections_b = [
        Detection(
            xyxy=(33, 31, 71, 92),
            class_id=56,
            label="chair",
            confidence=0.90,
            color=(255, 0, 0),
        )
    ]

    tracked_a = tracker.update(
        detections_a,
        camera_pose_world=_pose_world(),
        depth_map=depth_map,
        intrinsics=intrinsics,
    )
    tracked_b = tracker.update(
        detections_b,
        camera_pose_world=_pose_world(),
        depth_map=depth_map,
        intrinsics=intrinsics,
    )

    assert len(tracked_a) == 1
    assert len(tracked_b) == 1
    assert tracked_a[0].node_id == tracked_b[0].node_id
    assert tracked_b[0].label == "chair"
    assert tracked_b[0].state == "visible"
    assert tracked_b[0].world_centroid is not None


def test_scene_graph_memory_creates_ego_object_and_segment_relations() -> None:
    from obj_recog.scene_graph import SceneGraphMemory, SceneGraphQueryService

    memory = SceneGraphMemory()
    snapshot = memory.update(
        frame_index=1,
        detections=[
            Detection(
                xyxy=(34, 52, 90, 104),
                class_id=60,
                label="dining table",
                confidence=0.94,
                color=(255, 99, 71),
            ),
            Detection(
                xyxy=(52, 42, 68, 57),
                class_id=41,
                label="cup",
                confidence=0.89,
                color=(64, 196, 255),
            ),
        ],
        segments=[
            PanopticSegment(
                segment_id=1,
                label_id=3,
                label="floor",
                color_rgb=(255, 179, 71),
                mask=_mask(0, 82, 120, 120),
                area_pixels=120 * 38,
            )
        ],
        depth_map=_depth_map(),
        intrinsics=_intrinsics(),
        camera_pose_world=_pose_world(),
        slam_tracking_state="TRACKING",
    )

    labels = {(node.type, node.label) for node in snapshot.nodes}
    assert ("ego", "camera") in labels
    assert ("object", "dining table") in labels
    assert ("object", "cup") in labels
    assert ("segment", "floor") in labels

    relations = {(edge.relation, snapshot.node(edge.source).label, snapshot.node(edge.target).label) for edge in snapshot.edges}
    assert ("front", "camera", "dining table") in relations
    assert ("front", "camera", "cup") in relations
    assert ("on", "cup", "dining table") in relations
    assert ("near", "cup", "dining table") in relations
    assert ("on", "dining table", "floor") in relations

    query = SceneGraphQueryService(snapshot)
    on_table = {node.label for node in query.objects_on("dining table")}
    assert on_table == {"cup"}


def test_scene_graph_memory_transitions_visible_to_occluded_to_lost() -> None:
    from obj_recog.scene_graph import SceneGraphMemory

    memory = SceneGraphMemory(occlusion_ttl_frames=15)
    depth_map = _depth_map()
    intrinsics = _intrinsics()

    snapshot = memory.update(
        frame_index=1,
        detections=[
            Detection(
                xyxy=(36, 40, 78, 98),
                class_id=56,
                label="chair",
                confidence=0.91,
                color=(255, 0, 0),
            )
        ],
        segments=[],
        depth_map=depth_map,
        intrinsics=intrinsics,
        camera_pose_world=_pose_world(),
        slam_tracking_state="TRACKING",
    )
    chair_id = next(node.id for node in snapshot.nodes if node.label == "chair")

    for frame_index in range(2, 16):
        snapshot = memory.update(
            frame_index=frame_index,
            detections=[],
            segments=[],
            depth_map=depth_map,
            intrinsics=intrinsics,
            camera_pose_world=_pose_world(),
            slam_tracking_state="TRACKING",
        )

    assert snapshot.node(chair_id).state == "occluded"

    snapshot = memory.update(
        frame_index=17,
        detections=[],
        segments=[],
        depth_map=depth_map,
        intrinsics=intrinsics,
        camera_pose_world=_pose_world(),
        slam_tracking_state="TRACKING",
    )

    assert snapshot.node(chair_id).state == "lost"


def test_scene_graph_memory_keeps_last_world_centroid_when_slam_is_not_tracking() -> None:
    from obj_recog.scene_graph import SceneGraphMemory

    memory = SceneGraphMemory()
    depth_map = _depth_map()
    intrinsics = _intrinsics()

    snapshot = memory.update(
        frame_index=1,
        detections=[
            Detection(
                xyxy=(52, 42, 68, 57),
                class_id=41,
                label="cup",
                confidence=0.89,
                color=(64, 196, 255),
            )
        ],
        segments=[],
        depth_map=depth_map,
        intrinsics=intrinsics,
        camera_pose_world=_pose_world(),
        slam_tracking_state="TRACKING",
    )
    cup_id = next(node.id for node in snapshot.nodes if node.label == "cup")
    original_centroid = np.asarray(snapshot.node(cup_id).world_centroid, dtype=np.float32)

    snapshot = memory.update(
        frame_index=2,
        detections=[
            Detection(
                xyxy=(52, 42, 68, 57),
                class_id=41,
                label="cup",
                confidence=0.89,
                color=(64, 196, 255),
            )
        ],
        segments=[],
        depth_map=depth_map,
        intrinsics=intrinsics,
        camera_pose_world=_pose_world(tx=1.0),
        slam_tracking_state="INITIALIZING",
    )

    assert np.allclose(snapshot.node(cup_id).world_centroid, original_centroid)
    assert snapshot.node(cup_id).last_seen_direction == "front"


def test_ego_direction_bucket_updates_with_camera_translation() -> None:
    from obj_recog.scene_graph import bucket_ego_direction

    world_centroid = np.array([0.0, 0.0, 2.0], dtype=np.float32)

    assert bucket_ego_direction(world_centroid, _pose_world()) == "front"
    assert bucket_ego_direction(world_centroid, _pose_world(tx=1.0)) == "front-left"


def test_query_service_returns_right_side_objects_and_last_seen_direction() -> None:
    from obj_recog.scene_graph import SceneGraphMemory, SceneGraphQueryService

    memory = SceneGraphMemory()
    depth_map = _depth_map()
    intrinsics = _intrinsics()

    snapshot = memory.update(
        frame_index=1,
        detections=[
            Detection(
                xyxy=(10, 40, 30, 96),
                class_id=56,
                label="chair",
                confidence=0.88,
                color=(255, 0, 0),
            ),
            Detection(
                xyxy=(72, 38, 108, 96),
                class_id=39,
                label="bottle",
                confidence=0.84,
                color=(0, 255, 0),
            ),
        ],
        segments=[],
        depth_map=depth_map,
        intrinsics=intrinsics,
        camera_pose_world=_pose_world(),
        slam_tracking_state="TRACKING",
    )

    query = SceneGraphQueryService(snapshot)
    right_labels = {node.label for node in query.objects_right_of_ego()}
    assert right_labels == {"bottle"}

    snapshot = memory.update(
        frame_index=2,
        detections=[],
        segments=[],
        depth_map=depth_map,
        intrinsics=intrinsics,
        camera_pose_world=_pose_world(),
        slam_tracking_state="TRACKING",
    )
    query = SceneGraphQueryService(snapshot)
    assert query.last_seen_direction("bottle") == "front-right"
