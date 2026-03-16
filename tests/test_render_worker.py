from __future__ import annotations

import time

import numpy as np

from obj_recog.render_worker import (
    ReconstructionRenderWorker,
    _LatestRenderSnapshotMailbox,
)
from obj_recog.types import RenderSnapshot


def _snapshot(*, geometry_revision: int, color_revision: int | None = None) -> RenderSnapshot:
    return RenderSnapshot(
        mesh_vertices_xyz=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        mesh_triangles=np.array([[0, 0, 0]], dtype=np.int32),
        mesh_vertex_colors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        mesh_geometry_revision=geometry_revision,
        mesh_color_revision=(geometry_revision if color_revision is None else color_revision),
    )


def test_latest_render_snapshot_mailbox_keeps_only_latest_snapshot() -> None:
    mailbox = _LatestRenderSnapshotMailbox()
    first = _snapshot(geometry_revision=1)
    second = _snapshot(geometry_revision=2)

    mailbox.publish(first)
    mailbox.publish(second)

    latest = mailbox.take_latest()

    assert latest is second
    assert mailbox.dropped_count() == 1
    assert mailbox.take_latest() is None


class _FakeRenderViewer:
    instances: list["_FakeRenderViewer"] = []

    def __init__(self, **_kwargs) -> None:
        self.snapshots: list[RenderSnapshot] = []
        self.tick_count = 0
        self.closed = False
        self.mode = "empty"
        _FakeRenderViewer.instances.append(self)

    def apply_render_snapshot(self, snapshot: RenderSnapshot) -> None:
        self.snapshots.append(snapshot)
        self.mode = "full_mesh" if snapshot.mesh_triangles.size > 0 else "empty"

    def tick(self) -> bool:
        self.tick_count += 1
        return True

    def current_render_mode(self) -> str:
        return self.mode

    def close(self) -> None:
        self.closed = True


def test_reconstruction_render_worker_ticks_without_waiting_for_new_snapshots() -> None:
    _FakeRenderViewer.instances.clear()
    worker = ReconstructionRenderWorker(
        viewer_factory=_FakeRenderViewer,
        tick_hz=120.0,
    )
    try:
        worker.publish(_snapshot(geometry_revision=1))
        worker.publish(_snapshot(geometry_revision=2))
        time.sleep(0.05)

        viewer = _FakeRenderViewer.instances[-1]

        assert worker.is_active() is True
        assert viewer.tick_count >= 3
        assert viewer.snapshots[-1].mesh_geometry_revision == 2
        assert len(viewer.snapshots) <= 2
    finally:
        worker.close()
        assert _FakeRenderViewer.instances[-1].closed is True
