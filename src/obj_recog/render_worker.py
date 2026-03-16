from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from obj_recog.types import FrameArtifacts, RenderSnapshot
from obj_recog.visualization import Open3DMeshViewer


@dataclass(slots=True)
class _MailboxState:
    latest_snapshot: RenderSnapshot | None = None
    dropped_count: int = 0


class _LatestRenderSnapshotMailbox:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = _MailboxState()

    def publish(self, snapshot: RenderSnapshot) -> None:
        with self._lock:
            if self._state.latest_snapshot is not None:
                self._state.dropped_count += 1
            self._state.latest_snapshot = snapshot

    def take_latest(self) -> RenderSnapshot | None:
        with self._lock:
            snapshot = self._state.latest_snapshot
            self._state.latest_snapshot = None
            return snapshot

    def dropped_count(self) -> int:
        with self._lock:
            return int(self._state.dropped_count)


def render_snapshot_from_artifacts(artifacts: FrameArtifacts) -> RenderSnapshot:
    return RenderSnapshot(
        mesh_vertices_xyz=np.asarray(artifacts.mesh_vertices_xyz, dtype=np.float32).copy(),
        mesh_triangles=np.asarray(artifacts.mesh_triangles, dtype=np.int32).copy(),
        mesh_vertex_colors=np.asarray(artifacts.mesh_vertex_colors, dtype=np.float32).copy(),
        scene_graph_snapshot=artifacts.scene_graph_snapshot,
        mesh_geometry_revision=artifacts.mesh_geometry_revision,
        mesh_color_revision=artifacts.mesh_color_revision,
    )


class ReconstructionRenderWorker:
    def __init__(
        self,
        *,
        window_name: str = "3D Reconstruction",
        viewer_factory=Open3DMeshViewer,
        layout_primary_width: int = 640,
        layout_primary_height: int = 360,
        debug_log=None,
        tick_hz: float = 60.0,
    ) -> None:
        self._window_name = str(window_name)
        self._viewer_factory = viewer_factory
        self._layout_primary_width = int(layout_primary_width)
        self._layout_primary_height = int(layout_primary_height)
        self._debug_log = debug_log or (lambda _message: None)
        self._tick_interval_sec = 1.0 / max(1.0, float(tick_hz))
        self._mailbox = _LatestRenderSnapshotMailbox()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._active = True
        self._error: Exception | None = None
        self._current_mode = "empty"
        self._geometry_update_count = 0
        self._color_update_count = 0
        self._last_geometry_revision: int | None = None
        self._last_color_revision: int | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="reconstruction-render-worker",
            daemon=True,
        )
        self._thread.start()

    def publish(self, snapshot: RenderSnapshot) -> bool:
        self._record_snapshot_revisions(snapshot)
        self._mailbox.publish(snapshot)
        self._wake_event.set()
        return self.is_active()

    def is_active(self) -> bool:
        with self._state_lock:
            return bool(self._active and self._error is None)

    def close(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        self._thread.join(timeout=2.0)

    def _record_snapshot_revisions(self, snapshot: RenderSnapshot) -> None:
        geometry_revision = snapshot.mesh_geometry_revision
        color_revision = snapshot.mesh_color_revision
        if geometry_revision is None or geometry_revision != self._last_geometry_revision:
            self._geometry_update_count += 1
            self._last_geometry_revision = geometry_revision
            self._last_color_revision = color_revision
            if color_revision is not None:
                self._color_update_count += 1
            return
        if color_revision is None or color_revision != self._last_color_revision:
            self._color_update_count += 1
            self._last_color_revision = color_revision

    def _set_state(
        self,
        *,
        active: bool | None = None,
        error: Exception | None = None,
        current_mode: str | None = None,
    ) -> None:
        with self._state_lock:
            if active is not None:
                self._active = bool(active)
            if error is not None:
                self._error = error
            if current_mode is not None:
                self._current_mode = str(current_mode)

    def _run(self) -> None:
        viewer = None
        tick_count = 0
        last_log_at = time.perf_counter()
        try:
            viewer = self._viewer_factory(
                window_name=self._window_name,
                layout_primary_width=self._layout_primary_width,
                layout_primary_height=self._layout_primary_height,
            )
            self._set_state(active=True)
            while not self._stop_event.is_set():
                started_at = time.perf_counter()
                snapshot = self._mailbox.take_latest()
                if snapshot is not None:
                    apply_snapshot = getattr(viewer, "apply_render_snapshot", None)
                    if callable(apply_snapshot):
                        apply_snapshot(snapshot)
                    else:
                        viewer.update(
                            snapshot.mesh_vertices_xyz,
                            snapshot.mesh_triangles,
                            snapshot.mesh_vertex_colors,
                            snapshot.scene_graph_snapshot,
                            snapshot.mesh_geometry_revision,
                            snapshot.mesh_color_revision,
                        )
                tick = getattr(viewer, "tick", None)
                if callable(tick):
                    active = bool(tick())
                else:
                    active = True
                current_mode_getter = getattr(viewer, "current_render_mode", None)
                current_mode = (
                    str(current_mode_getter()) if callable(current_mode_getter) else self._current_mode
                )
                self._set_state(active=active, current_mode=current_mode)
                if not active:
                    break
                tick_count += 1
                now = time.perf_counter()
                if (now - last_log_at) >= 1.0:
                    self._debug_log(
                        "render worker "
                        f"ticks={tick_count} "
                        f"dropped={self._mailbox.dropped_count()} "
                        f"mode={current_mode} "
                        f"geometry_updates={self._geometry_update_count} "
                        f"color_updates={self._color_update_count}"
                    )
                    tick_count = 0
                    last_log_at = now
                elapsed = time.perf_counter() - started_at
                wait_timeout = max(0.0, self._tick_interval_sec - elapsed)
                self._wake_event.wait(timeout=wait_timeout)
                self._wake_event.clear()
        except Exception as exc:  # pragma: no cover - exercised in dedicated tests with fake viewers.
            self._set_state(active=False, error=exc)
            self._debug_log(f"render worker failed ({exc})")
        finally:
            if viewer is not None:
                close = getattr(viewer, "close", None)
                if callable(close):
                    close()
            self._set_state(active=False)
