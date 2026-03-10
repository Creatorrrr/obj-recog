# Object Recognition Dense Map Only Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Strip the runtime down to cumulative 3D mapping plus object recognition, removing debug overlays and extra 3D helper geometry.

**Architecture:** Keep the ORB-SLAM3 + MiDaS + YOLO pipeline intact for cumulative mapping, but simplify presentation and runtime plumbing so the 2D view only renders detections and the 3D view only renders the dense accumulated point cloud. Preserve calibration setup because ORB-SLAM3 still depends on it.

**Tech Stack:** Python, OpenCV, Open3D, NumPy, PyTorch, Ultralytics, ORB-SLAM3 bridge

---

### Task 1: Lock the simplified UI behavior with tests

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_visualization.py`
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_main_smoke.py`

**Step 1: Write failing tests**
- Add a visualization test that asserts `draw_detections(...)` only emits detection labels, not FPS/tracking/debug text.
- Update the Open3D viewer test to expect only the dense point cloud geometry, not frustum/trajectory line sets.
- Update the main-loop smoke test that inspected overlay kwargs so it no longer expects camera/debug overlay metadata.

**Step 2: Run tests to verify they fail**

Run:
```bash
./.venv/bin/python -m pytest -q tests/test_visualization.py tests/test_main_smoke.py
```

### Task 2: Remove debug-heavy rendering and plumbing

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/visualization.py`
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/main.py`

**Step 1: Write minimal implementation**
- Make `draw_detections(...)` render only boxes and labels.
- Make `Open3DPointCloudViewer` manage only the dense point cloud.
- Update `_update_viewer(...)` and the main loop to stop passing debug overlay metadata.

**Step 2: Run focused tests**

Run:
```bash
./.venv/bin/python -m pytest -q tests/test_visualization.py tests/test_main_smoke.py
```

### Task 3: Verify the whole project still passes

**Files:**
- No additional file edits expected

**Step 1: Run the full suite**

Run:
```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m compileall src
```
