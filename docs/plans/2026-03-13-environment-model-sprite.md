# Environment Model Sprite Rendering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Render preview sprites in the environment-model panel so chairs, desks, backpacks, and similar objects no longer appear only as generic cuboids.

**Architecture:** Keep the existing environment panel and object-snapshot flow, but enrich snapshots with sprite metadata and let the visualization layer alpha-composite preview sprites into the isometric panel. Preserve the old cuboid renderer as a safe fallback when sprite metadata is unavailable.

**Tech Stack:** Python, NumPy, OpenCV, pytest

---

### Task 1: Add failing metadata and rendering tests

**Files:**
- Modify: `tests/test_simulation.py`
- Modify: `tests/test_visualization.py`

**Step 1: Write a failing simulation test**

Assert that environment-object snapshots include `preview_sprite_path` for scenario assets.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_simulation.py -k "environment_object"`

Expected: FAIL because snapshots do not yet expose the sprite path.

**Step 3: Write a failing visualization test**

Render the environment panel once with a sprite path and once without it, then assert the two images differ.

**Step 4: Run test to verify it fails**

Run: `pytest -q tests/test_visualization.py -k "environment_model"`

Expected: FAIL because sprite metadata is ignored by the panel.

### Task 2: Propagate sprite metadata and yaw through runtime snapshots

**Files:**
- Modify: `src/obj_recog/simulation.py`
- Test: `tests/test_simulation.py`

**Step 1: Add minimal implementation**

Expose `preview_sprite_path` and `yaw_deg` from scene objects and asset placements in `_environment_object_snapshots()`. Preserve yaw on runtime `_SceneObject` instances.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_simulation.py -k "environment_object or blender_realtime_frame_source"`

Expected: PASS

### Task 3: Render preview sprites in the isometric panel

**Files:**
- Modify: `src/obj_recog/visualization.py`
- Test: `tests/test_visualization.py`

**Step 1: Write minimal implementation**

Load preview sprites with alpha, scale them from object footprint/height, rotate by yaw when available, and alpha-composite them into the panel. If loading fails, fall back to the old box renderer.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_visualization.py -k "environment_model"`

Expected: PASS

### Task 4: Run cross-check regression tests

**Files:**
- Test: `tests/test_simulation.py`
- Test: `tests/test_visualization.py`
- Test: `tests/test_main_smoke.py`

**Step 1: Run combined verification**

Run: `pytest -q tests/test_simulation.py tests/test_visualization.py tests/test_main_smoke.py -k "environment_model or blender_realtime_frame_source or photoreal"`

Expected: PASS
