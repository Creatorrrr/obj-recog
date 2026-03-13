# Living Room Natural Background Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the `Environment Model` view for `living_room_navigation_v1` clearly show an outdoor natural setting through the front window using sky, grass, and tree background geometry.

**Architecture:** Keep simulation and Blender rendering unchanged, and extend the Open3D environment-view mesh assembly path to add a non-colliding outdoor backdrop layer behind the front glass. Build the backdrop from simple triangle meshes with vertex colors so the viewer can render sky, grass, and tree silhouettes without introducing image-texture infrastructure.

**Tech Stack:** Python, Open3D, NumPy, pytest

---

### Task 1: Add a failing environment-viewer test for outdoor backdrop geometry

**Files:**
- Modify: `tests/test_visualization.py`
- Modify: `src/obj_recog/visualization.py`

**Step 1: Write the failing test**

Add a test that updates `Open3DEnvironmentViewer` with `living_room_navigation_v1` scene state and asserts the room mesh now contains extra backdrop geometry beyond the interior floor, walls, and ceiling baseline.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_visualization.py -k "environment_viewer and backdrop"`

Expected: `FAIL` because no outdoor backdrop geometry exists yet.

**Step 3: Write minimal implementation**

Add a helper that builds sky, grass, and tree silhouette meshes outside the front window and include those meshes in the room mesh assembly path for scene-spec-backed environment views.

**Step 4: Run test to verify it passes**

Run the same pytest target and confirm `PASS`.

### Task 2: Add a failing test for backdrop color composition

**Files:**
- Modify: `tests/test_visualization.py`
- Modify: `src/obj_recog/visualization.py`

**Step 1: Write the failing test**

Add assertions that the generated room mesh colors include:
- a sky-dominant blue range
- a grass-dominant green range
- a darker tree-toned green/brown range

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_visualization.py -k "environment_viewer and backdrop_colors"`

Expected: `FAIL` because the viewer currently only contains interior surface colors.

**Step 3: Write minimal implementation**

Assign deterministic vertex colors to the new backdrop meshes so the outdoor layers are visually distinct and testable.

**Step 4: Run test to verify it passes**

Run the same pytest target and confirm `PASS`.

### Task 3: Verify the full visualization slice

**Files:**
- Modify: `src/obj_recog/visualization.py`
- Modify: `tests/test_visualization.py`

**Step 1: Run focused verification**

Run: `pytest -q tests/test_visualization.py`

Expected: `PASS`.

**Step 2: Run a simulation smoke check**

Run:

```bash
PYTHONPATH=src python -m obj_recog.main \
  --input-source sim \
  --scenario living_room_navigation_v1 \
  --blender-exec /Applications/Blender.app/Contents/MacOS/Blender \
  --width 640 \
  --height 360 \
  --device auto \
  --depth-profile fast \
  --segmentation-mode panoptic \
  --sim-planner-model gpt-5-mini \
  --sim-planner-timeout-sec 8 \
  --sim-replan-interval-sec 4 \
  --sim-selfcal-max-sec 6 \
  --sim-action-batch-size 6 \
  --explanation-mode off
```

Confirm that the `Environment Model` window shows sky, grass, and trees outside the front glass while the rest of the environment view still renders normally.
