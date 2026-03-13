# Environment Open3D View Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the `Environment Model` OpenCV panel with a separate Open3D environment viewer while keeping the existing `3D Reconstruction` SLAM viewer intact.

**Architecture:** Extend `SimulationScenarioState` and environment-object snapshots with the geometry metadata the new environment viewer needs. Add `Open3DEnvironmentViewer` that renders room surfaces, procedural asset meshes, and camera pose markers, then wire `main.run()` to manage this viewer instead of `cv2.imshow()` for the environment window.

**Tech Stack:** Python, Open3D, NumPy, pytest

---

### Task 1: Add failing tests for scenario-state geometry metadata

**Files:**
- Modify: `tests/test_simulation.py`
- Modify: `src/obj_recog/simulation.py`

**Step 1: Write the failing test**

Add a test asserting `environment_objects` snapshots include `asset_id` and `yaw_deg`, and the emitted `SimulationScenarioState` includes room dimensions.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_simulation.py::test_simulation_frame_source_environment_state_includes_room_and_asset_geometry`

**Step 3: Write minimal implementation**

Extend `SimulationScenarioState` and `_environment_object_snapshots()` so the metadata is always present.

**Step 4: Run test to verify it passes**

Run the same pytest target and confirm `PASS`.

### Task 2: Add failing tests for the new Open3D environment viewer

**Files:**
- Modify: `tests/test_visualization.py`
- Modify: `src/obj_recog/visualization.py`

**Step 1: Write the failing test**

Add tests that:
- the new viewer opens a window named `Environment Model`
- room surfaces and object meshes are populated from scenario state
- the camera marker geometry is updated

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_visualization.py -k "environment_viewer"`

**Step 3: Write minimal implementation**

Add `Open3DEnvironmentViewer` and any small reusable mesh helpers needed.

**Step 4: Run test to verify it passes**

Run the same pytest target and confirm `PASS`.

### Task 3: Add failing runtime wiring tests

**Files:**
- Modify: `tests/test_main_smoke.py`
- Modify: `tests/test_main_simulation.py`
- Modify: `src/obj_recog/main.py`

**Step 1: Write the failing test**

Add tests asserting `run()` updates/closes a separate environment viewer and no longer requires the OpenCV environment panel renderer path.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_main_smoke.py -k "environment_viewer" tests/test_main_simulation.py -k "environment_viewer"`

**Step 3: Write minimal implementation**

Wire the new viewer into `run()`, preserve window positioning behavior for remaining OpenCV windows, and close both viewers on shutdown.

**Step 4: Run test to verify it passes**

Run the same pytest target and confirm `PASS`.

### Task 4: Remove obsolete panel path and verify the whole slice

**Files:**
- Modify: `src/obj_recog/visualization.py`
- Modify: `src/obj_recog/main.py`
- Modify: `tests/test_visualization.py`
- Modify: `tests/test_main_smoke.py`
- Modify: `tests/test_main_simulation.py`
- Modify: `tests/test_simulation.py`

**Step 1: Run focused verification**

Run:

```bash
pytest -q tests/test_simulation.py tests/test_visualization.py tests/test_main_smoke.py tests/test_main_simulation.py
```

**Step 2: Run a manual runtime smoke check**

Start the sim and verify:
- `Environment Model` is an Open3D window
- `3D Reconstruction` still appears separately
- room surfaces and asset meshes are visible
