# Photoreal Asset Mesh Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-populate missing Blender asset-cache libraries with recognizable procedural meshes and lock in that YOLO uses the raw camera frame buffer.

**Architecture:** Extend the Blender realtime worker with testable procedural mesh blueprints and a runtime cache-writer path for missing `.blend` libraries. Add detector-input regression tests in the runtime pipeline so the displayed camera image and YOLO source stay aligned.

**Tech Stack:** Python, Blender `bpy`, NumPy, pytest

---

### Task 1: Add failing detector-input regression test

**Files:**
- Modify: `tests/test_frame_source.py`
- Modify: `src/obj_recog/main.py`

**Step 1: Write the failing test**

Add a test that captures the exact frame passed into `detector.detect()` and asserts it is the resized raw `frame_bgr`, not a GT overlay or alternate render.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_frame_source.py::test_process_frame_passes_resized_raw_camera_frame_to_detector`

**Step 3: Write minimal implementation**

Adjust `process_frame()` only if needed to make the contract explicit.

**Step 4: Run test to verify it passes**

Run the same pytest target and confirm `PASS`.

### Task 2: Add failing procedural-asset blueprint tests

**Files:**
- Modify: `tests/test_blender_worker.py`
- Modify: `scripts/blender/realtime_worker.py`

**Step 1: Write the failing test**

Add tests that key assets such as `chair_modern` and `desk_basic` resolve to multi-part procedural blueprints rather than a single cube fallback.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_blender_worker.py::test_procedural_asset_blueprint_builds_recognizable_chair_and_desk`

**Step 3: Write minimal implementation**

Add pure blueprint helpers first, then connect them to the Blender worker.

**Step 4: Run test to verify it passes**

Run the same pytest target and confirm `PASS`.

### Task 3: Implement Blender cache population for missing libraries

**Files:**
- Modify: `scripts/blender/realtime_worker.py`

**Step 1: Implement the cache-writer path**

When a library file is missing or unusable, procedurally create the object, save it to `placement.blender_library_path`, and use that object for the current render.

**Step 2: Verify locally**

Run targeted worker/runtime tests, then run a short `photoreal` simulation command to confirm cached `.blend` files are created and visible in the camera view.

### Task 4: Final verification

**Files:**
- Modify: `tests/test_frame_source.py`
- Modify: `tests/test_blender_worker.py`
- Modify: `scripts/blender/realtime_worker.py`
- Modify: `src/obj_recog/main.py`

**Step 1: Run verification**

Run:

```bash
pytest -q tests/test_frame_source.py tests/test_blender_worker.py
```

Then run a short Blender-backed simulation and inspect the generated cache files under `~/.cache/obj-recog/assets/libraries/`.
