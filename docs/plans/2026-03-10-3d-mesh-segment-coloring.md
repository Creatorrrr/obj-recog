# 3D Mesh Segment Coloring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Project recent Mask2Former panoptic segments back onto the existing TSDF mesh and show them as a translucent vertex-color overlay in the 3D viewer.

**Architecture:** Keep TSDF geometry generation unchanged and treat semantic coloring as a separate cached mesh-color pass. Segmentation worker results are tagged with frame indices, matched back to the source frame pose/intrinsics, and stored as recent observations that can recolor cached mesh vertices without rebuilding mesh geometry.

**Tech Stack:** Python, NumPy, OpenCV, PyTorch/transformers Mask2Former, existing ORB-SLAM3 bridge, existing Open3D TSDF mesh viewer.

---

### Task 1: Segmentation result carries a dense id map

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/types.py`
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/segmenter.py`
- Test: `/Users/chasoik/Projects/obj-recog/tests/test_segmenter.py`

**Step 1: Write the failing test**

Add assertions that `SegmentationResult.segment_id_map` exists, has `int32` dtype, contains accepted segment ids, and leaves filtered segments as `-1`.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_segmenter.py`

Expected: FAIL because `SegmentationResult` has no `segment_id_map`.

**Step 3: Write minimal implementation**

Build `segment_id_map` during panoptic post-processing and attach it to `SegmentationResult`.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_segmenter.py`

Expected: PASS

### Task 2: Frame-indexed segmentation worker results

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/segmenter.py`
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/main.py`
- Test: `/Users/chasoik/Projects/obj-recog/tests/test_main_smoke.py`

**Step 1: Write the failing test**

Add a smoke test proving worker submissions include `frame_index`, that completed results are matched back to source-frame metadata, and stale/missing metadata is ignored safely.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_main_smoke.py -k segmentation`

Expected: FAIL because worker submit/poll signatures do not include `frame_index`.

**Step 3: Write minimal implementation**

Update `SegmentationWorker.submit(frame_index, frame_bgr)` and `poll()` to return `(frame_index, SegmentationResult)`. Keep a small metadata cache in `main.run()` keyed by frame index.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_main_smoke.py -k segmentation`

Expected: PASS

### Task 3: TSDF mesh recolors from recent segmentation observations

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/mapping.py`
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/main.py`
- Test: `/Users/chasoik/Projects/obj-recog/tests/test_tsdf_mapping.py`

**Step 1: Write the failing test**

Add tests that:
- no observations keep base mesh colors
- a segmentation observation recolors vertices with `base*0.55 + segment*0.45`
- multiple observations prefer recent colors via weighted averaging

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_tsdf_mapping.py`

Expected: FAIL because `TsdfMeshMapBuilder` has no segmentation observation ingestion or recolor pass.

**Step 3: Write minimal implementation**

Store recent segmentation observations, project cached mesh vertices into those frames, and recompute cached vertex colors without changing geometry.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_tsdf_mapping.py`

Expected: PASS

### Task 4: Runtime integration and verification

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/main.py`
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/types.py`
- Test: `/Users/chasoik/Projects/obj-recog/tests/test_main_smoke.py`

**Step 1: Write the failing test**

Add a smoke test that a tracked segmentation observation updates mesh vertex colors without regenerating geometry arrays.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_main_smoke.py -k mesh`

Expected: FAIL because runtime does not hand segmentation observations to the map builder.

**Step 3: Write minimal implementation**

Add `intrinsics` to `FrameArtifacts`, plumb frame metadata through the main loop, and refresh cached mesh colors after observation ingestion.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_main_smoke.py -k mesh`

Expected: PASS

### Task 5: Final verification

**Files:**
- Verify only

**Step 1: Run focused test suites**

Run:
- `./.venv/bin/python -m pytest -q tests/test_segmenter.py tests/test_tsdf_mapping.py tests/test_main_smoke.py`
- `./.venv/bin/python -m compileall src`

**Step 2: Run full test suite**

Run: `./.venv/bin/python -m pytest -q`

Expected: PASS
