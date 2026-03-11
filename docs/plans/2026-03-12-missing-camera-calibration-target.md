# Missing Camera Calibration Target Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make a missing explicit calibration path act as the destination for generated calibration settings instead of raising an error.

**Architecture:** Keep `ensure_runtime_calibration()` as the single decision point. Existing files stay on the explicit-load path; missing files become output targets for either approximate disabled-mode settings or full self-calibration output, with the final YAML written to the requested location.

**Tech Stack:** Python 3.11, pathlib, pytest, existing ORB-SLAM3 calibration helpers

---

### Task 1: Missing explicit path becomes generated output

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_auto_calibration.py`
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/auto_calibration.py`

**Step 1: Write the failing test**

Add a test showing that when `camera_calibration` points to a missing nested path, `ensure_runtime_calibration()` runs warm-up, writes the YAML to that exact path, and returns that same path in the runtime state.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_auto_calibration.py -k missing_explicit`

Expected: FAIL because the current code raises `RuntimeError` for a missing explicit path.

**Step 3: Write minimal implementation**

Detect the missing explicit path, skip the immediate error, run the existing calibration flow, and write the final YAML to the requested path using the existing YAML writer.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_auto_calibration.py -k missing_explicit`

Expected: PASS

### Task 2: Disabled mode also writes to the requested explicit path

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_auto_calibration.py`
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/auto_calibration.py`

**Step 1: Write the failing test**

Add a test showing that when `disable_slam_calibration=True` and `camera_calibration` points to a missing file, the approximate calibration YAML is written to that requested path.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_auto_calibration.py -k disabled_missing_explicit`

Expected: FAIL because the current code raises `RuntimeError` before disabled-mode handling.

**Step 3: Write minimal implementation**

Route the disabled-mode branch through the explicit target path when one was requested.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_auto_calibration.py -k disabled_missing_explicit`

Expected: PASS

### Task 3: Final verification

**Files:**
- Verify only

**Step 1: Run focused verification**

Run:
- `./.venv/bin/python -m pytest -q tests/test_auto_calibration.py -k "explicit_path or missing_explicit or disabled"`

**Step 2: Run full touched suite**

Run:
- `./.venv/bin/python -m pytest -q tests/test_auto_calibration.py`

Expected: PASS
