# Dotenv Camera Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow `.env` to provide the camera calibration path through `CAMERA_CALIBRATION` while preserving CLI override behavior and current auto-calibration fallback.

**Architecture:** Keep the existing `.env` loading in `main.run()` unchanged and resolve the config default from the environment only when `--camera-calibration` is not provided. Validation stays in the runtime calibration path so config parsing remains a lightweight source-resolution step.

**Tech Stack:** Python 3.11, argparse, python-dotenv fallback loader, pytest

---

### Task 1: Config environment fallback

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/config.py`
- Test: `/Users/chasoik/Projects/obj-recog/tests/test_config.py`

**Step 1: Write the failing test**

Add tests that:
- `CAMERA_CALIBRATION` populates `config.camera_calibration` when the CLI flag is absent.
- `--camera-calibration` still overrides `CAMERA_CALIBRATION`.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_config.py`

Expected: FAIL because `parse_config()` currently defaults `camera_calibration` to `None`.

**Step 3: Write minimal implementation**

Update `build_parser()` so `--camera-calibration` defaults to `os.getenv("CAMERA_CALIBRATION")`.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_config.py`

Expected: PASS

### Task 2: Runtime smoke coverage for dotenv-provided calibration

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_main_smoke.py`

**Step 1: Write the failing test**

Add a smoke test that:
- writes `.env` with `CAMERA_CALIBRATION=<path>`
- loads `.env` with `_load_app_dotenv()`
- builds config via `parse_config([])`
- verifies the resulting config carries the `.env` calibration path

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_main_smoke.py -k calibration`

Expected: FAIL because config parsing does not yet read `CAMERA_CALIBRATION`.

**Step 3: Write minimal implementation**

No additional runtime changes should be required beyond Task 1. Keep the test as proof that the startup flow already supports the new source.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_main_smoke.py -k calibration`

Expected: PASS

### Task 3: Final verification

**Files:**
- Verify only

**Step 1: Run focused tests**

Run:
- `./.venv/bin/python -m pytest -q tests/test_config.py`
- `./.venv/bin/python -m pytest -q tests/test_main_smoke.py -k calibration`

**Step 2: Run combined verification**

Run:
- `./.venv/bin/python -m pytest -q tests/test_config.py tests/test_main_smoke.py -k "calibration or dotenv"`

Expected: PASS
