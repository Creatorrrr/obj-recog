# Explanation Auto Refresh Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the one-shot Explain trigger with an ON/OFF toggle that auto-refreshes situation explanations on a configurable interval from `.env`.

**Architecture:** Keep the feature inside the existing `run()` frame loop. Store toggle state and the last auto-refresh timestamp in `main.py`, parse the interval into `AppConfig`, and pass toggle state to the overlay renderer so the button label reflects `ON` or `OFF`.

**Tech Stack:** Python, pytest, OpenCV UI loop, dotenv-backed environment configuration

---

### Task 1: Add regression tests

**Files:**
- Modify: `tests/test_main_smoke.py`
- Modify: `tests/test_config.py`

**Step 1: Write the failing test**

Add a smoke test that clicks the Explain button, advances fake time past the refresh interval, and expects more than one explanation submission.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_main_smoke.py::test_run_auto_refreshes_explanation_while_toggle_is_on -q`
Expected: FAIL because no repeated submission happens.

**Step 3: Write minimal implementation**

Do not implement here.

**Step 4: Run test to verify it passes**

Run after implementation.

**Step 5: Commit**

Not requested.

### Task 2: Parse refresh interval from config

**Files:**
- Modify: `src/obj_recog/config.py`
- Modify: `tests/test_config.py`

**Step 1: Write the failing test**

Add a config test asserting `EXPLANATION_REFRESH_INTERVAL_SEC` is reflected in `parse_config([])`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_parse_config_uses_explanation_refresh_interval_from_environment -q`
Expected: FAIL because `AppConfig` has no such field yet.

**Step 3: Write minimal implementation**

Add an `AppConfig.explanation_refresh_interval_sec` field with a positive float default, and load it from environment during `parse_config`.

**Step 4: Run test to verify it passes**

Run the config test again.

**Step 5: Commit**

Not requested.

### Task 3: Implement toggle + auto refresh UI flow

**Files:**
- Modify: `src/obj_recog/main.py`
- Modify: `src/obj_recog/visualization.py`
- Modify: `tests/test_main_smoke.py`

**Step 1: Write the failing test**

Use the smoke test from Task 1 as the regression driver.

**Step 2: Run test to verify it fails**

Run the targeted smoke test and confirm the missing repeated submission.

**Step 3: Write minimal implementation**

Track whether explanation auto-refresh is enabled, toggle it on mouse click or `e`, submit immediately when turning on, and re-submit only when the interval elapses and the worker is idle. Update the button label to `Explain ON` or `Explain OFF`.

**Step 4: Run test to verify it passes**

Run the targeted smoke/config/visualization tests.

**Step 5: Commit**

Not requested.
