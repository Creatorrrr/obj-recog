# Explanation Refresh Status Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve the previous situation explanation body during background refreshes and show request progress separately in the panel header.

**Architecture:** Add a separate refresh request state in `main.py` and keep `ExplanationResult` as the displayed body state. Extend the panel renderer to show a `Refresh:` status line above the body while leaving the body unchanged during in-flight or failed background refreshes when a prior successful explanation exists.

**Tech Stack:** Python, pytest, OpenCV panel rendering

---

### Task 1: Add regression tests for preserved body behavior

**Files:**
- Modify: `tests/test_main_smoke.py`
- Modify: `tests/test_visualization.py`

**Step 1: Write the failing test**

Add one smoke test that confirms a previous successful explanation body remains visible during auto-refresh and another that confirms it remains visible after a failed refresh. Add a visualization test for the new `Refresh:` line.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_main_smoke.py::test_run_preserves_previous_explanation_body_while_auto_refresh_is_pending tests/test_main_smoke.py::test_run_preserves_previous_explanation_body_when_auto_refresh_fails tests/test_visualization.py::test_render_explanation_panel_renders_refresh_status_line -q`
Expected: FAIL because the current code replaces the body with loading/error content and has no refresh status line.

**Step 3: Write minimal implementation**

Do not implement here.

**Step 4: Run test to verify it passes**

Run after implementation.

**Step 5: Commit**

Not requested.

### Task 2: Implement separate request status

**Files:**
- Modify: `src/obj_recog/main.py`
- Modify: `src/obj_recog/visualization.py`

**Step 1: Write the failing test**

Use the Task 1 tests as the regression driver.

**Step 2: Run test to verify it fails**

Run the targeted tests and confirm the body is still being replaced.

**Step 3: Write minimal implementation**

Track refresh status separately from displayed explanation content, preserve the last successful body during background refreshes, and expose the refresh status line to the panel renderer.

**Step 4: Run test to verify it passes**

Run the targeted tests plus existing explanation-related smoke tests.

**Step 5: Commit**

Not requested.
