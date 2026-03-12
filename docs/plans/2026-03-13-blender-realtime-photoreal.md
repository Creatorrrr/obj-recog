# Blender Realtime Photoreal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current photoreal stub with a live Blender-based realtime frame source that uses real 3D mesh assets and supports an interactive macOS GUI preview.

**Architecture:** Add a persistent Blender worker process, a Blender-backed `FrameSource`, and a mesh-first asset catalog. Keep the existing `FramePacket` pipeline and validation harness, and add a new isometric environment-model window driven by simulation world state.

**Tech Stack:** Python, Blender, Eevee, OpenCV, NumPy, existing simulation/runtime pipeline, pytest

---

### Task 1: Add failing config and routing tests for the realtime Blender path

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/tests/test_config.py`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/tests/test_simulation.py`

**Step 1: Write the failing test**

Add tests that assert:

- `render_profile=photoreal` and `blender_exec` parse correctly.
- `SimulationRuntime.create_frame_source()` selects a realtime Blender-backed frame source instead of raising the current manifest-only error for `photoreal`.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_config.py tests/test_simulation.py -k "photoreal or blender"`

Expected: FAIL because the current photoreal path still prepares a manifest and raises.

**Step 3: Write minimal implementation**

Create the smallest routing changes needed so tests can target a new realtime worker path without invoking Blender yet.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_config.py tests/test_simulation.py -k "photoreal or blender"`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_config.py tests/test_simulation.py src/obj_recog/config.py src/obj_recog/simulation.py
git commit -m "test: route photoreal config to realtime blender path"
```

### Task 2: Build the Blender worker protocol and lifecycle wrapper

**Files:**
- Create: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/blender_worker.py`
- Create: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/tests/test_blender_worker.py`

**Step 1: Write the failing test**

Add tests for a pure-Python worker wrapper:

- starts a subprocess with the expected command
- sends frame requests as JSON
- parses worker responses
- surfaces startup failure and timeout errors cleanly

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_blender_worker.py`

Expected: FAIL because the module does not exist yet.

**Step 3: Write minimal implementation**

Add:

- request/response dataclasses
- worker process launcher
- send/receive helpers
- timeout and restart handling scaffolding

Use fake subprocesses in tests. Do not touch Blender yet.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_blender_worker.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/obj_recog/blender_worker.py tests/test_blender_worker.py
git commit -m "feat: add blender worker protocol wrapper"
```

### Task 3: Convert the asset catalog from sprite-first to mesh-first metadata

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/sim_assets.py`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/tests/test_sim_assets.py`

**Step 1: Write the failing test**

Add tests that assert each asset entry includes:

- Blender library path metadata
- object name
- semantic class
- recommended scale
- lod/material metadata

Also add a test that `studio_open_v1` resolves to mesh placements, not preview-only placements.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_sim_assets.py`

Expected: FAIL because current entries are preview-sprite oriented.

**Step 3: Write minimal implementation**

Refactor asset entries and manifest generation for mesh-first use, while keeping existing fields only where still needed for compatibility.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_sim_assets.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/obj_recog/sim_assets.py tests/test_sim_assets.py
git commit -m "feat: add mesh-first asset catalog metadata"
```

### Task 4: Add BlenderRealtimeFrameSource for a single scenario

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/simulation.py`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/tests/test_simulation.py`

**Step 1: Write the failing test**

Add tests that:

- instantiate a Blender-backed frame source with a fake worker
- receive RGB, depth, semantic mask metadata, and GT pose
- emit a valid `FramePacket` for `studio_open_v1`

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_simulation.py -k "blender realtime"`

Expected: FAIL because the frame source does not exist yet.

**Step 3: Write minimal implementation**

Add `BlenderRealtimeFrameSource` and route `render_profile=photoreal` to it. Keep the first version scoped to `studio_open_v1`.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_simulation.py -k "blender realtime"`

Expected: PASS

**Step 5: Commit**

```bash
git add src/obj_recog/simulation.py tests/test_simulation.py
git commit -m "feat: add blender realtime frame source"
```

### Task 5: Add Blender-side scene template and request script

**Files:**
- Create: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/scripts/blender/realtime_worker.py`
- Create: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/scripts/blender/scene_template/README.md`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/tests/test_blender_worker.py`

**Step 1: Write the failing test**

Add tests that validate the worker wrapper builds the correct Blender command line and scene bootstrap arguments.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_blender_worker.py -k command`

Expected: FAIL because the wrapper does not yet know how to invoke the Blender worker script.

**Step 3: Write minimal implementation**

Add the Blender-side Python worker entrypoint and command builder. Keep it focused on:

- loading the template scene
- accepting requests
- updating camera/object transforms
- writing response files

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_blender_worker.py`

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/blender/realtime_worker.py scripts/blender/scene_template/README.md src/obj_recog/blender_worker.py tests/test_blender_worker.py
git commit -m "feat: add blender realtime worker entrypoint"
```

### Task 6: Connect RGB, depth, semantic mask, and instance mask to FramePacket

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/frame_source.py`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/simulation.py`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/tests/test_simulation.py`

**Step 1: Write the failing test**

Add tests that assert photoreal packets carry:

- RGB image
- depth map
- GT intrinsics
- GT pose
- mask-derived or metadata-derived detections
- enriched scenario state

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_simulation.py -k "photoreal packet"`

Expected: FAIL

**Step 3: Write minimal implementation**

Wire worker responses into `FramePacket` creation and maintain compatibility with the rest of the runtime.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_simulation.py -k "photoreal packet"`

Expected: PASS

**Step 5: Commit**

```bash
git add src/obj_recog/frame_source.py src/obj_recog/simulation.py tests/test_simulation.py
git commit -m "feat: map blender render passes into frame packets"
```

### Task 7: Add the isometric environment-model window

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/visualization.py`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/main.py`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/tests/test_main_smoke.py`

**Step 1: Write the failing test**

Add a smoke test that verifies a second environment-model window is rendered when simulation data is available.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_main_smoke.py -k isometric`

Expected: FAIL because the extra window does not exist yet.

**Step 3: Write minimal implementation**

Add a fixed-isometric renderer that draws:

- room footprint
- static mesh placements
- dynamic actors
- current camera pose and heading
- mission target highlight

Display it in a dedicated OpenCV window separate from the reconstruction viewer.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_main_smoke.py -k isometric`

Expected: PASS

**Step 5: Commit**

```bash
git add src/obj_recog/visualization.py src/obj_recog/main.py tests/test_main_smoke.py
git commit -m "feat: add isometric environment model window"
```

### Task 8: Extend validation for Blender worker health and photoreal pass integrity

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/validation.py`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/tests/test_validation.py`

**Step 1: Write the failing test**

Add tests for new validation subsystems:

- `mesh_backed_environment`
- `worker_health`
- `frame_latency`
- `pass_output_presence`

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_validation.py -k "worker or latency or mesh_backed_environment"`

Expected: FAIL

**Step 3: Write minimal implementation**

Track worker startup state, restarts, render time, IPC time, and required-pass presence in the validation probe and reports.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_validation.py -k "worker or latency or mesh_backed_environment"`

Expected: PASS

**Step 5: Commit**

```bash
git add src/obj_recog/validation.py tests/test_validation.py
git commit -m "feat: validate blender worker health and photoreal passes"
```

### Task 9: Run the full verification set on the single-scenario reference path

**Files:**
- No code changes required unless failures appear.

**Step 1: Run the focused test set**

Run:

```bash
pytest -q tests/test_config.py tests/test_frame_source.py tests/test_sim_assets.py tests/test_blender_worker.py tests/test_simulation.py tests/test_validation.py tests/test_main_simulation.py
```

Expected: PASS

**Step 2: Run the smoke checks**

Run:

```bash
pytest -q tests/test_main_smoke.py -k 'validation_probe or explanation or scene_graph or segmentation or isometric'
```

Expected: PASS

**Step 3: Run compile and diff checks**

Run:

```bash
python -m compileall src/obj_recog
git diff --check
```

Expected: PASS

**Step 4: Manual verification**

Run:

```bash
OPENAI_API_KEY=... PYTHONPATH=src python -m obj_recog.main \
  --input-source sim \
  --scenario studio_open_v1 \
  --render-profile photoreal \
  --sim-perception-mode assisted \
  --width 640 \
  --height 360 \
  --device cpu \
  --segmentation-mode panoptic \
  --explanation-mode on
```

Confirm:

- `Object Recognition` window updates interactively
- `Situation Explanation` window shows text after `e`
- Open3D reconstruction viewer opens
- `Environment Isometric View` opens
- Blender worker stays alive during the run

**Step 5: Commit final polish if needed**

```bash
git add -A
git commit -m "feat: finish studio photoreal blender integration"
```

### Task 10: Expand from `studio_open_v1` to the remaining five scenarios

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/simulation.py`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/sim_assets.py`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/tests/test_simulation.py`
- Modify: `/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/tests/test_validation.py`

**Step 1: Write the failing test**

Add tests that each remaining scenario resolves to valid photoreal mesh placements and dynamic actor updates through the Blender worker path.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_simulation.py tests/test_validation.py -k "photoreal and scenario"`

Expected: FAIL

**Step 3: Write minimal implementation**

Port the remaining scenarios and tune asset selections and LODs only as needed to restore performance and validation health.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_simulation.py tests/test_validation.py -k "photoreal and scenario"`

Expected: PASS

**Step 5: Commit**

```bash
git add src/obj_recog/simulation.py src/obj_recog/sim_assets.py tests/test_simulation.py tests/test_validation.py
git commit -m "feat: expand photoreal blender path across all scenarios"
```
