# Living Room Procedural Furniture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the living-room simulation from single-box furniture meshes to richer shared procedural furniture assemblies without breaking renderer consistency or runtime stability.

**Architecture:** Keep [sim_scene.py](/Users/chasoik/Projects/obj-recog/src/obj_recog/sim_scene.py) as the single source of truth for furniture geometry, but expand each major object into multiple semantic-preserving subcomponents. Feed those components to both the Open3D environment viewer and the realtime camera renderer so the operator view and rendered camera frames stay aligned.

**Tech Stack:** Python, NumPy, Open3D viewer integration, procedural software rendering worker, pytest

---

### Task 1: Add scene-structure tests first

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_sim_scene.py`
- Reference: `/Users/chasoik/Projects/obj-recog/src/obj_recog/sim_scene.py`

**Step 1: Write the failing tests**

Add tests that assert:

- sofa returns multiple `SceneMeshComponent` entries
- dining table returns multiple entries including leg-like/support parts
- each dining chair returns multiple entries
- front window has frame parts separate from the glass panel

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_sim_scene.py -k "sofa or dining_table or dining_chair or window"`

Expected: FAIL because each furniture item is still represented by a single box component.

**Step 3: Do not implement yet**

Stop after confirming the new structure tests fail.

**Step 4: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/tests/test_sim_scene.py
git commit -m "test: add detailed furniture scene expectations"
```

### Task 2: Refactor procedural furniture generation in the shared scene builder

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/sim_scene.py`
- Test: `/Users/chasoik/Projects/obj-recog/tests/test_sim_scene.py`

**Step 1: Write the minimal production change**

Introduce helper builders in [sim_scene.py](/Users/chasoik/Projects/obj-recog/src/obj_recog/sim_scene.py), for example:

- `_make_sofa_components(...)`
- `_make_dining_table_components(...)`
- `_make_dining_chair_components(...)`
- `_make_tv_console_components(...)`
- `_make_window_frame_components(...)`

Keep `SceneMeshComponent.semantic_label` at the parent object category even when component IDs become more specific.

**Step 2: Run the focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_sim_scene.py -k "sofa or dining_table or dining_chair or window"`

Expected: PASS

**Step 3: Refactor only if still green**

If needed, extract a helper such as `_append_box_component(...)` or `_component_color(...)` to keep the file readable.

**Step 4: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/src/obj_recog/sim_scene.py /Users/chasoik/Projects/obj-recog/tests/test_sim_scene.py
git commit -m "feat: add procedural multi-part furniture meshes"
```

### Task 3: Keep the environment viewer compatible with detailed components

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_visualization.py`
- Reference: `/Users/chasoik/Projects/obj-recog/src/obj_recog/visualization.py`

**Step 1: Write the failing test**

Add or extend a visualization test to assert that:

- the environment viewer still combines all detailed components into room/object meshes
- increased object component counts do not break mesh upload or camera marker rendering

**Step 2: Run test to verify it fails if needed**

Run: `PYTHONPATH=src pytest -q tests/test_visualization.py -k "environment_viewer"`

Expected: If no change is needed, this may already pass. If it fails, the failure should identify an assumption about one-mesh-per-object behavior.

**Step 3: Apply the minimal fix only if the test fails**

Update [visualization.py](/Users/chasoik/Projects/obj-recog/src/obj_recog/visualization.py) only if the detailed component stream reveals a real bug.

**Step 4: Re-run**

Run: `PYTHONPATH=src pytest -q tests/test_visualization.py -k "environment_viewer"`

Expected: PASS

**Step 5: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/src/obj_recog/visualization.py /Users/chasoik/Projects/obj-recog/tests/test_visualization.py
git commit -m "test: validate environment viewer with detailed furniture components"
```

### Task 4: Update the realtime worker to consume the shared detailed geometry

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/scripts/blender/realtime_worker.py`
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_realtime_worker_script.py`
- Reference: `/Users/chasoik/Projects/obj-recog/src/obj_recog/sim_scene.py`

**Step 1: Write the failing worker test**

Add a test that confirms the worker builds a larger primitive set from the living-room scene and still emits:

- RGB
- depth
- semantic mask
- instance mask

with non-empty object visibility.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_realtime_worker_script.py -k "primitive or detailed or furniture"`

Expected: FAIL because the worker still constructs one primitive per furniture item.

**Step 3: Write the minimal implementation**

Refactor `_build_primitives(...)` in [realtime_worker.py](/Users/chasoik/Projects/obj-recog/scripts/blender/realtime_worker.py) to derive render primitives from shared component definitions rather than from raw object boxes alone.

Use the parent semantic label for semantic IDs so detection/segmentation summaries remain object-level.

**Step 4: Run the focused test**

Run: `PYTHONPATH=src pytest -q tests/test_realtime_worker_script.py -k "primitive or detailed or furniture"`

Expected: PASS

**Step 5: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/scripts/blender/realtime_worker.py /Users/chasoik/Projects/obj-recog/tests/test_realtime_worker_script.py
git commit -m "feat: render detailed procedural furniture in realtime worker"
```

### Task 5: Add end-to-end regression coverage for simulation views

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_main_simulation.py`
- Reference: `/Users/chasoik/Projects/obj-recog/src/obj_recog/main.py`
- Reference: `/Users/chasoik/Projects/obj-recog/src/obj_recog/simulation.py`

**Step 1: Write the failing test**

Add a regression test that sim mode still wires:

- camera view
- 3D reconstruction
- environment view

when detailed furniture components are present.

**Step 2: Run test to verify it fails only if integration assumptions broke**

Run: `PYTHONPATH=src pytest -q tests/test_main_simulation.py -k "sim"`

Expected: Either PASS already or fail with a concrete integration assumption to fix.

**Step 3: Apply the minimal integration fix if required**

Only touch [main.py](/Users/chasoik/Projects/obj-recog/src/obj_recog/main.py) or [simulation.py](/Users/chasoik/Projects/obj-recog/src/obj_recog/simulation.py) if the new component counts expose a real regression.

**Step 4: Re-run**

Run: `PYTHONPATH=src pytest -q tests/test_main_simulation.py -k "sim"`

Expected: PASS

**Step 5: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/src/obj_recog/main.py /Users/chasoik/Projects/obj-recog/src/obj_recog/simulation.py /Users/chasoik/Projects/obj-recog/tests/test_main_simulation.py
git commit -m "test: keep simulation views stable with detailed furniture meshes"
```

### Task 6: Run the full relevant verification suite

**Files:**
- No code changes required

**Step 1: Run focused suites**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_sim_scene.py tests/test_visualization.py tests/test_realtime_worker_script.py tests/test_main_simulation.py
```

Expected: PASS

**Step 2: Run broader simulation regression suites**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_blender_worker.py tests/test_frame_source.py tests/test_living_room_runtime.py
```

Expected: PASS

**Step 3: Run static verification**

Run:

```bash
python -m compileall /Users/chasoik/Projects/obj-recog/src/obj_recog /Users/chasoik/Projects/obj-recog/scripts/blender/realtime_worker.py
git diff --check
```

Expected: both commands succeed with no diff formatting errors.

**Step 4: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog
git commit -m "feat: enrich procedural living room furniture geometry"
```
