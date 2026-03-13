# Interior Test TV Scenario Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a second simulation scenario that runs against `/Users/chasoik/Downloads/InteriorTest.blend`, renders the real authored room in both the camera view and `Environment Model`, and succeeds when the robot reaches the hidden goal in front of the TV.

**Architecture:** Introduce a blend-backed scene path alongside the existing procedural living-room path. Extract normalized scene metadata from the `.blend` file, use that data for runtime goal and collision handling plus Open3D environment rendering, and make the Blender worker open the authored scene directly for sensor rendering while keeping planner input camera-only.

**Tech Stack:** Python, Blender subprocess scripting, NumPy, Open3D viewer integration, pytest

---

### Task 1: Add config and registry tests for the new scenario

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_config.py`
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_sim_scene.py`
- Reference: `/Users/chasoik/Projects/obj-recog/src/obj_recog/config.py`
- Reference: `/Users/chasoik/Projects/obj-recog/src/obj_recog/simulation.py`

**Step 1: Write the failing tests**

Add tests that assert:

- `--scenario interior_test_tv_navigation_v1` parses successfully
- the scenario registry includes the new scenario id
- the new scenario exposes a TV-target semantic class and a non-empty blend file path

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_config.py tests/test_sim_scene.py -k "interior_test_tv_navigation_v1 or tv"`

Expected: FAIL because the new scenario is not registered yet.

**Step 3: Write minimal implementation**

Update scenario choices and registry wiring only, without adding extraction behavior yet.

**Step 4: Run the tests again**

Run: `PYTHONPATH=src pytest -q tests/test_config.py tests/test_sim_scene.py -k "interior_test_tv_navigation_v1 or tv"`

Expected: PASS

**Step 5: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/tests/test_config.py /Users/chasoik/Projects/obj-recog/tests/test_sim_scene.py /Users/chasoik/Projects/obj-recog/src/obj_recog/config.py /Users/chasoik/Projects/obj-recog/src/obj_recog/simulation.py
git commit -m "feat: register interior test tv scenario"
```

### Task 2: Add a blend-backed scene spec and extraction manifest

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/sim_protocol.py`
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/sim_scene.py`
- Create: `/Users/chasoik/Projects/obj-recog/src/obj_recog/blend_scene_loader.py`
- Test: `/Users/chasoik/Projects/obj-recog/tests/test_sim_scene.py`

**Step 1: Write the failing tests**

Add tests that assert:

- the new scene spec resolves `/Users/chasoik/Downloads/InteriorTest.blend`
- the loader extracts the `TV` object anchor
- the normalized manifest includes room/furniture objects and a hidden TV-front goal pose

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_sim_scene.py -k "blend or TV or interior_test"`

Expected: FAIL because the loader and blend-backed spec do not exist.

**Step 3: Write minimal implementation**

Create `blend_scene_loader.py` that invokes Blender headless, reads object transforms, normalizes coordinates, and returns a lightweight manifest. Extend the sim scene layer to produce a scene spec for `interior_test_tv_navigation_v1`.

**Step 4: Run the focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_sim_scene.py -k "blend or TV or interior_test"`

Expected: PASS

**Step 5: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/src/obj_recog/sim_protocol.py /Users/chasoik/Projects/obj-recog/src/obj_recog/sim_scene.py /Users/chasoik/Projects/obj-recog/src/obj_recog/blend_scene_loader.py /Users/chasoik/Projects/obj-recog/tests/test_sim_scene.py
git commit -m "feat: add blend-backed scene extraction for tv scenario"
```

### Task 3: Add authored-scene worker startup and frame rendering tests

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_blender_worker.py`
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_realtime_worker_script.py`
- Reference: `/Users/chasoik/Projects/obj-recog/src/obj_recog/blender_worker.py`
- Reference: `/Users/chasoik/Projects/obj-recog/scripts/blender/realtime_worker.py`

**Step 1: Write the failing tests**

Add tests that assert:

- the worker command can point at a real `.blend` scene file
- the worker can build the authored scene without procedural mesh payload synthesis
- frame responses still include RGB, depth, semantic, instance, and intrinsics for the new scenario

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_blender_worker.py tests/test_realtime_worker_script.py -k "blend or interior_test or authored"`

Expected: FAIL because the worker only supports the procedural path.

**Step 3: Write minimal implementation**

Extend the worker request/build path so the new scenario can open `/Users/chasoik/Downloads/InteriorTest.blend`, identify the authored scene mode, and render from it.

**Step 4: Run the focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_blender_worker.py tests/test_realtime_worker_script.py -k "blend or interior_test or authored"`

Expected: PASS

**Step 5: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/src/obj_recog/blender_worker.py /Users/chasoik/Projects/obj-recog/scripts/blender/realtime_worker.py /Users/chasoik/Projects/obj-recog/tests/test_blender_worker.py /Users/chasoik/Projects/obj-recog/tests/test_realtime_worker_script.py
git commit -m "feat: render authored blender scene for tv scenario"
```

### Task 4: Make Environment Model render extracted real structure

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/visualization.py`
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_visualization.py`
- Reference: `/Users/chasoik/Projects/obj-recog/src/obj_recog/blend_scene_loader.py`

**Step 1: Write the failing tests**

Add tests that assert:

- `Open3DEnvironmentViewer` accepts extracted authored-scene mesh groups
- the new scenario draws actual extracted room/object meshes rather than procedural room geometry
- the robot pose marker still updates correctly in the authored scene

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_visualization.py -k "interior_test or authored or environment_viewer"`

Expected: FAIL because the environment viewer only understands the procedural scene spec.

**Step 3: Write minimal implementation**

Update the environment viewer so authored scenes can provide normalized room/object preview meshes directly, bypassing procedural component generation.

**Step 4: Run the focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_visualization.py -k "interior_test or authored or environment_viewer"`

Expected: PASS

**Step 5: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/src/obj_recog/visualization.py /Users/chasoik/Projects/obj-recog/tests/test_visualization.py
git commit -m "feat: show authored interior scene in environment model"
```

### Task 5: Add hidden goal and collision behavior for the TV-front scenario

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/simulation.py`
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/sim_scene.py`
- Test: `/Users/chasoik/Projects/obj-recog/tests/test_living_room_runtime.py`

**Step 1: Write the failing tests**

Add tests that assert:

- the new scenario starts from the fixed lower-room pose
- success is triggered only when the robot reaches the hidden TV-front target radius
- movement into extracted collision proxies is blocked

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_living_room_runtime.py -k "interior_test or collision or tv_front"`

Expected: FAIL because there is no authored-scene goal or collision handling.

**Step 3: Write minimal implementation**

Add hidden goal evaluation and collision proxy checks for the blend-backed scenario while preserving the existing procedural scenario path.

**Step 4: Run the focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_living_room_runtime.py -k "interior_test or collision or tv_front"`

Expected: PASS

**Step 5: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/src/obj_recog/simulation.py /Users/chasoik/Projects/obj-recog/src/obj_recog/sim_scene.py /Users/chasoik/Projects/obj-recog/tests/test_living_room_runtime.py
git commit -m "feat: add tv-front goal and collision checks for authored scene"
```

### Task 6: Preserve planner redaction boundaries for the authored scene

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/src/obj_recog/sim_planner.py`
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_sim_planner.py`

**Step 1: Write the failing tests**

Add tests that assert planner-visible context for `interior_test_tv_navigation_v1` does not include:

- exact TV goal coordinates
- full extracted object list
- hidden collision geometry
- authored objects not yet observed by camera

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_sim_planner.py -k "interior_test or TV or redaction"`

Expected: FAIL because the new scene metadata will not yet be filtered explicitly.

**Step 3: Write minimal implementation**

Extend planner-context building so authored-scene metadata is redacted the same way hidden procedural-scene truth is redacted today.

**Step 4: Run the focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_sim_planner.py -k "interior_test or TV or redaction"`

Expected: PASS

**Step 5: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/src/obj_recog/sim_planner.py /Users/chasoik/Projects/obj-recog/tests/test_sim_planner.py
git commit -m "test: keep authored scene truth out of planner context"
```

### Task 7: Add integration coverage and docs for the new scenario

**Files:**
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_main_simulation.py`
- Modify: `/Users/chasoik/Projects/obj-recog/tests/test_living_room_episode_integration.py`
- Modify: `/Users/chasoik/Projects/obj-recog/README.md`

**Step 1: Write the failing tests**

Add tests that assert:

- sim mode can select `interior_test_tv_navigation_v1`
- the environment viewer receives authored-scene state for that scenario
- the integration run can finish successfully at the TV-front target under offscreen execution

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_main_simulation.py tests/test_living_room_episode_integration.py -k "interior_test or TV"`

Expected: FAIL because the new scenario is not wired end-to-end yet.

**Step 3: Write minimal implementation**

Wire the scenario through main/runtime integration and document the new CLI usage in the README.

**Step 4: Run the focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_main_simulation.py tests/test_living_room_episode_integration.py -k "interior_test or TV"`

Expected: PASS

**Step 5: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog/tests/test_main_simulation.py /Users/chasoik/Projects/obj-recog/tests/test_living_room_episode_integration.py /Users/chasoik/Projects/obj-recog/README.md
git commit -m "feat: add interior test tv scenario documentation and integration"
```

### Task 8: Run the full relevant verification suite

**Files:**
- No code changes required

**Step 1: Run focused authored-scene suites**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_config.py tests/test_sim_scene.py tests/test_blender_worker.py tests/test_realtime_worker_script.py tests/test_visualization.py tests/test_living_room_runtime.py tests/test_sim_planner.py tests/test_main_simulation.py
```

Expected: PASS

**Step 2: Run integration coverage**

Run:

```bash
OPENAI_API_KEY=... PYTHONPATH=src pytest -q tests/test_living_room_episode_integration.py -k "interior_test or TV"
```

Expected: PASS

**Step 3: Run static verification**

Run:

```bash
python -m compileall /Users/chasoik/Projects/obj-recog/src/obj_recog /Users/chasoik/Projects/obj-recog/scripts/blender/realtime_worker.py
git diff --check
```

Expected: both commands succeed.

**Step 4: Commit**

```bash
git add /Users/chasoik/Projects/obj-recog
git commit -m "feat: add authored blender tv navigation scenario"
```
