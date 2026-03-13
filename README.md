# obj-recog

Real-time object recognition, monocular 3D reconstruction, and simulation testbed.

## Prerequisites

- Python 3.11+
- `pip` or another Python package manager
- Optional: `OPENAI_API_KEY` if you want the explanation window to call the LLM
- Optional for photoreal work: Blender is installed at `/Applications/Blender.app/Contents/MacOS/Blender`

## Install

From the repo root:

```bash
python -m pip install -e .[dev]
```

Check the CLI:

```bash
PYTHONPATH=src python -m obj_recog.main -h
```

## Basic Live Run

Use the default camera input:

```bash
PYTHONPATH=src python -m obj_recog.main \
  --input-source live \
  --width 640 \
  --height 360 \
  --device auto \
  --depth-profile fast \
  --segmentation-mode panoptic \
  --explanation-mode on
```

When running from this repository, the bundled ORB-SLAM3 vocabulary at
`third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt` is picked up automatically. If you
run from a different install layout, pass `--slam-vocabulary /absolute/path/to/ORBvoc.txt`.

Useful runtime controls:

- Press `e` or click the bottom-right toggle to turn the explanation window on or off.
- `Situation Explanation` appears in a separate OpenCV window.
- `Environment Model` is a separate Open3D third-person scene view when simulation data is available.

## Simulation Runs

The simulation path currently supports two scenarios:

- `living_room_navigation_v1`: procedural 20-pyeong-class apartment living room with dining-table-front goal
- `interior_test_tv_navigation_v1`: `/Users/chasoik/Downloads/InteriorTest.blend`-backed interior with a TV-front goal

Both scenarios use:

- a hidden scene world that is not exposed directly to the planner
- Blender for robot camera RGB/depth/semantic/instance rendering
- Open3D for the operator 3D room view
- a closed loop of self-calibration, perception, LLM planning, action execution, and hidden goal evaluation

Run it like this:

```bash
PYTHONPATH=src python -m obj_recog.main \
  --input-source sim \
  --scenario living_room_navigation_v1 \
  --blender-exec /Applications/Blender.app/Contents/MacOS/Blender \
  --width 640 \
  --height 360 \
  --device auto \
  --depth-profile fast \
  --segmentation-mode panoptic \
  --sim-planner-model gpt-5-mini \
  --sim-planner-timeout-sec 8 \
  --sim-replan-interval-sec 4 \
  --sim-selfcal-max-sec 6 \
  --sim-action-batch-size 6 \
  --explanation-mode off
```

Useful sim flags:

- `--sim-headless`: run without the desktop windows.
- `--sim-open3d-view off`: disable the Open3D operator room view.
- `--sim-planner-model`: choose the LLM used for navigation planning.

Visible sim windows:

- `Object Recognition`
- `3D Reconstruction`
- `Environment Model`

The old simulation asset bootstrap flow, external manifests, multi-scenario validation suite, and assisted/ground-truth sim perception modes are retired.

To run the authored TV scenario:

```bash
PYTHONPATH=src python -m obj_recog.main \
  --input-source sim \
  --scenario interior_test_tv_navigation_v1 \
  --blender-exec /Applications/Blender.app/Contents/MacOS/Blender \
  --width 640 \
  --height 360 \
  --device auto \
  --depth-profile fast \
  --segmentation-mode panoptic \
  --sim-planner-model gpt-5-mini \
  --sim-planner-timeout-sec 8 \
  --sim-replan-interval-sec 4 \
  --sim-selfcal-max-sec 6 \
  --sim-action-batch-size 6 \
  --explanation-mode off
```

## Outputs

Each sim run writes episode artifacts under:

- `reports/sim/living_room_navigation_v1/.../episode_report.json`
- `reports/sim/living_room_navigation_v1/.../planner_turns.jsonl`
- `reports/sim/living_room_navigation_v1/.../self_calibration.json`
- `reports/sim/interior_test_tv_navigation_v1/.../episode_report.json`
- `reports/sim/interior_test_tv_navigation_v1/.../planner_turns.jsonl`
- `reports/sim/interior_test_tv_navigation_v1/.../self_calibration.json`

## Development Checks

Useful regression commands:

```bash
pytest -q tests/test_config.py tests/test_blender_worker.py tests/test_realtime_worker_script.py tests/test_main_simulation.py
pytest -q tests/test_sim_scene.py tests/test_sim_planner.py tests/test_living_room_runtime.py
python -m compileall src/obj_recog
git diff --check
```
