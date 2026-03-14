# obj-recog

Real-time object recognition, monocular 3D reconstruction, and simulation testbed.

## Prerequisites

- Python 3.11+
- `pip` or another Python package manager
- Optional: `OPENAI_API_KEY` if you want the explanation window to call the LLM
- Optional for simulation: a Unity player build that runs the RGB-only TCP server in `unity/Assets/Scripts`

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

The simulation path is now an RGB-only Unity interface:

- `Unity -> Python`: camera RGB frame and timestamp only
- `Python -> Unity`: movement and camera-pan action commands only
- online runtime uses the same monocular perception stack as live input
- hidden goal, collision truth, and evaluation stay outside the online inference loop

Run it like this:

```bash
PYTHONPATH=src python -m obj_recog.main \
  --input-source sim \
  --scenario living_room_navigation_v1 \
  --unity-player-path C:/path/to/obj-recog-unity.exe \
  --width 640 \
  --height 360 \
  --device auto \
  --depth-profile fast \
  --segmentation-mode panoptic \
  --camera-calibration calibration/calibration.yaml \
  --sim-planner-model gpt-5-mini \
  --sim-planner-timeout-sec 8 \
  --sim-replan-interval-sec 4 \
  --sim-selfcal-max-sec 6 \
  --sim-action-batch-size 6 \
  --explanation-mode off
```

Useful sim flags:

- `--sim-headless`: run without the desktop windows.
- `--unity-host` / `--unity-port`: connect to an already-running Unity player.
- `--sim-planner-model`: choose the LLM used for navigation planning.
- `--sim-interface-mode rgb_only`: enforce the RGB-only contract.

Visible sim windows:

- `Object Recognition`
- `3D Reconstruction`

## Outputs

Each sim run writes episode artifacts under:

- `reports/sim/living_room_navigation_v1/.../episode_report.json`
- `reports/sim/living_room_navigation_v1/.../planner_turns.jsonl`
- `reports/sim/living_room_navigation_v1/.../self_calibration.json`
- `episode_report.json` records the online run only; official success/failure belongs to offline evaluation.

## Development Checks

Useful regression commands:

```bash
pytest -q tests/test_config.py tests/test_main_simulation.py tests/test_living_room_runtime.py tests/test_unity_rgb.py tests/test_offline_benchmark.py
python -m compileall src/obj_recog
git diff --check
```
