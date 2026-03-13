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
- `Environment Model` is the isometric simulation-state view when simulation data is available.

## Fast Simulation Run

Run one scenario with the current lightweight renderer:

```bash
PYTHONPATH=src python -m obj_recog.main \
  --input-source sim \
  --scenario studio_open_v1 \
  --sim-seed 7 \
  --sim-max-steps 240 \
  --eval-budget-sec 20 \
  --sim-perception-mode assisted \
  --render-profile fast \
  --width 640 \
  --height 360 \
  --device cpu \
  --depth-profile fast \
  --point-stride 8 \
  --max-points 5000 \
  --segmentation-mode panoptic \
  --segmentation-interval 12 \
  --explanation-mode off
```

Supported scenarios:

- `studio_open_v1`
- `office_clutter_v1`
- `lab_corridor_v1`
- `showroom_occlusion_v1`
- `office_crossflow_v1`
- `warehouse_moving_target_v1`

`--sim-perception-mode` options:

- `runtime`: detector/depth/tracker only
- `assisted`: runtime path with simulator GT help
- `ground_truth`: simulator GT only

## Validate All Scenarios

Run the validation harness across all scenarios:

```bash
PYTHONPATH=src python -m obj_recog.main \
  --validate-all-scenarios \
  --validation-output-dir validation/manual-check \
  --input-source sim \
  --sim-seed 7 \
  --sim-camera-fps 4 \
  --sim-max-steps 80 \
  --eval-budget-sec 16 \
  --width 320 \
  --height 180 \
  --device auto \
  --depth-profile fast \
  --point-stride 12 \
  --max-points 2000 \
  --segmentation-mode panoptic \
  --segmentation-interval 12 \
  --explanation-mode on \
  --render-profile fast
```

This writes per-run reports and a summary under `validation/manual-check/`.

## Photoreal / Blender Notes

The photoreal path is wired to use Blender:

```bash
PYTHONPATH=src python -m obj_recog.main \
  --input-source sim \
  --scenario office_clutter_v1 \
  --render-profile photoreal \
  --blender-exec /Applications/Blender.app/Contents/MacOS/Blender \
  --sim-perception-mode assisted \
  --width 640 \
  --height 360 \
  --device cpu \
  --segmentation-mode panoptic \
  --explanation-mode off
```

Current status:

- `render_profile=photoreal` is routed through the Blender realtime frame-source code path.
- The Blender worker at `scripts/blender/realtime_worker.py` now processes realtime requests and writes RGB, depth, semantic mask, instance mask, and detection metadata bundles.
- Multi-scenario photoreal routing and validation are covered by tests.
- If the asset-library `.blend` files are missing from the asset cache, the worker falls back to procedurally generated Blender mesh objects built from the scene manifest. This still uses real 3D meshes, but not high-fidelity scanned/library assets yet.
- The first photoreal frame is currently expensive on this machine. Expect a cold-start render to take several seconds before performance work lands.
- If you already have an external render bundle manifest, you can use `--sim-external-manifest /absolute/path/to/manifest.json`.

## Outputs

- Scenario run reports: `reports/<scenario>-seed<seed>.json`
- Validation summary and per-run JSON: `validation/<name>/`
- Optional validation preview crops: enable `--scenario-preview-shots`

## Development Checks

Useful regression commands:

```bash
pytest -q tests/test_simulation.py tests/test_validation.py -k "photoreal and scenario"
pytest -q tests/test_config.py tests/test_frame_source.py tests/test_sim_assets.py tests/test_blender_worker.py tests/test_simulation.py tests/test_validation.py tests/test_main_simulation.py
pytest -q tests/test_main_smoke.py -k 'validation_probe or explanation or scene_graph or segmentation or isometric'
python -m compileall src/obj_recog
git diff --check
```
