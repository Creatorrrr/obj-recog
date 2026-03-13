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

Notes:

- `--render-profile fast` is the speed-oriented simulator path. Scene objects are rendered as simplified boxes or sprite-like stand-ins rather than high-fidelity asset meshes.
- `Environment Model` still shows the full room layout and object placements, but the camera view is not photoreal in this mode.

## Bootstrap Photoreal Assets

Photoreal benchmark runs now require normalized external-provenance assets in the local cache. Bootstrap the assets for a scenario before running `render_profile=photoreal` with `sim_perception_mode=runtime`:

```bash
PYTHONPATH=src python -m obj_recog.asset_bootstrap \
  --scenario studio_open_v1 \
  --asset-cache-dir ~/.cache/obj-recog/assets \
  --asset-quality low \
  --blender-exec /Applications/Blender.app/Contents/MacOS/Blender
```

You can prebuild every supported scenario at once:

```bash
PYTHONPATH=src python -m obj_recog.asset_bootstrap \
  --all-scenarios \
  --asset-cache-dir ~/.cache/obj-recog/assets \
  --asset-quality low \
  --blender-exec /Applications/Blender.app/Contents/MacOS/Blender
```

The bootstrap step exports two canonical cache artifacts per asset:

- a normalized Blender library `.blend` used by the camera renderer
- a preview `.ply` mesh used by the `Environment Model` Open3D scene view

If a catalog entry still points at a provider landing page instead of a direct archive, the bootstrap CLI now fails immediately with a direct-archive error instead of silently producing a procedural fallback. Curating all archive URLs and hashes is still required asset-content work.

## Photoreal Simulation Run

For a benchmark-valid camera-perception run, use `--sim-perception-mode runtime`. This path now fails fast if any required asset is missing, stale, or marked procedural in the cache metadata.

Use this when you want the simulator camera view to come from Blender-backed 3D mesh assets instead of the lightweight `fast` renderer:

```bash
PYTHONPATH=src python -m obj_recog.main \
  --input-source sim \
  --scenario studio_open_v1 \
  --sim-seed 7 \
  --sim-max-steps 240 \
  --eval-budget-sec 20 \
  --sim-perception-mode runtime \
  --render-profile photoreal \
  --blender-exec /Applications/Blender.app/Contents/MacOS/Blender \
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

If the required external assets are not present yet, run the bootstrap command above first. Benchmark photoreal runs no longer silently fall back to procedural meshes.

Supported scenarios:

- `studio_open_v1`
- `office_clutter_v1`
- `lab_corridor_v1`
- `showroom_occlusion_v1`
- `office_crossflow_v1`
- `warehouse_moving_target_v1`

`--sim-perception-mode` options:

- `runtime`: detector/depth/tracker only. This is the only benchmark-valid camera-perception mode.
- `assisted`: runtime path with simulator GT help. Useful for debugging, not a benchmark.
- `ground_truth`: simulator GT only. Useful for debugging, not a benchmark.

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

For a like-for-like single-scenario comparison with the `fast` example above, first bootstrap the assets, then switch the render path to Blender-backed `photoreal` and use `runtime` perception:

```bash
PYTHONPATH=src python -m obj_recog.asset_bootstrap \
  --scenario studio_open_v1 \
  --asset-cache-dir ~/.cache/obj-recog/assets \
  --asset-quality low \
  --blender-exec /Applications/Blender.app/Contents/MacOS/Blender

PYTHONPATH=src python -m obj_recog.main \
  --input-source sim \
  --scenario studio_open_v1 \
  --sim-seed 7 \
  --sim-max-steps 240 \
  --eval-budget-sec 20 \
  --render-profile photoreal \
  --blender-exec /Applications/Blender.app/Contents/MacOS/Blender \
  --sim-perception-mode runtime \
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

Current status:

- `render_profile=photoreal` is routed through the Blender realtime frame-source code path.
- `Environment Model` uses the same normalized asset family as the camera renderer by loading preview meshes from the asset cache.
- The Blender worker at `scripts/blender/realtime_worker.py` processes realtime requests and writes RGB, depth, semantic mask, instance mask, and detection metadata bundles.
- Benchmark photoreal runs require external-provenance cache metadata and fail immediately if the cache is missing or stale.
- Multi-scenario photoreal routing and validation are covered by tests.
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
