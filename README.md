# obj-recog

Real-time object recognition, monocular 3D reconstruction, and simulation testbed.

## Prerequisites

- Python 3.11+
- `pip` or another Python package manager
- Optional: `OPENAI_API_KEY` if you want the explanation window to call the LLM
- Optional for simulation: Unity 6 LTS to open and build the project in `unity`

## Install

From the repo root:

```bash
python -m pip install -e .[dev]
```

On Windows with Python 3.12 on x64, the project metadata now pins PyTorch and
TorchVision to the official CUDA 12.8 wheels, so a fresh install should pick up
GPU support automatically. On other platforms or Python versions it falls back
to the default PyPI packages.

You can verify the resolved runtime like this:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## Windows CUDA OpenCV

The default `opencv-python` wheel is CPU-only. If you want `--opencv-cuda on` to
work, build OpenCV from source with CUDA and install the resulting Python
binding into the current `.venv`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_opencv_cuda_windows.ps1
```

This script expects:

- NVIDIA CUDA Toolkit 12.8
- Visual Studio 2022 Build Tools with C++ workload
- CMake
- Ninja

The installer fetches the official `opencv` and `opencv_contrib` `4.13.0`
sources. `opencv_contrib` is required because OpenCV's CUDA image-processing
modules depend on `cudev`.

After the build, verify the runtime explicitly:

```powershell
.\.venv\Scripts\python .\scripts\verify_opencv_cuda.py
```

If you later rerun `python -m pip install -e .[dev]`, a CPU OpenCV wheel may be
installed again. In that case, rerun `.\scripts\install_opencv_cuda_windows.ps1`
to restore the CUDA build inside `.venv`.

## Windows ORB-SLAM3 Bridge

`3D Reconstruction` in `sim + rgb_only` now requires the native ORB-SLAM3 bridge.
On Windows, build it before launching `obj_recog.main`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_orbslam3_bridge_windows.ps1
```

This script looks for:

- CMake
- Visual Studio 2022 Build Tools with the C++ workload
- a local `third_party\ORB_SLAM3` checkout
- CMake package metadata for OpenCV, Eigen3, Boost, and OpenSSL

If your dependencies live outside the default search paths, pass them explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_orbslam3_bridge_windows.ps1 `
  -OpenCvDir C:\path\to\OpenCVConfig\dir `
  -Eigen3Dir C:\path\to\Eigen3Config\dir `
  -CMakePrefixPath C:\path\to\prefix1;C:\path\to\prefix2
```

The expected output binary is `native\orbslam3_bridge\build\orbslam3_bridge.exe`.

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
  --opencv-cuda on \
  --depth-profile fast \
  --segmentation-mode panoptic \
  --explanation-mode on
```

The startup log now shows both the requested and resolved device, for example:

```text
[obj-recog] runtime accel requested_device=auto resolved_device=cuda precision=fp16 ...
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

The Unity project lives in `unity`. `Assets/Scenes/LivingRoomMain.unity` is based on ApartmentKit `Scene_02` with obj-recog runtime wiring layered on top. Open it in Unity 6 LTS, then:

- run Play mode with no special arguments for `manual` keyboard/mouse control
- build a Windows standalone player and let Python launch it in `agent` mode

Manual mode controls:

- `W/S`: forward and backward
- `A/D`: strafe left and right
- `Q`: turn body right
- `E`: turn body left
- mouse X/Y: pan camera and look up/down
- `R`: reset
- `F1`: HUD toggle
- `Esc`: release cursor, then press again to quit

Run it like this:

```bash
PYTHONPATH=src python -m obj_recog.main \
  --input-source sim \
  --scenario living_room_navigation_v1 \
  --unity-player-path C:/path/to/obj-recog-unity.exe \
  --width 640 \
  --height 360 \
  --device auto \
  --opencv-cuda on \
  --detector-backend torch \
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
- `--unity-player-path`: launch the same player build in `agent` mode with `--obj-recog-mode=agent`.

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
.venv/Scripts/python scripts/verify_opencv_cuda.py
python -m compileall src/obj_recog
git diff --check
```
