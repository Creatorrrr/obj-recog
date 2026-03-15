# Unity Living Room Project

This folder is now the Unity project root for the RGB-only living-room simulator.

`Assets/Scenes/LivingRoomMain.unity` is based on ApartmentKit `Scene_02` and the runtime bootstrap only adds the robot rig, hidden goal, HUD, and TCP/manual control wiring.

`Apartment Kit` is a local Asset Store dependency and is intentionally excluded from Git. Import `Apartment Kit` version `4.2` into Unity 6 LTS `6000.3.11f1` and keep the default asset paths:

- `Assets/Brick Project Studio/Apartment Kit`
- `Assets/Brick Project Studio/_BPS Basic Assets`

Before running the scene or the Python simulator, validate the local setup from the repo root:

```bash
PYTHONPATH=src python -m obj_recog.unity_vendor_check --unity-project-root unity
```

Do not edit files under `Assets/Brick Project Studio`. If you need to customize the environment, create tracked copies or prefab/material variants outside the vendor folder.

## Modes

- `manual`: keyboard and mouse drive the robot directly.
- `agent`: Python launches the same player and drives it through the RGB-only TCP contract.

The player chooses `manual` by default. It switches to `agent` when launched with:

- `--obj-recog-mode=agent`
- `--obj-recog-host=<ip>`
- `--obj-recog-port=<port>`

## Manual Controls

- `W/S`: move forward and backward
- `A/D`: strafe left and right
- `Q`: body turn right
- `E`: body turn left
- mouse X/Y: camera pan and look up/down
- `R`: reset
- `F1`: toggle HUD
- `Esc`: release cursor, then press again to quit

## Python Contract

- `Unity -> Python`: `rgb_frame` JSON payload with `timestamp_sec`, `image_encoding=png`, and `image_bytes_b64`
- `Python -> Unity`: `reset_episode`, `action`, and `shutdown`

No depth, semantic mask, instance mask, pose, intrinsics, or hidden-goal state are sent to Python.

## Opening the Project

1. Run the vendor check command from the repo root and confirm it passes.
2. Open this `unity` folder in Unity 6 LTS `6000.3.11f1`.
3. Import Asset Store package `Apartment Kit` version `4.2` if the vendor folders are missing.
4. Keep the imported assets at the default path under `Assets/Brick Project Studio/`.
5. Load `Assets/Scenes/LivingRoomMain.unity`.
6. Press Play to test `manual` mode in the editor.
7. Build a Windows standalone player from the same scene.

The bootstrap keeps the ApartmentKit environment intact and wires the robot rig, hidden goal trigger, HUD, and TCP server automatically.
