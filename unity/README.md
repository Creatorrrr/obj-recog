# Unity Living Room Project

This folder is now the Unity project root for the RGB-only living-room simulator.

`Assets/Scenes/LivingRoomMain.unity` is based on ApartmentKit `Scene_02` and the runtime bootstrap only adds the robot rig, hidden goal, HUD, and TCP/manual control wiring.

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

1. Open this `unity` folder in Unity 6 LTS.
2. Load `Assets/Scenes/LivingRoomMain.unity`.
3. Press Play to test `manual` mode in the editor.
4. Build a Windows standalone player from the same scene.

The bootstrap keeps the ApartmentKit environment intact and wires the robot rig, hidden goal trigger, HUD, and TCP server automatically.
