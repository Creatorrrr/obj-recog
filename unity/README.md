# Unity Living Room Project

This folder is now the Unity project root for the RGB-only living-room simulator.

The runtime scripts are render-pipeline neutral. If URP is installed, the scene builder prefers `Universal Render Pipeline/Lit`; otherwise it falls back to built-in `Standard`.

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
- `Q/E`: body turn left and right
- mouse X: camera pan
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

The bootstrap scene creates the room, robot rig, hidden goal trigger, HUD, and TCP server wiring automatically.
