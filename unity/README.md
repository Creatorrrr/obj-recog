# Unity RGB-Only Server

This folder holds reference Unity scripts for the RGB-only simulator contract.

- `RgbOnlyRobotRig.cs`: applies movement and camera-pan primitives to the robot rig.
- `RgbOnlyTcpServer.cs`: exposes the TCP server used by `src/obj_recog/unity_rgb.py`.

Runtime contract:

- `Unity -> Python`: PNG-encoded RGB frame plus timestamp.
- `Python -> Unity`: `reset_episode`, `shutdown`, or one action command with `primitive` and `value`.

The Python runtime assumes Unity camera intrinsics are pre-matched to the local calibration file and are not transmitted over the socket.
