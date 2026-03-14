`third_party/ORB_SLAM3` is expected to contain a pinned checkout of the official ORB-SLAM3 source.

Fetch it with:

```bash
./scripts/fetch_orbslam3.sh
```

Then build the bridge with:

```bash
./scripts/build_orbslam3_bridge.sh
```

On Windows PowerShell, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_orbslam3_bridge_windows.ps1
```
