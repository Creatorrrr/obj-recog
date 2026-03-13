# Environment Open3D View Design

**Problem**

`Environment Model` is currently an OpenCV isometric panel. The user wants that window replaced with a separate Open3D 3D view that shows the placed environment assets from a third-person perspective, including walls, floor, and ceiling, while preserving the existing `3D Reconstruction` SLAM result window.

**Decision**

Replace the OpenCV `Environment Model` panel with a dedicated Open3D environment viewer. Keep `3D Reconstruction` unchanged as the SLAM/map reconstruction view.

**Approach**

- Add a new `Open3DEnvironmentViewer` in `visualization.py`.
- Feed it room dimensions, object placements, asset identity, target/visible state, and camera pose from `SimulationScenarioState`.
- Render room surfaces as simple static meshes.
- Render objects using the same procedural asset blueprint definitions used by the photoreal Blender fallback/cache path, so the environment viewer reflects the intended asset shape rather than generic boxes.
- Render the runtime camera pose as a small camera marker/frustum inside the scene.
- Update `main.run()` to manage a second Open3D viewer for `Environment Model` instead of calling `cv2.imshow()` for that window.

**Data Model Changes**

`SimulationScenarioState` needs environment dimensions.
`environment_objects` snapshots need `asset_id` and `yaw_deg` so the viewer can instantiate and orient the correct asset mesh.

**Tradeoffs**

- The new environment view will be heavier than the old OpenCV panel, but it matches the user’s intent and shares asset logic with the photoreal path.
- The environment viewer will use procedural Open3D meshes, not imported `.blend` meshes, because Open3D does not directly consume the Blender libraries already cached for the camera renderer.

**Success Criteria**

- `Environment Model` becomes a separate Open3D 3D window.
- `3D Reconstruction` remains available and unchanged.
- Room geometry includes floor, walls, and ceiling.
- Scene objects appear as recognizable procedural meshes with correct placement/orientation.
- Runtime tests cover the new viewer wiring and the extra scenario-state data needed to drive it.
