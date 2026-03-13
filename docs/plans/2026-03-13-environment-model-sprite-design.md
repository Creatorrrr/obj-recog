# Environment Model Sprite Rendering Design

**Goal:** Replace box-only environment-model objects with asset preview sprites so the isometric panel communicates object identity more clearly.

**Context:** The current `Environment Model` panel renders every object as a shaded cuboid from `center_world` and `size_xyz`. That is useful for occupancy and scale, but it hides the semantic difference between furniture and props. At the same time, the simulation already has per-object preview sprites in the asset manifest and in scene objects.

## Current Behavior

- `render_environment_model_panel()` in `src/obj_recog/visualization.py` draws every object with `_draw_isometric_object()`.
- `_draw_isometric_object()` ignores sprite metadata and only paints three box faces.
- `scenario_state.environment_objects` does not currently expose `preview_sprite_path`, and runtime scene objects do not preserve yaw metadata.

## Desired Behavior

- When an environment object has a preview sprite path, the environment panel should render a sprite-based marker anchored to the floor position instead of a plain cuboid.
- The sprite should scale from the object footprint and height so larger furniture still reads larger.
- Transparent sprite regions should preserve the background grid.
- If sprite loading fails or metadata is missing, the existing cuboid rendering should remain as a fallback.

## Approach

1. Extend environment-object snapshots to carry sprite metadata needed by the visualization layer.
2. Preserve per-object yaw in runtime scene objects so the visualization can rotate sprites consistently.
3. Update `render_environment_model_panel()` to try sprite rendering first, then fall back to the existing cuboid painter.
4. Add regression tests for snapshot metadata propagation and for sprite-vs-box rendering differences.

## Non-Goals

- Replacing the panel with a realtime Open3D or Blender viewport.
- Changing the camera-view renderer or YOLO inference behavior.
- Downloading or generating missing `.blend` asset libraries.
