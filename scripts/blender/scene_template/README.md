# Blender Scene Template

This directory is reserved for the realtime Blender scene template used by the
photoreal simulation path.

Expected contents in later tasks:

- `base_scene.blend`
  Base room, camera rig, render layers, and reusable collections.
- linked collections for environment meshes, static props, and dynamic actors
- render layer setup for:
  - RGB
  - depth
  - semantic mask
  - instance mask

The command builder already assumes a `.blend` file in this directory can be
passed to Blender as the scene template entrypoint.
