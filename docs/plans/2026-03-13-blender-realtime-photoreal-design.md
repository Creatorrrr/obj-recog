# Blender Realtime Photoreal Simulation Design

## Summary

Move the simulator from the current `fast` sprite-based approximation to a Blender-driven realtime photoreal path for local macOS GUI use. The new path will use real 3D mesh assets for walls, furniture, props, and dynamic actors, and will render RGB, depth, semantic mask, instance mask, and ground-truth camera pose in realtime through a persistent Blender worker.

The existing `FramePacket` contract, detector/depth/SLAM/scene-graph/LLM pipeline, and validation harness stay in place. The main change is the frame source: `render_profile=photoreal` stops being a manifest-preparation stub and becomes a live Blender-backed source.

## Goals

- Replace sprite/billboard scene rendering with real 3D mesh rendering for all simulation objects.
- Support interactive preview on local macOS GUI with a target of `10-15 FPS` for early scenarios.
- Keep the downstream runtime unchanged by continuing to emit `FramePacket`.
- Produce RGB, depth, semantic mask, instance mask, and GT pose from the same rendered frame.
- Add a second OpenCV window showing an isometric environment-model view built from the simulation state.

## Non-Goals

- Linux/headless support in the first version.
- Path-traced photorealism or film-quality rendering.
- General-purpose asset authoring or in-repo storage of large mesh libraries.
- Multi-target missions or arbitrary user-authored scenarios.

## Environment Constraints

- Platform: local macOS GUI only.
- Renderer: Blender Eevee, not Cycles.
- Performance target: `10-15 FPS` for `studio_open_v1` and `office_clutter_v1`, with degraded but interactive performance acceptable for the harder scenarios.
- Failure behavior: fail fast if Blender worker, asset cache, or render passes are unavailable. Do not silently fall back to `fast`.

## Current State

- [simulation.py](/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/simulation.py) currently supports `render_profile=fast|photoreal`.
- `fast` uses `AnalyticSceneRenderer`, which projects simplified assets with 2.5D sprite-like rendering.
- `photoreal` only writes a scene manifest and expects an external render bundle.
- [sim_assets.py](/Users/chasoik/Projects/obj-recog/.worktrees/codex-blender-realtime-photoreal/src/obj_recog/sim_assets.py) currently models a sprite-oriented asset catalog, preview sprites, and scene manifests.
- The validation harness already tracks render realism, reconstruction, calibration, detection, segmentation, graph construction, and LLM explanation.

## Proposed Architecture

### 1. Persistent Blender Worker

Introduce a long-lived Blender subprocess that stays open for the duration of the app session.

- New module: `src/obj_recog/blender_worker.py`
- Process responsibilities:
  - Load a base `.blend` template once.
  - Import or link mesh assets from the asset cache.
  - Instantiate the current scenario scene graph.
  - Accept per-frame camera pose and actor transform updates over IPC.
  - Render RGB, depth, semantic mask, and instance mask.
  - Return frame metadata and pass locations to the app.

The app never launches Blender per frame. It starts Blender once and streams frame requests.

### 2. BlenderRealtimeFrameSource

Add a `FrameSource` implementation that wraps the worker.

- New class: `BlenderRealtimeFrameSource`
- Inputs:
  - scenario ID
  - rig spec
  - current dynamic actor transforms
  - current camera pose
  - frame index and timestamp
- Outputs:
  - `FramePacket.frame_bgr`
  - `FramePacket.depth_map`
  - `FramePacket.pose_world_gt`
  - `FramePacket.intrinsics_gt`
  - semantic detections or GT metadata derived from object visibility
  - `scenario_state` enriched with `render_profile="photoreal"` and worker metadata

`SimulationRuntime.create_frame_source()` will route `render_profile=photoreal` to this new source instead of writing a manifest and throwing an error.

### 3. Asset and Scene System

Replace sprite-first asset usage with Blender-first mesh asset usage.

- Extend the asset catalog to include:
  - `blend_library_path`
  - `object_name`
  - `lod_group`
  - `semantic_class`
  - `recommended_scale`
  - `collision_footprint`
  - `material_profile`
- Keep assets outside git under `~/.cache/obj-recog/blender-assets`.
- Store only manifests/specs in the repository.

Each scenario object placement becomes a mesh placement with:

- `asset_id`
- `semantic_class`
- `target_role`
- `transform`
- `dynamic_role`
- `instance_id`

## Blender Scene Layout

Use a single template scene with named collections:

- `Environment`
- `StaticProps`
- `DynamicActors`
- `SemanticMaskPass`
- `InstanceMaskPass`
- `CameraRig`

Every Blender object gets custom properties:

- `semantic_class`
- `target_role`
- `instance_id`
- `scenario_id`

Rendering outputs:

- RGB: Eevee materials and scene lighting
- Depth: Z/depth pass converted to meters
- Semantic mask: material override or dedicated view layer
- Instance mask: unique flat color/material pass

## IPC Design

Use a local IPC channel suitable for macOS GUI:

- Preferred: Unix domain socket
- Acceptable fallback: localhost TCP

Request payload per frame:

- `frame_index`
- `timestamp_sec`
- `scenario_id`
- `camera_pose_world`
- `intrinsics`
- `dynamic_actor_transforms`
- `lighting_seed`

Response payload per frame:

- `rgb_path`
- `depth_path`
- `semantic_mask_path`
- `instance_mask_path`
- `pose_world_gt`
- `intrinsics_gt`
- `render_time_ms`
- `worker_state`

The first version may use temporary file paths for image transfer. Shared-memory optimization can come later.

## Performance Strategy

To hit the interactive target:

- Use Eevee only.
- Default to `640x360` preview resolution.
- Support `asset_quality=low|high`.
- Preload asset libraries and keep object handles cached.
- Update transforms only per frame; do not rebuild scenes per frame.
- Cap triangle count and draw-call cost per scenario.
- Use low-poly or reduced-Lod humans for moving actors.
- Render auxiliary mask passes from lightweight view layers.

Expected performance tiers:

- `studio_open_v1`, `office_clutter_v1`: `10-15 FPS`
- harder scenarios: `8+ FPS` acceptable in v1

## Scenario Model Changes

All six current scenarios remain, but every object becomes a real mesh placement.

- `studio_open_v1`: desk, chair, potted plant, backpack meshes
- `office_clutter_v1`: desk, chair, monitor, laptop, cabinet, books, backpack
- `lab_corridor_v1`: doorframe, partition wall, cart, pillar, suitcase/backpack
- `showroom_occlusion_v1`: showroom props plus moving `person` occluder
- `office_crossflow_v1`: office props plus two moving `person` distractors
- `warehouse_moving_target_v1`: shelves, crates, boxes, moving suitcase/backpack, occluder, distractors

## UI Changes

Keep the existing windows and add one more:

- `Object Recognition`
- `Situation Explanation`
- existing 3D reconstruction/Open3D viewer
- new `Environment Isometric View`

The isometric window is not a reconstruction. It is a model-state visualization derived from the simulation world state and actor transforms, rendered in a fixed isometric projection.

## Failure Handling

- If Blender is missing, worker startup fails with a clear error.
- If the asset cache is incomplete, startup fails with the missing `asset_id`.
- If the worker dies mid-run:
  - try one restart
  - if restart fails, terminate the session and report the worker failure
- If frame latency exceeds budget:
  - mark the frame as degraded
  - report timing metrics
  - do not silently substitute old frames
- `photoreal` must not silently downgrade to `fast`.

## Validation Changes

Extend the existing validation harness with photoreal-specific checks:

- `mesh_backed_environment`
  - verify all scenario placements resolve to real mesh objects
- `worker_health`
  - startup success, restarts, timeouts
- `frame_latency`
  - render time, IPC time, total frame time
- `pass_output_presence`
  - RGB/depth/semantic/instance output presence per frame

Existing validation categories remain:

- reconstruction
- calibration
- object detection
- segmentation
- scene graph
- llm explanation

## Delivery Phases

### Phase 1

Deliver one scenario end-to-end:

- `studio_open_v1`
- Blender worker process
- realtime RGB/depth/semantic/instance rendering
- `BlenderRealtimeFrameSource`
- environment isometric view
- validation for one scenario

### Phase 2

Expand to the remaining five scenarios and tune performance.

### Phase 3

Raise realism and throughput:

- better masks
- better actor movement
- asset LOD tuning
- richer lighting/material variants

## Risks

- Eevee quality may still fall short of “real world” expectations if assets/materials are weak.
- `10-15 FPS` can collapse if draw-call count or render pass count grows uncontrolled.
- Moving actors with rigged animation can become the main bottleneck.
- macOS GUI automation and Blender lifecycle management may be fragile without careful worker restart logic.

## Recommendation

Implement the Blender realtime path first for `studio_open_v1` only, and treat it as the reference architecture for the rest of the ladder. Do not attempt to convert all six scenarios simultaneously before the worker, pass extraction, and UI are proven stable.
