# Interior Test TV Scenario Design

**Date:** 2026-03-13

**Goal:** Add a second simulation scenario that uses `/Users/chasoik/Downloads/InteriorTest.blend` as the actual living-room world, starts from a fixed point, and ends when the robot reaches the space in front of the TV.

## Context

The current simulation path is centered on a single procedural scene, `living_room_navigation_v1`. That scene is useful for validating the closed-loop architecture, but it does not support authored Blender interiors. The new request is materially different:

- the world source must be an existing `.blend` file
- the camera renderer must use that real authored scene
- the Open3D `Environment Model` must also show that same real structure
- the robot must still plan from camera-derived observations only
- the test ends when the robot reaches a hidden goal in front of the TV

This means the authored Blender scene has to become a first-class simulation scene source rather than just a rendering template.

## Chosen Direction

Use the `.blend` file as the single scene truth source for the new scenario `interior_test_tv_navigation_v1`.

- Blender remains the authoritative scene source for camera rendering.
- A new extraction layer reads mesh/object transforms from the `.blend` file and converts them into an internal scene description.
- Open3D uses that extracted scene description to display the same room structure in `Environment Model`.
- Hidden evaluation, start pose, and collision proxies are derived from the extracted scene, but planner-visible state remains camera-only.

This preserves the core closed-loop rule while keeping the authored scene and operator view consistent.

## Scenario Definition

### Scenario ID

- `interior_test_tv_navigation_v1`

### Scene Source

- `/Users/chasoik/Downloads/InteriorTest.blend`

### Anchor Objects

The inspected Blender file contains a `TV` mesh at approximately:

- Blender world location: `(0.0, 3.926, 1.322)`

This object is the goal anchor for the scenario.

### Start Pose

Use a fixed start pose in the open lower area of the room:

- sim pose: `(x=0.0, y=1.25, z=-2.8, yaw_deg=0.0, camera_pan_deg=0.0)`

This places the robot away from the TV and requires meaningful navigation through the room.

### Goal Pose

Use a hidden target point approximately `0.8m` in front of the TV toward the room interior.

- target description: `arrive in front of the TV`
- success radius: `0.5m`

The exact coordinates remain hidden runtime truth and are not shown to the planner.

## Scene Abstraction

The existing `LivingRoomSceneSpec` is procedural-room specific. The new scenario needs a scene abstraction that can represent:

- authored `.blend` world source
- extracted static mesh preview data
- hidden goal anchor metadata
- collision proxy geometry
- operator-view mesh groups

The implementation should generalize the sim scene spec so both scenario families can coexist:

- procedural room scenes
- blend-backed authored scenes

The current procedural scenario stays intact.

## Coordinate Handling

Blender and the current sim runtime do not use the same semantic axis conventions. The runtime should standardize all extracted authored-scene coordinates into the sim convention before they are used by:

- hidden world state
- camera pose updates
- collision checks
- Open3D environment display
- goal evaluation

The conversion layer should be explicit and tested. The main goal is consistency, not preserving raw Blender axes.

## Extraction Layer

Add a loader that opens the `.blend` file in Blender headless mode and emits a normalized scene manifest.

The manifest should include:

- object name
- object type
- world transform
- dimensions or bounds
- mesh preview geometry or a lightweight proxy reference
- semantic hints for key objects like `TV`, walls, windows, floor, and large furniture

This extracted manifest becomes the internal description used by simulation and visualization.

## Rendering Path

### Camera Rendering

The Blender worker should support an authored-scene mode:

- open the specified `.blend`
- reuse its actual objects, materials, and lighting
- place the robot camera according to runtime pose
- render RGB, depth, semantic, and instance outputs

This is different from the current procedural worker path, which builds the scene from code.

### Environment Model

The Open3D `Environment Model` must use the extracted authored-scene mesh data instead of procedural room geometry. It should show:

- real walls and openings
- real furniture placement
- TV position
- current robot pose marker

This view is for the operator only. It can display hidden structure, but that information must not leak into planner input.

## Runtime Behavior

The runtime loop stays the same at the top level:

- self calibration
- perceive and plan
- execute schedule
- reassess
- succeed or fail

What changes is the world source and evaluator:

- hidden goal is in front of the authored `TV`
- hidden world geometry comes from the authored scene
- camera renders from the authored scene

## Collision Handling

The current runtime updates robot position directly without authored-scene collision checks. The new scenario requires hidden collision proxies for major static geometry, especially:

- walls
- TV wall zone
- table and large furniture
- piano and large corner objects

These collision checks are runtime-only truth. They prevent the robot from moving through real scene structure but are not exposed directly to the planner.

## Planner Boundary

The planner-visible rule does not change.

The planner receives only:

- current RGB-derived perception outputs
- depth summaries
- reconstruction summaries
- graph relations
- object and segment observations
- recent action history
- calibration/tracking health

The planner does not receive:

- exact goal coordinates
- object list from the extracted scene manifest
- hidden collision geometry
- unseen authored objects

If the TV is not visible from the camera, the planner should not be allowed to know its exact position.

## Testing Strategy

Add coverage in these areas:

- config and scenario registry
- authored-scene extraction and normalization
- Blender worker authored-scene startup and rendering
- Open3D environment rendering from extracted real meshes
- runtime success evaluation for the TV-front goal
- planner redaction boundary for authored scenes

The tests should prove that:

- the new scenario is selectable
- the authored `.blend` is parsed successfully
- the worker can render frames from it
- the environment view uses extracted real structure
- final success is based on hidden goal proximity only
- planner context does not expose authored-scene truth directly

## Non-Goals

- No replacement of the procedural scenario
- No manual remodeling of the `.blend` scene into procedural geometry
- No dependence on external web assets
- No change to the planner protocol itself beyond scenario metadata support

## Acceptance Criteria

- `--scenario interior_test_tv_navigation_v1` runs successfully.
- Camera rendering uses `/Users/chasoik/Downloads/InteriorTest.blend`.
- `Environment Model` displays the actual extracted scene structure from the same `.blend`.
- The robot starts from the fixed lower-room position and succeeds only when reaching the hidden TV-front target.
- Planner-visible state remains observation-only and does not expose authored-scene truth.
