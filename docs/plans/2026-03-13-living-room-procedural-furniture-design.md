# Living Room Procedural Furniture Detail Design

**Date:** 2026-03-13

**Goal:** Replace single-box furniture meshes in the living-room simulation with richer procedural furniture assemblies that look closer to real household objects while preserving renderer consistency and real-time performance.

## Context

The current living-room scene uses mostly one-box geometry for each large object. That keeps the renderer simple but makes the environment view and camera frames look synthetic in an unhelpful way. The scene already has the right object categories and placement; the problem is geometric fidelity, not layout.

The simulation must keep these constraints:

- No external 3D assets.
- The Open3D environment view and the camera renderer must describe the same room contents.
- Semantic reasoning remains at the object level, not at the per-part level.
- Real-time responsiveness must stay acceptable on CPU.

## Chosen Direction

Use a shared procedural furniture builder and a mixed detail strategy.

- Build furniture from multiple simple parts instead of one box per object.
- Keep both renderers aligned by deriving their geometry from the same procedural definitions.
- Spend more geometry budget on camera-salient furniture and less on background items.

This keeps the visual improvement large without introducing the maintenance cost of divergent renderer-specific scene logic.

## Detail Levels

### High-detail objects

- Sofa
- Dining table
- Dining chairs
- TV console
- Front window frame

These objects are frequently visible and visually dominant. They should be assembled from multiple slabs, panels, legs, and cushions.

### Medium-detail objects

- Coffee table
- TV panel
- Floor, walls, ceiling

These remain simpler but should still have visible structural separation where it matters.

## Geometry Strategy

### Shared scene builder

Keep [sim_scene.py](/Users/chasoik/Projects/obj-recog/src/obj_recog/sim_scene.py) as the source of truth for procedural geometry. Instead of returning one `SceneMeshComponent` per furniture item, it should return multiple components per object.

Examples:

- Sofa:
  - base plinth
  - seat platform
  - backrest
  - left arm
  - right arm
  - seat cushions
  - back cushions
- Dining table:
  - tabletop
  - apron frame
  - four legs
- Dining chair:
  - seat
  - backrest
  - four legs
- TV console:
  - top slab
  - side panels
  - center shelf or cabinet blocks
- Front window:
  - glass panel
  - left, right, top, bottom frame members

### Semantic behavior

Subcomponents must keep the parent semantic category. A sofa cushion should still be labeled `sofa`, not a new category. This avoids confusing segmentation, object summaries, and LLM planning.

### Primitive budget

Target part counts:

- Sofa: 7-9 parts
- Dining table: 5-6 parts
- Dining chair: about 6 parts each
- TV console: 4-5 parts
- Coffee table: about 4 parts

No curved meshes or imported assets are needed. The target style is "architectural blockout with furniture structure," not photoreal CAD.

## Renderer Alignment

### Open3D environment view

[visualization.py](/Users/chasoik/Projects/obj-recog/src/obj_recog/visualization.py) should keep combining all returned components into room and object meshes exactly as it does now, but it will receive many more object components.

### Camera renderer

[realtime_worker.py](/Users/chasoik/Projects/obj-recog/scripts/blender/realtime_worker.py) should stop generating one axis-aligned primitive per furniture object. It should instead consume the same procedural part definitions and emit one render primitive per returned component.

The camera renderer can stay box-based internally, but boxes now correspond to meaningful furniture parts instead of whole objects.

## Materials

Keep materials lightweight and procedural.

- Sofa: slight cushion/base contrast in the same fabric family
- Wood furniture: top/frame/leg tone variation
- TV console: darker body with lighter top or shelf accents
- Window frame: off-white or muted aluminum tone

No UV workflow or image textures are needed for this change.

## Testing Strategy

Add tests that verify structure, not just existence.

- [tests/test_sim_scene.py](/Users/chasoik/Projects/obj-recog/tests/test_sim_scene.py)
  - sofa expands into multiple components
  - dining table expands into tabletop plus support parts
  - dining chair expands into seat/back/leg structure
  - front window frame parts exist separately from glass
- [tests/test_visualization.py](/Users/chasoik/Projects/obj-recog/tests/test_visualization.py)
  - environment viewer still combines detailed components successfully
- [tests/test_realtime_worker_script.py](/Users/chasoik/Projects/obj-recog/tests/test_realtime_worker_script.py)
  - worker still emits RGB/depth/semantic/instance outputs with increased primitive counts

## Non-Goals

- No imported Blender models
- No curved subdivision surfaces
- No animation or soft-body behavior
- No change to room layout, robot behavior, planner protocol, or success criteria

## Acceptance Criteria

- Environment Model shows visibly more realistic furniture structure.
- Camera frames show furniture with recognizable legs, arms, backs, frames, and cushions.
- Open3D and camera renderer remain scene-consistent.
- Semantic labels remain object-level.
- Existing simulation runtime remains stable and tests cover the new component structure.
