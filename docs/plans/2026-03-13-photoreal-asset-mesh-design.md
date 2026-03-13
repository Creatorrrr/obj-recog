# Photoreal Asset Mesh Design

**Problem**

`photoreal` rendering currently links cached `.blend` asset libraries when available, but the cache only guarantees preview sprites. When the `.blend` files are missing, the Blender worker falls back to coarse primitives, so the camera view is not a good proxy for real object-recognition performance.

**Recommended Approach**

Generate recognizable procedural asset meshes on demand inside the Blender worker, save them into the existing asset-cache library path, and reuse those cached `.blend` files on later runs. Keep YOLO inference bound to the raw camera frame buffer, with only the existing inference resize step.

**Alternatives Considered**

1. Download and import third-party asset packs directly.
This would provide higher fidelity, but the current catalog URLs are landing pages rather than stable direct-download/import recipes. It is brittle and not a good fit for an immediate runtime fix.

2. Improve the current primitive fallback only in-memory.
This would help visuals during one run, but it would not populate the asset cache and would repeat the work every time.

**Design**

- Add procedural mesh blueprints for common semantic assets used in the simulation scenes, especially `chair`, `desk`, `potted plant`, `backpack`, `suitcase`, and `laptop`.
- When the Blender worker cannot link `placement.blender_library_path`, procedurally build the mesh, save it as a `.blend` library at that path, then use that object for rendering.
- Keep the YOLO path unchanged in principle: detect from the raw `frame_bgr` that also underlies the user-visible camera view, not from an alternate synthetic view or GT overlay.
- Add regression tests for:
  - procedural asset blueprints being nontrivial for key assets
  - `process_frame()` using the raw camera frame buffer for detector input, with only inference resizing

**Success Criteria**

- First `photoreal` run populates missing cached `.blend` libraries automatically.
- Camera renders show recognizable mesh silhouettes for key scene objects instead of plain cuboids.
- YOLO continues to run on the same raw camera pixels the user sees in the object-recognition view, aside from resizing for inference.
