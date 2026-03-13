from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and export a single asset into cache artifacts")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--archive-type", required=True)
    parser.add_argument("--extract-member-glob", required=True)
    parser.add_argument("--import-format", required=True)
    parser.add_argument("--source-object-name", default="")
    parser.add_argument("--export-object-name", required=True)
    parser.add_argument("--uniform-scale", type=float, default=1.0)
    parser.add_argument("--up-axis", default="Z")
    parser.add_argument("--forward-axis", default="-Y")
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--output-preview-mesh", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_args:
        raw_args = raw_args[raw_args.index("--") + 1 :]
    args = build_arg_parser().parse_args(raw_args)

    try:
        import bpy  # type: ignore
    except Exception as exc:  # pragma: no cover - only exercised inside Blender.
        raise RuntimeError("asset bootstrap worker requires Blender bpy runtime") from exc

    input_root = Path(args.input_root)
    source_candidates = sorted(input_root.rglob("*"))
    source_path = next((path for path in source_candidates if path.is_file()), None)
    if source_path is None:
        raise RuntimeError(f"no source asset found under {input_root}")

    _import_source_asset(
        bpy=bpy,
        source_path=source_path,
        import_format=str(args.import_format),
        up_axis=str(args.up_axis),
        forward_axis=str(args.forward_axis),
    )

    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not imported_objects:
        raise RuntimeError(f"no mesh objects were imported from {source_path}")

    if args.source_object_name:
        target = next((obj for obj in imported_objects if obj.name == args.source_object_name), None)
        if target is None:
            raise RuntimeError(f"source object {args.source_object_name!r} not found in imported asset")
        imported_objects = [target]

    target = _join_mesh_objects(bpy=bpy, objects=imported_objects, export_object_name=str(args.export_object_name))
    target.scale = (float(args.uniform_scale), float(args.uniform_scale), float(args.uniform_scale))
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.ops.export_mesh.ply(filepath=str(args.output_preview_mesh))
    bpy.data.libraries.write(str(Path(args.output_blend)), {target})
    return 0


def _import_source_asset(*, bpy, source_path: Path, import_format: str, up_axis: str, forward_axis: str) -> None:
    if import_format == "blend":
        with bpy.data.libraries.load(str(source_path), link=False) as (data_from, data_to):
            if not data_from.objects:
                raise RuntimeError(f"blend asset {source_path} contains no objects")
            data_to.objects = list(data_from.objects)
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.scene.collection.objects.link(obj)
        return
    if import_format == "glb":
        bpy.ops.import_scene.gltf(filepath=str(source_path))
        return
    if import_format == "fbx":
        bpy.ops.import_scene.fbx(filepath=str(source_path), axis_up=up_axis, axis_forward=forward_axis)
        return
    if import_format == "obj":
        import_obj = getattr(getattr(bpy.ops, "wm", object()), "obj_import", None)
        if callable(import_obj):
            import_obj(filepath=str(source_path), up_axis=up_axis, forward_axis=forward_axis)
        else:
            bpy.ops.import_scene.obj(filepath=str(source_path), axis_up=up_axis, axis_forward=forward_axis)
        return
    raise RuntimeError(f"unsupported import format: {import_format}")


def _join_mesh_objects(*, bpy, objects, export_object_name: str):
    if len(objects) == 1:
        obj = objects[0]
        obj.name = export_object_name
        return obj
    for selected in tuple(bpy.context.selected_objects):
        selected.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = export_object_name
    return joined


if __name__ == "__main__":  # pragma: no cover - exercised manually inside Blender.
    raise SystemExit(main())
