from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return argv[1:]
    index = argv.index("--")
    return argv[index + 1 :]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(_argv_after_double_dash(argv or sys.argv))


def build_manifest_payload() -> dict[str, object]:
    import bpy

    depsgraph = bpy.context.evaluated_depsgraph_get()
    objects: list[dict[str, object]] = []
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        evaluated_object = obj.evaluated_get(depsgraph)
        mesh = evaluated_object.to_mesh()
        if mesh is None:
            continue

        mesh.calc_loop_triangles()
        vertices_xyz: list[list[float]] = []
        triangles: list[list[int]] = []
        for vertex in mesh.vertices:
            coordinate = evaluated_object.matrix_world @ vertex.co
            x = float(coordinate.x)
            y = float(coordinate.y)
            z = float(coordinate.z)
            vertices_xyz.append([x, y, z])
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)
        for triangle in mesh.loop_triangles:
            triangles.append([int(index) for index in triangle.vertices])

        dimensions = getattr(obj, "dimensions", (0.0, 0.0, 0.0))
        objects.append(
            {
                "object_id": obj.name,
                "object_type": obj.type,
                "center_xyz": [
                    float(obj.location.x),
                    float(obj.location.y),
                    float(obj.location.z),
                ],
                "size_xyz": [
                    float(dimensions[0]),
                    float(dimensions[1]),
                    float(dimensions[2]),
                ],
                "yaw_deg": float(obj.rotation_euler.z) * 57.29577951308232,
                "vertices_xyz": vertices_xyz,
                "triangles": triangles,
            }
        )
        evaluated_object.to_mesh_clear()

    room_size_xyz = (
        [0.0, 0.0, 0.0]
        if min_x == float("inf")
        else [
            float(max_x - min_x),
            float(max_z - min_z),
            float(max_y - min_y),
        ]
    )
    return {
        "blend_file_path": bpy.data.filepath,
        "room_size_xyz": room_size_xyz,
        "objects": objects,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_manifest_payload()
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_json": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
