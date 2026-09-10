#!/usr/bin/env python3
"""Bake Unity/SimLab Collada assemblies into single OBJ meshes for MuJoCo."""

from __future__ import annotations

from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "meshes"
DST = ROOT / "assets" / "meshes"

PARTS = [
    "armor_base",
    "armor_light",
    "base_link",
    "yaw_link",
    "pitch_link",
    "wheel_link",
    "camera_link",
]


def scene_to_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    pieces = [
        geom
        for geom in loaded.dump()
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
    ]
    if not pieces:
        raise RuntimeError(f"no triangles in {path}")
    return trimesh.util.concatenate(pieces)


def export_number_plate(path: Path) -> None:
    """Quad on the armor +X face, 120mm square, with UVs for the sticker."""
    path.write_text(
        "\n".join(
            [
                "v 0.001 -0.060 -0.060",
                "v 0.001  0.060 -0.060",
                "v 0.001  0.060  0.060",
                "v 0.001 -0.060  0.060",
                "vt 0 0",
                "vt 1 0",
                "vt 1 1",
                "vt 0 1",
                "vn 1 0 0",
                "f 1/1/1 2/2/1 3/3/1",
                "f 1/1/1 3/3/1 4/4/1",
                "",
            ]
        )
    )


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name in PARTS:
        src = SRC / f"{name}.dae"
        dst = DST / f"{name}.obj"
        print(f"converting {src.name} -> {dst.name}")
        mesh = scene_to_mesh(src)
        verts = mesh.vertices
        faces = mesh.faces + 1
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
        lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
        dst.write_text("\n".join(lines) + "\n")
        print(f"  faces={len(mesh.faces)} bounds={mesh.bounds.tolist()}")

    plate = DST / "armor_number.obj"
    export_number_plate(plate)
    print(f"wrote {plate.name}")


if __name__ == "__main__":
    main()
