# Reads every mesh file (.vtu / .vtp) found under MESH_DIR (recursively) and
# applies a uniform scaling to the point coordinates by MESH_FACTOR.
#
# All point/cell data arrays are preserved. By default the scaled meshes are
# written to OUT_DIR, mirroring the directory layout of MESH_DIR. Set
# IN_PLACE = True to overwrite the original files instead.
#
# Run with an environment that has pyvista, e.g.:
#   conda run -n fenicsx python physical_scale.py

import pathlib

import numpy as np
import pyvista as pv

MESH_DIR = pathlib.Path("./mesh")
OUT_DIR = pathlib.Path("./mesh_scaled")
MESH_FACTOR = 700.0

IN_PLACE = False
SCALE_CENTER = (0.0, 0.0, 0.0)  # point held fixed by the scaling

EXTENSIONS = (".vtu", ".vtp")


def find_mesh_files(root):
    files = []
    for ext in EXTENSIONS:
        files.extend(sorted(root.rglob(f"*{ext}")))
    return files


def scale_mesh(mesh, factor, center):
    center = np.asarray(center, dtype=float)
    mesh.points = (mesh.points - center) * factor + center
    return mesh


def main():
    if not MESH_DIR.is_dir():
        raise SystemExit(f"MESH_DIR does not exist: {MESH_DIR.resolve()}")

    mesh_files = find_mesh_files(MESH_DIR)
    if not mesh_files:
        raise SystemExit(f"No {'/'.join(EXTENSIONS)} files found under {MESH_DIR.resolve()}")

    print(f"Scaling {len(mesh_files)} mesh file(s) by factor {MESH_FACTOR} "
          f"about {SCALE_CENTER}")

    for src in mesh_files:
        if IN_PLACE:
            dst = src
        else:
            dst = OUT_DIR / src.relative_to(MESH_DIR)
            dst.parent.mkdir(parents=True, exist_ok=True)

        mesh = pv.read(src)
        b0 = mesh.bounds
        scale_mesh(mesh, MESH_FACTOR, SCALE_CENTER)
        b1 = mesh.bounds
        mesh.save(dst)

        print(f"  {src}  ->  {dst}")
        print(f"      bounds {tuple(round(v, 6) for v in b0)}")
        print(f"          -> {tuple(round(v, 6) for v in b1)}")

    print("Done.")


if __name__ == "__main__":
    main()
