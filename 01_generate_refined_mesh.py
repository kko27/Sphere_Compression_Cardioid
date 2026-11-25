"""
Generate a family of svFSI-ready sphere meshes for a mesh-refinement study.

Replaces the STL -> MeshMixer -> SimVascular GUI workflow with gmsh. Each
level is meshed from an ideal sphere (not the faceted coarse surface), so the
geometric error shrinks along with h.

Run with the fenicsx conda env:
    conda activate fenicsx
    python 01_generate_refined_mesh.py

For each target edge length h in MESH_SIZES this writes OUT_ROOT/h{h:.1f}/
with the same layout as mesh_scaled/:

    mesh-complete.mesh.vtu          linear tets, GlobalNodeID / GlobalElementID / ModelRegionID
    mesh-complete.domains.vtu       + DOMAIN_ID (cell centroid r <= RADIUS_THRESHOLD -> 2, else 1)
    mesh-complete.exterior.vtp      whole sphere surface
    walls_combined.vtp              same as exterior
    fibersLong.vtu / fibersSheet.vtu / fibersNormal.vtu   FIB_DIR cell data
    crescent_domains/mesh-complete.domains_rotXXX.vtu
    mesh-surfaces/
        sphere_wall.vtp             whole sphere surface
        sphere_top.vtp              z >= 0 hemisphere   (RigidPlanePenalty, wall_top)
        sphere_bottom.vtp           z <= 0 hemisphere   (RigidPlanePenalty, wall_bottom)
        pin_axis_top.vtp            1 triangle touching the node at (0, 0, +R)
        pin_axis_bottom.vtp         1 triangle touching the node at (0, 0, -R)
        cap_top.vtp / cap_bottom.vtp  cells near the poles (inspection only)

How the surfaces are kept consistent:
  * The sphere is built from 8 octant patches (gmsh built-in kernel). The
    equator is a geometric curve, so every surface triangle is in exactly one
    of sphere_top / sphere_bottom: no gap band and no overlap.
  * The poles (0, 0, +/-R) are geometric points, so a mesh node sits exactly
    on the compression axis at every refinement level. The pin triangle is
    built on that node.
  * Every face triangle carries the GlobalNodeID of its points and the
    GlobalElementID of the tet that owns it. That is what svFSI's face reader
    needs to match faces against the volume mesh. Face normals point outward.
"""

import pathlib

import gmsh
import numpy as np
import pyvista as pv

# --- configuration ----------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent
OUT_ROOT = HERE / "mesh_refined"

# Sphere in scaled (physical) units: unit-diameter sphere x 700 (physical_scale.py).
RADIUS = 350.0
CENTER = (0.0, 0.0, 0.0)

# Target edge lengths. H0 matches the mean surface edge of mesh_scaled/ (~43.3).
H0 = 43.3
MESH_SIZES = [H0, H0 / 2.0, H0 / 4.0]

# gmsh algorithms: 2D 6 = Frontal-Delaunay, 3D 1 = Delaunay (+ Netgen optimization).
GMSH_ALGO_2D = 6
GMSH_ALGO_3D = 1
GMSH_NUM_THREADS = 0  # 0 = gmsh default
GMSH_VERBOSITY = 2

# Pole caps (inspection only, as in 01_create_gen_pinpoints.py): a surface cell
# is in a cap if its centroid is within CAP_TOL_FRAC * (2R) of the pole plane.
CAP_TOL_FRAC = 0.02

# Domains (same logic as 00_assign_domains.py).
RADIUS_THRESHOLD = 300.0
CRESCENT_SHAPE = True
PERCENTAGE_COVERAGE = 0.7
MAX_ROTATION_DEG = 180.0
N_ROTATIONS = 4

# Fibers: L = e_phi (circumferential about z) tilted by independent Gaussian
# angles toward e_theta and e_r (std FIBER_NOISE_DEG), N = e_r orthogonalized
# against L, S = L x N. This matches the statistics of the mesh_scaled/ fibers
# (~10 deg std per component, Gaussian). Set FIBER_NOISE_DEG = 0 for pure fibers.
FIBER_NOISE_DEG = 10.0
FIBER_SEED = 0


# --- geometry + meshing -----------------------------------------------------


def mesh_sphere(radius, h, center=CENTER):
    """Mesh a sphere with gmsh. Returns (points, tets, tri_top, tri_bottom)
    with 0-based connectivity into `points`."""
    cx, cy, cz = center
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", GMSH_VERBOSITY)
        if GMSH_NUM_THREADS:
            gmsh.option.setNumber("General.NumThreads", GMSH_NUM_THREADS)
        gmsh.model.add("sphere")
        geo = gmsh.model.geo

        c = geo.addPoint(cx, cy, cz, h)
        px = geo.addPoint(cx + radius, cy, cz, h)
        py = geo.addPoint(cx, cy + radius, cz, h)
        mx = geo.addPoint(cx - radius, cy, cz, h)
        my = geo.addPoint(cx, cy - radius, cz, h)
        pz = geo.addPoint(cx, cy, cz + radius, h)  # top pole
        mz = geo.addPoint(cx, cy, cz - radius, h)  # bottom pole

        equator = [px, py, mx, my]
        eq_arcs = [geo.addCircleArc(equator[i], c, equator[(i + 1) % 4]) for i in range(4)]
        up_arcs = [geo.addCircleArc(p, c, pz) for p in equator]
        dn_arcs = [geo.addCircleArc(p, c, mz) for p in equator]

        top_surfs, bot_surfs = [], []
        for i in range(4):
            j = (i + 1) % 4
            lt = geo.addCurveLoop([eq_arcs[i], up_arcs[j], -up_arcs[i]], reorient=True)
            lb = geo.addCurveLoop([eq_arcs[i], dn_arcs[j], -dn_arcs[i]], reorient=True)
            top_surfs.append(geo.addSurfaceFilling([lt], sphereCenterTag=c))
            bot_surfs.append(geo.addSurfaceFilling([lb], sphereCenterTag=c))

        sl = geo.addSurfaceLoop(top_surfs + bot_surfs)
        vol = geo.addVolume([sl])
        geo.synchronize()

        gmsh.model.addPhysicalGroup(2, top_surfs, name="sphere_top")
        gmsh.model.addPhysicalGroup(2, bot_surfs, name="sphere_bottom")
        gmsh.model.addPhysicalGroup(3, [vol], name="sphere")

        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Algorithm", GMSH_ALGO_2D)
        gmsh.option.setNumber("Mesh.Algorithm3D", GMSH_ALGO_3D)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = coords.reshape(-1, 3)
        tag2idx = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
        tag2idx[tags.astype(np.int64)] = np.arange(len(tags))

        def tris_of(surfs):
            out = []
            for s in surfs:
                _, ntags = gmsh.model.mesh.getElementsByType(2, s)
                out.append(tag2idx[ntags.astype(np.int64)].reshape(-1, 3))
            return np.vstack(out)

        tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(4, vol)
        tets = tag2idx[tet_nodes.astype(np.int64)].reshape(-1, 4)
        q = gmsh.model.mesh.getElementQualities(tet_tags, "minSICN")
        tri_top, tri_bot = tris_of(top_surfs), tris_of(bot_surfs)
    finally:
        gmsh.finalize()

    # Compact: drop nodes not used by any tet (e.g. the sphere-center point).
    used = np.unique(tets)
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    points = coords[used]
    tets, tri_top, tri_bot = remap[tets], remap[tri_top], remap[tri_bot]
    assert (tri_top >= 0).all() and (tri_bot >= 0).all()

    # VTK wants positively oriented tets.
    a, b, cc, d = (points[tets[:, k]] for k in range(4))
    neg = np.einsum("ij,ij->i", np.cross(b - a, cc - a), d - a) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]

    print(f"  gmsh: {len(points)} nodes, {len(tets)} tets, "
          f"{len(tri_top)} top + {len(tri_bot)} bottom tris; "
          f"tet minSICN min {q.min():.3f} mean {q.mean():.3f}")
    return points, tets, tri_top, tri_bot


# --- svFSI-style VTK objects ------------------------------------------------


def face_owner_lookup(tets):
    """Map each tet face (sorted node triplet) to its owning tet index."""
    faces = np.vstack([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]],
                       tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]])
    owner = np.tile(np.arange(len(tets)), 4)
    keys = np.sort(faces, axis=1)
    return {tuple(k): o for k, o in zip(keys.tolist(), owner.tolist())}, keys


def boundary_face_keys(keys):
    """Sorted triplets of tet faces that appear exactly once (domain boundary)."""
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    return {tuple(k) for k in uniq[counts == 1].tolist()}


def volume_grid(points, tets):
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, points)
    grid.point_data["GlobalNodeID"] = np.arange(1, len(points) + 1, dtype=np.int32)
    grid.cell_data["ModelRegionID"] = np.ones(len(tets), dtype=np.int32)
    grid.cell_data["GlobalElementID"] = np.arange(1, len(tets) + 1, dtype=np.int32)
    return grid


def face_polydata(points, tris, owner_lookup, center=CENTER):
    """Build a face .vtp: outward-oriented triangles, local points, GlobalNodeID
    per point, and the owning tet's GlobalElementID per cell."""
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3).copy()
    p0, p1, p2 = (points[tris[:, k]] for k in range(3))
    normal = np.cross(p1 - p0, p2 - p0)
    inward = np.einsum("ij,ij->i", normal, (p0 + p1 + p2) / 3.0 - np.asarray(center)) < 0
    tris[inward] = tris[inward][:, [0, 2, 1]]

    owners = np.array([owner_lookup[tuple(sorted(t))] for t in tris.tolist()])

    gnodes, local = np.unique(tris, return_inverse=True)
    poly = pv.PolyData.from_regular_faces(points[gnodes], local.reshape(-1, 3))
    poly.point_data["GlobalNodeID"] = (gnodes + 1).astype(np.int32)
    poly.cell_data["GlobalElementID"] = (owners + 1).astype(np.int32)
    poly.cell_data["ModelFaceID"] = np.ones(len(tris), dtype=np.int32)
    return poly


def pin_triangle(points, tris, pole):
    """Index of one triangle in `tris` touching the node nearest `pole`."""
    node = int(np.argmin(np.linalg.norm(points - np.asarray(pole), axis=1)))
    dist = np.linalg.norm(points[node] - np.asarray(pole))
    if dist > 1e-6 * RADIUS:
        raise RuntimeError(f"No mesh node at pole {pole} (nearest is {dist:g} away)")
    touching = np.flatnonzero((tris == node).any(axis=1))
    return int(touching[0]), node


def cap_triangles(points, tris, plane_z, tol):
    cz = points[tris].mean(axis=1)[:, 2]
    return np.flatnonzero(np.abs(cz - plane_z) <= tol)


# --- domains + fibers ---------------------------------------------------------


def assign_domains(grid, out_dir):
    cc = grid.cell_centers().points - np.asarray(CENTER)
    radii = np.linalg.norm(cc, axis=1)
    dom = grid.copy()
    dom.cell_data["DOMAIN_ID"] = np.where(radii <= RADIUS_THRESHOLD, 2, 1).astype(np.int32)
    dom.save(out_dir / "mesh-complete.domains.vtu")

    if not CRESCENT_SHAPE:
        return
    cres_dir = out_dir / "crescent_domains"
    cres_dir.mkdir(exist_ok=True)
    half_angle = np.deg2rad(PERCENTAGE_COVERAGE * 360.0 / 2.0)
    outer = radii > RADIUS_THRESHOLD
    unit = cc / np.maximum(radii, 1e-12)[:, None]
    for rot_deg in np.linspace(0.0, MAX_ROTATION_DEG, N_ROTATIONS + 1):
        rot = np.deg2rad(rot_deg)
        axis = np.array([0.0, np.sin(rot), np.cos(rot)])
        polar = np.arccos(np.clip(unit @ axis, -1.0, 1.0))
        dom.cell_data["DOMAIN_ID"] = np.where(outer & (polar <= half_angle), 1, 2).astype(np.int32)
        dom.save(cres_dir / f"mesh-complete.domains_rot{int(round(rot_deg)):03d}.vtu")


def _unit(v):
    return v / np.linalg.norm(v, axis=1)[:, None]


def compute_fibers(grid, noise_deg=FIBER_NOISE_DEG, seed=FIBER_SEED):
    cc = grid.cell_centers().points - np.asarray(CENTER)
    e_r = _unit(cc)
    e_phi = np.cross([0.0, 0.0, 1.0], e_r)
    on_axis = np.linalg.norm(e_phi, axis=1) < 1e-8
    e_phi[on_axis] = np.cross([1.0, 0.0, 0.0], e_r[on_axis])  # arbitrary tangent on the axis
    e_phi = _unit(e_phi)
    e_theta = np.cross(e_phi, e_r)

    rng = np.random.default_rng(seed)
    sig = np.deg2rad(noise_deg)
    a_theta = rng.normal(0.0, sig, len(cc)) if sig > 0 else np.zeros(len(cc))
    a_r = rng.normal(0.0, sig, len(cc)) if sig > 0 else np.zeros(len(cc))

    long_ = _unit(e_phi + np.tan(a_theta)[:, None] * e_theta + np.tan(a_r)[:, None] * e_r)
    normal = _unit(e_r - np.einsum("ij,ij->i", e_r, long_)[:, None] * long_)
    sheet = np.cross(long_, normal)
    return long_, sheet, normal


def write_fibers(grid, out_dir):
    for name, vec in zip(("Long", "Sheet", "Normal"), compute_fibers(grid)):
        g = grid.copy()
        g.cell_data["FIB_DIR"] = vec.astype(np.float64)
        g.save(out_dir / f"fibers{name}.vtu")


# --- checks -------------------------------------------------------------------


def verify(points, tets, faces, boundary_keys):
    """Sanity checks that would otherwise only show up as svFSI read errors."""
    n_pts, n_tets = len(points), len(tets)
    for name, poly in faces.items():
        gn = poly.point_data["GlobalNodeID"]
        ge = poly.cell_data["GlobalElementID"]
        assert gn.min() >= 1 and gn.max() <= n_pts, name
        assert ge.min() >= 1 and ge.max() <= n_tets, name
        assert np.allclose(poly.points, points[gn - 1]), name
        tri = poly.regular_faces
        for t, e in zip(gn[tri] - 1, ge - 1):
            key = tuple(sorted(t.tolist()))
            assert key in boundary_keys, f"{name}: triangle is not a boundary face"
            assert set(key) <= set(tets[e].tolist()), f"{name}: GlobalElementID mismatch"

    top = {tuple(sorted(t)) for t in (faces["sphere_top"].point_data["GlobalNodeID"]
                                      [faces["sphere_top"].regular_faces]).tolist()}
    bot = {tuple(sorted(t)) for t in (faces["sphere_bottom"].point_data["GlobalNodeID"]
                                      [faces["sphere_bottom"].regular_faces]).tolist()}
    assert not (top & bot), "top/bottom overlap"
    assert len(top | bot) == len(boundary_keys), "top + bottom do not cover the whole surface"
    print(f"  verified: {len(top)} top + {len(bot)} bottom = {len(boundary_keys)} boundary faces, "
          "no overlap, all GlobalElementIDs own their faces")


# --- driver -------------------------------------------------------------------


def build_level(h):
    out_dir = OUT_ROOT / f"h{h:.1f}"
    surf_dir = out_dir / "mesh-surfaces"
    surf_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== h = {h:.3f} -> {out_dir.relative_to(HERE)}")

    points, tets, tri_top, tri_bot = mesh_sphere(RADIUS, h)
    owner_lookup, keys = face_owner_lookup(tets)
    boundary_keys = boundary_face_keys(keys)

    grid = volume_grid(points, tets)
    grid.save(out_dir / "mesh-complete.mesh.vtu")

    tri_all = np.vstack([tri_top, tri_bot])
    cx, cy, cz = CENTER
    pin_top, node_top = pin_triangle(points, tri_top, (cx, cy, cz + RADIUS))
    pin_bot, node_bot = pin_triangle(points, tri_bot, (cx, cy, cz - RADIUS))
    tol = CAP_TOL_FRAC * 2.0 * RADIUS

    faces = {
        "sphere_wall": face_polydata(points, tri_all, owner_lookup),
        "sphere_top": face_polydata(points, tri_top, owner_lookup),
        "sphere_bottom": face_polydata(points, tri_bot, owner_lookup),
        "pin_axis_top": face_polydata(points, tri_top[[pin_top]], owner_lookup),
        "pin_axis_bottom": face_polydata(points, tri_bot[[pin_bot]], owner_lookup),
        "cap_top": face_polydata(points, tri_top[cap_triangles(points, tri_top, cz + RADIUS, tol)], owner_lookup),
        "cap_bottom": face_polydata(points, tri_bot[cap_triangles(points, tri_bot, cz - RADIUS, tol)], owner_lookup),
    }
    verify(points, tets, faces, boundary_keys)

    for name, poly in faces.items():
        poly.save(surf_dir / f"{name}.vtp", binary=not name.startswith("pin_"))
    faces["sphere_wall"].save(out_dir / "mesh-complete.exterior.vtp")
    faces["sphere_wall"].save(out_dir / "walls_combined.vtp")

    for tag, node in (("top", node_top), ("bottom", node_bot)):
        gids = faces[f"pin_axis_{tag}"].point_data["GlobalNodeID"]
        print(f"  pin_axis_{tag}: pole node GlobalNodeID {node + 1} at {points[node]}, "
              f"triangle GlobalNodeIDs {gids.tolist()}")

    assign_domains(grid, out_dir)
    write_fibers(grid, out_dir)

    edges = grid.extract_all_edges().compute_cell_sizes(length=True, area=False, volume=False)
    s_edges = faces["sphere_wall"].extract_all_edges().compute_cell_sizes(length=True, area=False, volume=False)
    print(f"  edge length: volume mean {edges.cell_data['Length'].mean():.2f}, "
          f"surface mean {s_edges.cell_data['Length'].mean():.2f} (target {h:.2f})")
    return out_dir


def main():
    OUT_ROOT.mkdir(exist_ok=True)
    for h in MESH_SIZES:
        build_level(h)


if __name__ == "__main__":
    main()
