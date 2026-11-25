# Post-processing across ALL saved timesteps.
#
# Adapted from the older axisymmetric (r, z) version for this workspace,
# where stresses/principal directions are already computed in full 3D
# XYZ cartesian coordinates by 02_principal_stress.py (Principal_stress_1/2/3,
# Principal_direction_1/2/3, sorted descending: 1 = most tensile,
# 3 = most compressive).
#
# Adds, for every *_principal.vtu step file in INPUT_DIR:
#   1. Sign-harmonized principal direction glyphs (tension + compression),
#      colored by stress sign/magnitude.
#   2. Mean-principal-stress domain classification (tension / compression /
#      neutral), same idea as the old post_process.py.
#
# Sign convention ("reharmonize with respect to center coordinates"):
# principal directions are headless (v and -v represent the same axis), so
# the eigensolver has no reason to pick a consistent sign from one point to
# the next. Rather than propagate signs from neighbor to neighbor (which
# depends on traversal order and can drift), each vector is flipped so it
# points RADIALLY OUTWARD from the sphere's center:
#
#     if dot(v, position - center) < 0: v = -v
#
# For a point to the right of center (+X) this makes the vector point
# toward +X ("right"); to the left (-X), toward -X ("left"); above center
# (+Z), "up"; below (-Z), "down" -- and likewise for Y -- because the
# dominant component of (position - center) sets the sign of the whole
# vector. This is a purely geometric rule (no mesh connectivity / KD-tree
# needed) and is exact by construction for a problem that is symmetric
# about its center.
#
# Exception: direction 3 (most compressive) instead points TOWARD the
# z = Z_PLANE plane -- -Z for points above it, +Z for points below -- so the
# arrows visually "squeeze" inward and emphasize compression. Nearly
# horizontal dir3 vectors fall back to pointing radially inward.
#
# Outputs are grouped by file type into their own subdirectory under
# OUTPUT_DIR (domain/, glyphs_tension/, glyphs_compression/,
# glyphs_compression_dir3/, glyph_source/), one file per timestep, with the
# zero-padded step number at the end of each filename (e.g.
# glyphs_tension_001.vtp) so a whole subdirectory can be opened in
# ParaView as a time series.

import glob
import os
import re

import numpy as np
import pyvista as pv

# ============================================================
# 0. CONFIGURATION
# ============================================================

# INPUT_DIR = "./4-procs"
# INPUT_DIR = "/Users/kbko/Documents/Dani_Simulations/LoadAndUnloadButRapid/4-procs"
# INPUT_DIR = "/Users/kbko/Documents/Dani_Simulations/PlatesMoving/1-procs"
INPUT_DIR = "/Users/kbko/Documents/Dani_Simulations/LoadTwoStepsRapid/4-procs"
# OUTPUT_DIR = "./sign_harmonized_v2"
OUTPUT_DIR = "/Users/kbko/Documents/Dani_Simulations/LoadTwoStepsRapid/sign_harmonized"


OUT_SUBDIRS = {
    "domain": "domain",
    "glyphs_tension": "glyphs_tension",
    "glyphs_compression": "glyphs_compression",
    "glyphs_tension_dir2": "glyphs_tension_dir2",
    "glyphs_compression_dir2": "glyphs_compression_dir2",
    "glyphs_compression_dir3": "glyphs_compression_dir3",
    "glyph_source": "glyph_source",
}

# Fraction of the sphere radius used to select "near center" points for
# the orientation diagnostic printout.
CENTER_TOL_FRACTION = 0.1

# Arrow/glyph size, as a fraction of the mesh's bounding-box diagonal
# (keeps glyphs a sensible size regardless of model units). Arrow length
# within each subset is directly proportional to the raw |sigma| value --
# no percentile capping/normalization.
GLYPH_SCALE_FRACTION = 0.1

# Arrow thickness multiplier relative to PyVista's default arrow (shaft
# radius 0.05, tip radius 0.1, both relative to a unit-length arrow).
# 1.0 = default, 2.0 = twice as thick, 0.5 = half. Arrows are scaled
# uniformly by |sigma|, so thickness still grows with length; this only
# changes the thickness-to-length ratio.
GLYPH_THICKNESS = 1.0

DISPLACEMENT_FIELD_NAME = "Displacement"

# Coarsen the glyphs: instead of one arrow per node, bin the nodes into a
# uniform 3D grid with cell edge GLYPH_BIN_FRACTION * (bbox diagonal),
# average the Cauchy stress TENSOR over the nodes in each bin, re-solve the
# eigenproblem, and draw one arrow per bin at the bin's node centroid.
# (Averaging the tensor, not the eigenvectors, is required because
# eigenvectors are headless and can swap order between neighbors.) Larger
# fraction -> fewer, coarser arrows. Set COARSEN_GLYPHS = False to go back
# to one arrow per node.
COARSEN_GLYPHS = False
GLYPH_BIN_FRACTION = 0.04  # ~650 arrows here; 0.08 -> ~300, 0.04 -> ~1900
STRESS_TENSOR_FIELD_NAME = "Cauchy_stress"  # Voigt: XX, YY, ZZ, XY, YZ, XZ

# Alternative to COARSEN_GLYPHS: down-sample AFTER the nodal eigen-
# decomposition and sign harmonization. Uses the same GLYPH_BIN_FRACTION
# grid, but instead of averaging, keeps the single node closest to each
# bin's node centroid, so every arrow shows exact nodal sigma/direction
# values. Bins are built from the reference (undeformed) node coordinates,
# so the same nodes are kept at every timestep. Mutually exclusive with
# COARSEN_GLYPHS.
DOWNSAMPLE_GLYPHS = True 

# Direction-3 (compressive) glyphs point toward this plane: -Z above it,
# +Z below it.
Z_PLANE = 0.0

# Print the near-center orientation diagnostic table for every timestep
# (verbose) or only for the last (loaded) one.
PRINT_DIAGNOSTIC_EVERY_STEP = False


def get_point_array(dataset, name):
    """Fetch a named array, explicitly from point data (02_principal_stress.py
    writes everything as point data)."""
    return np.asarray(dataset.point_data[name])


def orient_radially_outward(vectors, radial):
    """Flip each headless vector so it points away from the center,
    i.e. dot(vector, radial) >= 0."""
    oriented = vectors.copy()
    sign = np.sign(np.einsum("ij,ij->i", oriented, radial))
    # Points exactly at the center (sign == 0) are left unflipped.
    flip = sign < 0
    oriented[flip] *= -1.0
    return oriented


def orient_toward_z_plane(vectors, points, radial, z_plane=0.0, eps=1.0e-6):
    """Flip each headless vector so it points TOWARD the plane z = z_plane
    (points above it get a -Z component, points below get +Z), to emphasize
    compression. Vectors that are nearly horizontal (|v_z| < eps), where the
    Z rule is ambiguous, fall back to pointing radially inward."""
    oriented = vectors.copy()
    dz = points[:, 2] - z_plane
    vz = oriented[:, 2]

    use_z = np.abs(vz) >= eps
    flip_z = use_z & (vz * dz > 0)

    inward = np.einsum("ij,ij->i", oriented, radial)
    flip_radial = ~use_z & (inward > 0)

    oriented[flip_z | flip_radial] *= -1.0
    return oriented


def voigt_to_tensor(voigt):
    """(N, 6) Voigt [XX, YY, ZZ, XY, YZ, XZ] -> (N, 3, 3) symmetric tensor."""
    xx, yy, zz, xy, yz, xz = voigt.T
    tensor = np.zeros((voigt.shape[0], 3, 3))
    tensor[:, 0, 0] = xx
    tensor[:, 1, 1] = yy
    tensor[:, 2, 2] = zz
    tensor[:, 0, 1] = tensor[:, 1, 0] = xy
    tensor[:, 1, 2] = tensor[:, 2, 1] = yz
    tensor[:, 0, 2] = tensor[:, 2, 0] = xz
    return tensor


def principal_from_tensor(tensor):
    """Principal stresses/directions sorted descending (same convention as
    02_principal_stress.py: 1 = most tensile, 3 = most compressive)."""
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    order = np.argsort(eigenvalues, axis=1)[:, ::-1]
    eigenvalues = np.take_along_axis(eigenvalues, order, axis=1)
    eigenvectors = np.take_along_axis(
        eigenvectors, order[:, np.newaxis, :], axis=2
    )
    sigmas = [eigenvalues[:, k] for k in range(3)]
    dirs = [eigenvectors[..., k] for k in range(3)]
    return sigmas, dirs


def bin_points(points, bin_size):
    """Assign each point to a uniform grid of cubes of edge `bin_size`.
    Returns (per-point bin index, per-bin point count)."""
    ijk = np.floor((points - points.min(axis=0)) / bin_size).astype(np.int64)
    _, inverse, counts = np.unique(
        ijk, axis=0, return_inverse=True, return_counts=True
    )
    return inverse.ravel(), counts


def bin_average(points, bin_size, fields):
    """Group points into a uniform grid of cubes of edge `bin_size` and
    return (bin centroids, {name: per-bin mean of each field})."""
    inverse, counts = bin_points(points, bin_size)
    n_bins = counts.size

    def mean(values):
        out = np.zeros((n_bins,) + values.shape[1:])
        np.add.at(out, inverse, values)
        return out / counts.reshape((-1,) + (1,) * (values.ndim - 1))

    return mean(points), {name: mean(v) for name, v in fields.items()}


def downsample_indices(points, bin_size):
    """Indices of one representative node per grid bin: the node closest
    to the centroid of the nodes in that bin."""
    inverse, counts = bin_points(points, bin_size)
    centroids = np.zeros((counts.size, 3))
    np.add.at(centroids, inverse, points)
    centroids /= counts[:, np.newaxis]
    dist = np.linalg.norm(points - centroids[inverse], axis=1)
    # Sort by bin, then by distance; the first entry of each bin wins.
    order = np.lexsort((dist, inverse))
    first = np.r_[True, inverse[order][1:] != inverse[order][:-1]]
    return np.sort(order[first])


def copy_time_value(src, dst):
    """Copy the solver's TimeValue field data so ParaView places this file
    at the same physical time as the *_principal.vtu / .pvd series. Without
    it ParaView falls back to the file index (0, 1, 2, ...) as time, which
    misaligns the glyphs against the other series."""
    if "TimeValue" in src.field_data.keys():
        # Explicit copy: np.asarray would return a view into src's VTK
        # buffer, and sharing that across datasets corrupts the heap.
        dst.field_data["TimeValue"] = np.array(
            src.field_data["TimeValue"], dtype=np.float64, copy=True
        )


def build_glyphs(glyph_source, bbox_diag, mask, out_path, label,
                  orient="dir1_oriented", abs_array="abs_sigma1"):
    """Build arrow glyphs for the subset selected by `mask`, oriented along
    `orient` and scaled by the raw (uncapped) `abs_array` values -- arrow
    length is directly proportional to |sigma| within this subset, with no
    percentile clipping of outliers."""
    if not np.any(mask):
        print(f"    No points in the '{label}' subset -- skipping.")
        return

    subset = glyph_source.extract_points(mask, adjacent_cells=False)
    raw_abs = np.asarray(subset[abs_array])

    raw_max = max(raw_abs.max(), 1.0e-12)
    factor = GLYPH_SCALE_FRACTION * bbox_diag / raw_max

    glyphs = subset.glyph(
        orient=orient, scale=abs_array, factor=factor,
        geom=pv.Arrow(
            shaft_radius=0.05 * GLYPH_THICKNESS,
            tip_radius=0.1 * GLYPH_THICKNESS,
        ),
    )
    copy_time_value(glyph_source, glyphs)
    glyphs.save(out_path)

    print(
        f"    Saved {out_path} ({mask.sum()} {label} arrows; raw "
        f"|{abs_array[4:]}| range [{raw_abs.min():.4g}, {raw_abs.max():.4g}] "
        f"Pa, no capping)"
    )


def step_suffix(step_file, fallback_index):
    """Zero-padded step number parsed from the input filename (e.g.
    "result_001_principal.vtu" -> "001"), so it can be appended to the end
    of each output filename. Falls back to the file's position in the
    input list if no number is found."""
    match = re.search(r"(\d+)", os.path.basename(step_file))
    number = int(match.group(1)) if match else fallback_index
    return f"{number:03d}"


def process_file(step_file, out_dirs, print_diagnostic, fallback_index):
    print(f"\n=== Processing {step_file} ===")
    results = pv.read(step_file)
    step_id = step_suffix(step_file, fallback_index)

    sigma1 = get_point_array(results, "Principal_stress_1")
    sigma2 = get_point_array(results, "Principal_stress_2")
    sigma3 = get_point_array(results, "Principal_stress_3")
    dir1_raw = get_point_array(results, "Principal_direction_1")
    dir2_raw = get_point_array(results, "Principal_direction_2")
    dir3_raw = get_point_array(results, "Principal_direction_3")

    # --- Mean-principal-stress domain classification -------------------
    mean_principal_stress = (sigma1 + sigma2 + sigma3) / 3.0
    domain = np.zeros_like(mean_principal_stress)
    domain[mean_principal_stress > 0] = 1
    domain[mean_principal_stress < 0] = 2
    domain[mean_principal_stress == 0] = 3
    results.point_data["mean_principal_stress"] = mean_principal_stress
    results.point_data["Domain"] = domain

    out_domain = os.path.join(out_dirs["domain"], f"with_domain_{step_id}.vtu")
    results.save(out_domain)
    print(f"    Saved {out_domain}")

    # --- Center-based sign harmonization --------------------------------
    points = results.points
    bounds = results.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
    center = np.array(
        [0.5 * (bounds[0] + bounds[1]),
         0.5 * (bounds[2] + bounds[3]),
         0.5 * (bounds[4] + bounds[5])]
    )
    radial = points - center

    dir1_oriented = orient_radially_outward(dir1_raw, radial)
    dir2_oriented = orient_radially_outward(dir2_raw, radial)
    # dir3 (most compressive) uses its own convention: arrows point toward
    # the Z_PLANE (down above it, up below it) to emphasize compression.
    dir3_oriented = orient_toward_z_plane(dir3_raw, points, radial, Z_PLANE)

    # --- Near-center diagnostic printout ---------------------------------
    if print_diagnostic:
        radius = np.linalg.norm(radial, axis=1)
        sphere_radius = 0.5 * max(
            bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
        )
        center_tol = CENTER_TOL_FRACTION * sphere_radius
        near_center = radius < center_tol

        print(
            f"    Orientation diagnostic near the sphere center "
            f"({near_center.sum()} points, radius < {center_tol:.4g}):"
        )
        for idx in np.where(near_center)[0]:
            px, py, pz = points[idx]
            vx, vy, vz = dir1_oriented[idx]
            print(
                f"      x={px:9.4g} y={py:9.4g} z={pz:9.4g} "
                f"sigma1={sigma1[idx]:10.4g} sigma3={sigma3[idx]:10.4g} "
                f"dir1=({vx:6.3f},{vy:6.3f},{vz:6.3f})"
            )

    bbox_diag = np.linalg.norm(
        [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
    )
    has_disp = DISPLACEMENT_FIELD_NAME in results.point_data.keys()
    displacement = (
        get_point_array(results, DISPLACEMENT_FIELD_NAME) if has_disp else None
    )

    # --- Optional coarsening of the glyph locations ----------------------
    if COARSEN_GLYPHS and DOWNSAMPLE_GLYPHS:
        raise ValueError(
            "COARSEN_GLYPHS and DOWNSAMPLE_GLYPHS are mutually exclusive."
        )
    if DOWNSAMPLE_GLYPHS:
        keep = downsample_indices(
            np.asarray(results.points), GLYPH_BIN_FRACTION * bbox_diag
        )
        points = np.asarray(points)[keep]
        sigma1, sigma2, sigma3 = sigma1[keep], sigma2[keep], sigma3[keep]
        dir1_oriented = dir1_oriented[keep]
        dir2_oriented = dir2_oriented[keep]
        dir3_oriented = dir3_oriented[keep]
        if has_disp:
            displacement = displacement[keep]
        print(
            f"    Down-sampled glyphs: {results.n_points} nodes -> "
            f"{keep.size} nodes (bin edge "
            f"{GLYPH_BIN_FRACTION * bbox_diag:.4g})"
        )
    elif COARSEN_GLYPHS:
        fields = {
            "tensor": voigt_to_tensor(
                get_point_array(results, STRESS_TENSOR_FIELD_NAME)
            )
        }
        if has_disp:
            fields["disp"] = displacement
        points, binned = bin_average(
            np.asarray(results.points), GLYPH_BIN_FRACTION * bbox_diag, fields
        )
        (sigma1, sigma2, sigma3), (d1, d2, d3) = principal_from_tensor(
            binned["tensor"]
        )
        if has_disp:
            displacement = binned["disp"]
        radial = points - center
        dir1_oriented = orient_radially_outward(d1, radial)
        dir2_oriented = orient_radially_outward(d2, radial)
        dir3_oriented = orient_toward_z_plane(d3, points, radial, Z_PLANE)
        print(
            f"    Coarsened glyphs: {results.n_points} nodes -> "
            f"{points.shape[0]} bins (bin edge "
            f"{GLYPH_BIN_FRACTION * bbox_diag:.4g})"
        )

    # --- Glyphs of the principal directions ------------------------------
    glyph_source = pv.PolyData(points)
    glyph_source["dir1_oriented"] = dir1_oriented
    glyph_source["sigma1"] = sigma1
    glyph_source["abs_sigma1"] = np.abs(sigma1)

    glyph_source["dir2_oriented"] = dir2_oriented
    glyph_source["sigma2"] = sigma2
    glyph_source["abs_sigma2"] = np.abs(sigma2)

    glyph_source["dir3_oriented"] = dir3_oriented
    glyph_source["sigma3"] = sigma3
    glyph_source["abs_sigma3"] = np.abs(sigma3)

    if has_disp:
        glyph_source[DISPLACEMENT_FIELD_NAME] = displacement

    copy_time_value(results, glyph_source)

    out_glyph_source = os.path.join(
        out_dirs["glyph_source"], f"glyph_source_{step_id}.vtp"
    )
    glyph_source.save(out_glyph_source)
    print(f"    Saved {out_glyph_source}")

    tension_mask = sigma1 > 0
    compression_mask = ~tension_mask

    build_glyphs(
        glyph_source, bbox_diag, tension_mask,
        os.path.join(
            out_dirs["glyphs_tension"], f"glyphs_tension_{step_id}.vtp"
        ),
        "tensile",
    )
    build_glyphs(
        glyph_source, bbox_diag, compression_mask,
        os.path.join(
            out_dirs["glyphs_compression"], f"glyphs_compression_{step_id}.vtp"
        ),
        "compressive",
    )

    # Same tension/compression split, but for the 2nd (intermediate)
    # principal stress/direction.
    tension_mask_dir2 = sigma2 > 0
    compression_mask_dir2 = ~tension_mask_dir2

    build_glyphs(
        glyph_source, bbox_diag, tension_mask_dir2,
        os.path.join(
            out_dirs["glyphs_tension_dir2"],
            f"glyphs_tension_dir2_{step_id}.vtp",
        ),
        "tensile (dir2)",
        orient="dir2_oriented",
        abs_array="abs_sigma2",
    )
    build_glyphs(
        glyph_source, bbox_diag, compression_mask_dir2,
        os.path.join(
            out_dirs["glyphs_compression_dir2"],
            f"glyphs_compression_dir2_{step_id}.vtp",
        ),
        "compressive (dir2)",
        orient="dir2_oriented",
        abs_array="abs_sigma2",
    )

    # Compressive-stress glyphs proper: arrows along dir3 (the most
    # compressive principal axis) scaled by |sigma3|, wherever sigma3 is
    # actually compressive.
    compression_mask_dir3 = sigma3 < 0
    build_glyphs(
        glyph_source, bbox_diag, compression_mask_dir3,
        os.path.join(
            out_dirs["glyphs_compression_dir3"],
            f"glyphs_compression_dir3_{step_id}.vtp",
        ),
        "compressive (dir3)",
        orient="dir3_oriented",
        abs_array="abs_sigma3",
    )


def main():
    step_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_principal.vtu")))
    if not step_files:
        raise SystemExit(f"No *_principal.vtu files found in {INPUT_DIR}")

    out_dirs = {
        key: os.path.join(OUTPUT_DIR, subdir)
        for key, subdir in OUT_SUBDIRS.items()
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"Found {len(step_files)} step file(s) in {INPUT_DIR}")
    for i, step_file in enumerate(step_files):
        is_last = i == len(step_files) - 1
        process_file(
            step_file, out_dirs,
            print_diagnostic=PRINT_DIAGNOSTIC_EVERY_STEP or is_last,
            fallback_index=i,
        )

    print(
        f"\nDone. Each output type is grouped in its own subdirectory under "
        f"{OUTPUT_DIR}/, one file per timestep, named with the zero-padded "
        f"step number at the end of the filename (e.g. "
        f"glyphs_tension_001.vtp) so a subdirectory can be opened directly "
        f"in ParaView as a time series (File > Open, select all files in "
        f"the folder).\n"
        "\nAll principal-direction vectors were sign-harmonized to point "
        "radially outward from the sphere's center (right of center -> "
        "points right, left of center -> points left, above -> up, below "
        "-> down), rather than by nearest-neighbor propagation, since this "
        "problem is symmetric about its center. Exception: direction 3 "
        f"(compressive) points toward z = {Z_PLANE} (down above it, up "
        "below it) to emphasize compression.\n"
        f"\nTo view everything in the deformed (loaded) configuration, "
        f"apply ParaView's 'Warp By Vector' filter (Vectors = "
        f"'{DISPLACEMENT_FIELD_NAME}') to every source you have open, with "
        "the same Scale Factor."
    )


if __name__ == "__main__":
    main()
