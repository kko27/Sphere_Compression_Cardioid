"""
Post-processing of the LOADED (final compression) state only.

Builds on post_process.py's mean-principal-stress domain
classification and adds:

    1. Glyphs            -- arrows at cell centers along the in-plane
                            principal directions, colored by
                            tension/compression
    2. Orientation angle -- the theta1 field itself, plus a printed
                            diagnostic table near the sphere's center
                            to directly check the tension direction

Input:
    Reads a single .vtu step file (the final/loaded state written by
    full_model_pipeline.py) rather than the whole results.pvd time
    series, since only the loaded state is of interest here.

If the .vtu already contains the orientation fields written by the
patched full_model_pipeline.py (principal_angle_inplane_deg,
sigma1_inplane, sigma2_inplane, out_of_plane_governs,
principal_dir1_inplane, principal_dir2_inplane), those are used
directly. Otherwise everything is recomputed here from
sigma_rr/sigma_tt/sigma_zz/sigma_rz, so this script also works on
older result files that predate the orientation-angle patch.
"""

import numpy as np
import pyvista as pv

# ============================================================
# 0. CONFIGURATION
# ============================================================

# The loaded state = the last load step written by
# full_model_pipeline.py (n_steps = 30 there). Update this if you
# change n_steps, or point it at a different step to compare.
STEP_FILE = "./results/results_p0_000030.vtu"

OUT_DOMAIN_VTU = "results_p0_000030_with_domain.vtu"
OUT_GLYPHS_TENSION_VTP = "results_p0_000030_glyphs_tension.vtp"
OUT_GLYPHS_COMPRESSION_VTP = "results_p0_000030_glyphs_compression.vtp"
# Glyphs along the SECOND in-plane principal direction (dir2), i.e.
# the minor / most-compressive in-plane principal axis. dir1 points
# along sigma1 (major); dir2 is perpendicular to it and points along
# sigma2 (minor), so these arrows show the compressive-stress
# direction and magnitude directly, everywhere sigma2_inplane < 0.
OUT_GLYPHS_COMPRESSION_DIR2_VTP = \
    "results_p0_000030_glyphs_compression_dir2.vtp"
OUT_GLYPH_SOURCE_VTP = "results_p0_000030_glyph_source.vtp"

# Fraction of the sphere radius used to select "near center" cells
# for the orientation-angle diagnostic printout.
CENTER_TOL_FRACTION = 0.05

# Number of nearest neighbors used when harmonizing eigenvector
# sign for the glyphs (see orient_eigenvector_field).
SIGN_HARMONIZE_K = 6

# Arrow/glyph size, as a fraction of the mesh's bounding-box
# diagonal (keeps glyphs a sensible size regardless of model units).
GLYPH_SCALE_FRACTION = 0.03

# Percentile (not the raw max) used to normalize glyph length. Both
# Hertzian contact and diametral-compression solutions have sharp
# stress concentrations right at the contact-patch edge; letting a
# handful of outlier cells set the whole scale makes every other
# arrow in the plot invisible. Values above this percentile are
# capped (saturate at maximum arrow length) rather than stretching
# the scale to fit them.
GLYPH_SCALE_PERCENTILE = 95.0

# Name of the nodal displacement field written by
# full_model_pipeline.py's u_vis Function (see section 15 there).
DISPLACEMENT_FIELD_NAME = "displacement"


# ============================================================
# 1. LOAD THE LOADED-STATE FILE
# ============================================================

results = pv.read(STEP_FILE)


def get_cell_array(dataset, name):
    """Fetch a named array, explicitly from cell data (DG0 fields
    are cell-wise constant, so this avoids any ambiguity with
    point-data arrays of the same name)."""
    return np.asarray(dataset.cell_data[name])


has_orientation_fields = "principal_angle_inplane_deg" in results.cell_data.keys()

sigma_rr = get_cell_array(results, "sigma_rr")
sigma_tt = get_cell_array(results, "sigma_tt")
sigma_zz = get_cell_array(results, "sigma_zz")
sigma_rz = get_cell_array(results, "sigma_rz")

if has_orientation_fields:
    print("Using orientation fields already written by "
          "full_model_pipeline.py.")

    theta1_deg = get_cell_array(results, "principal_angle_inplane_deg")
    sigma1_inplane = get_cell_array(results, "sigma1_inplane")
    sigma2_inplane = get_cell_array(results, "sigma2_inplane")
    out_of_plane_governs = get_cell_array(results, "out_of_plane_governs")
    dir1_raw = get_cell_array(results, "principal_dir1_inplane")[:, :2]

    # dir2 = minor in-plane principal direction (along sigma2). Use
    # the field written by full_model_pipeline.py if present,
    # otherwise just rotate dir1 by 90 deg in the (r, z) plane.
    if "principal_dir2_inplane" in results.cell_data.keys():
        dir2_raw = get_cell_array(results, "principal_dir2_inplane")[:, :2]
    else:
        dir2_raw = np.column_stack((-dir1_raw[:, 1], dir1_raw[:, 0]))

else:
    print("Orientation fields not found in this .vtu -- "
          "recomputing from sigma_rr/sigma_tt/sigma_zz/sigma_rz.")

    avg = 0.5 * (sigma_rr + sigma_zz)
    diff = 0.5 * (sigma_rr - sigma_zz)
    in_plane_radius = np.sqrt(diff**2 + sigma_rz**2)

    sigma1_inplane = avg + in_plane_radius
    sigma2_inplane = avg - in_plane_radius

    theta1 = 0.5 * np.arctan2(2.0 * sigma_rz, sigma_rr - sigma_zz)
    theta1_deg = np.degrees(theta1)

    out_of_plane_governs = (
        (sigma_tt > sigma1_inplane) | (sigma_tt < sigma2_inplane)
    ).astype(np.float64)

    dir1_raw = np.column_stack((np.cos(theta1), np.sin(theta1)))
    # dir2 is perpendicular to dir1 in the (r, z) plane (along sigma2).
    dir2_raw = np.column_stack((-np.sin(theta1), np.cos(theta1)))

    # Stash these back onto the mesh so downstream steps (domain
    # classification, saving) can treat both cases identically.
    results.cell_data["principal_angle_inplane_deg"] = theta1_deg
    results.cell_data["sigma1_inplane"] = sigma1_inplane
    results.cell_data["sigma2_inplane"] = sigma2_inplane
    results.cell_data["out_of_plane_governs"] = out_of_plane_governs

n_cells = len(sigma_rr)


# ============================================================
# 2. MEAN-PRINCIPAL-STRESS DOMAIN CLASSIFICATION
#    (same logic as the original post_process.py)
# ============================================================

mean_principal_stress = results["mean_principal_stress"]

domain = np.zeros_like(mean_principal_stress)
domain[mean_principal_stress > 0] = 1
domain[mean_principal_stress < 0] = 2
domain[mean_principal_stress == 0] = 3
results.cell_data["Domain"] = domain

results.save(OUT_DOMAIN_VTU)
print(f"Saved {OUT_DOMAIN_VTU}")


# ============================================================
# 3. ORIENTATION ANGLE: NEAR-CENTER DIAGNOSTIC PRINTOUT
# ============================================================
#
# Sanity check for "is the tension at the center pointing the right
# way": near the axis of symmetry AND near the sphere's equatorial
# center, expect
#   sigma_rr ~ sigma_tt  (both tensile, positive)
#   sigma_zz              compressive, negative
#   theta1 ~ 0 deg         (larger in-plane principal axis along r)
# because sigma_rz -> 0 there by symmetry.

centers = results.cell_centers().points
r_coord = centers[:, 0]
z_coord = centers[:, 1]

bounds = results.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
r_extent = bounds[1] - bounds[0]
z_extent = bounds[3] - bounds[2]
z_mid = 0.5 * (bounds[2] + bounds[3])

r_tol = CENTER_TOL_FRACTION * r_extent
z_tol = CENTER_TOL_FRACTION * z_extent

near_center = (r_coord < r_tol) & (np.abs(z_coord - z_mid) < z_tol)

print(f"\nOrientation-angle diagnostic near the sphere center "
      f"({near_center.sum()} cells, r < {r_tol:.4g}, "
      f"|z - {z_mid:.4g}| < {z_tol:.4g}):")
print(f"{'r':>10} {'z':>10} {'sigma_rr':>12} {'sigma_tt':>12} "
      f"{'sigma_zz':>12} {'theta1_deg':>12} {'tension_axis':>14} "
      f"{'out_of_plane':>13}")

for idx in np.where(near_center)[0]:
    tension_axis = "radial" if sigma1_inplane[idx] > 0 else "axial/none"
    print(f"{r_coord[idx]:10.4g} {z_coord[idx]:10.4g} "
          f"{sigma_rr[idx]:12.4g} {sigma_tt[idx]:12.4g} "
          f"{sigma_zz[idx]:12.4g} {theta1_deg[idx]:12.4g} "
          f"{tension_axis:>14} "
          f"{bool(out_of_plane_governs[idx]):>13}")

print(
    "\nExpected physical picture (Hiramatsu-Oka / Hertz double-"
    "contact): sigma_rr = sigma_tt > 0 (tensile, hoop/radial), "
    "sigma_zz < 0 (compressive, along the load axis), theta1 ~ 0 "
    "deg (larger in-plane principal axis pointing radially, i.e. "
    "perpendicular to the compression axis). A theta1 far from 0 "
    "deg, or sigma1_inplane < 0, near the center indicates the "
    "tensile direction is NOT where it should be."
)


# ============================================================
# 4. SIGN-HARMONIZED IN-PLANE PRINCIPAL DIRECTION FIELD
# ============================================================
#
# Eigenvectors are "headless": dir1 and -dir1 represent the same
# principal axis, and the eigensolver/formula used per-cell has no
# reason to pick consistent signs between neighboring cells. Naive
# averaging on such a field produces spurious cancellation/kinks at
# sign-flip boundaries. This does a greedy nearest-neighbor sign
# harmonization (same idea used to consistently orient point-cloud
# normals) so that the glyphs below are visually coherent.

try:
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise ImportError(
        "scipy is required for eigenvector sign harmonization "
        "(pip install scipy)"
    ) from exc


def orient_eigenvector_field(points_2d, vectors_2d, k=6):
    """
    Greedy BFS sign-harmonization of a headless (eigenvector)
    vector field, so neighboring vectors point "the same way"
    along their shared axis wherever the field varies smoothly.

    points_2d  : (N, 2) array of sample locations
    vectors_2d : (N, 2) array of unit (or near-unit) directions,
                 each defined only up to an overall sign
    k          : number of nearest neighbors used to propagate
                 orientation

    Returns a new (N, 2) array with harmonized signs.
    """
    n = points_2d.shape[0]
    oriented = vectors_2d.copy()
    visited = np.zeros(n, dtype=bool)

    tree = cKDTree(points_2d)
    _, neighbor_idx = tree.query(points_2d, k=min(k + 1, n))

    # Process connected regions independently (guards against
    # duplicate/coincident points).
    for start in range(n):
        if visited[start]:
            continue

        visited[start] = True
        queue = [start]

        while queue:
            i = queue.pop()
            for j in neighbor_idx[i]:
                if j == i or visited[j]:
                    continue
                # Flip vector j if it points "against" the already-
                # oriented vector i along their shared axis.
                if np.dot(oriented[i], oriented[j]) < 0.0:
                    oriented[j] = -oriented[j]
                visited[j] = True
                queue.append(j)

    return oriented


dir1_oriented = orient_eigenvector_field(
    centers[:, :2], dir1_raw, k=SIGN_HARMONIZE_K
)
dir1_oriented_3d = np.column_stack(
    (dir1_oriented, np.zeros(n_cells))
)  # pad to 3 components (r, z, 0) for VTK vectors

# Same sign-harmonization for the minor (compressive) in-plane
# principal direction, so the compression glyphs below are visually
# coherent too.
dir2_oriented = orient_eigenvector_field(
    centers[:, :2], dir2_raw, k=SIGN_HARMONIZE_K
)
dir2_oriented_3d = np.column_stack(
    (dir2_oriented, np.zeros(n_cells))
)


# ============================================================
# 5. GLYPHS OF THE PRINCIPAL EIGENVECTOR
# ============================================================
#
# Split into tension (sigma1_inplane > 0) and compression
# (sigma1_inplane < 0) so direction AND sign are both visible at a
# glance -- this is the direct visual check for "is the center
# really in tension, in the right direction".

glyph_source = pv.PolyData(centers)

# Deformed-configuration consistency: `centers` above are cell
# centroids of the REFERENCE (undeformed) mesh -- the same
# coordinates results.vtu itself uses before you apply "Warp By
# Vector" to it. Sampling `results` onto these points pulls in the
# nodal "displacement" field (interpolated, exact for the CG1
# u_vis field written by full_model_pipeline.py) so that applying
# the identical "Warp By Vector" filter (array = "displacement",
# same Scale Factor) to these glyph files moves each arrow's
# anchor point to its correct deformed-configuration location.
# This is a pure translation of each arrow -- it does not affect
# glyph orientation/length, which are already computed from the
# (already-deformed) Cauchy stress.
glyph_source = glyph_source.sample(results)

glyph_source["dir1_oriented"] = dir1_oriented_3d
glyph_source["sigma1_inplane"] = sigma1_inplane
glyph_source["abs_sigma1"] = np.abs(sigma1_inplane)

# Minor (compressive) in-plane principal axis: direction along
# sigma2, plus its stress value and magnitude, for the compressive-
# stress glyphs built below.
glyph_source["dir2_oriented"] = dir2_oriented_3d
glyph_source["sigma2_inplane"] = sigma2_inplane
glyph_source["abs_sigma2"] = np.abs(sigma2_inplane)

# Save the raw (point + orientation vector + scalar) data too, in
# case you'd rather drive ParaView's own Glyph filter interactively
# (arrow style, density, live scale factor) instead of using the
# pre-baked arrow geometry saved below.
#   Orientation Array = dir1_oriented   (unit vector -- direction only)
#   Scale Array        = abs_sigma1      (NOT dir1_oriented's magnitude,
#                                          which is always 1)
# For the compressive stresses use dir2_oriented / abs_sigma2 instead.
glyph_source.save(OUT_GLYPH_SOURCE_VTP)
print(f"Saved {OUT_GLYPH_SOURCE_VTP} "
      f"(raw points + dir1_oriented/sigma1_inplane + "
      f"dir2_oriented/sigma2_inplane, for ParaView's own Glyph "
      f"filter)")

bbox_diag = np.linalg.norm(
    [bounds[1] - bounds[0], bounds[3] - bounds[2]]
)


def build_glyphs(mask, out_path, label, orient="dir1_oriented",
                 abs_array="abs_sigma1"):
    """
    Build arrow glyphs for the subset of cells selected by `mask`,
    oriented along `orient` and scaled by `abs_array` WITHIN THAT
    SUBSET ONLY (not the global tension+compression range), and
    capping at GLYPH_SCALE_PERCENTILE so a single stress-
    concentration cell (e.g. right at the contact-patch edge) can't
    shrink every other arrow in the plot to invisibility.

    orient    : name of the unit direction array to point along
                ("dir1_oriented" for the major/tensile axis,
                 "dir2_oriented" for the minor/compressive axis)
    abs_array : name of the |sigma| magnitude array to scale by
                (must correspond to `orient`)
    """
    if not np.any(mask):
        print(f"No cells in the '{label}' subset -- skipping.")
        return

    subset = glyph_source.extract_points(mask, adjacent_cells=False)
    raw_abs = np.asarray(subset[abs_array])

    cap = np.percentile(raw_abs, GLYPH_SCALE_PERCENTILE)
    cap = max(cap, 1.0e-12)
    capped_name = abs_array + "_capped"
    subset[capped_name] = np.minimum(raw_abs, cap)

    factor = GLYPH_SCALE_FRACTION * bbox_diag / cap

    glyphs = subset.glyph(
        orient=orient, scale=capped_name,
        factor=factor, geom=pv.Arrow(),
    )
    glyphs.save(out_path)

    n_saturated = int(np.sum(raw_abs > cap))
    print(f"Saved {out_path} ({mask.sum()} {label} arrows; "
          f"scaled to the {GLYPH_SCALE_PERCENTILE:.0f}th percentile "
          f"|{abs_array[4:]}| = {cap:.4g} Pa, raw range "
          f"[{raw_abs.min():.4g}, {raw_abs.max():.4g}] Pa, "
          f"{n_saturated} cell(s) saturated at max arrow length)")


tension_mask = sigma1_inplane > 0
compression_mask = ~tension_mask

build_glyphs(tension_mask, OUT_GLYPHS_TENSION_VTP, "tensile")
build_glyphs(compression_mask, OUT_GLYPHS_COMPRESSION_VTP, "compressive")

# Compressive-stress glyphs proper: arrows along dir2 (the minor
# in-plane principal axis, i.e. the compression direction) scaled by
# |sigma2_inplane|, wherever sigma2_inplane is actually compressive.
# For a diametral/Hertzian compression problem this is nearly the
# whole domain, and directly visualizes the compressive stress flow.
compression_mask_dir2 = sigma2_inplane < 0
build_glyphs(compression_mask_dir2, OUT_GLYPHS_COMPRESSION_DIR2_VTP,
             "compressive (dir2)", orient="dir2_oriented",
             abs_array="abs_sigma2")

print(
    "\nOpen the *_domain.vtu in ParaView for coloring, and the "
    "*_glyphs_*.vtp on top of it as separate sources (they share "
    "the same coordinate system). Color the tension/compression "
    "glyphs by 'sigma1_inplane' with a diverging colormap "
    "(e.g. blue-white-red centered at 0) to see sign at a glance. "
    "Arrow LENGTH is capped at the "
    f"{GLYPH_SCALE_PERCENTILE:.0f}th percentile of |sigma1_inplane| "
    "within each tension/compression subset separately -- check the "
    "printed 'saturated at max arrow length' counts above; if you "
    "want to see those outlier cells' true relative magnitude "
    "instead, lower GLYPH_SCALE_PERCENTILE towards 100, or raise "
    "GLYPH_SCALE_FRACTION and re-run.\n"
    "\nThe "
    f"{OUT_GLYPHS_COMPRESSION_DIR2_VTP} file holds arrows along "
    "dir2_oriented (the minor / compressive in-plane principal "
    "axis) scaled by |sigma2_inplane|, for visualizing the "
    "compressive stress direction and magnitude directly; color it "
    "by 'sigma2_inplane'.\n"
    "\nAll of these outputs carry the same '"
    f"{DISPLACEMENT_FIELD_NAME}"
    "' point field as results_p0_000030.vtu itself. To view "
    "everything in the deformed (loaded) configuration rather than "
    "the reference one, apply ParaView's 'Warp By Vector' filter "
    "(Vectors = '"
    f"{DISPLACEMENT_FIELD_NAME}"
    "') to EVERY source you have open -- the domain mesh AND each "
    "glyph file -- with "
    "the SAME Scale Factor (1.0 for true physical displacement; "
    "keep it identical across all sources if you exaggerate it, "
    "or the glyphs will visibly separate from the mesh again)."
)
