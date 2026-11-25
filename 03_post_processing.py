# Plots the applied plate deformation together with the resulting first
# principal stress in the organoid's equatorial region, one curve per axis
# on a shared time axis.
#
# Deformation: read directly from top_plate_step.dat (the full imposed
# displacement history of the top plate, in the reference/undeformed length
# units of the mesh).
#
# Stress: read from each ./4-procs/*_principal.vtu file (written by
# 02_principal_stress.py). At every saved timestep, the "equatorial region"
# is defined as the points within a thin band around the Z mid-plane of the
# *reference* (undeformed) mesh -- the point coordinates in these files are
# the reference configuration, with the actual deformation stored in the
# "Displacement" array. The first principal stress (Principal_stress_1) is
# averaged over the points that fall in that band.
#
# Run with an environment that has pyvista + matplotlib, e.g. the sibling
# Sphere_Compression/.venv used elsewhere in this repo.

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

INPUT_DIR = "./4-procs"
TOP_PLATE_FILE = "./top_plate_step.dat"
OUTPUT_PLOT = "03_deformation_vs_equatorial_stress_2.svg"

# Equatorial band half-width, expressed as a fraction of the sphere radius.
# Points within +/- this fraction of the radius of the Z mid-plane are
# treated as "equatorial" and averaged together at each timestep.
EQUATOR_BAND_FRACTION = 0.02

# How to combine sigma_1 across the points inside the equatorial band.
AGGREGATION = "mean"  # one of: "mean", "max", "median"


def load_top_plate_deformation(path):
    """Read the full imposed top-plate displacement history."""
    data = np.loadtxt(path)
    nt, _nf = data[0]
    nt = int(nt)
    time_disp = data[1 : nt + 1, :]  # rows of [time, displacement]
    return time_disp[:, 0], time_disp[:, 1]


def aggregate(values, how):
    if how == "mean":
        return float(np.mean(values))
    if how == "max":
        return float(np.max(values))
    if how == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown aggregation method: {how}")


def equatorial_sigma1(vtu_file, band_fraction, how):
    """Return (time, aggregated sigma_1, n_points_in_band) for one file."""
    mesh = pv.read(vtu_file)

    time_value = np.asarray(mesh["TimeValue"]).ravel()[0]

    z = mesh.points[:, 2]
    z_min, z_max = z.min(), z.max()
    z_mid = 0.5 * (z_min + z_max)
    radius = 0.5 * (z_max - z_min)
    tol = band_fraction * radius

    in_band = np.abs(z - z_mid) <= tol
    n_points = int(in_band.sum())
    if n_points == 0:
        raise RuntimeError(
            f"No points fell inside the equatorial band for {vtu_file} "
            f"(z_mid={z_mid:.4g}, tol={tol:.4g}). Widen EQUATOR_BAND_FRACTION."
        )

    sigma1 = mesh["Principal_stress_1"][in_band]
    return time_value, aggregate(sigma1, how), n_points


def main():
    # --- Applied deformation (full time history) -----------------------
    plate_time, plate_disp = load_top_plate_deformation(TOP_PLATE_FILE)

    # --- Equatorial first principal stress at each saved timestep ------
    vtu_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_principal.vtu")))
    if not vtu_files:
        raise SystemExit(f"No *_principal.vtu files found in {INPUT_DIR}")

    stress_time, stress_sigma1 = [], []
    for vtu_file in vtu_files:
        t, sigma1_agg, n_points = equatorial_sigma1(
            vtu_file, EQUATOR_BAND_FRACTION, AGGREGATION
        )
        stress_time.append(t)
        stress_sigma1.append(sigma1_agg)
        print(
            f"{os.path.basename(vtu_file)}: t={t:g}, "
            f"{AGGREGATION}(sigma_1)={sigma1_agg:.4g} Pa "
            f"({n_points} points in equatorial band)"
        )

    order = np.argsort(stress_time)
    stress_time = np.asarray(stress_time)[order]
    stress_sigma1 = np.asarray(stress_sigma1)[order]

    # --- Plot: deformation and stress vs. time, shared x-axis ----------
    fig, ax_deform = plt.subplots(figsize=(9, 5.5))

    (line_deform,) = ax_deform.plot(
        plate_time, plate_disp, color="tab:blue", lw=1.5, label="Top plate displacement"
    )
    ax_deform.set_xlabel("Time", fontsize = 14)
    ax_deform.set_ylabel("Top plate displacement ($\\mu $m)", color="tab:blue", fontsize = 14)
    ax_deform.tick_params(axis="y", labelcolor="tab:blue", labelsize = 12)
    ax_deform.set_xlim(plate_time.min(), plate_time.max())

    ax_stress = ax_deform.twinx()
    (line_stress,) = ax_stress.plot(
        stress_time,
        stress_sigma1,
        color="tab:red",
        marker="o",
        lw=1.5,
        label=f"Equatorial {AGGREGATION}",
    )
    ax_stress.set_ylabel("Equatorial tensile stress (Pa)", color="tab:red", fontsize = 14)
    ax_stress.tick_params(axis="y", labelcolor="tab:red", labelsize = 12)

    ax_deform.legend(
        handles=[line_deform, line_stress],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize = 12
    )
    ax_deform.set_title("Applied deformation vs. equatorial tensile stress", pad=32, fontsize = 16)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT)
    print(f"\nSaved plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
