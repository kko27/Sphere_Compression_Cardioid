# Python code that reads in hoop stress data (found under ./sign_harmonized/glyphs_tension/*.vtp)
# And at the following time points [003, 008, 013, 018, 023, 028]
# Define an equitorial region as an input for the anaysis

# Plot the mean hoop stress in the equitorial region over compressive strain
# The corresponding strains as [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
#
# The glyph .vtp files contain arrow *geometry*, so each tensile node is
# replicated over every vertex of its arrow and the point coordinates are
# arrow vertices, not node positions. Each arrow vertex carries
# vtkOriginalPointIds, which indexes the matching node in
# ./sign_harmonized/glyph_source/glyph_source_XXX.vtp. We collapse the
# arrows back to one value per node with that id, and take the node's
# reference (undeformed) position from glyph_source to define the band.

import os

from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv

INPUT_DIR = "./sign_harmonized/glyphs_tension/"
SOURCE_DIR = "./sign_harmonized/glyph_source/"
OUTPUT_PLOT = "10_hoop_stress_vs_strain.svg"
OUTPUT_TEXT = "10_hoop_stress_vs_strain.csv"

# Equatorial band half-width, as a fraction of the sphere radius: nodes with
# |z - z_mid| <= EQUATOR_BAND_FRACTION * R (reference configuration) are kept.
EQUATOR_BAND_FRACTION = 0.02

TIME_STEPS = ["003", "008", "013", "018", "023", "028"]
STRAINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]


def equatorial_hoop_stress(step, band_fraction):
    """Return (mean, std, n_nodes) of sigma_1 over tensile nodes in the band."""
    glyphs = pv.read(os.path.join(INPUT_DIR, f"glyphs_tension_{step}.vtp"))
    source = pv.read(os.path.join(SOURCE_DIR, f"glyph_source_{step}.vtp"))

    # One entry per tensile node (drop the duplicated arrow vertices).
    node_ids, first = np.unique(
        np.asarray(glyphs["vtkOriginalPointIds"]), return_index=True
    )
    sigma1 = np.asarray(glyphs["sigma1"])[first]

    # Band defined on the full reference mesh so it is the same every step.
    z_all = source.points[:, 2]
    z_mid = 0.5 * (z_all.min() + z_all.max())
    radius = 0.5 * (z_all.max() - z_all.min())
    in_band = np.abs(z_all[node_ids] - z_mid) <= band_fraction * radius

    n_nodes = int(in_band.sum())
    if n_nodes == 0:
        raise RuntimeError(
            f"No tensile nodes in the equatorial band at step {step}. "
            "Widen EQUATOR_BAND_FRACTION."
        )
    return float(np.mean(sigma1[in_band])), float(np.std(sigma1[in_band])), n_nodes


def main():
    means, stds = [], []
    for step, strain in zip(TIME_STEPS, STRAINS):
        mean, std, n_nodes = equatorial_hoop_stress(step, EQUATOR_BAND_FRACTION)
        means.append(mean)
        stds.append(std)
        print(
            f"step {step}: strain={strain:.2f}, mean hoop stress={mean:.4g} Pa "
            f"(std {std:.4g}, {n_nodes} nodes in band)"
        )

    strains_pct = 100.0 * np.asarray(STRAINS)
    means = np.asarray(means)
    stds = np.asarray(stds)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(strains_pct, means - stds, means + stds,
                    color="tab:red", alpha=0.2, lw=0, label="$\\pm$1 SD")
    ax.plot(strains_pct, means, color="tab:red", marker="o", lw=1.5,
            label="Mean")

    ax.set_xlabel("Compressive strain (%)", fontsize=26)
    ax.set_ylabel("Equatorial hoop stress (Pa)", fontsize=26)
    ax.tick_params(labelsize=18)
    ax.set_xlim(strains_pct.min(), strains_pct.max())
    ax.legend(frameon=False, fontsize=16)
    # ax.set_title("Equatorial hoop stress vs. compressive strain", fontsize=30)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT)
    # Save data to CSV
    with open(OUTPUT_TEXT, "w") as f:
        f.write("Strain (%),Mean Hoop Stress (Pa),Std Dev (Pa),Number of Nodes\n")
        for i in range(len(STRAINS)):
            f.write(f"{STRAINS[i]*100:.2f},{means[i]:.4g},{stds[i]:.4g},{n_nodes}\n")
    print(f"\nSaved plot to {OUTPUT_PLOT}")
    print(f"Saved data to {OUTPUT_TEXT}")


if __name__ == "__main__":
    main()
