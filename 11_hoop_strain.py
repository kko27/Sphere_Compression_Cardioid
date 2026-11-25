# Computes the Hoop (circumferential) Strain from the Cartesian Strain Tensor in Voigt Notation
# Opens the result_XXX.vtu files from the 4-procs directory and extracts the Strain Tensor
# that is in Voigt Notation [xx,yy,zz,xy,yz,xz] and the reference coordinates of the nodes.
# The circumferential direction is e_theta about the z-axis (the compression axis) through
# the centre of the reference mesh.
#
# Both methods are evaluated at every node and written, together with the original
# arrays, to ./circumferential_strain/result_XXX.vtu (plus a .pvd collection for ParaView).
# The mean +/- SD over the equatorial band of the skin layer is plotted against
# compressive strain.
#
# Notes on the solver output (checked against Def_grad, E = 0.5*(F^T F - I)):
#   - "Strain" is the Green-Lagrange strain and its shear entries are *tensor* shear
#     (E_xy, not gamma_xy = 2*E_xy), so no factor of 1/2 is applied.
#   - The mesh points are the reference configuration; deformed = points + Displacement.

import os

from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv

INPUT_DIR = "./4-procs/"
OUTPUT_DIR = "./circumferential_strain/"
OUTPUT_PVD = "circumferential_strain.pvd"
OUTPUT_PLOT = "11_hoop_strain_vs_compression.svg"
OUTPUT_MODEL_PLOT = "11_hoop_strain_vs_models.svg"
OUTPUT_TEXT ="11_hoop_strain_vs_compression.csv"

N_STEPS = 55

# Equatorial band half-width, as a fraction of the sphere radius: nodes with
# |Z - z_mid| <= EQUATOR_BAND_FRACTION * R (reference configuration) are flagged.
EQUATOR_BAND_FRACTION = 0.02

# Skin layer thickness, as a fraction of the sphere radius: nodes with
# R - |X - center| <= SKIN_LAYER_FRACTION * R (reference configuration) are flagged.
SKIN_LAYER_FRACTION = 0.05

# Steps used for the summary plot and their nominal compressive strains.
TIME_STEPS = ["003", "008", "013", "018", "023", "028"]
STRAINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

# Fit windows for the analytical models in the model-comparison plot (compressive strain, fraction).
HERTZ_FIT_MAX = 0.10   # Hertz C fitted to eps_z <= this
INCOMP_FIT_MIN = 0.15  # Incompressible scale K fitted to eps_z >= this


def hoop_strain_tensor(X, E):
    """Option 1: rotate the Green-Lagrange tensor into e_theta.

    X: (n,3) reference coords relative to the axis; E: (n,6) Voigt [xx,yy,zz,xy,yz,xz].
    E_tt = e_theta . E . e_theta with e_theta = (-sin t, cos t, 0).
    """
    theta = np.arctan2(X[:, 1], X[:, 0])
    s, c = np.sin(theta), np.cos(theta)
    exy = E[:, 3]  # tensor shear (see notes above)
    return E[:, 0] * s**2 + E[:, 1] * c**2 - 2 * exy * s * c


def hoop_strain_geometric(X, u):
    """Option 2: hoop stretch r/R of the material circle about the z-axis.

    r = sqrt(x^2+y^2) (deformed), R = sqrt(X^2+Y^2) (reference). Exact for an
    axisymmetric deformation with no twist. Returns (engineering, Green-Lagrange).
    """
    x = X + u
    r = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    R = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        stretch = np.where(R > 0, r / R, np.nan)
    return stretch - 1, 0.5 * (stretch**2 - 1)


def process_step(step, center, in_band, in_skin, in_region):
    mesh = pv.read(os.path.join(INPUT_DIR, f"result_{step}.vtu"))
    X = mesh.points - center
    E = np.asarray(mesh.point_data["Strain"])
    u = np.asarray(mesh.point_data["Displacement"])

    eps_tt = hoop_strain_tensor(X, E)
    eps_tt2_eng, eps_tt2_gl = hoop_strain_geometric(X, u)

    mesh.point_data["Hoop_strain_tensor_GL"] = eps_tt
    mesh.point_data["Hoop_strain_geometric_eng"] = eps_tt2_eng
    mesh.point_data["Hoop_strain_geometric_GL"] = eps_tt2_gl
    mesh.point_data["Hoop_strain_GL_difference"] = eps_tt - eps_tt2_gl
    mesh.point_data["Equatorial_band"] = in_band.astype(np.uint8)
    mesh.point_data["Skin_layer"] = in_skin.astype(np.uint8)
    mesh.point_data["Equatorial_skin"] = in_region.astype(np.uint8)
    mesh.save(os.path.join(OUTPUT_DIR, f"result_{step}.vtu"))

    return {
        "tensor_GL": eps_tt[in_region],
        "geometric_eng": eps_tt2_eng[in_region],
        "geometric_GL": eps_tt2_gl[in_region],
    }


def write_pvd(steps):
    with open(os.path.join(OUTPUT_DIR, OUTPUT_PVD), "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n  <Collection>\n')
        for i, step in enumerate(steps):
            f.write(f'    <DataSet timestep="{i + 1}" part="0" file="result_{step}.vtu"/>\n')
        f.write("  </Collection>\n</VTKFile>\n")


def plot_model_comparison(eps_z, mu, sd):
    """Tensor GL hoop strain vs. two analytical forms explaining the non-linearity.

    Hertz contact:            eps_t = C * eps_z^1.5, C fitted to the low-strain means
                              (eps_z <= HERTZ_FIT_MAX).
    Incompressibility (GL):   lambda_t = (1 - eps_z)^-0.5  =>  eps_t = K * 0.5*(1/(1 - eps_z) - 1),
                              K fitted to the high-strain means (eps_z >= INCOMP_FIT_MIN).
    Both fits are least squares through the origin. All strains are fractions here; plotted in %.
    """
    def hertz_form(e):
        return e**1.5

    def incomp_form(e):
        return 0.5 * (1.0 / (1.0 - e) - 1.0)

    def fit_scale(f, mask):
        g = f(eps_z[mask])
        return np.sum(mu[mask] * g) / np.sum(g**2)

    C = fit_scale(hertz_form, eps_z <= HERTZ_FIT_MAX + 1e-12)
    K = fit_scale(incomp_form, eps_z >= INCOMP_FIT_MIN - 1e-12)

    # Combined (illustrative, not a standard closed form): eps_t = C * eps_z^1.5 * (1 - eps_z)^-n.
    # C is kept from the low-strain Hertz fit, so this reduces to Hertz as eps_z -> 0; only n is
    # fitted, by least squares in log space over all eps_z > 0:
    #   log(mu / (C eps_z^1.5)) = -n log(1 - eps_z).
    pos = eps_z > 0
    L = np.log(1.0 - eps_z[pos])
    n = -np.sum(np.log(mu[pos] / (C * hertz_form(eps_z[pos]))) * L) / np.sum(L**2)

    def combined_form(e):
        return C * hertz_form(e) * (1.0 - e)**(-n)

    z = np.linspace(eps_z.min(), eps_z.max(), 200)
    hertz = C * hertz_form(z)
    incomp = K * incomp_form(z)
    combined = combined_form(z)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(100 * eps_z, 100 * (mu - sd), 100 * (mu + sd), color="tab:blue", alpha=0.15, lw=0)
    ax.plot(100 * eps_z, 100 * mu, color="tab:blue", marker="o", ls="-", lw=1.5,
            label="FE: tensor (Green–Lagrange)")
    ax.plot(100 * z, 100 * hertz, color="tab:orange", ls="--", lw=2,
            label=rf"Hertz: $\epsilon_\theta = {C:.3g}\,\epsilon_z^{{1.5}}$"
                  rf" (fit $\epsilon_z \leq {100 * HERTZ_FIT_MAX:.0f}\%$)")
    ax.plot(100 * z, 100 * incomp, color="tab:green", ls="-.", lw=2,
            label=rf"Incompressible: ${K:.3g}\cdot\frac{{1}}{{2}}[(1-\epsilon_z)^{{-1}}-1]$"
                  rf" (fit $\epsilon_z \geq {100 * INCOMP_FIT_MIN:.0f}\%$)")
    ax.plot(100 * z, 100 * combined, color="tab:purple", ls=":", lw=2.5,
            label=rf"Combined: ${C:.3g}\,\epsilon_z^{{1.5}}(1-\epsilon_z)^{{-{n:.2f}}}$")

    ax.set_xlabel("Compressive strain (%)", fontsize=26)
    ax.set_ylabel("Equatorial hoop strain (%)", fontsize=26)
    ax.tick_params(labelsize=18)
    ax.set_xlim(100 * eps_z.min(), 100 * eps_z.max())
    ax.legend(frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, OUTPUT_MODEL_PLOT))
    print(f"Hertz fit (eps_z <= {HERTZ_FIT_MAX}): C = {C:.4g}")
    print(f"Incompressible fit (eps_z >= {INCOMP_FIT_MIN}): K = {K:.4g}")
    print(f"Combined fit (C fixed, all eps_z > 0): n = {n:.4g}")
    for e, m, h, i, cb in zip(eps_z, mu, C * hertz_form(eps_z), K * incomp_form(eps_z),
                              combined_form(eps_z)):
        print(f"  eps_z={e:.2f}: FE={100 * m:.3f}%  Hertz={100 * h:.3f}%  "
              f"Incomp={100 * i:.3f}%  Combined={100 * cb:.3f}%")
    print(f"Saved model comparison to {os.path.join(OUTPUT_DIR, OUTPUT_MODEL_PLOT)}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Axis, equatorial band and skin layer from the reference mesh (same points every step).
    ref = pv.read(os.path.join(INPUT_DIR, "result_001.vtu"))
    bounds = np.asarray(ref.bounds)
    center = 0.5 * (bounds[0::2] + bounds[1::2])
    radius = 0.5 * (bounds[5] - bounds[4])
    in_band = np.abs(ref.points[:, 2] - center[2]) <= EQUATOR_BAND_FRACTION * radius
    r_ref = np.linalg.norm(ref.points - center, axis=1)
    in_skin = radius - r_ref <= SKIN_LAYER_FRACTION * radius
    in_region = in_band & in_skin
    if not in_region.any():
        raise RuntimeError("No nodes in the equatorial skin layer. "
                           "Widen EQUATOR_BAND_FRACTION or SKIN_LAYER_FRACTION.")
    n_region = int(in_region.sum())
    print(f"Axis through ({center[0]:.3g}, {center[1]:.3g}), "
          f"{int(in_band.sum())} nodes in equatorial band, {int(in_skin.sum())} in skin layer, "
          f"{n_region} in both")

    steps = [f"{i:03d}" for i in range(1, N_STEPS + 1)]
    band_values = {}
    for step in steps:
        band_values[step] = process_step(step, center, in_band, in_skin, in_region)
    write_pvd(steps)
    print(f"Wrote {len(steps)} .vtu files and {OUTPUT_PVD} to {OUTPUT_DIR}")

    methods = ["tensor_GL", "geometric_GL", "geometric_eng"]
    means = {m: np.array([np.nanmean(band_values[s][m]) for s in TIME_STEPS]) for m in methods}
    stds = {m: np.array([np.nanstd(band_values[s][m]) for s in TIME_STEPS]) for m in methods}
    for k, (step, strain) in enumerate(zip(TIME_STEPS, STRAINS)):
        print(f"step {step}: strain={strain:.2f}, "
              + ", ".join(f"{m}={means[m][k]:.4g} (std {stds[m][k]:.3g})" for m in methods))

    strains_pct = 100.0 * np.asarray(STRAINS)
    styles = {
        "tensor_GL": ("tab:blue", "o", "-", "Option 1: tensor (Green–Lagrange)"),
        "geometric_GL": ("tab:red", "s", "--", "Option 2: geometric (Green–Lagrange)"),
        "geometric_eng": ("tab:gray", "^", ":", "Option 2: geometric ($r/R-1$)"),
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in methods:
        color, marker, ls, label = styles[m]
        mu, sd = 100.0 * means[m], 100.0 * stds[m]
        ax.fill_between(strains_pct, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
        ax.plot(strains_pct, mu, color=color, marker=marker, ls=ls, lw=1.5, label=label)

    ax.set_xlabel("Compressive strain (%)", fontsize=26)
    ax.set_ylabel("Equatorial hoop strain (%)", fontsize=26)
    ax.tick_params(labelsize=18)
    ax.set_xlim(strains_pct.min(), strains_pct.max())
    ax.legend(frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PLOT))

    plot_model_comparison(np.asarray(STRAINS), means["tensor_GL"], stds["tensor_GL"])

    with open(os.path.join(OUTPUT_DIR, OUTPUT_TEXT), "w") as f:
        f.write("Strain (%),"
                + ",".join(f"Mean {m},Std {m}" for m in methods)
                + ",Number of Nodes\n")
        for k, strain in enumerate(STRAINS):
            f.write(f"{strain * 100:.2f},"
                    + ",".join(f"{means[m][k]:.4g},{stds[m][k]:.4g}" for m in methods)
                    + f",{n_region}\n")
    print(f"\nSaved plot to {os.path.join(OUTPUT_DIR, OUTPUT_PLOT)}")
    print(f"Saved data to {os.path.join(OUTPUT_DIR, OUTPUT_TEXT)}")


if __name__ == "__main__":
    main()
