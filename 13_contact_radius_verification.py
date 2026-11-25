# Code verification using the Tatara Model
# Compare the contact radius predicted by the FEM solver with the contact radius
# from the analytical implicit solution provided by Tatara (formalized by
# Giannopoulos et al. (2025)).
#
# Analytical model
# ----------------
#   delta = 3(1-nu^2) F / (4 E a)  -  F g(a) / (pi E)                        (Tatara)
#   g(a)  = 2(1+nu) R^2 / (a^2+R^2)^(3/2)  +  (1-nu^2) / (a^2+4R^2)^(1/2)
# delta is the approach at ONE contact (half the total plate approach).  F is
# eliminated with the Hertz contact-radius relation
#   a^3 = 3(1-nu^2) F R / (4 E)    =>    F = 4 E a^3 / (3 (1-nu^2) R)
# so that for each delta the contact radius a is the root of a single equation:
#   f(a; delta) = a^2/R - 4 a^3 g(a) / (3 pi (1-nu^2) R) - delta = 0
#
# FEM contact radius
# ------------------
# The plates are RigidPlanePenalty BCs: a surface node carries contact force only
# when it has penetrated the plate plane.  For each result_*.vtu the mesh is
# warped by Displacement and, on the exterior surface, nodes past the plane are
# "in contact".  The contact radius is bracketed by
#   a_in  = largest radial distance sqrt(x^2+y^2) of a contacting node
#   a_out = smallest radial distance of a non-contacting node beyond a_in
# and reported as their midpoint (the bracket is the mesh resolution).
#
# The top-plate position is evaluated from the Fourier series in
# top_plate_step.fcs, which is what svMultiPhysics actually applies (it
# FFTs top_plate_step.dat with 64 modes), not from the raw .dat values.
#
# Frames where the plate is ramping are not quasi-static: the FEM sphere runs
# ahead of / behind the plate and partially loses contact.  They are plotted with
# open markers and excluded from the error statistics; only hold frames
# (plate essentially stationary) are a fair test of the static Tatara solution.
#
# Units: um, Pa.
#
# Writes (in contact_radius_verification/):
#   contact_radius.csv
#   contact_radius_vs_time.png / .svg
#   contact_radius_vs_delta.png / .svg
#   contact_radius_rel_error.png / .svg

import glob
import os
import re

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar

from fft_temporal_values import recon_fft

# ---------------------------------------------------------------- parameters
nu = 0.45        # Poisson ratio
E = 200.0        # Elastic modulus (Pa)
R = 350.0        # Radius of sphere before compression (um)
step = 20        # rows of top_plate_step.dat per saved result (dt_dat = 0.05 s, results every 1 s)

Z_TOP0 = 348.0       # Plane_point z of wall_top    (solver_contact.xml)
Z_BOT0 = -348.0      # Plane_point z of wall_bottom (fixed: bottom_plate_step.dat is all zero)
HOLD_VEL = 0.5       # |plate velocity| (um/s) below which a frame counts as a "hold" frame

RESULT_DIR = "4-procs"
DAT_FILE = "top_plate_step.dat"
FCS_FILE = "top_plate_step.fcs"
OUT_DIR = "contact_radius_verification"

# Plot styling: axes ticks 12, titles 18, axis labels 14, legend 12
plt.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize": 12,
                     "axes.titlesize": 18, "axes.labelsize": 14,
                     "legend.fontsize": 12})

def g(x):
    return (2*(1 + nu)*R**2)/((x**2 + 4*R**2)**1.5) + (1 - nu**2)/(x**2 + 4*R**2)**0.5

def hertz_force(x):
    """Force (Pa.um^2 = pN) giving Hertz contact radius x."""
    return 4*E*x**3/(3*(1 - nu**2)*R)

def f(x, delta):
    F = hertz_force(x)
    return (3*(1 - nu**2)*F)/(4*E*x) - (g(x)*F)/(np.pi*E) - delta

def tatara_contact_radius(delta):
    """Root a of f(a; delta) = 0.  delta(a) is monotonic on (0, 2R], so brentq is safe."""
    if delta <= 0:
        return 0.0
    return root_scalar(f, args=(delta,), method="brentq", bracket=(1e-9, 2*R)).root


# ---------------------------------------------------------------- plate motion
def load_fourier_plate(fcs_file):
    """Callable t -> top-plate displacement, reconstructed from the solver's .fcs file."""
    with open(fcs_file) as fh:
        lines = [l for l in fh.read().splitlines() if l.strip()]
    ti, T = map(float, lines[0].split())
    qi, qs = map(float, lines[1].split())
    n = int(lines[2])
    c = np.array([list(map(float, l.split())) for l in lines[3:3 + n]])
    return lambda t: recon_fft(qi, qs, ti, T, c[:, 0], c[:, 1], np.atleast_1d(t), n)


# ---------------------------------------------------------------- FEM contact
def contact_radius(r, gap):
    """Bracket the contact radius from surface nodes.

    r   : radial distance of each surface node from the z axis
    gap : signed distance from node to plate (<= 0 means penetrated -> in contact)
    Returns (a_mid, a_in, a_out); all zero if no node is in contact.
    """
    inside = gap <= 0.0
    if not np.any(inside):
        return 0.0, 0.0, 0.0
    a_in = r[inside].max()
    beyond = (~inside) & (r > a_in)
    a_out = r[beyond].min() if np.any(beyond) else a_in
    return 0.5*(a_in + a_out), a_in, a_out


def fem_contact(fn, z_top):
    vol = pv.read(fn)
    t = float(vol.field_data["TimeValue"][0])
    surf = vol.extract_surface(algorithm="dataset_surface").warp_by_vector("Displacement")
    P = np.asarray(surf.points)
    r = np.hypot(P[:, 0], P[:, 1])
    top, bot = P[:, 2] > 0, P[:, 2] < 0
    top_c = contact_radius(r[top], z_top - P[top, 2])     # penetrated when z > z_top
    bot_c = contact_radius(r[bot], P[bot, 2] - Z_BOT0)    # penetrated when z < z_bot
    return t, top_c, bot_c


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Raw prescribed displacement (used to flag ramp vs hold frames)
    z_data = np.loadtxt(DAT_FILE, skiprows=1)
    plate_fcs = load_fourier_plate(FCS_FILE)

    # Result files in numeric order (result_001 ... result_055)
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "result_[0-9][0-9][0-9].vtu")),
                   key=lambda p: int(re.search(r"result_(\d+)\.vtu", p).group(1)))
    if not files:
        raise SystemExit(f"no result_###.vtu files in {RESULT_DIR}")

    rows = []
    for fn in files:
        i = int(re.search(r"result_(\d+)\.vtu", fn).group(1))
        # Save every 20 displacement data where i is the 20th
        z_prescribed = z_data[step*i, 1]
        u_top = float(plate_fcs(z_data[step*i, 0])[0])     # displacement the solver applied
        z_top = Z_TOP0 + u_top

        # Plate velocity from the .dat (central difference over one row)
        j = step*i
        lo, hi = max(j - 1, 0), min(j + 1, len(z_data) - 1)
        vel = (z_data[hi, 1] - z_data[lo, 1])/(z_data[hi, 0] - z_data[lo, 0])
        hold = abs(vel) < HOLD_VEL

        # Approach per contact: half of (undeformed diameter - plate separation).
        # Plates start at +-348 with R = 350, so delta(0) = 2 um of pre-compression.
        delta = 0.5*(2*R - (z_top - Z_BOT0))

        t, (a_top, a_top_in, a_top_out), (a_bot, a_bot_in, a_bot_out) = fem_contact(fn, z_top)
        a_tat = tatara_contact_radius(delta)
        F_tat = hertz_force(a_tat)

        rows.append([t, z_prescribed, u_top, delta, int(hold), int(vel <= 0),
                     a_top, a_top_in, a_top_out, a_bot, a_bot_in, a_bot_out, a_tat, F_tat])
        print(f"t={t:5.1f}  delta={delta:7.2f}  {'hold' if hold else 'ramp'}  "
              f"a_top={a_top:6.1f} [{a_top_in:6.1f},{a_top_out:6.1f}]  "
              f"a_bot={a_bot:6.1f}  a_Tatara={a_tat:6.1f}")

    cols = ["time", "z_prescribed_dat", "u_top_fcs", "delta", "hold", "loading",
            "a_top", "a_top_in", "a_top_out", "a_bot", "a_bot_in", "a_bot_out",
            "a_tatara", "F_tatara_pN"]
    D = np.array(rows)
    np.savetxt(os.path.join(OUT_DIR, "contact_radius.csv"), D, delimiter=",",
               header=",".join(cols), comments="", fmt="%.6g")
    C = {c: D[:, k] for k, c in enumerate(cols)}
    hold = C["hold"].astype(bool)
    loading = C["loading"].astype(bool)

    # Smooth analytical curve a(delta)
    d_curve = np.linspace(1e-3, 1.05*C["delta"].max(), 300)
    a_curve = np.array([tatara_contact_radius(d) for d in d_curve])

    # Hertz reference (a = sqrt(R delta)), to show what Tatara's correction adds
    a_hertz = np.sqrt(R*d_curve)

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=300)
        plt.close(fig)

    # ---- 1) a vs time
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(C["time"], C["a_tatara"], "k-", lw=2, label="Tatara (analytical)")
    for key, colr, lab in (("top", "C0", "FEM top contact"), ("bot", "C1", "FEM bottom contact")):
        a, lo, hi = C[f"a_{key}"], C[f"a_{key}_in"], C[f"a_{key}_out"]
        yerr = np.vstack([a - lo, hi - a])
        ax.errorbar(C["time"][hold], a[hold], yerr=yerr[:, hold], fmt="o", color=colr,
                    ms=5, capsize=2, label=f"{lab} (hold)")
        ax.plot(C["time"][~hold], a[~hold], "o", mfc="none", color=colr, ms=5,
                label=f"{lab} (ramp)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Contact radius $a$ ($\\mu$m)")
    ax.set_title("Contact radius vs time")
    ax.legend(frameon=False)
    save(fig, "contact_radius_vs_time")

    # ---- 2) a vs delta
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(d_curve, a_curve, "k-", lw=2, label="Tatara (analytical)")
    ax.plot(d_curve, a_hertz, "k--", lw=1, label="Hertz, $a=\\sqrt{R\\delta}$")
    for key, colr, lab in (("top", "C0", "FEM top"), ("bot", "C1", "FEM bottom")):
        a, lo, hi = C[f"a_{key}"], C[f"a_{key}_in"], C[f"a_{key}_out"]
        for branch, mk, blab in ((loading, "o", "loading"), (~loading, "s", "unloading")):
            m = hold & branch
            ax.errorbar(C["delta"][m], a[m], yerr=np.vstack([a[m] - lo[m], hi[m] - a[m]]),
                        fmt=mk, color=colr, ms=5, capsize=2, label=f"{lab}, {blab} (hold)")
    ax.set_xlabel("Approach per contact $\\delta$ ($\\mu$m)")
    ax.set_ylabel("Contact radius $a$ ($\\mu$m)")
    ax.set_title("Contact radius vs displacement")
    ax.legend(frameon=False)
    save(fig, "contact_radius_vs_delta")

    # ---- 3) relative error (hold frames with contact)
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, colr, lab in (("top", "C0", "Top contact"), ("bot", "C1", "Bottom contact")):
        a = C[f"a_{key}"]
        m = hold & (a > 0) & (C["a_tatara"] > 0)
        err = 100*(a[m] - C["a_tatara"][m])/C["a_tatara"][m]
        ax.plot(C["delta"][m & loading], err[loading[m]], "o", color=colr, label=f"{lab}, loading")
        ax.plot(C["delta"][m & ~loading], err[~loading[m]], "s", mfc="none", color=colr,
                label=f"{lab}, unloading")
        print(f"{lab}: hold-frame relative error  mean |e| = {np.mean(np.abs(err)):.1f}%  "
              f"max |e| = {np.max(np.abs(err)):.1f}%")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Approach per contact $\\delta$ ($\\mu$m)")
    ax.set_ylabel("$(a_{FEM} - a_{Tatara})/a_{Tatara}$ (%)")
    ax.set_title("Contact radius relative error")
    ax.legend(frameon=False)
    save(fig, "contact_radius_rel_error")

    print(f"wrote results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
