# This code reads in the results.vtu file in 4-procs and converts the
# Cauchy Stress Tensor and Strain into cylindrical coordinates
# The fields are called Cauchy_stress and Strain respectively
# The .vtu file contains both data in Voigt Notation (XX, YY, ZZ, XY, YZ, XZ)
# The results are saved in a new directory called './cylindrical_coords'
#
# Cylindrical axis: the z-axis (compression axis) through the centre of the reference
# mesh, same as 11_hoop_strain.py. Basis e_r = (c, s, 0), e_theta = (-s, c, 0), e_z.
# Each tensor is rotated in the configuration it lives in:
#   - "Strain" (Green-Lagrange, tensor shear) is material -> theta from reference X.
#   - "Cauchy_stress" is spatial -> theta from deformed x = X + u.
# T_cyl = Q T Q^T with rows of Q = (e_r, e_theta, e_z).
#
# Output arrays (Voigt order RR, TT, ZZ, RT, TZ, RZ):
#   Strain_cyl, Cauchy_stress_cyl, plus one scalar array per component
#   (e.g. Strain_TT, Cauchy_stress_RZ) for easy coloring in ParaView.

import os

import numpy as np
import pyvista as pv

INPUT_DIR = "./4-procs/"
OUTPUT_DIR = "./cylindrical_coords/"
OUTPUT_PVD = "cylindrical_coords.pvd"

N_STEPS = 55

CYL_COMPONENTS = ["RR", "TT", "ZZ", "RT", "TZ", "RZ"]


def voigt_to_tensor(V):
    """(n,6) Voigt [xx,yy,zz,xy,yz,xz] -> (n,3,3) symmetric tensor."""
    xx, yy, zz, xy, yz, xz = V.T
    T = np.empty((V.shape[0], 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = xx, yy, zz
    T[:, 0, 1] = T[:, 1, 0] = xy
    T[:, 1, 2] = T[:, 2, 1] = yz
    T[:, 0, 2] = T[:, 2, 0] = xz
    return T


def tensor_to_voigt(T):
    """(n,3,3) -> (n,6) in the same index order: [11,22,33,12,23,13]."""
    return np.column_stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2],
                            T[:, 0, 1], T[:, 1, 2], T[:, 0, 2]])


def cylindrical_basis(X):
    """Rotation Q (n,3,3) with rows e_r, e_theta, e_z for coords X relative to the axis."""
    theta = np.arctan2(X[:, 1], X[:, 0])
    c, s = np.cos(theta), np.sin(theta)
    Q = np.zeros((X.shape[0], 3, 3))
    Q[:, 0, 0], Q[:, 0, 1] = c, s
    Q[:, 1, 0], Q[:, 1, 1] = -s, c
    Q[:, 2, 2] = 1.0
    return Q


def to_cylindrical(V, X):
    """Rotate a Cartesian Voigt tensor field into cylindrical Voigt [RR,TT,ZZ,RT,TZ,RZ]."""
    Q = cylindrical_basis(X)
    T = voigt_to_tensor(V)
    return tensor_to_voigt(np.einsum("nij,njk,nlk->nil", Q, T, Q))


def process_step(step, center):
    mesh = pv.read(os.path.join(INPUT_DIR, f"result_{step}.vtu"))
    X_ref = mesh.points - center
    x_def = X_ref + np.asarray(mesh.point_data["Displacement"])

    for name, coords in (("Strain", X_ref), ("Cauchy_stress", x_def)):
        V = np.asarray(mesh.point_data[name])
        V_cyl = to_cylindrical(V, coords)

        # Trace is invariant under rotation.
        tr_err = np.abs(V[:, :3].sum(1) - V_cyl[:, :3].sum(1)).max()
        if tr_err > 1e-8 * max(np.abs(V[:, :3]).max(), 1.0):
            raise RuntimeError(f"step {step}: trace mismatch for {name} ({tr_err:.3g})")

        mesh.point_data[f"{name}_cyl"] = V_cyl
        for k, comp in enumerate(CYL_COMPONENTS):
            mesh.point_data[f"{name}_{comp}"] = V_cyl[:, k]

    mesh.save(os.path.join(OUTPUT_DIR, f"result_{step}.vtu"))


def write_pvd(steps):
    with open(os.path.join(OUTPUT_DIR, OUTPUT_PVD), "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n  <Collection>\n')
        for i, step in enumerate(steps):
            f.write(f'    <DataSet timestep="{i + 1}" part="0" file="result_{step}.vtu"/>\n')
        f.write("  </Collection>\n</VTKFile>\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Axis through the centre of the reference mesh (same points every step).
    ref = pv.read(os.path.join(INPUT_DIR, "result_001.vtu"))
    bounds = np.asarray(ref.bounds)
    center = 0.5 * (bounds[0::2] + bounds[1::2])
    center[2] = 0.0  # only x, y matter for theta; keep z untouched
    print(f"Axis through ({center[0]:.3g}, {center[1]:.3g})")

    steps = [f"{i:03d}" for i in range(1, N_STEPS + 1)]
    for step in steps:
        process_step(step, center)
    write_pvd(steps)
    print(f"Wrote {len(steps)} .vtu files and {OUTPUT_PVD} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
