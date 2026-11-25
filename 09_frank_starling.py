"""Frank-Starling-style curve: top-platen reaction force vs. nodal Z-displacement.

x-axis : |U_z| at a tracked node, read from each result_*.vtu (um)
y-axis : platen F_z (mid-height cut) from section_force.csv (written by 05), in uN

NODE_IDS holds one 0-based VTK point index (same as "Point ID" in ParaView's
spreadsheet view) per WINDOW_SIZE-frame window, so the tracked node can change
between plateaus (e.g. once a node is pinned by the platen).  Frames are
matched to CSV rows by the TimeValue field.

Writes:  frank_starling_trajectory.png   (every frame, coloured by time)
         frank_starling_curve.png        (one point per displacement plateau)
         frank_starling_nodes.csv        (time, node, Uz, Fz_top per frame)
"""

import glob
import os
import re

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt


# --- user inputs -------------------------------------------------------------
WINDOW_SIZE = 50             # frames per displacement plateau (as in 07)
NODE_IDS = [353, 352, 369, 369, 917, 421]   # 0-based point index, one per window
DIR_PATH = "../Sphere_Compression_FrankStarling/24-procs"
CSV_NAME = "section_force.csv"


def read_all_uz(files):
    """Return (times, Uz) with Uz of shape (n_frames, n_points)."""
    times, uz = [], []
    for fn in files:
        mesh = pv.read(fn)
        times.append(float(mesh.field_data["TimeValue"][0]))
        uz.append(mesh.point_data["Displacement"][:, 2])
    return np.asarray(times), np.asarray(uz)


def main():
    files = sorted(glob.glob(os.path.join(DIR_PATH, "result_*.vtu")),
                   key=lambda p: int(re.search(r"result_(\d+)", p).group(1)))
    if not files:
        raise SystemExit(f"no result_*.vtu in {DIR_PATH}")

    ref_points = pv.read(files[0]).points
    times, uz_all = read_all_uz(files)
    order = np.argsort(times)
    times, uz_all = times[order], uz_all[order]

    # --- pick the tracked node for each frame -----------------------------------
    n_windows = -(-len(times) // WINDOW_SIZE)       # ceil
    if len(NODE_IDS) != n_windows:
        raise SystemExit(f"NODE_IDS has {len(NODE_IDS)} entries but there are "
                         f"{n_windows} windows of {WINDOW_SIZE} frames")
    for k, nid in enumerate(NODE_IDS):
        if not 0 <= nid < len(ref_points):
            raise SystemExit(f"NODE_IDS[{k}] = {nid} out of range "
                             f"(mesh has {len(ref_points)} points)")
        print(f"window {k}: node {nid} at reference position {ref_points[nid]}")

    frame_node = np.asarray(NODE_IDS)[np.arange(len(times)) // WINDOW_SIZE]
    uz = uz_all[np.arange(len(times)), frame_node]

    # --- match reaction force to the vtu frames by time ------------------------
    data = np.loadtxt(os.path.join(DIR_PATH, CSV_NAME), delimiter=",", skiprows=1)
    csv_t, csv_fz = data[:, 0], data[:, 1]
    idx = np.searchsorted(csv_t, times).clip(0, len(csv_t) - 1)
    idx_prev = (idx - 1).clip(0)
    idx = np.where(np.abs(csv_t[idx_prev] - times) < np.abs(csv_t[idx] - times), idx_prev, idx)
    if np.abs(csv_t[idx] - times).max() > 1e-6:
        raise SystemExit("CSV times do not match vtu TimeValues -- rerun 05?")
    fz_top = csv_fz[idx] / 1e6                     # pN -> uN

    disp = np.abs(uz)
    force = np.abs(fz_top)

    out_csv = os.path.join(DIR_PATH, "frank_starling_nodes.csv")
    np.savetxt(out_csv, np.column_stack([times, frame_node, uz, fz_top]), delimiter=",",
               header="time_s,node_id,Uz_um,sphere_top_Fz_uN", comments="",
               fmt=["%.6g", "%d", "%.6e", "%.6e"])
    print(f"wrote {out_csv}")

    # --- 1) full trajectory ------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(disp, force, color="0.7", linewidth=0.8, zorder=1)
    sc = plt.scatter(disp, force, c=times, cmap="viridis", s=10, zorder=2)
    plt.colorbar(sc, label="Time (s)")
    plt.xlabel(r"$|U_z|$ at tracked node (µm)", fontsize=14)
    plt.ylabel(r"Reaction force $|F_z|$ (µN)", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PATH, "frank_starling_trajectory.png"), dpi=450)
    plt.close()

    # --- 2) per-plateau summary --------------------------------------------------
    plateau_disp, plateau_ptp = [], []
    for k, i in enumerate(range(0, len(times), WINDOW_SIZE)):
        sl = slice(i, i + WINDOW_SIZE)
        plateau_disp.append(disp[sl].mean())
        plateau_ptp.append(np.ptp(fz_top[sl]))
        print(f"  frames {i:3d}-{min(i + WINDOW_SIZE, len(times)) - 1:3d} (node {NODE_IDS[k]}): "
              f"|Uz| = {plateau_disp[-1]:8.3f} um   ptp Fz = {plateau_ptp[-1]:8.3f} uN")

    plt.figure(figsize=(6, 6))
    plt.plot(plateau_disp, plateau_ptp, marker="o", color="r", linewidth=1)
    plt.xlabel(r"$|U_z|$ at tracked node (µm)", fontsize=14)
    plt.ylabel("Peak to Peak Force (µN)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PATH, "frank_starling_curve.png"), dpi=450)
    plt.close()
    print(f"wrote plots to {DIR_PATH}")


if __name__ == "__main__":
    main()
