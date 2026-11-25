# Python Code that Reads in the Cauchy Stress Tensor (provided in Voigt notation)
# and computes the principal stresses and directions.
#
# Runs for every .vtu file inside ./4-procs/ (excluding files this script
# generates, i.e. *_principal.vtu). For each input file it writes a companion
# <stem>_principal.vtu with the principal stresses/directions attached and a
# <stem>_principal.png screenshot.

import glob
import os

import pyvista as pv
import numpy as np


# INPUT_DIR = "./4-procs"
# INPUT_DIR = "/Users/kbko/Documents/Dani_Simulations/LV_sv0D/24-procs"
# INPUT_DIR = "/Users/kbko/Documents/Dani_Simulations/LoadAndUnloadButRapid/4-procs"
INPUT_DIR = "/Users/kbko/Documents/Dani_Simulations/LoadTwoStepsRapid/4-procs"
# Results are reported in Pa for this simulation
SCALE_DYN_PER_CM2_TO_PA = 1.0


def process_file(result_file):
    stem, _ = os.path.splitext(os.path.basename(result_file))
    output_file = os.path.join(INPUT_DIR, f"{stem}_principal.vtu")
    screenshot_file = os.path.join(INPUT_DIR, f"{stem}_principal.png")

    # Load in the .vtu file from the simulation
    result = pv.read(result_file)
    cauchy_voigt = result["Cauchy_stress"] * SCALE_DYN_PER_CM2_TO_PA

    # Components are saved as XX, YY, ZZ, XY, YZ, XZ in Voigt notation.
    xx, yy, zz, xy, yz, xz = cauchy_voigt.T

    # Assemble the full symmetric 3x3 Cauchy stress tensor at every point.
    n_pts = cauchy_voigt.shape[0]
    stress_tensor = np.zeros((n_pts, 3, 3))
    stress_tensor[:, 0, 0] = xx
    stress_tensor[:, 1, 1] = yy
    stress_tensor[:, 2, 2] = zz
    stress_tensor[:, 0, 1] = stress_tensor[:, 1, 0] = xy
    stress_tensor[:, 1, 2] = stress_tensor[:, 2, 1] = yz
    stress_tensor[:, 0, 2] = stress_tensor[:, 2, 0] = xz

    # Perform the eigenvalue decomposition of the stress tensor to get the
    # principal stresses and directions. np.linalg.eigh is used since the stress
    # tensor is symmetric; it returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(stress_tensor)

    # Reorder to descending order so index 0 is the major (most tensile/least
    # compressive) principal stress and index 2 is the minor principal stress.
    order = np.argsort(eigenvalues, axis=1)[:, ::-1]
    eigenvalues = np.take_along_axis(eigenvalues, order, axis=1)
    eigenvectors = np.take_along_axis(eigenvectors, order[:, np.newaxis, :], axis=2)

    sigma1, sigma2, sigma3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    dir1, dir2, dir3 = eigenvectors[..., 0], eigenvectors[..., 1], eigenvectors[..., 2]

    # Save the eigenvalues and eigenvectors to the result object for visualization in Paraview
    result["Principal_stress_1"] = sigma1
    result["Principal_stress_2"] = sigma2
    result["Principal_stress_3"] = sigma3
    result["Principal_direction_1"] = dir1
    result["Principal_direction_2"] = dir2
    result["Principal_direction_3"] = dir3

    result.save(output_file)
    print(f"Wrote principal stresses/directions to {output_file}")

    # Quick in-script visualization: glyph the major principal direction, scaled
    # by its magnitude, colored by sigma1, on top of the deformed sphere. Glyphs
    # use |sigma1| (magnitude, not signed value) so compressive stresses don't
    # flip the arrow direction, and the scale factor is relative to the mesh
    # size rather than the raw stress units.
    result["Principal_stress_1_abs"] = np.abs(sigma1)

    mesh_size = result.length  # bounding-box diagonal
    glyph_factor = 0.3 * mesh_size / max(np.abs(sigma1).max(), 1e-12)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(result, scalars="Principal_stress_1", cmap="coolwarm", opacity=0.5)

    glyphs = result.glyph(
        orient="Principal_direction_1",
        scale="Principal_stress_1_abs",
        factor=glyph_factor,
        geom=pv.Line(),
    )
    plotter.add_mesh(glyphs, scalars="Principal_stress_1", cmap="coolwarm", line_width=2)

    plotter.add_scalar_bar(title="Sigma_1 (major principal stress) [Pa]")
    plotter.reset_camera()
    plotter.screenshot(screenshot_file)
    plotter.close()
    print(f"Saved screenshot to {screenshot_file}")


def main():
    result_files = sorted(
        f
        for f in glob.glob(os.path.join(INPUT_DIR, "*.vtu"))
        if not f.endswith("_principal.vtu")
    )
    if not result_files:
        print(f"No .vtu files found in {INPUT_DIR}")
        return

    print(f"Found {len(result_files)} .vtu file(s) in {INPUT_DIR}")
    for result_file in result_files:
        print(f"\n=== Processing {result_file} ===")
        process_file(result_file)


if __name__ == "__main__":
    main()
