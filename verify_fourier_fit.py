import csv
import os
import sys

import numpy as np

# Make the svMultiPhysics fft utility importable (it lives in ./utilities)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utilities"))
from fft_temporal_values import fft, visualize_fft, write_fourier_coeff_file

# OUT_PATH = "/Users/kbko/Documents/EP_init_src/Sphere_Compression_Cardioid"
OUT_PATH = "/Users/kbko/Documents/EP_init_src/CEM/waveform_generation"
FILE_OUT = "top_plate_step.dat"

file_values = np.loadtxt(f"{OUT_PATH}/{FILE_OUT}")

nt, nf = file_values[0]
nt, nf = int(nt), int(nf)

temporal_values = file_values[1:nt + 1, :]        # rows of [time, displacement]
d = temporal_values.shape[1] - 1                  # data dimensions (1 here)

result = fft(nt, temporal_values, d, nf)

# Write the .fcs file svMultiPhysics consumes (optional, handy to have)
fcs_out = os.path.join(OUT_PATH, os.path.splitext(FILE_OUT)[0] + ".fcs")
write_fourier_coeff_file(fcs_out, result, d, nf)
print(f"Fourier coefficient file written to {fcs_out}")

# Plot raw data vs. Fourier reconstruction (saves fft_reconstruction.png + shows)
visualize_fft(result, temporal_values, nf) 