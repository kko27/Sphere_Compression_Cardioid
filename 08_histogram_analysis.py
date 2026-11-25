import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt


numstr = ['003', '008', '013', '018', '023']
deformation = ['0%', '5%', '10%', '15%', '20%']

# Load the principal stresses for every deformation step
stresses = {1: [], 2: [], 3: []}
for i in range(len(numstr)):

    RESULT_FILE_NAME = "./4-procs/result_" + numstr[i] + "_principal.vtu"

    # Load in the .vtu file from the simulation
    result = pv.read(RESULT_FILE_NAME)
    for k in stresses:
        stresses[k].append(result["Principal_stress_" + str(k)].flatten())

# Ordered colors so increasing deformation reads as a progression
colors = plt.cm.viridis(np.linspace(0, 0.9, len(numstr)))

# One histogram per principal stress, overlaying all deformation steps
for k in stresses:

    plt.figure(figsize=(8, 6))
    for i in range(len(numstr)):
        plt.hist(stresses[k][i], bins=50, range = [-30, 10], histtype='step', linewidth=2, color=colors[i], label=deformation[i])
    plt.xlabel('Principal Stress ' + str(k) + ' (Pa)', fontsize=16)
    plt.ylabel('Mesh Elements', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Deformation', fontsize = 14, title_fontsize = 14)
    plt.savefig("./PaperFigs/histogram_principal_stress_" + str(k) + ".svg")
