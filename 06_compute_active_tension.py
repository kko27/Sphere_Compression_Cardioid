import pyvista as pv 
from matplotlib import pyplot as plt 
import numpy as np 
import os 

# Dir
DIR_PATH = "../FrankStarlingSquish_70/24-procs"
# List all *.vtu files in the directory by time order 
Folder_list = sorted([f for f in os.listdir(DIR_PATH) if f.endswith('.vtu')], key=lambda x: float(x.split('_')[-1].replace('.vtu', '')))
Max_active_tension_ls = []
Mean_active_tension_ls = []
time_list = 0.1*np.array([float(f.split('_')[-1].replace('.vtu', '')) for f in Folder_list])

# Iterate through each file and extract "Active_tension_fibers" field data 
for file in Folder_list:
    file_path = os.path.join(DIR_PATH, file)
    mesh = pv.read(file_path)
    
    # Check if the field exists in the mesh
    if "Active_tension_fibers" in mesh.point_data:
        active_tension_data = mesh.point_data["Active_tension_fibers"]
        print(f"File: {file}, Active Tension Fibers Data: {active_tension_data}")
    else:
        print(f"File: {file} does not contain 'Active_tension_fibers' field.")

    Max_active_tension = np.max(active_tension_data)
    Mean_active_tension = np.mean(active_tension_data)

    Max_active_tension_ls.append(Max_active_tension)
    Mean_active_tension_ls.append(Mean_active_tension)

    print(f"File: {file}, Max Active Tension: {Max_active_tension}, Mean Active Tension: {Mean_active_tension}")

# Plotting the results 
plt.figure(figsize=(8, 6))
# Plot Max Active Tension
plt.plot(time_list, Max_active_tension_ls, color='r', linewidth=2, label='Max Active Tension')
plt.plot(time_list, Mean_active_tension_ls, color='b', linewidth=2, label='Mean Active Tension')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Active Tension (Pa)', fontsize=14)
plt.title('Active Tension Fibers Over Time', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
# Save the plot as a PNG file
output_plot_path = os.path.join(DIR_PATH, "active_tension_fibers_over_time.png")
plt.savefig(output_plot_path, dpi=450)
