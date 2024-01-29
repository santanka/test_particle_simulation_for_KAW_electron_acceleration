import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import datetime
import os

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

initial_Kperp_eq_min_eV = 1E0
initial_Kperp_eq_max_eV = 1.1E1
initial_Kperp_eq_mesh_number = 11

initial_S_value_min = 1E-2
initial_S_value_max = 1E0
initial_S_value_mesh_number = 50

separate_number_psi = 30

grid_scale = 'linear'

#figure_suffix = r'_initial_mlat_deg.png'
#figure_suffix = r'_initial_S_value.png'
#figure_suffix = r'_initial_psi.png'
figure_suffix = r'_initial_K.png'
#color_bar_label = r'$\lambda_{\mathrm{i}}$ [deg]'
#color_bar_label = r'$S_{\mathrm{i}}$'
#color_bar_label = r'$\psi_{\mathrm{i}}$ [$\pi$ rad]'
color_bar_label = r'$K_{\mathrm{i}}$ [eV]'

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/test_particle_simulation_detrapped_phase_edgedefinition/auroral_precipitation'
file_path = f'{dir_name}/Kperp_eq_{initial_Kperp_eq_min_eV:.4f}_{initial_Kperp_eq_max_eV:.4f}_eV_{initial_Kperp_eq_mesh_number}_S_{initial_S_value_min:.4f}_{initial_S_value_max:.4f}_{initial_S_value_mesh_number}_{separate_number_psi}_{grid_scale}.csv'
figure_name = os.path.basename(file_path).replace('.csv', figure_suffix)

data = np.loadtxt(file_path, delimiter=',', skiprows=1)
data_initial_Kperp_eq = data[:, 0]
data_initial_S_value = data[:, 1]
data_initial_psi = data[:, 2]
data_initial_capital_theta = data[:, 3]
data_initial_mlat_rad = data[:, 4]
data_initial_energy_perp_eV = data[:, 5]
data_initial_energy_para_eV = data[:, 6]
data_initial_energy_eV = data[:, 7]
data_energy_perp_ionospheric_end_eV = data[:, 8]
data_energy_para_ionospheric_end_eV = data[:, 9]
data_energy_ionospheric_end_eV = data[:, 10]

color_bar_target = data_initial_energy_eV
color_bar_target_min = np.min(color_bar_target)
color_bar_target_max = np.max(color_bar_target)

fig = plt.figure(figsize=(20, 20), dpi=100)

ax = fig.add_subplot(111)
ax.set_xlabel(r'$K_{\perp} (\lambda = 0) [ \mathrm{eV} ]$')
ax.set_ylabel(r'$K (\lambda = \lambda_{\mathrm{iono}}) [ \mathrm{eV} ]$')
ax.set_xscale('linear')
ax.set_yscale('linear')

im = ax.scatter(data_initial_Kperp_eq, data_energy_ionospheric_end_eV, s=100, c=color_bar_target, cmap=cm.jet, vmin=color_bar_target_min, vmax=color_bar_target_max, marker='o', edgecolors='k', linewidths=1, zorder=2, alpha=0.51)
ax.plot(data_initial_Kperp_eq, data_energy_perp_ionospheric_end_eV, c='k', lw=4, label=r'$K_{\perp} (\lambda = \lambda_{\mathrm{iono}})$', zorder=1)

ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

ax.legend()

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(color_bar_label)
#cbar.set_ticks(np.linspace(-0.9, 0, 10))

plt.tight_layout()
plt.savefig(f'{dir_name}/{figure_name}')
plt.savefig(f'{dir_name}/{figure_name.replace(".png", ".pdf")}')
plt.close()