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
plt.rcParams["font.size"] = 55

initial_Kperp_eq_min_eV = 1E0
initial_Kperp_eq_max_eV = 1.1E1
initial_Kperp_eq_mesh_number = 11

initial_S_value_min = 1E-2
initial_S_value_max = 1E0
initial_S_value_mesh_number = 50

separate_number_psi = 30

grid_scale = 'linear'

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/test_particle_simulation_detrapped_phase_edgedefinition/auroral_precipitation'
file_path = f'{dir_name}/Kperp_eq_{initial_Kperp_eq_min_eV:.4f}_{initial_Kperp_eq_max_eV:.4f}_eV_{initial_Kperp_eq_mesh_number}_S_{initial_S_value_min:.4f}_{initial_S_value_max:.4f}_{initial_S_value_mesh_number}_{separate_number_psi}_{grid_scale}.csv'
figure_name = os.path.basename(file_path).replace('.csv', '_for_paper.png')

data = np.loadtxt(file_path, delimiter=',', skiprows=1)

# Kperp_eq, S_value, psi, capital_theta, mlat_rad, energy_perp_eV, energy_para_eV, energy_eV, energy_perp_ionospheric_end_eV, energy_para_ionospheric_end_eV, energy_ionospheric_end_eV
# Kperp_eq = 1のみを抽出する。
#data = data[data[:, 0] == 1.0, :]

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

B_ratio = data_energy_perp_ionospheric_end_eV / data_initial_Kperp_eq
B_ratio_constant = np.mean(B_ratio)


##### additional parameters #####
# constants
speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]
electric_constant = 8.8541878128E-12  #[F m-1]
magnetic_constant = 1.25663706212E-6  #[N A-2]

#planet condition
planet_radius = 6.3781E6 #[m]
planet_radius_polar = 6.3568E6 #[m]
lshell_number = 9E0
r_eq = planet_radius * lshell_number

# background plasma parameters
number_density_eq = 1E6    # [m^-3]
ion_mass = 1.672621898E-27   # [kg]
ion_temperature_eq = 1E3   # [eV]
electron_mass = 9.10938356E-31    # [kg]
electron_temperature_eq = 1E2  # [eV]

tau_eq = ion_temperature_eq / electron_temperature_eq

def number_density(mlat_rad):
    return number_density_eq

def ion_temperature(mlat_rad):
    return ion_temperature_eq

def tau(mlat_rad):
    return tau_eq

# magnetic field
dipole_moment   = 7.75E22 #[Am]
B0_eq           = (1E-7 * dipole_moment) / r_eq**3E0

def magnetic_flux_density(mlat_rad):
    return B0_eq / np.cos(mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)     #[T]

def Alfven_speed(mlat_rad):
    return magnetic_flux_density(mlat_rad) / np.sqrt(magnetic_constant * number_density(mlat_rad) * ion_mass)    #[m/s]

def plasma_beta_ion(mlat_rad):
    return 2E0 * magnetic_constant * number_density(mlat_rad) * ion_temperature(mlat_rad) * elementary_charge / magnetic_flux_density(mlat_rad)**2E0  #[]

# wave parameters
kperp_rhoi = 2E0 * np.pi    #[rad]
wave_frequency = 2E0 * np.pi * 0.15    #[rad/s]

def wave_phase_speed(mlat_rad):
    return Alfven_speed(mlat_rad) * kperp_rhoi * np.sqrt((1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))) * np.sign(mlat_rad)    #[m/s]

def kpara(mlat_rad):
    return wave_frequency / wave_phase_speed(mlat_rad)   #[rad/m]

wave_scalar_potential = 2000E0   #[V]

def wave_modified_potential(mlat_rad):
    return wave_scalar_potential * (2E0 + 1E0 / tau(mlat_rad))    #[V]

def energy_wave_potential(mlat_rad):
    return elementary_charge * wave_modified_potential(mlat_rad)    #[J]

def trapping_frequency(mlat_rad):
    return np.abs(kpara(mlat_rad)) * np.sqrt(energy_wave_potential(mlat_rad) / electron_mass)   #[rad/s]

def d_Ktotal_d_t(theta, psi, mlat_rad):
    return - energy_wave_potential(mlat_rad) * (theta + wave_frequency) * np.sin(psi)   #[J/s]

##### additional parameters end #####


trapping_frequency_initial = np.zeros_like(data_initial_Kperp_eq)
dK_dt_initial = np.zeros_like(data_initial_Kperp_eq)
K_E_initial = np.zeros_like(data_initial_Kperp_eq)

for count_i in range(len(data_initial_Kperp_eq)):
    trapping_frequency_initial[count_i] = trapping_frequency(data_initial_mlat_rad[count_i])
    dK_dt_initial[count_i] = d_Ktotal_d_t(data_initial_capital_theta[count_i] * 2E0 * trapping_frequency_initial[count_i], data_initial_psi[count_i], data_initial_mlat_rad[count_i])
    K_E_initial[count_i] = energy_wave_potential(data_initial_mlat_rad[count_i])


fig = plt.figure(figsize=(30, 40), dpi=100)
gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 0.05])

cmap_color_1 = cm.gnuplot
cmap_color_2 = cm.turbo

color_target_1 = data_initial_mlat_rad * 180 / np.pi    # [deg]
vmin_1 = np.min(color_target_1)
vmax_1 = np.max(color_target_1)
norm_1 = mpl.colors.Normalize(vmin=vmin_1, vmax=vmax_1)
sm_1 = plt.cm.ScalarMappable(cmap=cmap_color_1, norm=norm_1)
sm_1.set_array([])
cbarax_1 = fig.add_subplot(gs[5, 0])
cbar_1 = fig.colorbar(sm_1, cax=cbarax_1, orientation='horizontal')
cbar_1.set_label(r'$\lambda_{\mathrm{i}}$ [deg]')

color_target_2 = data_energy_ionospheric_end_eV    # [eV]
vmin_2 = np.min(color_target_2)
vmax_2 = np.max(color_target_2)
norm_2 = mpl.colors.Normalize(vmin=vmin_2, vmax=vmax_2)
sm_2 = plt.cm.ScalarMappable(cmap=cmap_color_2, norm=norm_2)
sm_2.set_array([])
cbarax_2 = fig.add_subplot(gs[5, 1])
cbar_2 = fig.colorbar(sm_2, cax=cbarax_2, orientation='horizontal')
cbar_2.set_label(r'$K (\lambda = \lambda_{\mathrm{iono}})$ [$\mathrm{eV}$]')

ax_1_1 = fig.add_subplot(gs[0, 0], xlabel=r'$K_{\perp} (\lambda = 0)$ [$\mathrm{eV}$]', ylabel=r'$K_{\mathrm{i}}$ [$\mathrm{eV}$]')
ax_1_1.scatter(data_initial_Kperp_eq, data_initial_energy_eV, s=50, c=color_target_1, cmap=cmap_color_1, vmin=vmin_1, vmax=vmax_1, zorder=2, alpha=0.8)

ax_1_2 = fig.add_subplot(gs[1, 0], xlabel=r'$K_{\perp} (\lambda = 0)$ [$\mathrm{eV}$]', ylabel=r'$S_{\mathrm{i}}$')
ax_1_2.scatter(data_initial_Kperp_eq, data_initial_S_value, s=50, c=color_target_1, cmap=cmap_color_1, vmin=vmin_1, vmax=vmax_1, zorder=2, alpha=0.8)

ax_1_3 = fig.add_subplot(gs[2, 0], xlabel=r'$K_{\perp} (\lambda = 0)$ [$\mathrm{eV}$]', ylabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]')
ax_1_3.scatter(data_initial_Kperp_eq, data_initial_psi / np.pi, s=50, c=color_target_1, cmap=cmap_color_1, vmin=vmin_1, vmax=vmax_1, zorder=2, alpha=0.8)

ax_1_4 = fig.add_subplot(gs[3, 0], xlabel=r'$K_{\perp} (\lambda = 0)$ [$\mathrm{eV}$]', ylabel=r'$(\mathrm{d} K / \mathrm{d} t)_{\mathrm{i}}$ [$\mathrm{eV} / \mathrm{s}$]')
ax_1_4.scatter(data_initial_Kperp_eq, dK_dt_initial / elementary_charge, s=50, c=color_target_1, cmap=cmap_color_1, vmin=vmin_1, vmax=vmax_1, zorder=2, alpha=0.8)

ax_1_5 = fig.add_subplot(gs[4, 0], xlabel=r'$K_{\perp} (\lambda = 0)$ [$\mathrm{eV}$]', ylabel=r'$K (\lambda = \lambda_{\mathrm{iono}})$ [$\mathrm{eV}$]')
ax_1_5.scatter(data_initial_Kperp_eq, data_energy_ionospheric_end_eV, s=50, c=color_target_1, cmap=cmap_color_1, vmin=vmin_1, vmax=vmax_1, zorder=2, alpha=0.8)

ax_2_1 = fig.add_subplot(gs[0, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$K_{\mathrm{i}}$ [$\mathrm{eV}$]')
ax_2_1.scatter(data_initial_psi / np.pi, data_initial_energy_eV, s=50, c=color_target_2, cmap=cmap_color_2, vmin=vmin_2, vmax=vmax_2, zorder=2, alpha=0.8)

ax_2_2 = fig.add_subplot(gs[1, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$S_{\mathrm{i}}$')
ax_2_2.scatter(data_initial_psi / np.pi, data_initial_S_value, s=50, c=color_target_2, cmap=cmap_color_2, vmin=vmin_2, vmax=vmax_2, zorder=2, alpha=0.8)

ax_2_3 = fig.add_subplot(gs[2, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\Theta_{\mathrm{i}}$')
ax_2_3.scatter(data_initial_psi / np.pi, data_initial_capital_theta, s=50, c=color_target_2, cmap=cmap_color_2, vmin=vmin_2, vmax=vmax_2, zorder=2, alpha=0.8)

ax_2_4 = fig.add_subplot(gs[3, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$(\mathrm{d} K / \mathrm{d} t)_{\mathrm{i}}$ [$\mathrm{eV} / \mathrm{s}$]')
ax_2_4.scatter(data_initial_psi / np.pi, dK_dt_initial / elementary_charge, s=50, c=color_target_2, cmap=cmap_color_2, vmin=vmin_2, vmax=vmax_2, zorder=2, alpha=0.8)

ax_2_5 = fig.add_subplot(gs[4, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\lambda_{\mathrm{i}}$ [deg]')
ax_2_5.scatter(data_initial_psi / np.pi, data_initial_mlat_rad * 180 / np.pi, s=50, c=color_target_2, cmap=cmap_color_2, vmin=vmin_2, vmax=vmax_2, zorder=2, alpha=0.8)

axes = [ax_1_1, ax_1_2, ax_1_3, ax_1_4, ax_1_5, ax_2_1, ax_2_2, ax_2_3, ax_2_4, ax_2_5]
for ax in axes:
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.text(-0.2, 1.0, '(' + chr(97 + axes.index(ax)) + ')', transform=ax.transAxes)

fig.tight_layout(w_pad=0.3, h_pad=0)
plt.savefig(f'{dir_name}/{figure_name}')
plt.savefig(f'{dir_name}/{figure_name.replace(".png", ".pdf")}')
plt.close()




quit()

ax_1 = fig.add_subplot(gs[0:5, 0], xlabel=r'$K_{\perp} (\lambda = 0)$ [$\mathrm{eV}$]', ylabel=r'$K (\lambda = \lambda_{\mathrm{iono}})$ [$\mathrm{eV}$]')
ax_1.scatter(data_initial_Kperp_eq, data_energy_ionospheric_end_eV, s=50, c=color_target_1, cmap=cmap_color_1, vmin=vmin_1, vmax=vmax_1, zorder=2, alpha=0.8)
ax_1.plot(data_initial_Kperp_eq, data_energy_perp_ionospheric_end_eV, c='k', lw=4, label=r'$K_{\perp} (\lambda = \lambda_{\mathrm{iono}})$', zorder=1)
ax_1.text(0.99, 0.1, r'$\frac{K_{\perp} (\lambda = \lambda_{\mathrm{iono}})}{K_{\perp} (\lambda = 0)} = \frac{B_{0}(\lambda = \lambda_{\mathrm{iono}})}{B_{0}(\lambda = 0)} =$'+f' {B_ratio_constant:.2f}', transform=ax_1.transAxes, ha='right', va='bottom', fontsize=60)
ax_1.legend()

ax_2 = fig.add_subplot(gs[0, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\theta_{\mathrm{i}} / 2 \omega_{\mathrm{t}} (\lambda_{\mathrm{i}})$')
ax_2.scatter(data_initial_psi / np.pi, data_initial_capital_theta, s=50, c=color_target_2, cmap=cmap_color_2, vmin=vmin_2, vmax=vmax_2, zorder=2, alpha=0.8)

ax_3 = fig.add_subplot(gs[1, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$S_{\mathrm{i}}$')
ax_3.scatter(data_initial_psi / np.pi, data_initial_S_value, s=50, c=color_target_2, cmap=cmap_color_2, vmin=vmin_2, vmax=vmax_2, zorder=2, alpha=0.8)
ax_3.set_yscale('log')

ax_4 = fig.add_subplot(gs[2, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\lambda_{\mathrm{i}}$ [deg]')
ax_4.scatter(data_initial_psi / np.pi, data_initial_mlat_rad * 180 / np.pi, s=50, c=color_target_2, cmap=cmap_color_2, vmin=vmin_2, vmax=vmax_2, zorder=2, alpha=0.8)

ax_5 = fig.add_subplot(gs[3, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$K_{\mathrm{i}}$ [$\mathrm{eV}$]')
ax_5.scatter(data_initial_psi / np.pi, data_initial_energy_eV, s=50, c=color_target_2, cmap=cmap_color_2, vmin=vmin_2, vmax=vmax_2, zorder=2, alpha=0.8)

ax_6 = fig.add_subplot(gs[4, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$(\mathrm{d} K / \mathrm{d} t)_{\mathrm{i}}$ [$\mathrm{eV} / \mathrm{s}$]')
ax_6.scatter(data_initial_psi / np.pi, dK_dt_initial / elementary_charge, s=50, c=color_target_2, cmap=cmap_color_2, vmin=vmin_2, vmax=vmax_2, zorder=2, alpha=0.8)

axes = [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6]
for ax in axes:
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.text(-0.15, 1.0, '(' + chr(97 + axes.index(ax)) + ')', transform=ax.transAxes)


plt.tight_layout()
plt.savefig(f'{dir_name}/{figure_name}')
plt.savefig(f'{dir_name}/{figure_name.replace(".png", ".pdf")}')
plt.close()

quit()

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