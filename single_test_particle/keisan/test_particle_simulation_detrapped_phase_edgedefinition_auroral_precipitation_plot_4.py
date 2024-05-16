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


# data_initial_Kperp_eqの値ごとに、データを分割する。
data_initial_Kperp_eq_unique = np.unique(data_initial_Kperp_eq)
data_initial_Kperp_eq_unique_number = len(data_initial_Kperp_eq_unique)

data_initial_Kperp_eq_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
data_initial_S_value_times_sign_Theta_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
data_initial_psi_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
data_initial_capital_theta_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
data_initial_mlat_rad_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
data_initial_energy_perp_eV_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
data_initial_energy_para_eV_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
data_initial_energy_eV_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
data_energy_perp_ionospheric_end_eV_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
data_energy_para_ionospheric_end_eV_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
data_energy_ionospheric_end_eV_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])

for count_i in range(len(data_initial_Kperp_eq)):
    for count_j in range(data_initial_Kperp_eq_unique_number):
        if data_initial_Kperp_eq[count_i] == data_initial_Kperp_eq_unique[count_j]:
            data_initial_Kperp_eq_separated[count_j, count_i] = data_initial_Kperp_eq[count_i]
            data_initial_S_value_times_sign_Theta_separated[count_j, count_i] = data_initial_S_value[count_i] * np.sign(data_initial_capital_theta[count_i])
            data_initial_psi_separated[count_j, count_i] = data_initial_psi[count_i]
            data_initial_capital_theta_separated[count_j, count_i] = data_initial_capital_theta[count_i]
            data_initial_mlat_rad_separated[count_j, count_i] = data_initial_mlat_rad[count_i]
            data_initial_energy_perp_eV_separated[count_j, count_i] = data_initial_energy_perp_eV[count_i]
            data_initial_energy_para_eV_separated[count_j, count_i] = data_initial_energy_para_eV[count_i]
            data_initial_energy_eV_separated[count_j, count_i] = data_initial_energy_eV[count_i]
            data_energy_perp_ionospheric_end_eV_separated[count_j, count_i] = data_energy_perp_ionospheric_end_eV[count_i]
            data_energy_para_ionospheric_end_eV_separated[count_j, count_i] = data_energy_para_ionospheric_end_eV[count_i]
            data_energy_ionospheric_end_eV_separated[count_j, count_i] = data_energy_ionospheric_end_eV[count_i]
        else:
            data_initial_Kperp_eq_separated[count_j, count_i] = np.nan
            data_initial_S_value_times_sign_Theta_separated[count_j, count_i] = np.nan
            data_initial_psi_separated[count_j, count_i] = np.nan
            data_initial_capital_theta_separated[count_j, count_i] = np.nan
            data_initial_mlat_rad_separated[count_j, count_i] = np.nan
            data_initial_energy_perp_eV_separated[count_j, count_i] = np.nan
            data_initial_energy_para_eV_separated[count_j, count_i] = np.nan
            data_initial_energy_eV_separated[count_j, count_i] = np.nan
            data_energy_perp_ionospheric_end_eV_separated[count_j, count_i] = np.nan
            data_energy_para_ionospheric_end_eV_separated[count_j, count_i] = np.nan
            data_energy_ionospheric_end_eV_separated[count_j, count_i] = np.nan




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


trapping_frequency_initial_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
d_Ktotal_d_t_initial_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
K_E_initial_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
for count_i in range(data_initial_Kperp_eq_unique_number):
    for count_j in range(len(data_initial_Kperp_eq)):
        trapping_frequency_initial_separated[count_i, count_j] = trapping_frequency(data_initial_mlat_rad_separated[count_i, count_j])
        d_Ktotal_d_t_initial_separated[count_i, count_j] = d_Ktotal_d_t(data_initial_capital_theta_separated[count_i, count_j], data_initial_psi_separated[count_i, count_j], data_initial_mlat_rad_separated[count_i, count_j])
        K_E_initial_separated[count_i, count_j] = data_initial_energy_eV_separated[count_i, count_j] - energy_wave_potential(data_initial_mlat_rad_separated[count_i, count_j])


fig = plt.figure(figsize=(22.5, 15), dpi=100)
gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.05])

cmap_color = cm.turbo

color_target = data_energy_ionospheric_end_eV   # [eV]
vmin = np.min(color_target)
vmax = np.max(color_target)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
sm.set_array([])
cbarax = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(sm, cax=cbarax, orientation='horizontal')
cbar.set_label(r'$K (\lambda = \lambda_{\mathrm{iono}})$ [$\mathrm{eV}$]')

ax = fig.add_subplot(gs[0, 0], title=r'$K_{\perp} (\lambda = 0) =$ '+f'{data_initial_Kperp_eq_unique[-9]:.0f} eV', xlabel=r'MLAT $\lambda_{\mathrm{i}}$ [deg]', ylabel=r'$K_{\mathrm{i}}$ [eV]')
ax.scatter(data_initial_mlat_rad_separated[-9, :] / np.pi * 180E0, data_initial_energy_eV_separated[-9, :], s=200, alpha=0.8, color='white', zorder=1)
ax.scatter(data_initial_mlat_rad_separated[-9, :] / np.pi * 180E0, data_initial_energy_eV_separated[-9, :], c=color_target, cmap=cmap_color, s=200, vmin=vmin, vmax=vmax, alpha=0.8, zorder=10)
ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

xlim_ax = ax.get_xlim()
ylim_ax = ax.get_ylim()
if ylim_ax[0] > 0:
    ylim_ax = (0, ylim_ax[1])

mlat_deg_array = np.linspace(1E0, 70E0, 10000)
mlat_rad_array = mlat_deg_array / 180E0 * np.pi
Kperp_eV_array = 1E0  / B0_eq * magnetic_flux_density(mlat_rad_array)
Kpara_Vph_eV_array = electron_mass / 2E0 * wave_phase_speed(mlat_rad_array)**2E0 / elementary_charge
ax.plot(mlat_deg_array, Kperp_eV_array + Kpara_Vph_eV_array, color='k', lw=2, alpha=0.5, label=r'$K_{\perp} + K_{\mathrm{ph} \parallel}$')

ax.set_xlim(xlim_ax)
ax.set_ylim(ylim_ax)

ax.legend(loc='upper right')

fig.tight_layout()
plt.savefig(f'{dir_name}/{figure_name.replace(".png", "_MLAT_1eV.png")}')
plt.savefig(f'{dir_name}/{figure_name.replace(".png", "_MLAT_1eV.pdf")}')

quit()


fig = plt.figure(figsize=(30, 40), dpi=100)
gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 0.1])

cmap_color = cm.turbo

color_target = data_energy_ionospheric_end_eV   # [eV]
vmin = np.min(color_target)
vmax = np.max(color_target)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
sm.set_array([])
cbarax = fig.add_subplot(gs[5, :])
cbar = plt.colorbar(sm, cax=cbarax, orientation='horizontal')
cbar.set_label(r'$K (\lambda = \lambda_{\mathrm{iono}})$ [$\mathrm{eV}$]')

print(vmax, vmin)

ax_1_1 = fig.add_subplot(gs[0, 0], title=r'$K_{\perp} (\lambda = 0) =$ '+f'{data_initial_Kperp_eq_unique[-1]:.0f} eV', xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\Theta_{\mathrm{i}}$')
ax_1_2 = fig.add_subplot(gs[1, 0], title=r'$K_{\perp} (\lambda = 0) =$ '+f'{data_initial_Kperp_eq_unique[-3]:.0f} eV', xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\Theta_{\mathrm{i}}$')
ax_1_3 = fig.add_subplot(gs[2, 0], title=r'$K_{\perp} (\lambda = 0) =$ '+f'{data_initial_Kperp_eq_unique[-5]:.0f} eV', xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\Theta_{\mathrm{i}}$')
ax_1_4 = fig.add_subplot(gs[3, 0], title=r'$K_{\perp} (\lambda = 0) =$ '+f'{data_initial_Kperp_eq_unique[-7]:.0f} eV', xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\Theta_{\mathrm{i}}$')
ax_1_5 = fig.add_subplot(gs[4, 0], title=r'$K_{\perp} (\lambda = 0) =$ '+f'{data_initial_Kperp_eq_unique[-8]:.0f} eV', xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\Theta_{\mathrm{i}}$')
ax_2_1 = fig.add_subplot(gs[0, 1], title=r'$K_{\perp} (\lambda = 0) =$ '+f'{data_initial_Kperp_eq_unique[-9]:.0f} eV', xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\Theta_{\mathrm{i}}$')
ax_2_2 = fig.add_subplot(gs[1, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$K_{\mathrm{i}}$ [eV]')
ax_2_3 = fig.add_subplot(gs[2, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$S_{\mathrm{i}} \, \mathrm{sign}(\Theta_{\mathrm{i}})$')
ax_2_4 = fig.add_subplot(gs[3, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\lambda_{\mathrm{i}}$ [deg]')
ax_2_5 = fig.add_subplot(gs[4, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$(\mathrm{d} K / \mathrm{d} t)_{\mathrm{i}}$ [eV/s]')

ax_1_1.scatter(data_initial_psi_separated[-1, :] / np.pi, data_initial_capital_theta_separated[-1, :], c=color_target, cmap=cmap_color, s=50, vmin=vmin, vmax=vmax, alpha=0.8)
ax_1_2.scatter(data_initial_psi_separated[-3, :] / np.pi, data_initial_capital_theta_separated[-3, :], c=color_target, cmap=cmap_color, s=50, vmin=vmin, vmax=vmax, alpha=0.8)
ax_1_3.scatter(data_initial_psi_separated[-5, :] / np.pi, data_initial_capital_theta_separated[-5, :], c=color_target, cmap=cmap_color, s=50, vmin=vmin, vmax=vmax, alpha=0.8)
ax_1_4.scatter(data_initial_psi_separated[-7, :] / np.pi, data_initial_capital_theta_separated[-7, :], c=color_target, cmap=cmap_color, s=50, vmin=vmin, vmax=vmax, alpha=0.8)
ax_1_5.scatter(data_initial_psi_separated[-8, :] / np.pi, data_initial_capital_theta_separated[-8, :], c=color_target, cmap=cmap_color, s=50, vmin=vmin, vmax=vmax, alpha=0.8)
ax_2_1.scatter(data_initial_psi_separated[-9, :] / np.pi, data_initial_capital_theta_separated[-9, :], c=color_target, cmap=cmap_color, s=50, vmin=vmin, vmax=vmax, alpha=0.8)
ax_2_2.scatter(data_initial_psi_separated[-9, :] / np.pi, data_initial_energy_eV_separated[-9, :], c=color_target, cmap=cmap_color, s=50, vmin=vmin, vmax=vmax, alpha=0.8)
ax_2_3.scatter(data_initial_psi_separated[-9, :] / np.pi, data_initial_S_value_times_sign_Theta_separated[-9, :], c=color_target, cmap=cmap_color, s=50, vmin=vmin, vmax=vmax, alpha=0.8)
ax_2_4.scatter(data_initial_psi_separated[-9, :] / np.pi, data_initial_mlat_rad_separated[-9, :] / np.pi * 180E0, c=color_target, cmap=cmap_color, s=50, vmin=vmin, vmax=vmax, alpha=0.8)
ax_2_5.scatter(data_initial_psi_separated[-9, :] / np.pi, d_Ktotal_d_t_initial_separated[-9, :] / elementary_charge, c=color_target, cmap=cmap_color, s=50, vmin=vmin, vmax=vmax, alpha=0.8)

axes_Theta = [ax_1_1, ax_1_2, ax_1_3, ax_1_4, ax_1_5, ax_2_1, ax_2_3]
for ax in axes_Theta:
    xlim_ax = ax.get_xlim()
    ax.hlines(0, -1, 1, color='k', alpha=0.6)
    ax.set_xlim(xlim_ax)

xlim_ax_2_1 = ax_2_1.get_xlim()
ylim_ax_2_1 = ax_2_1.get_ylim()

axes = [ax_1_1, ax_1_2, ax_1_3, ax_1_4, ax_1_5, ax_2_1, ax_2_2, ax_2_3, ax_2_4, ax_2_5]
for ax in axes:
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.text(-0.2, 1.05, '(' + chr(97 + axes.index(ax)) + ')', transform=ax.transAxes)
    if ax == ax_1_1 or ax == ax_1_2 or ax == ax_1_3 or ax == ax_1_4 or ax == ax_1_5:
        ax.set_xlim(xlim_ax_2_1)
        ax.set_ylim(ylim_ax_2_1)

fig.tight_layout(w_pad=0.3, h_pad=0.0)
plt.savefig(f'{dir_name}/{figure_name}')
plt.savefig(f'{dir_name}/{figure_name.replace(".png", ".pdf")}')
plt.close()