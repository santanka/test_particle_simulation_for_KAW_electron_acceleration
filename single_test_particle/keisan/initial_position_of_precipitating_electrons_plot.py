import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import os

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 25

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
figure_name = os.path.basename(file_path).replace('.csv', '_for_paper_Section_5_2_1.png')

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

limit_altitude = 500E3 #[m]
a_req_b = 1E0**2E0 + 2E0 * lshell_number * limit_altitude / planet_radius_polar - planet_radius_polar**2E0 / planet_radius**2E0
mlat_upper_limit_rad = (a_req_b + np.sqrt(a_req_b**2E0 + 4E0 * lshell_number**2E0 * ((planet_radius_polar /planet_radius)**2E0 - (limit_altitude / planet_radius)**2E0))) / 2E0 / lshell_number**2E0
mlat_upper_limit_rad = np.arccos(np.sqrt(mlat_upper_limit_rad)) #[rad]
mlat_upper_limit_deg = mlat_upper_limit_rad * 180E0 / np.pi #[deg]

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

def energy_wave_phase_speed(mlat_rad):
    return 5E-1 * electron_mass * wave_phase_speed(mlat_rad)**2E0 #[J]

def delta(mlat_rad):
    return 3E0 / kpara(mlat_rad) / r_eq * np.sin(mlat_rad) * (3E0 + 5E0 * np.sin(mlat_rad)**2E0) / np.cos(mlat_rad)**2E0 / (1E0 + 3E0 * np.sin(mlat_rad)**2E0)**1.5E0    #[rad]

def Gamma(mlat_rad):
    return 1E0 + 2E0 * plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]

def Kperp_energy(mu, mlat_rad):
    return mu * magnetic_flux_density(mlat_rad) #[J]

def Kpara_energy(theta, mlat_rad):
    return (1E0 + theta / wave_frequency)**2E0 * energy_wave_phase_speed(mlat_rad) #[J]

def Ktotal_energy(mu, theta, mlat_rad):
    return Kperp_energy(mu, mlat_rad) + Kpara_energy(theta, mlat_rad) #[J]

def pitch_angle(mu, theta, mlat_rad):
    return np.arccos(np.sqrt(Kpara_energy(theta, mlat_rad)/ Ktotal_energy(mu, theta, mlat_rad)) * np.sign((theta + wave_frequency) / kpara(mlat_rad)))    #[rad]

def trapping_frequency(mlat_rad):
    return np.abs(kpara(mlat_rad)) * np.sqrt(energy_wave_potential(mlat_rad) / electron_mass)   #[rad/s]

def S_value(mu, theta, mlat_rad):
    return (Kpara_energy(theta, mlat_rad) / energy_wave_potential(mlat_rad) * (1E0 + Gamma(mlat_rad)) + Kperp_energy(mu, mlat_rad) / energy_wave_potential(mlat_rad))  * delta(mlat_rad)    #[]

##### additional parameters end #####

alpha_initial_separated = np.zeros([data_initial_Kperp_eq_unique_number, len(data_initial_Kperp_eq)])
for count_i in range(len(data_initial_Kperp_eq)):
    for count_j in range(data_initial_Kperp_eq_unique_number):
        if data_initial_Kperp_eq[count_i] == data_initial_Kperp_eq_unique[count_j]:
            mu = data_initial_Kperp_eq[count_i] / magnetic_flux_density(0) * elementary_charge
            theta = data_initial_capital_theta_separated[count_j, count_i] * 2E0 * trapping_frequency(data_initial_mlat_rad_separated[count_j, count_i])
            alpha_initial_separated[count_j, count_i] = pitch_angle(mu, theta, data_initial_mlat_rad_separated[count_j, count_i])
        else:
            alpha_initial_separated[count_j, count_i] = np.nan

fig = plt.figure(figsize=(15, 20), dpi=100)
gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 0.05])

cmap_color = cm.turbo

color_target = data_energy_ionospheric_end_eV
vmin = np.nanmin(color_target)
vmax = np.nanmax(color_target)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
sm.set_array([])
ax_cbar = fig.add_subplot(gs[4, :])
cbar = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal')
cbar.set_label(r'$K_{\mathrm{iono}}$ [eV]')

unique_number_list = [0, 2, 4, 8]

mu_for_ax_1 = data_initial_Kperp_eq_unique[unique_number_list[0]] / (magnetic_flux_density(0) * 1E9)
Kiono_min_for_ax_1 = data_initial_Kperp_eq_unique[unique_number_list[0]] * B_ratio_constant
mu_for_ax_2 = data_initial_Kperp_eq_unique[unique_number_list[1]] / (magnetic_flux_density(0) * 1E9)
Kiono_min_for_ax_2 = data_initial_Kperp_eq_unique[unique_number_list[1]] * B_ratio_constant
mu_for_ax_3 = data_initial_Kperp_eq_unique[unique_number_list[2]] / (magnetic_flux_density(0) * 1E9)
Kiono_min_for_ax_3 = data_initial_Kperp_eq_unique[unique_number_list[2]] * B_ratio_constant
mu_for_ax_4 = data_initial_Kperp_eq_unique[unique_number_list[3]] / (magnetic_flux_density(0) * 1E9)
Kiono_min_for_ax_4 = data_initial_Kperp_eq_unique[unique_number_list[3]] * B_ratio_constant

ax_1_1 = fig.add_subplot(gs[0, 0], title=r'$\mu = %.3f$ eV/nT, $\mathrm{min} K_{\mathrm{iono}} = %.0f$ eV' % (mu_for_ax_1, Kiono_min_for_ax_1), xlabel=r'$\lambda_{\mathrm{i}}$ [deg]', ylabel=r'$K_{\mathrm{i}}$ [eV]')
ax_2_1 = fig.add_subplot(gs[1, 0], title=r'$\mu = %.3f$ eV/nT, $\mathrm{min} K_{\mathrm{iono}} = %.0f$ eV' % (mu_for_ax_2, Kiono_min_for_ax_2), xlabel=r'$\lambda_{\mathrm{i}}$ [deg]', ylabel=r'$K_{\mathrm{i}}$ [eV]')
ax_3_1 = fig.add_subplot(gs[2, 0], title=r'$\mu = %.3f$ eV/nT, $\mathrm{min} K_{\mathrm{iono}} = %.0f$ eV' % (mu_for_ax_3, Kiono_min_for_ax_3), xlabel=r'$\lambda_{\mathrm{i}}$ [deg]', ylabel=r'$K_{\mathrm{i}}$ [eV]')
ax_4_1 = fig.add_subplot(gs[3, 0], title=r'$\mu = %.3f$ eV/nT, $\mathrm{min} K_{\mathrm{iono}} = %.0f$ eV' % (mu_for_ax_4, Kiono_min_for_ax_4), xlabel=r'$\lambda_{\mathrm{i}}$ [deg]', ylabel=r'$K_{\mathrm{i}}$ [eV]')
ax_1_2 = fig.add_subplot(gs[0, 1], title=r'$\mu = %.3f$ eV/nT, $\mathrm{min} K_{\mathrm{iono}} = %.0f$ eV' % (mu_for_ax_1, Kiono_min_for_ax_1), xlabel=r'$\lambda_{\mathrm{i}}$ [deg]', ylabel=r'$\alpha_{\mathrm{i}}$ [deg]')
ax_2_2 = fig.add_subplot(gs[1, 1], xlabel=r'$\lambda_{\mathrm{i}}$ [deg]', ylabel=r'$S_{\mathrm{i}}$', yscale='log')
ax_3_2 = fig.add_subplot(gs[2, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\theta_{\mathrm{i}} / 2 \omega_{\mathrm{t}} (\lambda_{\mathrm{i}})$')
ax_4_2 = fig.add_subplot(gs[3, 1], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\lambda_{\mathrm{i}}$ [deg]')

ax_1_1.scatter(data_initial_mlat_rad_separated[unique_number_list[0], :] * 180E0 / np.pi, data_initial_energy_eV_separated[unique_number_list[0], :], c=color_target, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.8, s=50)
ax_2_1.scatter(data_initial_mlat_rad_separated[unique_number_list[1], :] * 180E0 / np.pi, data_initial_energy_eV_separated[unique_number_list[1], :], c=color_target, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.8, s=50)
ax_3_1.scatter(data_initial_mlat_rad_separated[unique_number_list[2], :] * 180E0 / np.pi, data_initial_energy_eV_separated[unique_number_list[2], :], c=color_target, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.8, s=50)
ax_4_1.scatter(data_initial_mlat_rad_separated[unique_number_list[3], :] * 180E0 / np.pi, data_initial_energy_eV_separated[unique_number_list[3], :], c=color_target, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.8, s=50)
ax_1_2.scatter(data_initial_mlat_rad_separated[unique_number_list[0], :] * 180E0 / np.pi, alpha_initial_separated[unique_number_list[0], :] * 180E0 / np.pi, c=color_target, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.8, s=50)
ax_2_2.scatter(data_initial_mlat_rad_separated[unique_number_list[0], :] * 180E0 / np.pi, np.abs(data_initial_S_value_times_sign_Theta_separated[unique_number_list[0], :]), c=color_target, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.8, s=50)
ax_3_2.scatter(data_initial_psi_separated[unique_number_list[0], :] / np.pi, data_initial_capital_theta_separated[unique_number_list[0], :], c=color_target, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.8, s=50)
ax_4_2.scatter(data_initial_psi_separated[unique_number_list[0], :] / np.pi, data_initial_mlat_rad_separated[unique_number_list[0], :] * 180E0 / np.pi, c=color_target, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.8, s=50)


# energy lines & loss cone lines
ax_1_1_xlim = ax_1_1.get_xlim()
ax_1_1_ylim = ax_1_1.get_ylim()

if ax_1_1_ylim[0] != 0E0:
    ax_1_1_ylim = (0E0, ax_1_1_ylim[1])

ax_1_2_xlim = ax_1_2.get_xlim()
ax_1_2_ylim = ax_1_2.get_ylim()

mlat_deg_for_background = np.linspace(1E-1, mlat_upper_limit_deg, 1000)
mlat_rad_for_background = mlat_deg_for_background * np.pi / 180E0
energy_perp_for_background_1 = Kperp_energy(mu_for_ax_1*1E9*elementary_charge, mlat_rad_for_background) / elementary_charge
energy_perp_for_background_2 = Kperp_energy(mu_for_ax_2*1E9*elementary_charge, mlat_rad_for_background) / elementary_charge
energy_perp_for_background_3 = Kperp_energy(mu_for_ax_3*1E9*elementary_charge, mlat_rad_for_background) / elementary_charge
energy_perp_for_background_4 = Kperp_energy(mu_for_ax_4*1E9*elementary_charge, mlat_rad_for_background) / elementary_charge
energy_wave_phase_speed_for_background = energy_wave_phase_speed(mlat_rad_for_background) / elementary_charge
energy_wave_potential_for_background = energy_wave_potential(mlat_rad_for_background) / elementary_charge * np.ones_like(mlat_rad_for_background)

ax_1_1.plot(mlat_deg_for_background, (energy_wave_phase_speed_for_background + energy_perp_for_background_1), c=r'r', linewidth=4, alpha=0.6, zorder=0)
ax_1_1.plot(mlat_deg_for_background, energy_wave_potential_for_background, c=r'g', linewidth=4, alpha=0.6, zorder=0)
ax_1_1.plot(mlat_deg_for_background, energy_perp_for_background_1, c=r'orange', linewidth=4, alpha=0.6, zorder=0)

ax_2_1.plot(mlat_deg_for_background, (energy_wave_phase_speed_for_background + energy_perp_for_background_2), c=r'r', linewidth=4, alpha=0.6, zorder=0)
ax_2_1.plot(mlat_deg_for_background, energy_wave_potential_for_background, c=r'g', linewidth=4, alpha=0.6, zorder=0)
ax_2_1.plot(mlat_deg_for_background, energy_perp_for_background_2, c=r'orange', linewidth=4, alpha=0.6, zorder=0)

ax_3_1.plot(mlat_deg_for_background, (energy_wave_phase_speed_for_background + energy_perp_for_background_3), c=r'r', linewidth=4, alpha=0.6, zorder=0)
ax_3_1.plot(mlat_deg_for_background, energy_wave_potential_for_background, c=r'g', linewidth=4, alpha=0.6, zorder=0)
ax_3_1.plot(mlat_deg_for_background, energy_perp_for_background_3, c=r'orange', linewidth=4, alpha=0.6, zorder=0)

ax_4_1.plot(mlat_deg_for_background, (energy_wave_phase_speed_for_background + energy_perp_for_background_4), c=r'r', linewidth=4, alpha=0.6, zorder=0)
ax_4_1.plot(mlat_deg_for_background, energy_wave_potential_for_background, c=r'g', linewidth=4, alpha=0.6, zorder=0)
ax_4_1.plot(mlat_deg_for_background, energy_perp_for_background_4, c=r'orange', linewidth=4, alpha=0.6, zorder=0)

loss_cone_for_background = np.arcsin(np.sqrt(magnetic_flux_density(mlat_rad_for_background) / magnetic_flux_density(mlat_upper_limit_rad))) * 180E0 / np.pi
ax_1_2.plot(mlat_deg_for_background, loss_cone_for_background, c=r'k', linewidth=4, alpha=0.6, zorder=0)

ax_1_1.set_xlim(ax_1_1_xlim)
ax_1_1.set_ylim(ax_1_1_ylim)
ax_2_1.set_xlim(ax_1_1_xlim)
ax_2_1.set_ylim(ax_1_1_ylim)
ax_3_1.set_xlim(ax_1_1_xlim)
ax_3_1.set_ylim(ax_1_1_ylim)
ax_4_1.set_xlim(ax_1_1_xlim)
ax_4_1.set_ylim(ax_1_1_ylim)
ax_1_2.set_xlim(ax_1_2_xlim)
ax_1_2.set_ylim(ax_1_2_ylim)

ax_2_2_ylim = ax_2_2.get_ylim()
if ax_2_2_ylim[1] > 1E0:
    ax_2_2_ylim = (ax_2_2_ylim[0], 1E0)
ax_2_2.set_ylim(ax_2_2_ylim)

ax_3_2.axhline(y=0E0, color='k', linewidth=4E0, zorder=0, alpha=0.3, linestyle='--')


axes_list = [ax_1_1, ax_2_1, ax_3_1, ax_4_1, ax_1_2, ax_2_2, ax_3_2, ax_4_2]
for ax in axes_list:
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.text(-0.15, 0.9, '(' + chr(97 + axes_list.index(ax)) + ')', transform=ax.transAxes)

fig.tight_layout(w_pad=0.3, h_pad=0.0)

plt.savefig(f'{dir_name}/{figure_name}')
plt.savefig(f'{dir_name}/{figure_name.replace(".png", ".pdf")}')
plt.close()