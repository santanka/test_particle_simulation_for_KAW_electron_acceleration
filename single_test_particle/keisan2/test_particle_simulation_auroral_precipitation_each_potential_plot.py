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

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan2/test_particle_simulation_auroral_precipitation_each_potential'

Kperp_eq = 0E0 #[eV]
#Kperp_eq = 1E0 #[eV]

potential_value_list = [2E3, 2E2 * np.sqrt(1E1), 2E2, 2E1 * np.sqrt(1E1), 2E1]

list_length = len(potential_value_list)

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

mu = Kperp_eq / (B0_eq * 1E9)    #[eV/nT]

# wave parameters
kperp_rhoi = 2E0 * np.pi    #[rad]
wave_frequency = 2E0 * np.pi * 0.15    #[rad/s]
wave_period = 2E0 * np.pi / wave_frequency    #[s]

def wave_phase_speed(mlat_rad):
    return Alfven_speed(mlat_rad) * kperp_rhoi * np.sqrt((1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))) * np.sign(mlat_rad)    #[m/s]

def kpara(mlat_rad):
    return wave_frequency / wave_phase_speed(mlat_rad)   #[rad/m]

def wave_phase_speed(mlat_rad):
    return Alfven_speed(mlat_rad) * kperp_rhoi * np.sqrt((1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))) * np.sign(mlat_rad)    #[m/s]

def kpara(mlat_rad):
    return wave_frequency / wave_phase_speed(mlat_rad)    #[rad/m]

def wave_modified_potential(mlat_rad, wave_scalar_potential):
    return wave_scalar_potential * (2E0 + 1E0 / tau(mlat_rad))    #[V]

def energy_wave_phase_speed(mlat_rad):
    return 5E-1 * electron_mass * wave_phase_speed(mlat_rad)**2E0 #[J]

def energy_wave_potential(mlat_rad, wave_scalar_potential):
    return elementary_charge * wave_modified_potential(mlat_rad, wave_scalar_potential)    #[J]

def delta_1(mlat_rad):
    return 3E0 / kpara(mlat_rad) / r_eq * np.sin(mlat_rad) * (3E0 + 5E0 * np.sin(mlat_rad)**2E0) / np.cos(mlat_rad)**2E0 / (1E0 + 3E0 * np.sin(mlat_rad)**2E0)**1.5E0

diff_rad = 1E-6 #[rad]

def d_mlat_d_z(mlat_rad):
    return 1E0 / r_eq / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)    #[rad/m]

def delta_2(mlat_rad):
    delta_plus = delta_1(mlat_rad + diff_rad) * kpara(mlat_rad + diff_rad) * magnetic_flux_density(mlat_rad + diff_rad)
    delta_minus = delta_1(mlat_rad - diff_rad) * kpara(mlat_rad - diff_rad) * magnetic_flux_density(mlat_rad - diff_rad)
    return (delta_plus - delta_minus) / 2E0 / diff_rad / kpara(mlat_rad)**2E0 / magnetic_flux_density(mlat_rad) * d_mlat_d_z(mlat_rad)

def Gamma(mlat_rad):
    return 1E0 + 2E0 * plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]

def trapping_frequency(mlat_rad, wave_scalar_potential):
    return np.abs(kpara(mlat_rad)) * np.sqrt(energy_wave_potential(mlat_rad, wave_scalar_potential) / electron_mass)   #[rad/s]

def pitch_angle_function(capital_theta, mlat_rad, wave_scalar_potential):
    vpara = (capital_theta * 2E0 * trapping_frequency(mlat_rad, wave_scalar_potential) + wave_frequency) / kpara(mlat_rad)
    vperp = np.sqrt(2E0 * magnetic_flux_density(mlat_rad) * (mu * elementary_charge * 1E9) / electron_mass)
    return np.arccos(vpara / np.sqrt(vpara**2E0 + vperp**2E0))  #[rad]

def Delta_K_function(psi, capital_theta, mlat_rad, wave_scalar_potential):
    Kpara_KE = 2E0 * (capital_theta + np.sqrt(energy_wave_phase_speed(mlat_rad) / 2E0 / energy_wave_potential(mlat_rad, wave_scalar_potential)))**2E0
    Kperp_KE = magnetic_flux_density(mlat_rad) * (mu * elementary_charge * 1E9) / energy_wave_potential(mlat_rad, wave_scalar_potential)
    return np.sin(psi) / (Kpara_KE + Kperp_KE)

def Delta_alpha_function(psi, capital_theta, mlat_rad, wave_scalar_potential):
    alpha = pitch_angle_function(capital_theta, mlat_rad, wave_scalar_potential)
    Kpara_KE = 2E0 * (capital_theta + np.sqrt(energy_wave_phase_speed(mlat_rad) / 2E0 / energy_wave_potential(mlat_rad, wave_scalar_potential)))**2E0
    Kperp_KE = magnetic_flux_density(mlat_rad) * (mu * elementary_charge * 1E9) / energy_wave_potential(mlat_rad, wave_scalar_potential)
    return Gamma(mlat_rad) * np.sin(psi)**2E0 / (1E0 + Gamma(mlat_rad) * np.cos(alpha)**2E0) * (1E0 / (Kpara_KE + Kperp_KE) * np.sin(psi) + delta_1(mlat_rad))

def Delta_Gamma_function(capital_theta, mlat_rad, wave_scalar_potential):
    alpha = pitch_angle_function(capital_theta, mlat_rad, wave_scalar_potential)
    return (Gamma(mlat_rad) - 1E0) * (3E0 - Gamma(mlat_rad)) * np.cos(alpha)**2E0 / (1E0 + Gamma(mlat_rad) * np.cos(alpha)**2E0) * delta_1(mlat_rad)

def Delta_delta_function(mlat_rad):
    return - ((Gamma(mlat_rad) - 1E0) / 2E0 * delta_1(mlat_rad) + delta_2(mlat_rad) / delta_1(mlat_rad))

def Delta_S_function(psi, capital_theta, mlat_rad, wave_scalar_potential):
    return Delta_K_function(psi, capital_theta, mlat_rad, wave_scalar_potential) + Delta_alpha_function(psi, capital_theta, mlat_rad, wave_scalar_potential) + Delta_Gamma_function(capital_theta, mlat_rad, wave_scalar_potential) - Delta_delta_function(mlat_rad)

def d_S_d_t(psi, capital_theta, mlat_rad, wave_scalar_potential, vpara_per_light_speed, S_value):
    return - kpara(mlat_rad) * (vpara_per_light_speed * speed_of_light) * Delta_S_function(psi, capital_theta, mlat_rad, wave_scalar_potential) * S_value   #[s-1]

def d_f_d_t(S_value, psi, capital_theta, mlat_rad, wave_scalar_potential):
    kpara_vpara = capital_theta * 2E0 * trapping_frequency(mlat_rad, wave_scalar_potential) + wave_frequency
    return kpara_vpara * ((1E0 + Gamma(mlat_rad)) * delta_1(mlat_rad) * capital_theta**2E0 - 5E-1 * (psi + np.pi - np.arcsin(S_value)) * Delta_S_function(psi, capital_theta, mlat_rad, wave_scalar_potential) * S_value)

##### additional parameters #####

mlat_rad_background = np.linspace(0E0, mlat_upper_limit_rad, 1000)
Vph_array_background = wave_phase_speed(mlat_rad_background) / speed_of_light  #[/c]
Kph_array_background = energy_wave_phase_speed(mlat_rad_background) / elementary_charge    #[eV]
Kperp_array_background = magnetic_flux_density(mlat_rad_background) * (mu * 1E9)   #[eV]

def path_name(potential):
    return f'{dir_name}/mu_{mu:.4f}_wave_scalar_potential_{potential:.4f}.csv'

def path_name_figure(potential):
    return f'{dir_name}/mu_{mu:.4f}_wave_scalar_potential_{potential:.4f}.png'

def data_load(potential):
    path = path_name(potential)
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data

def plot(data, potential_value):
    initial_S_array = data[:, 0]
    psi_initial_array = data[:, 1]
    vpara_initial_array = data[:, 2]
    capital_theta_initial_array = data[:, 3]
    mlat_deg_initial_array = data[:, 4]
    energy_initial_array = data[:, 5]
    energy_ionosphere_array = data[:, 6]
    reach_time_array = data[:, 7]

    dS_dt_initial_array = np.zeros_like(initial_S_array)
    df_dt_initial_array = np.zeros_like(initial_S_array)
    for count_i in range(len(initial_S_array)):
        dS_dt_initial_array[count_i] = d_S_d_t(psi_initial_array[count_i] * np.pi, capital_theta_initial_array[count_i], mlat_deg_initial_array[count_i] * np.pi / 180E0, potential_value, vpara_initial_array[count_i], initial_S_array[count_i])
        df_dt_initial_array[count_i] = d_f_d_t(initial_S_array[count_i], psi_initial_array[count_i] * np.pi, capital_theta_initial_array[count_i], mlat_deg_initial_array[count_i] * np.pi / 180E0, potential_value)

    fig = plt.figure(figsize=(15, 20), dpi=100)
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 0.05], width_ratios=[1, 1])
    axes = [gs[0, 0], gs[1, 0], gs[2, 0], gs[3, 0], gs[0, 1], gs[1, 1], gs[2, 1], gs[3, 1], gs[4, :]]

    # colorbar
    cmap_color = cm.turbo
    color_target = energy_ionosphere_array
    vmin = np.nanmin(color_target)
    vmax = np.nanmax(color_target)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    #norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap_color, norm=norm)
    sm.set_array([])
    ax_cbar = fig.add_subplot(axes[8])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(r'$K_{\mathrm{iono}}$ [eV]')

    fig.suptitle(r'$\varphi_{0} = %.1f$ V, $\mu = %.3f$ eV/nT, $\min \, K_{\mathrm{iono}} = %.1f$ eV, $\max \, K_{\mathrm{iono}} = %.1f$ eV' % (potential_value, mu, vmin, vmax))

    ax_1 = fig.add_subplot(axes[0], xlabel=r'$\lambda_{\mathrm{i}}$ [deg]', ylabel=r'$K_{\mathrm{i}}$ [eV]')
    ax_2 = fig.add_subplot(axes[1], xlabel=r'$\lambda_{\mathrm{i}}$ [deg]', ylabel=r'$v_{\parallel \mathrm{i}}$ [/c]')
    ax_3 = fig.add_subplot(axes[2], xlabel=r'$\lambda_{\mathrm{i}}$ [deg]', ylabel=r'$S_{\mathrm{i}}$', yscale='log')
    ax_4 = fig.add_subplot(axes[3], xlabel=r'$\lambda_{\mathrm{i}}$ [deg]', ylabel=r'reach time [s]')
    ax_5 = fig.add_subplot(axes[4], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\theta_{\mathrm{i}} / 2 \omega_{\mathrm{t}} (\lambda_{\mathrm{i}})$')
    ax_6 = fig.add_subplot(axes[5], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\lambda_{\mathrm{i}}$ [deg]')
    ax_7 = fig.add_subplot(axes[6], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\mathrm{d}S/\mathrm{d}t$ [s$^{-1}$]')
    ax_8 = fig.add_subplot(axes[7], xlabel=r'$\psi_{\mathrm{i}}$ [$\pi$ rad]', ylabel=r'$\mathrm{d}f/\mathrm{d}t$ [s$^{-1}$]', yscale='log')

    ax_1.scatter(mlat_deg_initial_array, energy_initial_array, c=color_target, cmap=cmap_color, norm=norm, alpha=0.8, s=50)
    ax_2.scatter(mlat_deg_initial_array, vpara_initial_array, c=color_target, cmap=cmap_color, norm=norm, alpha=0.8, s=50)
    ax_3.scatter(mlat_deg_initial_array, initial_S_array, c=color_target, cmap=cmap_color, norm=norm, alpha=0.8, s=50)
    ax_4.scatter(mlat_deg_initial_array, reach_time_array, c=color_target, cmap=cmap_color, norm=norm, alpha=0.8, s=50)
    ax_5.scatter(psi_initial_array, capital_theta_initial_array, c=color_target, cmap=cmap_color, norm=norm, alpha=0.8, s=50)
    ax_6.scatter(psi_initial_array, mlat_deg_initial_array, c=color_target, cmap=cmap_color, norm=norm, alpha=0.8, s=50)
    ax_7.scatter(psi_initial_array, dS_dt_initial_array, c=color_target, cmap=cmap_color, norm=norm, alpha=0.8, s=50)
    ax_8.scatter(psi_initial_array, df_dt_initial_array, c=color_target, cmap=cmap_color, norm=norm, alpha=0.8, s=50)

    ax_1_xlim = ax_1.get_xlim()
    ax_1_ylim = ax_1.get_ylim()

    KE_array_background = energy_wave_potential(mlat_rad_background, potential_value) / elementary_charge * np.ones_like(mlat_rad_background)   #[eV]
    ax_1.plot(mlat_rad_background * 180E0 / np.pi, Kph_array_background, c=r'r', linewidth=4, alpha=0.6, zorder=0, linestyle='-.')
    ax_1.plot(mlat_rad_background * 180E0 / np.pi, KE_array_background, c=r'g', linewidth=4, alpha=0.6, zorder=0, linestyle='-.')
    ax_1.plot(mlat_rad_background * 180E0 / np.pi, Kperp_array_background, c=r'orange', linewidth=4, alpha=0.6, zorder=0, linestyle='-.')

    if ax_1_ylim[0] != 0:
        ax_1_ylim = (0E0, ax_1_ylim[1])
    if ax_1_xlim[0] != 0:
        ax_1_xlim = (0E0, ax_1_xlim[1])
    ax_1.set_xlim(ax_1_xlim)
    ax_1.set_ylim(ax_1_ylim)
    
    ax_2_xlim = ax_2.get_xlim()
    ax_2_ylim = ax_2.get_ylim()

    ax_2.plot(mlat_rad_background * 180E0 / np.pi, Vph_array_background, c=r'r', linewidth=4, alpha=0.6, zorder=0, linestyle='-.')
    if ax_2_xlim[0] != 0:
        ax_2_xlim = (0E0, ax_2_xlim[1])
    ax_2.set_xlim(ax_2_xlim)
    ax_2.set_ylim(ax_2_ylim)

    ax_3_xlim = ax_3.get_xlim()
    ax_3_ylim = ax_3.get_ylim()

    if ax_3_ylim[1] != 1:
        ax_3_ylim = (ax_3_ylim[0], 1E0)
    if ax_3_xlim[0] != 0:
        ax_3_xlim = (0E0, ax_3_xlim[1])
    ax_3.set_xlim(ax_3_xlim)
    ax_3.set_ylim(ax_3_ylim)    
    
    ax_4_xlim = ax_4.get_xlim()
    ax_4_ylim = ax_4.get_ylim()

    if ax_4_ylim[0] != 0:
        ax_4_ylim = (0E0, ax_4_ylim[1])
    if ax_4_xlim[0] != 0:
        ax_4_xlim = (0E0, ax_4_xlim[1])
    max_reach_time = np.nanmax(reach_time_array)
    integer_number_for_period = np.floor(max_reach_time / wave_period)
    for count_i in range(int(integer_number_for_period) + 1):
        ax_4.axhline(y=wave_period * count_i, color='k', linewidth=4E0, zorder=0, alpha=0.3, linestyle='--')
    ax_4.set_xlim(ax_4_xlim)
    ax_4.set_ylim(ax_4_ylim)

    ax_5_xlim = ax_5.get_xlim()
    ax_5_ylim = ax_5.get_ylim()
    ax_5.axhline(y=0E0, color='k', linewidth=4E0, zorder=0, alpha=0.3, linestyle='--')
    if ax_5_xlim[0] != -1E0:
        ax_5_xlim = (-1E0, ax_5_xlim[1])
    if ax_5_ylim[0] < -1E0:
        ax_5_ylim = (-1E0, ax_5_ylim[1])
    if ax_5_xlim[1] > 1E0:
        ax_5_xlim = (ax_5_xlim[0], 1E0)
    if ax_5_ylim[1] > 1E0:
        ax_5_ylim = (ax_5_ylim[0], 1E0)
    ax_5.set_xlim(ax_5_xlim)
    ax_5.set_ylim(ax_5_ylim)

    ax_6_xlim = ax_6.get_xlim()
    ax_6_ylim = ax_6.get_ylim()
    if ax_6_xlim[0] != -1E0:
        ax_6_xlim = (-1E0, ax_6_xlim[1])
    if ax_6_ylim[0] != 0:
        ax_6_ylim = (0E0, ax_6_ylim[1])
    if ax_6_xlim[1] > 1E0:
        ax_6_xlim = (ax_6_xlim[0], 1E0)
    if ax_6_ylim[1] > mlat_upper_limit_deg:
        ax_6_ylim = (ax_6_ylim[0], mlat_upper_limit_deg)
    ax_6.set_xlim(ax_6_xlim)
    ax_6.set_ylim(ax_6_ylim)

    ax_7_xlim = ax_7.get_xlim()
    ax_7_ylim = ax_7.get_ylim()
    if ax_7_xlim[0] != -1E0:
        ax_7_xlim = (-1E0, ax_7_xlim[1])
    if ax_7_xlim[1] > 1E0:
        ax_7_xlim = (ax_7_xlim[0], 1E0)
    ax_7.set_xlim(ax_7_xlim)

    ax_8_xlim = ax_8.get_xlim()
    ax_8_ylim = ax_8.get_ylim()
    if ax_8_xlim[0] != -1E0:
        ax_8_xlim = (-1E0, ax_8_xlim[1])
    if ax_8_xlim[1] > 1E0:
        ax_8_xlim = (ax_8_xlim[0], 1E0)
    ax_8.set_xlim(ax_8_xlim)

    axes_list = [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6, ax_7, ax_8]
    number_count = 0
    for ax in axes_list:
        ax.minorticks_on()
        ax.grid(which='both', alpha=0.3)
        ax.text(-0.15, 0.9, '(' + chr(97 + number_count) + ')', transform=ax.transAxes)
        number_count += 1
    
    fig.tight_layout()
    fig.savefig(path_name_figure(potential_value))

    plt.close(fig)
    
    return

number_count = 0
for count_i in range(list_length):
    data = data_load(potential_value_list[count_i])
    plot(data, potential_value_list[count_i])