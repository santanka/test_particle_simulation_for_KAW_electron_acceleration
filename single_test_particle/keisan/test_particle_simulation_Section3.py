import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import datetime
import os
from multiprocessing import Pool
from tqdm.auto import tqdm  # Ensure to import tqdm correctly
import netCDF4 as nc

# Font setting
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 55

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

def d_mlat_d_z(mlat_rad):
    return 1E0 / r_eq / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)    #[rad/m]


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

diff_rad = 1E-6 #[rad]


# wave parameters
kperp_rhoi = 2E0 * np.pi    #[rad]
wave_frequency = 2E0 * np.pi * 0.15    #[rad/s]

def wave_phase_speed(mlat_rad):
    return Alfven_speed(mlat_rad) * kperp_rhoi * np.sqrt((1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad)))# * np.sign(mlat_rad)    #[m/s]

def kpara(mlat_rad):
    return wave_frequency / wave_phase_speed(mlat_rad)    #[rad/m]

wave_scalar_potential = 2000E0   #[V]

def wave_modified_potential(mlat_rad):
    return wave_scalar_potential * (2E0 + 1E0 / tau(mlat_rad))    #[V]

def energy_wave_phase_speed(mlat_rad):
    return 5E-1 * electron_mass * wave_phase_speed(mlat_rad)**2E0 #[J]

def energy_wave_potential(mlat_rad):
    return elementary_charge * wave_modified_potential(mlat_rad)    #[J]

def delta(mlat_rad):
    return 3E0 / kpara(mlat_rad) / r_eq * np.sin(mlat_rad) * (3E0 + 5E0 * np.sin(mlat_rad)**2E0) / np.cos(mlat_rad)**2E0 / (1E0 + 3E0 * np.sin(mlat_rad)**2E0)**1.5E0    #[rad]

def Gamma(mlat_rad):
    return 1E0 + 2E0 * plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]


# energy trajectory
def Kperp_energy(mu, mlat_rad):
    return mu * magnetic_flux_density(mlat_rad) #[J]

def Kpara_energy(theta, mlat_rad):
    return (1E0 + theta / wave_frequency)**2E0 * energy_wave_phase_speed(mlat_rad) #[J]

def Ktotal_energy(mu, theta, mlat_rad):
    return Kperp_energy(mu, mlat_rad) + Kpara_energy(theta, mlat_rad) #[J]

def pitch_angle(mu, theta, mlat_rad):
    return np.arccos(np.sqrt(Kpara_energy(theta, mlat_rad)/ Ktotal_energy(mu, theta, mlat_rad)) * np.sign((theta + wave_frequency) / kpara(mlat_rad)))    #[rad]

def d_Ktotal_d_t(theta, psi, mlat_rad):
    return - energy_wave_potential(mlat_rad) * (theta + wave_frequency) * np.sin(psi)   #[J/s]

def trapping_frequency(mlat_rad):
    return np.abs(kpara(mlat_rad)) * np.sqrt(energy_wave_potential(mlat_rad) / electron_mass)   #[rad/s]

def S_value(mu, theta, mlat_rad):
    return (Kpara_energy(theta, mlat_rad) / energy_wave_potential(mlat_rad) * (1E0 + Gamma(mlat_rad)) + Kperp_energy(mu, mlat_rad) / energy_wave_potential(mlat_rad))  * delta(mlat_rad)    #[]

def d_psi_d_t(theta):
    return theta    #[rad/s]

def d_theta_d_t(mu, theta, psi, mlat_rad):
    return - trapping_frequency(mlat_rad)**2E0 * (np.sin(psi) + S_value(mu, theta, mlat_rad))    #[rad/s]

def d_mlat_rad_d_t(theta, mlat_rad):
    return (theta + wave_frequency) / kpara(mlat_rad) / r_eq / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)    #[rad/s]

def vpara(theta, mlat_rad):
    return (theta + wave_frequency) / kpara(mlat_rad)    #[m/s]

def Xi_dSdt(mlat_rad):
    return 2E0 * (1E0 + plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))) * delta(mlat_rad)    #[]

def d_Gamma_d_t(theta, mlat_rad):
    return - 8E0 * plasma_beta_ion(mlat_rad) * tau(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))**2E0 * (theta + wave_frequency) * delta(mlat_rad)    #[s^-1]

def d_alpha_d_t(mu, theta, psi, mlat_rad):
    pitch_angle_rad = np.arccos(np.sqrt(Kpara_energy(theta, mlat_rad) / Ktotal_energy(mu, theta, mlat_rad)))
    return 1E0 / 2E0 / np.cos(pitch_angle_rad) * (theta + wave_frequency) * (delta(mlat_rad) * np.sin(pitch_angle_rad) + energy_wave_potential(mlat_rad) / Ktotal_energy(mu, theta, mlat_rad) * np.sin(pitch_angle_rad) * np.sin(psi))    #[rad/s]

def d_delta_d_t(theta, mlat_rad):
    d_delta_d_z = (delta(mlat_rad + diff_rad) - delta(mlat_rad - diff_rad)) / 2E0 / diff_rad * d_mlat_d_z(mlat_rad)
    return (theta + wave_frequency) / kpara(mlat_rad) * d_delta_d_z    #[s^-1]

def d_S_d_t(mu, theta, psi, mlat_rad):
    pitch_angle_rad = np.arccos(np.sqrt(Kpara_energy(theta, mlat_rad) / Ktotal_energy(mu, theta, mlat_rad)))
    coef_dKdt = S_value(mu, theta, mlat_rad) / Ktotal_energy(mu, theta, mlat_rad) * d_Ktotal_d_t(theta, psi, mlat_rad)
    coef_dGammadt = S_value(mu, theta, mlat_rad) / (1E0 + Gamma(mlat_rad) * np.cos(pitch_angle_rad)**2E0) * d_Gamma_d_t(theta, mlat_rad) * np.cos(pitch_angle_rad)**2E0
    coef_dalphadt = S_value(mu, theta, mlat_rad) / (1E0 + Gamma(mlat_rad) * np.cos(pitch_angle_rad)**2E0) * d_alpha_d_t(mu, theta, psi, mlat_rad) * Gamma(mlat_rad) * np.sin(2E0 * pitch_angle_rad)
    coef_ddeltadt = S_value(mu, theta, mlat_rad) / delta(mlat_rad) * d_delta_d_t(theta, mlat_rad)
    return coef_dKdt + coef_dGammadt + coef_dalphadt + coef_ddeltadt    #[s^-1]

def W_value_dSdt(mu, theta, psi, mlat_rad):
    return - 1E0 / (theta + wave_frequency) / Xi_dSdt(mlat_rad) * d_S_d_t(mu, theta, psi, mlat_rad) - np.sin(psi)    #[]

def region_detection(mu, theta, psi, mlat_rad, time_now, time_end):
    S_value_now = S_value(mu, theta, mlat_rad)
    trapping_frequency_now = trapping_frequency(mlat_rad)
    plasma_beta_ion_now = plasma_beta_ion(mlat_rad)
    d_S_d_t_now = d_S_d_t(mu, theta, psi, mlat_rad)
    delta_now = delta(mlat_rad)
    if S_value_now <= 1E0 and S_value_now >= 0E0:
        function_phase_trapping = (theta / 2E0 / trapping_frequency_now)**2E0 - 5E-1 * (np.cos(psi) + np.sqrt(1E0 - S_value_now**2E0) - S_value_now * (psi + np.pi - np.arcsin(S_value_now)))
        function_phase_trapping_time_derivative = (theta / trapping_frequency_now)**2E0 * (plasma_beta_ion_now * (1E0 + tau(mlat_rad)) + tau(mlat_rad)) / (plasma_beta_ion_now * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad)) * (theta + wave_frequency) * delta_now + 5E-1 * (psi + np.pi - np.arcsin(S_value_now)) * d_S_d_t_now
        function_saddle_point = psi + np.pi - np.arcsin(S_value_now)
    else:
        function_phase_trapping = np.nan
        function_phase_trapping_time_derivative = np.nan
        function_saddle_point = np.nan
    function_resonant_scattering = (theta / 2E0 / trapping_frequency_now)**2E0 - 5E-1 * (np.cos(psi) + S_value_now * (np.pi - psi) + 1E0)

    #characterize the region
    if function_resonant_scattering > 0E0:
        region = 2.4 + time_now / time_end
    else:
        if S_value_now <= 1E0 and S_value_now >= 0E0:
            if function_phase_trapping <= 0E0 and function_saddle_point >= 0E0:
                region = 0 + time_now / time_end
            else:
                region = 1.2 + time_now / time_end
        else:
            region = 1.2 + time_now / time_end
    
    return function_phase_trapping, function_phase_trapping_time_derivative, function_saddle_point, function_resonant_scattering, region

def detrapped_point_detection(region_before, region_now):
    if region_before <= 1E0 and region_now >= 1.2E0 and region_now <= 2.4E0:
        return 1
    elif region_before >= 1.2E0 and region_before <= 2.4E0 and region_now <= 1E0:
        return 2
    else:
        return 0
    
def loss_cone(mlat_rad):
    return np.arcsin(np.sqrt(magnetic_flux_density(mlat_rad) / magnetic_flux_density(mlat_upper_limit_rad)))    #[rad]

def force_electric_field(mlat_rad, psi):
    return - kpara(mlat_rad) * energy_wave_potential(mlat_rad) / elementary_charge * np.sin(psi)    #[eV/m]

def initial_psi_at_equator(initial_psi, initial_mlat_rad):
    mlat_divide_number = 1E4
    d_mlat_rad = initial_mlat_rad / mlat_divide_number
    old_initial_psi = initial_psi
    for count in range(int(mlat_divide_number)):
        old_mlat_rad = initial_mlat_rad - d_mlat_rad * count
        new_mlat_rad = initial_mlat_rad - d_mlat_rad * (count + 1)
        new_initial_psi = old_initial_psi - (kpara(old_mlat_rad) / d_mlat_d_z(old_mlat_rad) + kpara(new_mlat_rad) / d_mlat_d_z(new_mlat_rad)) / 2E0 * d_mlat_rad
        old_initial_psi = new_initial_psi
    return new_initial_psi


# runge-kutta method
dt = 1E-3
def RK4(mlat_rad_0, theta_0, psi_0, mu):
    # 1st step
    k1_mlat_rad = d_mlat_rad_d_t(theta_0, mlat_rad_0)
    k1_theta = d_theta_d_t(mu, theta_0, psi_0, mlat_rad_0)
    k1_psi = d_psi_d_t(theta_0)

    # 2nd step
    k2_mlat_rad = d_mlat_rad_d_t(theta_0 + k1_theta * dt / 2E0, mlat_rad_0 + k1_mlat_rad * dt / 2E0)
    k2_theta = d_theta_d_t(mu, theta_0 + k1_theta * dt / 2E0, psi_0 + k1_psi * dt / 2E0, mlat_rad_0 + k1_mlat_rad * dt / 2E0)
    k2_psi = d_psi_d_t(theta_0 + k1_theta * dt / 2E0)

    # 3rd step
    k3_mlat_rad = d_mlat_rad_d_t(theta_0 + k2_theta * dt / 2E0, mlat_rad_0 + k2_mlat_rad * dt / 2E0)
    k3_theta = d_theta_d_t(mu, theta_0 + k2_theta * dt / 2E0, psi_0 + k2_psi * dt / 2E0, mlat_rad_0 + k2_mlat_rad * dt / 2E0)
    k3_psi = d_psi_d_t(theta_0 + k2_theta * dt / 2E0)

    # 4th step
    k4_mlat_rad = d_mlat_rad_d_t(theta_0 + k3_theta * dt, mlat_rad_0 + k3_mlat_rad * dt)
    k4_theta = d_theta_d_t(mu, theta_0 + k3_theta * dt, psi_0 + k3_psi * dt, mlat_rad_0 + k3_mlat_rad * dt)
    k4_psi = d_psi_d_t(theta_0 + k3_theta * dt)

    # update
    mlat_rad_1 = mlat_rad_0 + dt * (k1_mlat_rad + 2E0 * k2_mlat_rad + 2E0 * k3_mlat_rad + k4_mlat_rad) / 6E0
    theta_1 = theta_0 + dt * (k1_theta + 2E0 * k2_theta + 2E0 * k3_theta + k4_theta) / 6E0
    psi_1 = psi_0 + dt * (k1_psi + 2E0 * k2_psi + 2E0 * k3_psi + k4_psi) / 6E0

    return mlat_rad_1, theta_1, psi_1

# initial condition
initial_K_eV = np.linspace(1E2, 1E3, 10)
initial_pitch_angle_deg = np.linspace(5E0, 85E0, 17)
initial_mlat_deg = 1E0

background_spatial_number = 1000

initial_pitch_angle_rad = initial_pitch_angle_deg * np.pi / 180E0 #[rad]
initial_mlat_rad = initial_mlat_deg * np.pi / 180E0 #[rad]

INI_K_EV, INI_PITCH_ANGLE_RAD = np.meshgrid(initial_K_eV, initial_pitch_angle_rad)
INI_PITCH_ANGLE_DEG = INI_PITCH_ANGLE_RAD * 180E0 / np.pi

INITIAL_KPERP_EV = INI_K_EV * np.sin(INI_PITCH_ANGLE_RAD)**2E0 #[eV]
INITIAL_KPARA_EV = INI_K_EV * np.cos(INI_PITCH_ANGLE_RAD)**2E0 #[eV]

INITIAL_MU = INITIAL_KPERP_EV * elementary_charge / magnetic_flux_density(initial_mlat_rad) #[J/T]
INITIAL_THETA = kpara(initial_mlat_rad) * np.sqrt(2E0 * INITIAL_KPARA_EV * elementary_charge / electron_mass) - wave_frequency  #[rad/s]
initial_psi = -9E-1 * np.pi #[rad]

def wave_psi_spatial_time_variation_array(time_start, time_end, initial_psi, initial_mlat_rad):
    time_grid_number = 1000
    spatial_grid_number = background_spatial_number
    time_array = np.linspace(time_start, time_end, time_grid_number)
    time_diff = time_array[1] - time_array[0]
    mlat_rad_array_spatial = np.linspace(0, mlat_upper_limit_rad, spatial_grid_number)
    mlat_rad_diff = mlat_rad_array_spatial[1] - mlat_rad_array_spatial[0]
    psi_spatial_time_array = np.zeros((spatial_grid_number, time_grid_number))
    psi_spatial_time_array[0, 0] = initial_psi_at_equator(initial_psi, initial_mlat_rad)
    old_psi = psi_spatial_time_array[0, 0]
    for count_i in range(spatial_grid_number-1):
        old_kpara = kpara(mlat_rad_array_spatial[count_i]) / d_mlat_d_z(mlat_rad_array_spatial[count_i])
        new_kpara = kpara(mlat_rad_array_spatial[count_i+1]) / d_mlat_d_z(mlat_rad_array_spatial[count_i+1])
        new_psi = old_psi + (old_kpara + new_kpara) / 2E0 * mlat_rad_diff
        psi_spatial_time_array[count_i+1, 0] = new_psi
        old_psi = new_psi
    old_psi_array = psi_spatial_time_array[:, 0]
    for count_j in range(time_grid_number-1):
        new_psi_array = old_psi_array - wave_frequency * time_diff
        psi_spatial_time_array[:, count_j+1] = new_psi_array
        old_psi_array = new_psi_array
    return mlat_rad_array_spatial, time_array, psi_spatial_time_array

dt = 1E-3
time_end = 2E1

def funtion_unstable_and_another_point(S_value, psi):
    return np.cos(psi) + np.sqrt(1E0 - S_value**2E0) - S_value * (psi + np.pi - np.arcsin(S_value))

def iteration_another_point(S_value):
    if S_value > 1E0 or S_value < 0E0:
        return np.nan
    psi_old = 5E-1 * np.pi
    diff_psi = 1E-5
    while True:
        function_0 = funtion_unstable_and_another_point(S_value, psi_old)
        function_1 = (funtion_unstable_and_another_point(S_value, psi_old + diff_psi) - funtion_unstable_and_another_point(S_value, psi_old - diff_psi)) / 2E0 / diff_psi
        diff = function_0 / function_1
        if np.abs(diff) > 1E-3:
            diff = np.sign(diff) * 1E-3
        psi_new = psi_old - diff
        if np.abs(psi_new - psi_old) < 1E-5:
            break
        else:
            psi_old = psi_new
    return psi_new


# simulation
def main_calculation(args):
    initial_mlat_rad_main, initial_mu_main, initial_theta_main, initial_psi_main = args

    time = 0E0

    mlat_rad_array = np.asarray([initial_mlat_rad_main])
    theta_array = np.asarray([initial_theta_main])
    vpara_array = np.asarray([vpara(initial_theta_main, initial_mlat_rad_main)])
    psi_array = np.asarray([initial_psi_main])
    time_array = np.asarray([time])
    trapping_frequency_array = np.asarray([trapping_frequency(initial_mlat_rad_main)])
    Ktotal_energy_array = np.asarray([Ktotal_energy(initial_mu_main, initial_theta_main, initial_mlat_rad_main)])
    alpha_array = np.asarray([pitch_angle(initial_mu_main, initial_theta_main, initial_mlat_rad_main)])
    S_value_array = np.asarray([S_value(initial_mu_main, initial_theta_main, initial_mlat_rad_main)])
    region_array = np.asarray([region_detection(initial_mu_main, initial_theta_main, initial_psi_main, initial_mlat_rad_main, time, time_end)[-1]])
    detrapped_point_array = np.asarray([0])

    while time_array[-1] < time_end:
        mlat_rad_new, theta_new, psi_new = RK4(mlat_rad_array[-1], theta_array[-1], psi_array[-1], initial_mu_main)
        psi_new = np.mod(psi_new + np.pi, 2E0 * np.pi) - np.pi
        time += dt

        if mlat_rad_new > mlat_upper_limit_rad or mlat_rad_new <= 1E-1 * np.pi / 180E0:
            break

        trapping_frequency_new = trapping_frequency(mlat_rad_new)
        Ktotal_energy_new = Ktotal_energy(initial_mu_main, theta_new, mlat_rad_new)
        alpha_new = pitch_angle(initial_mu_main, theta_new, mlat_rad_new)
        S_value_new = S_value(initial_mu_main, theta_new, mlat_rad_new)
        region_new = region_detection(initial_mu_main, theta_new, psi_new, mlat_rad_new, time, time_end)[-1]
        detrapped_point_new = detrapped_point_detection(region_array[-1], region_new)

        mlat_rad_array = np.append(mlat_rad_array, mlat_rad_new)
        theta_array = np.append(theta_array, theta_new)
        vpara_array = np.append(vpara_array, vpara(theta_new, mlat_rad_new))
        psi_array = np.append(psi_array, psi_new)
        time_array = np.append(time_array, time)
        trapping_frequency_array = np.append(trapping_frequency_array, trapping_frequency_new)
        Ktotal_energy_array = np.append(Ktotal_energy_array, Ktotal_energy_new)
        alpha_array = np.append(alpha_array, alpha_new)
        S_value_array = np.append(S_value_array, S_value_new)
        region_array = np.append(region_array, region_new)
        detrapped_point_array = np.append(detrapped_point_array, detrapped_point_new)
    
    return mlat_rad_array, theta_array, vpara_array, psi_array, time_array, trapping_frequency_array, Ktotal_energy_array, alpha_array, S_value_array, region_array, detrapped_point_array

# fig path
def fig_path(initial_K_eV, initial_pitch_angle_deg, initial_mlat_deg, initial_psi):
    dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/test_particle_simulation_Section3/{initial_psi/np.pi:.2f}_pi'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_name = f'{dir_name}/{initial_K_eV:.2f}eV_{initial_pitch_angle_deg:.2f}deg_{initial_mlat_deg:.2f}deg'
    return fig_name


# plot
def main_plot(args):
    mlat_rad_array, theta_array, vpara_array, psi_array, time_array, trapping_frequency_array, Ktotal_energy_array, alpha_array, S_value_array, region_array, detrapped_point_array, mu = args

    detrapped_point_1 = np.where(detrapped_point_array == 1)[0]
    detrapped_point_2 = np.where(detrapped_point_array == 2)[0]

    fig = plt.figure(figsize=(25*1.5, 20*1.5), dpi=100)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.05])

    cmap_color = cm.turbo
    color_target = time_array
    vmin_color = np.min(color_target)
    vmax_color = np.max(color_target)
    norm_color = mpl.colors.Normalize(vmin=vmin_color, vmax=vmax_color)
    scalarMap_color = plt.cm.ScalarMappable(norm=norm_color, cmap=cmap_color)
    scalarMap_color.set_array([])

    ax_cbar = fig.add_subplot(gs[2, :])
    cbar = fig.colorbar(scalarMap_color, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(r'Time $t$ [s]')

    ax_1_1 = fig.add_subplot(gs[0, 0], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$v_{\parallel}$ [$c$]')
    ax_1_2 = fig.add_subplot(gs[0, 1], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$K$ [eV]', yscale='log')
    ax_1_3 = fig.add_subplot(gs[0, 2], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$\alpha$ [deg]', yticks=[0, 30, 60, 90, 120, 150, 180])
    ax_2_1 = fig.add_subplot(gs[1, 0], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$S$', yscale='log')
    ax_2_2 = fig.add_subplot(gs[1, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$', xlim=(-1E0, 1E0))
    ax_2_3 = fig.add_subplot(gs[1, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$K$ [eV]', yscale='log', xlim=(-1E0, 1E0))

    ax_1_1.scatter(mlat_rad_array * 180E0 / np.pi, vpara_array / speed_of_light, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_1_1.scatter(mlat_rad_array[0] * 180E0 / np.pi, vpara_array[0] / speed_of_light, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_1_1.scatter(mlat_rad_array[-1] * 180E0 / np.pi, vpara_array[-1] / speed_of_light, c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_1_1.scatter(mlat_rad_array[detrapped_point_1] * 180E0 / np.pi, vpara_array[detrapped_point_1] / speed_of_light, c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_1_1.scatter(mlat_rad_array[detrapped_point_2] * 180E0 / np.pi, vpara_array[detrapped_point_2] / speed_of_light, c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)

    ax_1_2.scatter(mlat_rad_array * 180E0 / np.pi, Ktotal_energy_array / elementary_charge, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_1_2.scatter(mlat_rad_array[0] * 180E0 / np.pi, Ktotal_energy_array[0] / elementary_charge, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_1_2.scatter(mlat_rad_array[-1] * 180E0 / np.pi, Ktotal_energy_array[-1] / elementary_charge, c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_1_2.scatter(mlat_rad_array[detrapped_point_1] * 180E0 / np.pi, Ktotal_energy_array[detrapped_point_1] / elementary_charge, c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_1_2.scatter(mlat_rad_array[detrapped_point_2] * 180E0 / np.pi, Ktotal_energy_array[detrapped_point_2] / elementary_charge, c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)

    ax_1_3.scatter(mlat_rad_array * 180E0 / np.pi, alpha_array * 180E0 / np.pi, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_1_3.scatter(mlat_rad_array[0] * 180E0 / np.pi, alpha_array[0] * 180E0 / np.pi, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_1_3.scatter(mlat_rad_array[-1] * 180E0 / np.pi, alpha_array[-1] * 180E0 / np.pi, c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_1_3.scatter(mlat_rad_array[detrapped_point_1] * 180E0 / np.pi, alpha_array[detrapped_point_1] * 180E0 / np.pi, c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_1_3.scatter(mlat_rad_array[detrapped_point_2] * 180E0 / np.pi, alpha_array[detrapped_point_2] * 180E0 / np.pi, c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)

    ax_2_1.scatter(mlat_rad_array * 180E0 / np.pi, S_value_array, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_2_1.scatter(mlat_rad_array[0] * 180E0 / np.pi, S_value_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_2_1.scatter(mlat_rad_array[-1] * 180E0 / np.pi, S_value_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_2_1.scatter(mlat_rad_array[detrapped_point_1] * 180E0 / np.pi, S_value_array[detrapped_point_1], c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_2_1.scatter(mlat_rad_array[detrapped_point_2] * 180E0 / np.pi, S_value_array[detrapped_point_2], c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_2_1.axhline(y=1E0, color='k', linewidth=4E0, zorder=0, alpha=0.3, linestyle='--')

    ax_2_2.scatter(psi_array / np.pi, theta_array / trapping_frequency_array / 2E0, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_2_2.scatter(psi_array[0] / np.pi, theta_array[0] / trapping_frequency_array[0] / 2E0, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_2_2.scatter(psi_array[-1] / np.pi, theta_array[-1] / trapping_frequency_array[-1] / 2E0, c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_2_2.scatter(psi_array[detrapped_point_1] / np.pi, theta_array[detrapped_point_1] / trapping_frequency_array[detrapped_point_1] / 2E0, c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_2_2.scatter(psi_array[detrapped_point_2] / np.pi, theta_array[detrapped_point_2] / trapping_frequency_array[detrapped_point_2] / 2E0, c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_2_2.axhline(y=0E0, color='k', linewidth=4E0, zorder=0, alpha=0.3, linestyle='--')

    ax_2_3.scatter(psi_array / np.pi, Ktotal_energy_array / elementary_charge, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_2_3.scatter(psi_array[0] / np.pi, Ktotal_energy_array[0] / elementary_charge, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_2_3.scatter(psi_array[-1] / np.pi, Ktotal_energy_array[-1] / elementary_charge, c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_2_3.scatter(psi_array[detrapped_point_1] / np.pi, Ktotal_energy_array[detrapped_point_1] / elementary_charge, c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_2_3.scatter(psi_array[detrapped_point_2] / np.pi, Ktotal_energy_array[detrapped_point_2] / elementary_charge, c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)


    axes = [ax_1_1, ax_1_2, ax_1_3, ax_2_1, ax_2_2, ax_2_3]

    mlat_deg_for_background = np.linspace(0E0, mlat_upper_limit_deg, 1000)
    mlat_rad_for_background = mlat_deg_for_background * np.pi / 180E0
    energy_wave_phase_speed_for_background = energy_wave_phase_speed(mlat_rad_for_background)
    wave_phase_speed_for_background = wave_phase_speed(mlat_rad_for_background)
    energy_wave_potential_for_background = energy_wave_potential(mlat_rad_for_background)
    energy_perp_for_background = Kperp_energy(mu, mlat_rad_for_background)
    loss_cone_for_background = loss_cone(mlat_rad_for_background)

    xlim_enlarged_1_1 = ax_1_1.get_xlim()
    ylim_enlarged_1_1 = ax_1_1.get_ylim()
    xlim_enlarged_1_2 = ax_1_2.get_xlim()
    ylim_enlarged_1_2 = ax_1_2.get_ylim()
    xlim_enlarged_1_3 = ax_1_3.get_xlim()
    ylim_enlarged_1_3 = ax_1_3.get_ylim()

    ylim_enlarged_1_3 = ax_1_3.get_ylim()
    ylim_enlarged_1_3 = [np.nanmax([ylim_enlarged_1_3[0], 0E0]), np.nanmin([ylim_enlarged_1_3[1], 180E0])]

    ax_1_1.plot(mlat_deg_for_background, wave_phase_speed_for_background / speed_of_light, c='r', linewidth=4E0, zorder=0, alpha=0.6, label=r'$V_{\mathrm{ph} \parallel}$')
    ax_1_1.axhline(y=0E0, color='k', linewidth=4E0, zorder=0, alpha=0.3, linestyle='--')

    ax_1_2.plot(mlat_deg_for_background, (energy_wave_phase_speed_for_background + energy_perp_for_background) / elementary_charge, c='r', linewidth=4E0, zorder=0, alpha=0.6, label=r'$K_{\perp} + K_{\mathrm{ph \parallel}}$')
    ax_1_2.plot(mlat_deg_for_background, energy_wave_potential_for_background / elementary_charge * np.ones(len(mlat_deg_for_background)), c='g', linewidth=4E0, label=r'$K_{\mathrm{E}}$', alpha=0.6)
    ax_1_2.plot(mlat_deg_for_background, energy_perp_for_background / elementary_charge, c='orange', linewidth=4E0, label=r'$K_{\perp}$', alpha=0.6)

    ax_1_3.plot(mlat_deg_for_background, loss_cone_for_background * 180E0 / np.pi, c='k', linewidth=4E0, zorder=0, alpha=0.6, label=r'Loss cone')
    ax_1_3.plot(mlat_deg_for_background, (np.pi - loss_cone_for_background) * 180E0 / np.pi, c='k', linewidth=4E0, zorder=0, alpha=0.6)
    ax_1_3.axhline(y=90E0, color='k', linewidth=4E0, zorder=0, alpha=0.3, linestyle='--')

    ax_1_1.set_xlim(xlim_enlarged_1_1)
    ax_1_1.set_ylim(ylim_enlarged_1_1)
    ax_1_2.set_xlim(xlim_enlarged_1_2)
    ax_1_2.set_ylim(ylim_enlarged_1_2)
    ax_1_3.set_xlim(xlim_enlarged_1_3)
    ax_1_3.set_ylim(ylim_enlarged_1_3)

    ylim_enlarged_2_2 = [np.nanmin(theta_array / 2E0 / trapping_frequency_array)-0.1, np.nanmax(theta_array / 2E0 / trapping_frequency_array)+0.1]
    if ylim_enlarged_2_2[0] < -3E0:
        ylim_enlarged_2_2[0] = -3E0
    if ylim_enlarged_2_2[1] > 3E0:
        ylim_enlarged_2_2[1] = 3E0
    ax_2_2.set_ylim(ylim_enlarged_2_2)

    #ax_1_1.legend()
    #ax_1_2.legend()
    #ax_1_3.legend()

    for ax in axes:
        ax.minorticks_on()
        ax.grid(which='both', alpha=0.3)
        ax.text(-0.20, 1.0, '(' + chr(97 + axes.index(ax)) + ')', transform=ax.transAxes)
    
    fig.suptitle(r'$K_{\mathrm{i}} = %.1f$ eV, $\alpha_{\mathrm{i}} = %.1f$ deg, $\lambda_{\mathrm{i}} = %.1f$ deg, $\psi_{\mathrm{i}} = %.1f \pi$ rad' % (Ktotal_energy_array[0] / elementary_charge, alpha_array[0] * 180E0 / np.pi, mlat_rad_array[0] * 180E0 / np.pi, psi_array[0] / np.pi))

    fig.tight_layout(w_pad=0.3, h_pad=0.0)

    fig_name = fig_path(Ktotal_energy_array[0] / elementary_charge, alpha_array[0] * 180E0 / np.pi, mlat_rad_array[0] * 180E0 / np.pi, psi_array[0])
    plt.savefig(fig_name + '.png')
    plt.savefig(fig_name + '.pdf')
    plt.close()

    return

def main_data_save(args):
    mlat_rad_array, theta_array, vpara_array, psi_array, time_array, trapping_frequency_array, Ktotal_energy_array, alpha_array, S_value_array, region_array, detrapped_point_array, mu = args

    mlat_deg_for_background = np.linspace(0E0, mlat_upper_limit_deg, background_spatial_number)
    mlat_rad_for_background = mlat_deg_for_background * np.pi / 180E0
    energy_wave_phase_speed_for_background = energy_wave_phase_speed(mlat_rad_for_background)
    wave_phase_speed_for_background = wave_phase_speed(mlat_rad_for_background)
    energy_wave_potential_for_background = energy_wave_potential(mlat_rad_for_background) * np.ones(len(mlat_rad_for_background))
    energy_perp_for_background = Kperp_energy(mu, mlat_rad_for_background)
    loss_cone_for_background = loss_cone(mlat_rad_for_background)
    kpara_for_background = kpara(mlat_rad_for_background)
    magnetic_flux_density_for_background = magnetic_flux_density(mlat_rad_for_background)
    delta_1 = delta(mlat_rad_for_background)
    plasma_beta_ion_for_background = plasma_beta_ion(mlat_rad_for_background)
    Gamma_for_background = Gamma(mlat_rad_for_background)
    

    mlat_rad_array_contour, time_array_contour, psi_array_contour = wave_psi_spatial_time_variation_array(time_array[0], time_array[-1], psi_array[0], mlat_rad_array[0])
    mesh_time_array_contour, mesh_mlat_rad_array_contour = np.meshgrid(time_array_contour, mlat_rad_array_contour)

    force_electric_field_array = np.zeros_like(psi_array_contour)
    for count_time in range(time_array_contour.size):
        for count_mlat in range(mlat_rad_array_contour.size):
            force_electric_field_array[count_mlat, count_time] = force_electric_field(mesh_mlat_rad_array_contour[count_mlat, count_time], psi_array_contour[count_mlat, count_time])

    #save data as netCDF4
    data_name = fig_path(Ktotal_energy_array[0] / elementary_charge, alpha_array[0] * 180E0 / np.pi, mlat_rad_array[0] * 180E0 / np.pi, psi_array[0])
    data_name = data_name + '.nc'

    nc_dataset = nc.Dataset(data_name, 'w', format='NETCDF4')
    nc_dataset.createDimension('time', len(time_array))
    nc_dataset.createDimension('MLAT_deg', len(mlat_deg_for_background))
    nc_dataset.createDimension('time_background', len(time_array_contour))
    
    nc_time = nc_dataset.createVariable('time', np.dtype('float64').char, ('time'))
    nc_time.units = 's'
    nc_time.long_name = 'time'
    nc_time[:] = time_array

    nc_mlat_rad = nc_dataset.createVariable('mlat_rad', np.dtype('float64').char, ('time'))
    nc_mlat_rad.units = 'rad'
    nc_mlat_rad.long_name = 'magnetic latitude location of the electron [rad]'
    nc_mlat_rad[:] = mlat_rad_array

    nc_theta = nc_dataset.createVariable('theta', np.dtype('float64').char, ('time'))
    nc_theta.units = 'rad/s'
    nc_theta.long_name = 'theta'
    nc_theta[:] = theta_array

    nc_vpara = nc_dataset.createVariable('vpara', np.dtype('float64').char, ('time'))
    nc_vpara.units = 'c'
    nc_vpara.long_name = 'parallel velocity'
    nc_vpara[:] = vpara_array / speed_of_light

    nc_psi = nc_dataset.createVariable('psi', np.dtype('float64').char, ('time'))
    nc_psi.units = 'rad'
    nc_psi.long_name = 'wave phase as viewed by the electron'
    nc_psi[:] = psi_array

    nc_trapping_frequency = nc_dataset.createVariable('trapping_frequency', np.dtype('float64').char, ('time'))
    nc_trapping_frequency.units = 'rad/s'
    nc_trapping_frequency.long_name = 'trapping frequency'
    nc_trapping_frequency[:] = trapping_frequency_array

    nc_Ktotal_energy = nc_dataset.createVariable('energy', np.dtype('float64').char, ('time'))
    nc_Ktotal_energy.units = 'eV'
    nc_Ktotal_energy.long_name = 'total energy of the electron'
    nc_Ktotal_energy[:] = Ktotal_energy_array / elementary_charge

    nc_alpha = nc_dataset.createVariable('alpha', np.dtype('float64').char, ('time'))
    nc_alpha.units = 'deg'
    nc_alpha.long_name = 'pitch angle'
    nc_alpha[:] = alpha_array * 180E0 / np.pi

    nc_S_value = nc_dataset.createVariable('S_value', np.dtype('float64').char, ('time'))
    nc_S_value.units = ''
    nc_S_value.long_name = 'inhomogeneity factor'
    nc_S_value[:] = S_value_array

    nc_detrap = nc_dataset.createVariable('detrapped_point', np.dtype('int').char, ('time'))
    nc_detrap.units = ''
    nc_detrap.long_name = 'detrapped or trapped points (1: detrapped, 2: trapped, 0: others)'
    nc_detrap[:] = detrapped_point_array

    nc_mu = nc_dataset.createVariable('mu', np.dtype('float64').char)
    nc_mu.units = 'eV/nT'
    nc_mu.long_name = 'magnetic moment of the electron'
    nc_mu = mu / elementary_charge / 1E9

    nc_mlat_deg_for_background = nc_dataset.createVariable('mlat_deg_for_background', np.dtype('float64').char, ('MLAT_deg',))
    nc_mlat_deg_for_background.units = 'deg'
    nc_mlat_deg_for_background.long_name = 'background magnetic latitude location [deg]'
    nc_mlat_deg_for_background[:] = mlat_deg_for_background

    nc_energy_wave_phase_speed_for_background = nc_dataset.createVariable('Kphpara_for_background', np.dtype('float64').char, ('MLAT_deg',))
    nc_energy_wave_phase_speed_for_background.units = 'eV'
    nc_energy_wave_phase_speed_for_background.long_name = 'parallel kinetic energy of an electron at the parallel phase speed'
    nc_energy_wave_phase_speed_for_background[:] = energy_wave_phase_speed_for_background / elementary_charge

    nc_wave_phase_speed_for_background = nc_dataset.createVariable('Vphpara_for_background', np.dtype('float64').char, ('MLAT_deg',))
    nc_wave_phase_speed_for_background.units = 'c'
    nc_wave_phase_speed_for_background.long_name = 'parallel wave phase speed'
    nc_wave_phase_speed_for_background[:] = wave_phase_speed_for_background / speed_of_light

    nc_energy_wave_potential_for_background = nc_dataset.createVariable('K_E_for_background', np.dtype('float64').char, ('MLAT_deg',))
    nc_energy_wave_potential_for_background.units = 'eV'
    nc_energy_wave_potential_for_background.long_name = 'effective wave potential energy'
    nc_energy_wave_potential_for_background[:] = energy_wave_potential_for_background / elementary_charge

    nc_energy_perp_for_background = nc_dataset.createVariable('Kperp_for_background', np.dtype('float64').char, ('MLAT_deg',))
    nc_energy_perp_for_background.units = 'eV'
    nc_energy_perp_for_background.long_name = 'perpendicular kinetic energy of an electron'
    nc_energy_perp_for_background[:] = energy_perp_for_background / elementary_charge

    nc_loss_cone_for_background = nc_dataset.createVariable('loss_cone_for_background', np.dtype('float64').char, ('MLAT_deg',))
    nc_loss_cone_for_background.units = 'deg'
    nc_loss_cone_for_background.long_name = 'loss cone angle'
    nc_loss_cone_for_background[:] = loss_cone_for_background * 180E0 / np.pi

    nc_kpara_for_background = nc_dataset.createVariable('kpara_for_background', np.dtype('float64').char, ('MLAT_deg',))
    nc_kpara_for_background.units = 'rad/m'
    nc_kpara_for_background.long_name = 'parallel wave number'
    nc_kpara_for_background[:] = kpara_for_background

    nc_magnetic_flux_density_for_background = nc_dataset.createVariable('magnetic_flux_density_for_background', np.dtype('float64').char, ('MLAT_deg',))
    nc_magnetic_flux_density_for_background.units = 'nT'
    nc_magnetic_flux_density_for_background.long_name = 'magnetic flux density'
    nc_magnetic_flux_density_for_background[:] = magnetic_flux_density_for_background * 1E9

    nc_delta_1 = nc_dataset.createVariable('delta_1', np.dtype('float64').char, ('MLAT_deg',))
    nc_delta_1.units = ''
    nc_delta_1.long_name = 'first-orderr magnetic field gradient scale'
    nc_delta_1[:] = delta_1

    nc_number_density_for_background = nc_dataset.createVariable('number_density_for_background', np.dtype('float64').char)
    nc_number_density_for_background.units = 'm^-3'
    nc_number_density_for_background.long_name = 'number density of the plasma'
    nc_number_density_for_background = number_density_eq

    nc_ion_temperature_for_background = nc_dataset.createVariable('ion_temperature_for_background', np.dtype('float64').char)
    nc_ion_temperature_for_background.units = 'eV'
    nc_ion_temperature_for_background.long_name = 'ion temperature'
    nc_ion_temperature_for_background = ion_temperature_eq

    nc_tau_for_background = nc_dataset.createVariable('tau_for_background', np.dtype('float64').char)
    nc_tau_for_background.units = ''
    nc_tau_for_background.long_name = 'ion-to-electron temperature ratio'
    nc_tau_for_background = tau_eq

    nc_plasma_beta_ion_for_background = nc_dataset.createVariable('plasma_beta_ion_for_background', np.dtype('float64').char, ('MLAT_deg',))
    nc_plasma_beta_ion_for_background.units = ''
    nc_plasma_beta_ion_for_background.long_name = 'ion plasma beta'
    nc_plasma_beta_ion_for_background[:] = plasma_beta_ion_for_background

    nc_Gamma_for_background = nc_dataset.createVariable('Gamma_for_background', np.dtype('float64').char, ('MLAT_deg',))
    nc_Gamma_for_background.units = ''
    nc_Gamma_for_background.long_name = 'pitch angle coefficient'
    nc_Gamma_for_background[:] = Gamma_for_background

    nc_time_background = nc_dataset.createVariable('time_background', np.dtype('float64').char, ('time_background'))
    nc_time_background.units = 's'
    nc_time_background.long_name = 'time for the force of the parallel electric field of KAW'
    nc_time_background[:] = time_array_contour

    nc_force_electric_field = nc_dataset.createVariable('F_Epara', np.dtype('float64').char, ('MLAT_deg', 'time_background'))
    nc_force_electric_field.units = 'mV/m'
    nc_force_electric_field.long_name = 'force of the parallel electric field of KAW'
    nc_force_electric_field[:, :] = force_electric_field_array * 1E3

    nc_dataset.close()



    return


def main(args):
    initial_mlat_rad_main, initial_mu_main, initial_theta_main, initial_psi_main = args

    mlat_rad_array, theta_array, vpara_array, psi_array, time_array, trapping_frequency_array, Ktotal_energy_array, alpha_array, S_value_array, region_array, detrapped_point_array = main_calculation(args)

    main_plot([mlat_rad_array, theta_array, vpara_array, psi_array, time_array, trapping_frequency_array, Ktotal_energy_array, alpha_array, S_value_array, region_array, detrapped_point_array, initial_mu_main])

    main_data_save([mlat_rad_array, theta_array, vpara_array, psi_array, time_array, trapping_frequency_array, Ktotal_energy_array, alpha_array, S_value_array, region_array, detrapped_point_array, initial_mu_main])

    return
    
#main([initial_mlat_rad, INITIAL_MU[0, 0], INITIAL_THETA[0, 0], initial_psi])

if __name__ == '__main__':

    num_processors = os.cpu_count()
    
    args = []

    #using multiprocessing & tqdm
    for i in range(len(INITIAL_MU)):
        for j in range(len(INITIAL_MU[i])):
            args.append([initial_mlat_rad, INITIAL_MU[i, j], INITIAL_THETA[i, j], initial_psi])
    
    results = []
    with Pool(num_processors) as p:
        for result in tqdm(p.imap_unordered(main, args), total=len(args)):
            results.append(result)
    
    print('Complete')