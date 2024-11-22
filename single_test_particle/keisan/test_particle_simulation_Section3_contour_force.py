import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import datetime
import os
from multiprocessing import Pool
from tqdm.auto import tqdm  # Ensure to import tqdm correctly

# Font setting
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
font_size = 55
plt.rcParams["font.size"] = font_size

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


def wave_psi_spatial_time_variation_array(time_start, time_end, initial_psi, initial_mlat_rad):
    time_grid_number = 1000
    spatial_grid_number = 1000
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

initial_pitch_angle_rad = initial_pitch_angle_deg * np.pi / 180E0 #[rad]
initial_mlat_rad = initial_mlat_deg * np.pi / 180E0 #[rad]

INI_K_EV, INI_PITCH_ANGLE_RAD = np.meshgrid(initial_K_eV, initial_pitch_angle_rad)
INI_PITCH_ANGLE_DEG = INI_PITCH_ANGLE_RAD * 180E0 / np.pi

INITIAL_KPERP_EV = INI_K_EV * np.sin(INI_PITCH_ANGLE_RAD)**2E0 #[eV]
INITIAL_KPARA_EV = INI_K_EV * np.cos(INI_PITCH_ANGLE_RAD)**2E0 #[eV]

INITIAL_MU = INITIAL_KPERP_EV * elementary_charge / magnetic_flux_density(initial_mlat_rad) #[J/T]
INITIAL_THETA = kpara(initial_mlat_rad) * np.sqrt(2E0 * INITIAL_KPARA_EV * elementary_charge / electron_mass) - wave_frequency  #[rad/s]
initial_psi = -9E-1 * np.pi #[rad]

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

def dir_path(initial_psi):
    dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/test_particle_simulation_Section3_contour_force/{initial_psi/np.pi:.2f}_pi'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return

dir_path(initial_psi)


# fig path
def fig_path(initial_K_eV, initial_pitch_angle_deg, initial_mlat_deg, initial_psi):
    dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/test_particle_simulation_Section3_contour_force/{initial_psi/np.pi:.2f}_pi'
    fig_name = f'{dir_name}/{initial_K_eV:.2f}eV_{initial_pitch_angle_deg:.2f}deg_{initial_mlat_deg:.2f}deg'
    return fig_name

def main_plot(args):
    mlat_rad_array, theta_array, vpara_array, psi_array, time_array, trapping_frequency_array, Ktotal_energy_array, alpha_array, S_value_array, region_array, detrapped_point_array, mu = args
    
    mlat_rad_array_contour, time_array_contour, psi_array_contour = wave_psi_spatial_time_variation_array(time_array[0], time_array[-1], psi_array[0], mlat_rad_array[0])
    mesh_time_array_contour, mesh_mlat_rad_array_contour = np.meshgrid(time_array_contour, mlat_rad_array_contour)

    force_electric_field_array = np.zeros_like(psi_array_contour)
    for count_time in range(time_array_contour.size):
        for count_mlat in range(mlat_rad_array_contour.size):
            force_electric_field_array[count_mlat, count_time] = force_electric_field(mesh_mlat_rad_array_contour[count_mlat, count_time], psi_array_contour[count_mlat, count_time])

    detrapped_point_1 = np.where(detrapped_point_array == 1)[0]
    detrapped_point_2 = np.where(detrapped_point_array == 2)[0]

    fig = plt.figure(figsize=(20, 20), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])

    cmap_color = cm.bwr
    color_target = force_electric_field_array * 1E3
    vmax_color = np.max(np.abs(color_target))
    vmin_color = - vmax_color
    norm_color = mpl.colors.Normalize(vmin=vmin_color, vmax=vmax_color)
    scalarMap_color = plt.cm.ScalarMappable(norm=norm_color, cmap=cmap_color)
    scalarMap_color.set_array([])

    ax_cbar = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(scalarMap_color, cax=ax_cbar, orientation='vertical')
    cbar.set_label(r'Force of $\delta E_{\parallel}$ [eV/km]')

    ax = fig.add_subplot(gs[0, 0], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'Time $t$ [s]')
    ax.contourf(mesh_mlat_rad_array_contour * 180E0 / np.pi, mesh_time_array_contour, force_electric_field_array*1E3, cmap=cmap_color, norm=norm_color, levels=1000)
    ax.scatter(mlat_rad_array * 180E0 / np.pi, time_array, c='k', s=1, zorder=1)
    ax.scatter(mlat_rad_array[0] * 180E0 / np.pi, time_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax.scatter(mlat_rad_array[-1] * 180E0 / np.pi, time_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax.scatter(mlat_rad_array[detrapped_point_1] * 180E0 / np.pi, time_array[detrapped_point_1], c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax.scatter(mlat_rad_array[detrapped_point_2] * 180E0 / np.pi, time_array[detrapped_point_2], c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)

    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    xlim_max = np.max(mlat_rad_array * 180E0 / np.pi)
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(time_array[0], time_array[-1])

    fig.suptitle(r'$K_{\mathrm{i}} = %.1f$ eV, $\alpha_{\mathrm{i}} = %.1f$ deg, $\lambda_{\mathrm{i}} = %.1f$ deg, $\psi_{\mathrm{i}} = %.1f \pi$ rad' % (Ktotal_energy_array[0] / elementary_charge, alpha_array[0] * 180E0 / np.pi, mlat_rad_array[0] * 180E0 / np.pi, psi_array[0] / np.pi), fontsize=font_size*0.9)

    fig.tight_layout()

    fig_name = fig_path(Ktotal_energy_array[0] / elementary_charge, alpha_array[0] * 180E0 / np.pi, mlat_rad_array[0] * 180E0 / np.pi, psi_array[0])
    plt.savefig(fig_name + '.png')
    plt.savefig(fig_name + '.pdf')
    plt.close()

    return


def main(args):
    initial_mlat_rad_main, initial_mu_main, initial_theta_main, initial_psi_main = args

    mlat_rad_array, theta_array, vpara_array, psi_array, time_array, trapping_frequency_array, Ktotal_energy_array, alpha_array, S_value_array, region_array, detrapped_point_array = main_calculation(args)

    main_plot([mlat_rad_array, theta_array, vpara_array, psi_array, time_array, trapping_frequency_array, Ktotal_energy_array, alpha_array, S_value_array, region_array, detrapped_point_array, initial_mu_main])

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