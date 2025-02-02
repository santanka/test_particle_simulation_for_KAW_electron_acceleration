import numpy as np
import datetime
import os
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

number_parallel = os.cpu_count()
print("number_parallel = ", number_parallel)

# constants
speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]
electric_constant = 8.8541878128E-12  #[F m-1]
magnetic_constant = 1.25663706212E-6  #[N A-2]

#planet condition
planet_radius = 6.3781E6 #[m]
planet_radius_polar = 6.3568E6 #[m]
lshell_number = 9E0

#field line coordinate condition
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
ion_mass = 1.672621898E-27   # [kg] proton mass
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

###### potential ######
wave_scalar_potential_array = np.array([2E3, 2E2*np.sqrt(1E1), 2E2, 2E1*np.sqrt(1E1), 2E1])  #[V]
#wave_scalar_potential_array = np.array([2E3])
print(wave_scalar_potential_array)
###### potential ######

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

def delta_2(mlat_rad):
    delta_plus = delta_1(mlat_rad + diff_rad) * kpara(mlat_rad + diff_rad) * magnetic_flux_density(mlat_rad + diff_rad)
    delta_minus = delta_1(mlat_rad - diff_rad) * kpara(mlat_rad - diff_rad) * magnetic_flux_density(mlat_rad - diff_rad)
    return (delta_plus - delta_minus) / 2E0 / diff_rad / kpara(mlat_rad)**2E0 / magnetic_flux_density(mlat_rad) * d_mlat_d_z(mlat_rad)

def Gamma(mlat_rad):
    return 1E0 + 2E0 * plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]

def trapping_frequency(mlat_rad, wave_scalar_potential):
    return np.abs(kpara(mlat_rad)) * np.sqrt(energy_wave_potential(mlat_rad, wave_scalar_potential) / electron_mass)   #[rad/s]


# initial condition

# mu = 0

initial_S_value_min = 1E-2
initial_S_value_max = 1E0

separate_number_mesh_S = 50
separate_number_mesh_psi = 30

initial_S_value_array = np.linspace(initial_S_value_min, initial_S_value_max, separate_number_mesh_S)

print(initial_S_value_array)

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan2/test_particle_simulation_auroral_precipitation_each_potential'
os.makedirs(dir_name, exist_ok=True)


# detrapped condition for psi
def psi_max_function(S_value, psi):
    return - 5E-1 * (np.cos(psi) + np.sqrt(1E0 - S_value**2E0) - S_value * (psi + np.pi - np.arcsin(S_value)))    #[]

def psi_max_detection(args):
    S_value = args
    psi_max_old = 0E0
    psi_diff = 1E-5
    count_iteration = 0
    while True:
        diff = (psi_max_function(S_value, psi_max_old + psi_diff) - psi_max_function(S_value, psi_max_old - psi_diff)) / 2E0 / psi_diff
        update = psi_max_function(S_value, psi_max_old) / diff
        if np.abs(update) > 1E-3:
            update = np.sign(update) * 1E-3
        elif np.abs(update) != np.abs(update):
            print('update != update')
            return np.nan
        psi_max_new = psi_max_old - update
        if np.abs(psi_max_new - psi_max_old) < 1E-4:
            return psi_max_new
        else:
            psi_max_old = psi_max_new
            count_iteration += 1
            if psi_max_new > np.pi:
                print('psi_max_new > np.pi')
                psi_max_old = 2E0 * np.pi - psi_max_old
            if psi_max_old < - np.pi:
                psi_max_old = - 2E0 * np.pi - psi_max_old

psi_min_array = - np.pi + np.arcsin(initial_S_value_array)
psi_max_array = np.zeros_like(initial_S_value_array)

if __name__ == '__main__':
    args = []
    for count_i in range(separate_number_mesh_S):
        args.append(initial_S_value_array[count_i])

    results = []
    with Pool(number_parallel) as p:
        results = p.map(psi_max_detection, args)
    
    for count_i in range(separate_number_mesh_S):
        psi_max_array[count_i] = results[count_i]

# detrapped condition for capital_theta

def capital_theta_plus(S_value, psi):
    capital_theta = 5E-1 * (np.cos(psi) + np.sqrt(1E0 - S_value**2E0) - S_value * (psi + np.pi - np.arcsin(S_value)))
    if capital_theta < 0E0:
        return np.nan
    else:
        return np.sqrt(capital_theta)

capital_theta_array = np.zeros((separate_number_mesh_S, separate_number_mesh_psi, 2))
for count_i in range(separate_number_mesh_S):
    for count_j in range(separate_number_mesh_psi):
        psi_separate = psi_min_array[count_i] + (psi_max_array[count_i] - psi_min_array[count_i]) * count_j / (separate_number_mesh_psi - 1)
        capital_theta_array[count_i, count_j, 0] = capital_theta_plus(initial_S_value_array[count_i], psi_separate)
        capital_theta_array[count_i, count_j, 1] = - capital_theta_plus(initial_S_value_array[count_i], psi_separate)

# detrapped condition for mlat_rad

def S_value_function(S_value, capital_theta, mlat_rad, wave_scalar_potential):
    return S_value - energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad, wave_scalar_potential) * (1E0 + Gamma(mlat_rad)) * delta_1(mlat_rad) * (1E0 + 2E0 * trapping_frequency(mlat_rad, wave_scalar_potential) / wave_frequency * capital_theta)**2E0

def initial_mlat_rad_iteration(args):
    S_value, capital_theta, wave_scalar_potential = args
    if capital_theta != capital_theta:
        return np.nan
    mlat_rad_old = mlat_upper_limit_rad
    #Newton method
    count_iteration = 0
    while True:
        S_value_function_diff = (S_value_function(S_value, capital_theta, mlat_rad_old + diff_rad, wave_scalar_potential) - S_value_function(S_value, capital_theta, mlat_rad_old - diff_rad, wave_scalar_potential)) / 2E0 / diff_rad
        diff = S_value_function(S_value, capital_theta, mlat_rad_old, wave_scalar_potential) / S_value_function_diff
        if np.abs(diff) > 1E-1:
            diff = np.sign(diff) * 1E-1
        elif np.abs(diff) != np.abs(diff):
            print('diff != diff')
            return np.nan
        mlat_rad_new = mlat_rad_old - diff
        if np.abs(mlat_rad_new - mlat_rad_old) < 1E-6:
            return mlat_rad_new
        else:
            if mlat_rad_new < 0E0:
                mlat_rad_new = -mlat_rad_new
            if mlat_rad_new > mlat_upper_limit_rad:
                print('mlat_rad_new > mlat_upper_limit_rad')
                return np.nan
            if count_iteration > 1000:
                #print('count_iteration > 1000')
                return np.nan
            mlat_rad_old = mlat_rad_new
            count_iteration += 1

def make_initial_mlat_rad_array(wave_scalar_potential):
    initial_mlat_rad_array = np.zeros((separate_number_mesh_S, separate_number_mesh_psi, 2))
    args = []

    for count_i in range(separate_number_mesh_S):
        for count_j in range(separate_number_mesh_psi):
            for count_k in range(2):
                args.append([initial_S_value_array[count_i], capital_theta_array[count_i, count_j, count_k], wave_scalar_potential])

    results = []
    with Pool(number_parallel) as p:
        results = p.map(initial_mlat_rad_iteration, args)

    for count_i in range(separate_number_mesh_S):
        for count_j in range(separate_number_mesh_psi):
            for count_k in range(2):
                count_ijk = count_i * separate_number_mesh_psi * 2 + count_j * 2 + count_k
                initial_mlat_rad_array[count_i, count_j, count_k] = results[count_ijk]

    return initial_mlat_rad_array

def Delta_K_function(psi, capital_theta, mlat_rad, wave_scalar_potential):
    Kpara_KE = 2E0 * (capital_theta + np.sqrt(energy_wave_phase_speed(mlat_rad) / 2E0 / energy_wave_potential(mlat_rad, wave_scalar_potential)))**2E0
    return np.sin(psi) / Kpara_KE

# Delta_alpha = 0 because alpha = 0
def Delta_Gamma_function(mlat_rad):
    return (Gamma(mlat_rad) - 1E0) * (3E0 - Gamma(mlat_rad)) / (1E0 + Gamma(mlat_rad)) * delta_1(mlat_rad)

def Delta_delta_function(mlat_rad):
    return - ((Gamma(mlat_rad) - 1E0) / 2E0 * delta_1(mlat_rad) + delta_2(mlat_rad) / delta_1(mlat_rad))

def Delta_S_function(psi, capital_theta, mlat_rad, wave_scalar_potential):
    return Delta_K_function(psi, capital_theta, mlat_rad, wave_scalar_potential) + Delta_Gamma_function(mlat_rad) - Delta_delta_function(mlat_rad)

def d_f_d_t(S_value, psi, capital_theta, mlat_rad, wave_scalar_potential):
    kpara_vpara = capital_theta * 2E0 * trapping_frequency(mlat_rad, wave_scalar_potential) + wave_frequency
    return kpara_vpara * ((1E0 + Gamma(mlat_rad)) * delta_1(mlat_rad) * capital_theta**2E0 - 5E-1 * (psi + np.pi - np.arcsin(S_value)) * Delta_S_function(psi, capital_theta, mlat_rad, wave_scalar_potential) * S_value)

def detrapped_condition_check(S_value, psi, capital_theta, mlat_rad, wave_scalar_potential):
    if abs(S_value) > 1E0:
        return False, np.nan
    elif psi != psi:
        return False, np.nan
    elif capital_theta != capital_theta or capital_theta < -1E0 or capital_theta > 1E0:
        return False, np.nan
    elif mlat_rad != mlat_rad or mlat_rad < 0E0 or mlat_rad > mlat_upper_limit_rad:
        return False, np.nan
    elif d_f_d_t(S_value, psi, capital_theta, mlat_rad, wave_scalar_potential) <= 0E0:
        return False, d_f_d_t(S_value, psi, capital_theta, mlat_rad, wave_scalar_potential)
    else:
        return True, d_f_d_t(S_value, psi, capital_theta, mlat_rad, wave_scalar_potential)

# test particle simulation using 4th order Runge-Kutta method
def S_value_for_TPS(theta, mlat_rad, wave_scalar_potential):
    return energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad, wave_scalar_potential) * (1E0 + Gamma(mlat_rad)) * delta_1(mlat_rad) * (1E0 + theta / wave_frequency)**2E0

def d_psi_d_t(theta):
    return theta

def d_theta_d_t(psi, theta, mlat_rad, wave_scalar_potential):
    return - trapping_frequency(mlat_rad, wave_scalar_potential)**2E0 * (np.sin(psi) + S_value_for_TPS(theta, mlat_rad, wave_scalar_potential))

def d_mlat_rad_d_t(theta, mlat_rad):
    return (theta + wave_frequency) / kpara(mlat_rad) / r_eq / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)

def RK4_TPS(mlat_rad_0, theta_0, psi_0, wave_scalar_potential, time):
    dt = 1E-3

    # 1st step
    k1_mlat_rad = d_mlat_rad_d_t(theta_0, mlat_rad_0)
    k1_theta = d_theta_d_t(psi_0, theta_0, mlat_rad_0, wave_scalar_potential)
    k1_psi = d_psi_d_t(theta_0)

    # 2nd step
    k2_mlat_rad = d_mlat_rad_d_t(theta_0 + dt / 2E0 * k1_theta, mlat_rad_0 + dt / 2E0 * k1_mlat_rad)
    k2_theta = d_theta_d_t(psi_0 + dt / 2E0 * k1_psi, theta_0 + dt / 2E0 * k1_theta, mlat_rad_0 + dt / 2E0 * k1_mlat_rad, wave_scalar_potential)
    k2_psi = d_psi_d_t(theta_0 + dt / 2E0 * k1_theta)

    # 3rd step
    k3_mlat_rad = d_mlat_rad_d_t(theta_0 + dt / 2E0 * k2_theta, mlat_rad_0 + dt / 2E0 * k2_mlat_rad)
    k3_theta = d_theta_d_t(psi_0 + dt / 2E0 * k2_psi, theta_0 + dt / 2E0 * k2_theta, mlat_rad_0 + dt / 2E0 * k2_mlat_rad, wave_scalar_potential)
    k3_psi = d_psi_d_t(theta_0 + dt / 2E0 * k2_theta)

    # 4th step
    k4_mlat_rad = d_mlat_rad_d_t(theta_0 + dt * k3_theta, mlat_rad_0 + dt * k3_mlat_rad)
    k4_theta = d_theta_d_t(psi_0 + dt * k3_psi, theta_0 + dt * k3_theta, mlat_rad_0 + dt * k3_mlat_rad, wave_scalar_potential)
    k4_psi = d_psi_d_t(theta_0 + dt * k3_theta)

    # update
    mlat_rad_1 = mlat_rad_0 + dt / 6E0 * (k1_mlat_rad + 2E0 * k2_mlat_rad + 2E0 * k3_mlat_rad + k4_mlat_rad)
    theta_1 = theta_0 + dt / 6E0 * (k1_theta + 2E0 * k2_theta + 2E0 * k3_theta + k4_theta)
    psi_1 = psi_0 + dt / 6E0 * (k1_psi + 2E0 * k2_psi + 2E0 * k3_psi + k4_psi)

    return mlat_rad_1, theta_1, psi_1, time + dt

def reach_ionosphere_check(args):
    initial_mlat_rad, initial_theta, initial_psi, wave_scalar_potential, detrap_flag = args
    if detrap_flag == False:
        return False, np.nan, np.nan
    
    time = 0E0
    while True:
        mlat_rad_new, theta_new, psi_new, time_new = RK4_TPS(initial_mlat_rad, initial_theta, initial_psi, wave_scalar_potential, time)

        if mlat_rad_new / np.pi * 180E0 < 0E0:
            return False, np.nan, np.nan
        
        elif mlat_rad_new != mlat_rad_new or theta_new != theta_new or psi_new != psi_new:
            print('mlat_rad_new != mlat_rad_new or theta_new != theta_new or psi_new != psi_new')
            quit()

        elif mlat_rad_new / np.pi * 180E0 >= mlat_upper_limit_deg:
            energy_ionosphere = 5E-1 * electron_mass * ((theta_new + wave_frequency) / kpara(mlat_rad_new))**2E0 / elementary_charge   #[eV]
            return True, time_new, energy_ionosphere
        
        else:
            initial_mlat_rad = mlat_rad_new
            initial_theta = theta_new
            initial_psi = psi_new
            time = time_new


# output csv file
def output_data_path(wave_scalar_potential):
    return f'{dir_name}/wave_scalar_potential_{wave_scalar_potential:.1e}.csv'

# main

data_header = 'initial_S, initial_psi [pi rad], initial_vpara [/c], initial_capital_theta, mlat_deg [deg],energy [eV], energy_ionosphere [eV], reach time [s]'

def main_each_potential(wave_scalar_potential):
    initial_mlat_rad_array = make_initial_mlat_rad_array(wave_scalar_potential)

    S_initial_array = np.array([])
    psi_initial_array = np.array([])
    vpara_initial_array = np.array([])
    capital_theta_initial_array = np.array([])
    mlat_deg_initial_array = np.array([])
    energy_initial_array = np.array([])
    energy_ionosphere_array = np.array([])
    reach_time_array = np.array([])

    args = []
    for count_i in range(separate_number_mesh_S):
        S_count_i = initial_S_value_array[count_i]
        for count_j in range(separate_number_mesh_psi):
            psi_count_j = psi_min_array[count_i] + (psi_max_array[count_i] - psi_min_array[count_i]) * count_j / (separate_number_mesh_psi - 1)
            for count_k in range(2):
                capital_theta_count_k = capital_theta_array[count_i, count_j, count_k]
                mlat_rad_count_k = initial_mlat_rad_array[count_i, count_j, count_k]
                initial_theta = capital_theta_count_k * 2E0 * trapping_frequency(mlat_rad_count_k, wave_scalar_potential)
                detrap_flag, d_f_d_t_value = detrapped_condition_check(S_count_i, psi_count_j, capital_theta_count_k, mlat_rad_count_k, wave_scalar_potential)
                args.append([mlat_rad_count_k, initial_theta, psi_count_j, wave_scalar_potential, detrap_flag])
                #print(mlat_rad_count_k / np.pi * 180E0, initial_S_value_array[count_i], psi_count_j, capital_theta_count_k, detrap_flag, d_f_d_t_value)

    results = []
    with Pool(number_parallel) as p:
        results = p.map(reach_ionosphere_check, args)
    
    for count_i in range(separate_number_mesh_S):
        S_count_i = initial_S_value_array[count_i]
        for count_j in range(separate_number_mesh_psi):
            psi_count_j = psi_min_array[count_i] + (psi_max_array[count_i] - psi_min_array[count_i]) * count_j / (separate_number_mesh_psi - 1)
            for count_k in range(2):
                capital_theta_count_k = capital_theta_array[count_i, count_j, count_k]
                mlat_rad_count_k = initial_mlat_rad_array[count_i, count_j, count_k]
                theta_count_k = capital_theta_count_k * 2E0 * trapping_frequency(mlat_rad_count_k, wave_scalar_potential)
                vpara_initial = (theta_count_k + wave_frequency) / kpara(mlat_rad_count_k) / speed_of_light
                energy_initial = 5E-1 * electron_mass * (vpara_initial * speed_of_light)**2E0 / elementary_charge
                
                count_ijk = count_i * separate_number_mesh_psi * 2 + count_j * 2 + count_k
                if results[count_ijk][0]:
                    S_initial_array = np.append(S_initial_array, S_count_i)
                    psi_initial_array = np.append(psi_initial_array, psi_count_j / np.pi)
                    vpara_initial_array = np.append(vpara_initial_array, vpara_initial)
                    capital_theta_initial_array = np.append(capital_theta_initial_array, capital_theta_count_k)
                    mlat_deg_initial_array = np.append(mlat_deg_initial_array, mlat_rad_count_k / np.pi * 180E0)
                    energy_initial_array = np.append(energy_initial_array, energy_initial)
                    energy_ionosphere_array = np.append(energy_ionosphere_array, results[count_ijk][2])
                    reach_time_array = np.append(reach_time_array, results[count_ijk][1])
    
    np.savetxt(output_data_path(wave_scalar_potential), np.array([S_initial_array, psi_initial_array, vpara_initial_array, capital_theta_initial_array, mlat_deg_initial_array, energy_initial_array, energy_ionosphere_array, reach_time_array]).T, delimiter=',', header=data_header, comments='')

    return

if __name__ == '__main__':
    for wave_scalar_potential in wave_scalar_potential_array:
        main_each_potential(wave_scalar_potential)

print('finish')