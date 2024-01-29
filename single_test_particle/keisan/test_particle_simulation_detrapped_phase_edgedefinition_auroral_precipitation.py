import numpy as np
import datetime
import os
from multiprocessing import Pool

number_parallel = os.cpu_count()

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
    return Alfven_speed(mlat_rad) * kperp_rhoi * np.sqrt((1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))) * np.sign(mlat_rad)    #[m/s]

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
    grad_magnetic_flux_density = (magnetic_flux_density(mlat_rad + diff_rad) - magnetic_flux_density(mlat_rad - diff_rad)) / 2E0 / diff_rad * d_mlat_d_z(mlat_rad)    #[T/m]
    return 1E0 / kpara(mlat_rad) / magnetic_flux_density(mlat_rad) * grad_magnetic_flux_density    #[rad^-1]

def delta_2(mlat_rad):
    delta_plus = delta(mlat_rad + diff_rad) * kpara(mlat_rad + diff_rad) * magnetic_flux_density(mlat_rad + diff_rad)    #[rad]
    delta_minus = delta(mlat_rad - diff_rad) * kpara(mlat_rad - diff_rad) * magnetic_flux_density(mlat_rad - diff_rad)    #[rad]
    return (delta_plus - delta_minus) / 2E0 / diff_rad / kpara(mlat_rad)**2E0 / magnetic_flux_density(mlat_rad)    #[rad^-1]

def Gamma(mlat_rad):
    return 1E0 + 2E0 * plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]

def trapping_frequency(mlat_rad):
    return np.abs(kpara(mlat_rad)) * np.sqrt(energy_wave_potential(mlat_rad) / electron_mass)   #[rad/s]


# initial condition

initial_Kperp_eq_min_eV = 1E0    #[eV]
initial_Kperp_eq_max_eV = 1.1E1    #[eV]

initial_S_value_min = 1E-2
initial_S_value_max = 1E0

separate_number_mesh_Kperp = 11
separate_number_mesh_S = 50
separate_number_psi = 30

grid_scale = 'linear' # 'linear' or 'log'

if grid_scale == 'log':
    initial_Kperp_eq_array = np.logspace(np.log10(initial_Kperp_eq_min_eV), np.log10(initial_Kperp_eq_max_eV), separate_number_mesh_Kperp)
    initial_S_value_array = np.logspace(np.log10(initial_S_value_min), np.log10(initial_S_value_max), separate_number_mesh_S)
    initial_Kperp_eq_mesh, initial_S_value_mesh = np.meshgrid(initial_Kperp_eq_array, initial_S_value_array)
elif grid_scale == 'linear':
    initial_Kperp_eq_array = np.linspace(initial_Kperp_eq_min_eV, initial_Kperp_eq_max_eV, separate_number_mesh_Kperp)
    initial_S_value_array = np.linspace(initial_S_value_min, initial_S_value_max, separate_number_mesh_S)
    initial_Kperp_eq_mesh, initial_S_value_mesh = np.meshgrid(initial_Kperp_eq_array, initial_S_value_array)
else:
    print('Error: grid_scale')
    quit()

initial_Kperp_eq_array = initial_Kperp_eq_mesh.flatten()
initial_S_value_array = initial_S_value_mesh.flatten()
Kperp_S_value_grid_number = initial_Kperp_eq_array.size

initial_mu_array = elementary_charge * initial_Kperp_eq_array / magnetic_flux_density(0E0)  #[J/T]

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/test_particle_simulation_detrapped_phase_edgedefinition/auroral_precipitation'
os.makedirs(dir_name, exist_ok=True)

output_data_path = f'{dir_name}/Kperp_eq_{initial_Kperp_eq_min_eV:.4f}_{initial_Kperp_eq_max_eV:.4f}_eV_{separate_number_mesh_Kperp}_S_{initial_S_value_min:.4f}_{initial_S_value_max:.4f}_{separate_number_mesh_S}_{separate_number_psi}_{grid_scale}.csv'

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
psi_max = np.zeros_like(initial_S_value_array)

if __name__ == '__main__':
    args = []
    for count_i in range(Kperp_S_value_grid_number):
        args.append([initial_S_value_array[count_i]])

    results = []
    with Pool(number_parallel) as p:
        results = p.starmap(psi_max_detection, args)
    
    for count_i in range(Kperp_S_value_grid_number):
        psi_max[count_i] = results[count_i]

initial_psi_array = np.zeros((Kperp_S_value_grid_number, separate_number_psi))
for count_i in range(Kperp_S_value_grid_number):
    initial_psi_array[count_i, :] = np.linspace(psi_min_array[count_i], psi_max[count_i], separate_number_psi)

def capital_theta_function(S_value, psi, sign):
    base = 5E-1 * (np.cos(psi) + np.sqrt(1E0 - S_value**2E0) - S_value * (psi + np.pi - np.arcsin(S_value)))    #[]
    if base < 0E0:
        return np.nan
    else:
        return sign * np.sqrt(base)

capital_theta_array = np.zeros((Kperp_S_value_grid_number, separate_number_psi, 2))
for count_i in range(Kperp_S_value_grid_number):
    nan_flag = separate_number_psi
    for count_j in range(separate_number_psi):
        base = capital_theta_function(initial_S_value_array[count_i], initial_psi_array[count_i, count_j], 1E0)
        if base != base:
            nan_flag = nan_flag - 1
            capital_theta_array[count_i, count_j, 0] = np.nan
            capital_theta_array[count_i, count_j, 1] = np.nan
        else:
            capital_theta_array[count_i, count_j, 0] = base
            capital_theta_array[count_i, count_j, 1] = capital_theta_function(initial_S_value_array[count_i], initial_psi_array[count_i, count_j], -1E0)
    if nan_flag == 0:
        initial_Kperp_eq_array[count_i] = np.nan
        initial_S_value_array[count_i] = np.nan
        initial_mu_array[count_i] = np.nan

def S_value_function(mu, S_value, capital_theta, mlat_rad):
    return S_value - energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad) * (1E0 + Gamma(mlat_rad)) * delta(mlat_rad) * (1E0 + np.sqrt(2E0 * energy_wave_potential(mlat_rad) / energy_wave_phase_speed(mlat_rad)) * capital_theta)**2E0 - magnetic_flux_density(mlat_rad) * mu / energy_wave_potential(mlat_rad) * delta(mlat_rad)    #[]

def initial_mlat_rad_iteration(mu, S_value, capital_theta):
    if capital_theta != capital_theta:
        return np.nan
    mlat_rad_old = mlat_upper_limit_rad
    #Newton method
    count_iteration = 0
    while True:
        S_value_function_diff = (S_value_function(mu, S_value, capital_theta, mlat_rad_old + diff_rad) - S_value_function(mu, S_value, capital_theta, mlat_rad_old - diff_rad)) / 2E0 / diff_rad
        diff = S_value_function(mu, S_value, capital_theta, mlat_rad_old) / S_value_function_diff
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

initial_mlat_rad_array = np.zeros_like(capital_theta_array)
for count in range(2):
    if __name__ == '__main__':
        args = []
        for count_i in range(Kperp_S_value_grid_number):
            for count_j in range(separate_number_psi):
                args.append((initial_mu_array[count_i], initial_S_value_array[count_i], capital_theta_array[count_i, count_j, count]))

        results = []
        with Pool(number_parallel) as p:
            results = p.starmap(initial_mlat_rad_iteration, args)

        for count_i in range(Kperp_S_value_grid_number):
            for count_j in range(separate_number_psi):
                initial_mlat_rad_array[count_i, count_j, count] = results[count_i * separate_number_psi + count_j]

initial_mlat_deg_array = initial_mlat_rad_array * 180E0 / np.pi

def dS_dt_W_value(S_value, theta, mlat_rad):
    return (1E0 + Gamma(mlat_rad)) * (((1E0 + Gamma(mlat_rad)) / 2E0 - delta_2(mlat_rad) / delta(mlat_rad)**2E0) * S_value - energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad) * (Gamma(mlat_rad)**2E0 - 3E0 * Gamma(mlat_rad) + 3E0) * delta(mlat_rad) * (1E0 + theta / wave_frequency)**2E0)    #[]

def theta_check(S_value, capital_theta, psi, mlat_rad):
    if mlat_rad != mlat_rad:
        return np.nan
    theta = capital_theta * 2E0 * trapping_frequency(mlat_rad) #[rad/s]
    check_function = (np.cos(psi) + np.sqrt(1E0 - S_value**2E0) + (psi + np.pi - np.arcsin(S_value)) * (np.sin(psi) + dS_dt_W_value(S_value, capital_theta, mlat_rad) - S_value)) * (theta + wave_frequency) * (1E0 + Gamma(mlat_rad)) * delta(mlat_rad)
    if check_function <= 0E0:
        return theta
    else:
        return np.nan

initial_theta_array = np.zeros_like(capital_theta_array)
for count_i in range(Kperp_S_value_grid_number):
    nan_flag = separate_number_psi*2
    for count_j in range(separate_number_psi):
        nan_flag_j = 2
        initial_theta_array[count_i, count_j, 0] = theta_check(initial_S_value_array[count_i], capital_theta_array[count_i, count_j, 0], initial_psi_array[count_i, count_j], initial_mlat_rad_array[count_i, count_j, 0])
        if initial_theta_array[count_i, count_j, 0] != initial_theta_array[count_i, count_j, 0]:
            nan_flag = nan_flag - 1
            nan_flag_j = nan_flag_j - 1
            initial_mlat_rad_array[count_i, count_j, 0] = np.nan
        initial_theta_array[count_i, count_j, 1] = theta_check(initial_S_value_array[count_i], capital_theta_array[count_i, count_j, 1], initial_psi_array[count_i, count_j], initial_mlat_rad_array[count_i, count_j, 1])
        if initial_theta_array[count_i, count_j, 1] != initial_theta_array[count_i, count_j, 1]:
            nan_flag = nan_flag - 1
            nan_flag_j = nan_flag_j - 1
            initial_mlat_rad_array[count_i, count_j, 1] = np.nan
        if nan_flag_j == 0:
            initial_psi_array[count_i, count_j] = np.nan
    if nan_flag == 0:
        initial_Kperp_eq_array[count_i] = np.nan
        initial_S_value_array[count_i] = np.nan
        initial_mu_array[count_i] = np.nan



# runge-kutta method
def S_value_for_TPS(mu, theta, mlat_rad):
    return energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad) * (1E0 + Gamma(mlat_rad)) * delta(mlat_rad) * (1E0 + theta / wave_frequency)**2E0 + magnetic_flux_density(mlat_rad) * mu / energy_wave_potential(mlat_rad) * delta(mlat_rad)    #[]

def d_psi_d_t(theta):
    return theta    #[rad/s]

def d_theta_d_t(mu, theta, psi, mlat_rad):
    return - trapping_frequency(mlat_rad)**2E0 * (np.sin(psi) + S_value_for_TPS(mu, theta, mlat_rad))    #[rad/s^2]

def d_mlat_rad_d_t(theta, mlat_rad):
    return (theta + wave_frequency) / kpara(mlat_rad) / r_eq / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)    #[rad/s]

dt = 1E-3    #[s]
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


def main(args):
    initial_Kperp_eq_main = args[0]
    initial_S_value_main = args[1]
    mu_main = args[2]
    initial_psi_main = args[3]
    initial_mlat_rad_main = args[4]
    initial_theta_main = args[5]

    if initial_Kperp_eq_main != initial_Kperp_eq_main or initial_S_value_main != initial_S_value_main or initial_mlat_rad_main != initial_mlat_rad_main or initial_theta_main != initial_theta_main:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mlat_old = initial_mlat_rad_main
    theta_old = initial_theta_main
    psi_old = initial_psi_main
    time_old = 0E0

    #now = datetime.datetime.now()
    #print(now, r'calculation start', initial_Kperp_eq_main, initial_S_value_main, S_value_for_TPS(mu_main, theta_old, mlat_old), initial_psi_main, initial_mlat_rad_main, initial_theta_main)

    count_iteration = 0

    while True:
        count_iteration += 1
        mlat_new, theta_new, psi_new = RK4(mlat_old, theta_old, psi_old, mu_main)

        if mlat_new / np.pi * 180E0 < 1E0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        elif mlat_new >= mlat_upper_limit_rad:
            break

        else:
        
            S_new = S_value_for_TPS(mu_main, theta_new, mlat_new)
            time_new = time_old + dt

            if S_new < 1E0:
                # psi_new_modは-piからpiの範囲に収まるようにする
                psi_new_mod = np.mod(psi_new + np.pi, 2E0 * np.pi) - np.pi
                phase_trapping_edge_trajectory = (theta_new / trapping_frequency(mlat_new) / 2E0)**2E0 - 5E-1 * (np.cos(psi_new_mod) + np.sqrt(1E0 - S_new**2E0) - S_new * (psi_new_mod + np.pi - np.arcsin(S_new)))    #[]
                if phase_trapping_edge_trajectory < 0E0:
                    return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
            if psi_new != psi_new:
                print('Error: NaN')
                quit()
            
            mlat_old = mlat_new
            theta_old = theta_new
            psi_old = psi_new
            time_old = time_new
    
    energy_perp_new_eV = mu_main * magnetic_flux_density(mlat_new) / elementary_charge #[eV]
    energy_para_new_eV = 5E-1 * electron_mass * ((theta_new + wave_frequency) / kpara(mlat_new))**2E0 / elementary_charge #[eV]
    energy_new_eV = energy_perp_new_eV + energy_para_new_eV #[eV]

    energy_perp_ionospheric_end_eV = mu_main * magnetic_flux_density(mlat_upper_limit_rad) / elementary_charge #[eV]
    energy_para_ionospheric_end_eV = energy_new_eV - energy_perp_ionospheric_end_eV #[eV]

    initial_capital_theta = initial_theta_main / 2E0 / trapping_frequency(initial_mlat_rad_main)
    initial_energy_perp_eV = mu_main * magnetic_flux_density(initial_mlat_rad_main) / elementary_charge #[eV]
    initial_energy_para_eV = 5E-1 * electron_mass * ((initial_theta_main + wave_frequency) / kpara(initial_mlat_rad_main))**2E0 / elementary_charge #[eV]
    initial_energy_eV = initial_energy_perp_eV + initial_energy_para_eV #[eV]

    return initial_Kperp_eq_main, initial_S_value_main, initial_psi_main, initial_capital_theta, initial_mlat_rad_main, initial_energy_perp_eV, initial_energy_para_eV, initial_energy_eV, energy_perp_ionospheric_end_eV, energy_para_ionospheric_end_eV, energy_new_eV

output_data_array = np.array([])

for count_i in range(Kperp_S_value_grid_number):
    initial_Kperp_eq_main = initial_Kperp_eq_array[count_i]
    if initial_Kperp_eq_main != initial_Kperp_eq_main:
        continue

    initial_S_value_main = initial_S_value_array[count_i]
    initial_mu_main = initial_mu_array[count_i]
    initial_psi_main = initial_psi_array[count_i, :]
    initial_mlat_rad_main = initial_mlat_rad_array[count_i, :, :]
    initial_theta_main = initial_theta_array[count_i, :, :]

    now = datetime.datetime.now()
    print(now, r'calculation start', initial_Kperp_eq_main, initial_S_value_main)

    for count_j in range(2):
        if __name__ == '__main__':
            args = []
            for count_k in range(separate_number_psi):
                args.append((initial_Kperp_eq_main, initial_S_value_main, initial_mu_main, initial_psi_main[count_k], initial_mlat_rad_main[count_k, count_j], initial_theta_main[count_k, count_j]))
            
            results = []
            with Pool(number_parallel) as p:
                results = p.map(main, args)
            
            for result in results:
                if result[0] == result[0]:
                    result_array = np.array(result)
                    if output_data_array.size == 0:
                        output_data_array = result_array
                    else:
                        output_data_array = np.vstack((output_data_array, result_array))
    

np.savetxt(output_data_path, output_data_array, delimiter=',', header='Kperp_eq [eV], S_initial, psi_initial [rad], capital_theta_initial, mlat_rad_initial [rad], K_perp_initial [eV], K_para_initial [eV], K_initial [eV], K_perp_ionospheric_end [eV], K_para_ionospheric_end [eV], K_ionospheric_end [eV]')
print(output_data_path)