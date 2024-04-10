import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import datetime
import os
from multiprocessing import Pool

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 40


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

def Gamma(mlat_rad):
    return 1E0 + 2E0 * plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]


# S_critical
def S_critical_function(S_value):
    return np.sqrt(2E0 * np.pi * S_value) - np.sqrt(np.sqrt(1E0 - S_value**2E0) + S_value * (np.pi + np.arcsin(S_value)) + 1E0)    #[]

def S_critical_obtain():
    S_critical_old = 0.7246E0
    diff_S_critical = 1E-8
    while True:
        func_1 = S_critical_function(S_critical_old)
        func_2 = (S_critical_function(S_critical_old + diff_S_critical) - S_critical_function(S_critical_old - diff_S_critical)) / 2E0 / diff_S_critical
        diff = func_1 / func_2
        if abs(diff) > 1E-3:
            diff = np.sign(diff) * 1E-3
        S_critical_new = S_critical_old - diff
        if abs(S_critical_new - S_critical_old) < 1E-8:
            break
        S_critical_old = S_critical_new
    return S_critical_new

S_critical = S_critical_obtain()
print(S_critical)


# iteration

def h_function(S_value, mlat_rad, sign_theta):
    energy_ratio = energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)
    if 0 <= S_value and S_value <= S_critical:
        return S_value - energy_ratio * (1E0 + sign_theta * np.sqrt(1E0 / energy_ratio) * np.sqrt(np.sqrt(1E0 - S_value**2E0) + S_value * (np.pi + np.arcsin(S_value)) + 1E0))**2E0 * (1E0 + Gamma(mlat_rad)) * delta(mlat_rad)
    elif S_critical < S_value:
        return S_value - energy_ratio * (1E0 + sign_theta * np.sqrt(2E0 * np.pi / energy_ratio) * np.sqrt(S_value))**2E0 * (1E0 + Gamma(mlat_rad)) * delta(mlat_rad)
    else:
        return np.nan

def vpara_r(S_value, mlat_rad, sign_theta):
    if 0 <= S_value and S_value <= S_critical:
        return wave_phase_speed(mlat_rad) + sign_theta * np.sqrt(2E0 * energy_wave_potential(mlat_rad) / electron_mass) * np.sqrt(np.sqrt(1E0 - S_value**2E0) + S_value * (np.pi + np.arcsin(S_value)) + 1E0)
    elif S_critical < S_value:
        return wave_phase_speed(mlat_rad) + sign_theta * np.sqrt(np.pi * S_value * energy_wave_potential(mlat_rad) / electron_mass)
    else:
        return np.nan

def iteration(mlat_rad, sign_theta, sign_S):
    energy_ratio = energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)
    if sign_S == 1:
        S_value_old = 1E1
        #S_value_old = energy_ratio * np.pi * (1E0 + Gamma(mlat_rad)) * delta(mlat_rad) / (1E0 - 2E0 * np.pi * (1E0 + Gamma(mlat_rad)) * delta(mlat_rad)) + 1E-6
    elif sign_S == -1:
        S_value_old = 1E-6
    else:
        return np.nan, np.nan
    
    diff_S_value = 1E-8
    count_iteration = 0
    while True:
        func_1 = h_function(S_value_old, mlat_rad, sign_theta)
        func_2 = (h_function(S_value_old + diff_S_value, mlat_rad, sign_theta) - h_function(S_value_old - diff_S_value, mlat_rad, sign_theta)) / 2E0 / diff_S_value
        diff = func_1 / func_2
        S_value_new = S_value_old - diff
        if abs(S_value_new - S_value_old) < 1E-8:
            break
        S_value_old = S_value_new
        count_iteration += 1
        if count_iteration > 10000:
            return np.nan, np.nan
        if S_value_new < 0:
            return np.nan, np.nan
        if S_value_new != S_value_new:
            return np.nan, np.nan
    
    vpara_r_value = vpara_r(S_value_new, mlat_rad, sign_theta)

    return S_value_new, vpara_r_value

mlat_deg_array = np.linspace(0.01E0, 50E0, 100)
mlat_rad_array = mlat_deg_array * np.pi / 180E0

def main(args):
    mlat_rad = args[0]
    sign_theta = args[1]
    sign_S = args[2]

    S_value, vpara_r_value = iteration(mlat_rad, sign_theta, sign_S)

    return S_value, vpara_r_value


S_value_plus_max_array = np.zeros(mlat_rad_array.shape)
S_value_plus_min_array = np.zeros(mlat_rad_array.shape)
S_value_minus_max_array = np.zeros(mlat_rad_array.shape)
S_value_minus_min_array = np.zeros(mlat_rad_array.shape)

vpara_r_plus_max_array = np.zeros(mlat_rad_array.shape)
vpara_r_plus_min_array = np.zeros(mlat_rad_array.shape)
vpara_r_minus_max_array = np.zeros(mlat_rad_array.shape)
vpara_r_minus_min_array = np.zeros(mlat_rad_array.shape)

if __name__ == '__main__':

    num_process = os.cpu_count()

    args = []
    for mlat_rad in mlat_rad_array:
        args.append([mlat_rad, 1, 1])
        args.append([mlat_rad, 1, -1])
        args.append([mlat_rad, -1, 1])
        args.append([mlat_rad, -1, -1])
    
    results = []
    with Pool(num_process) as p:
        results = p.map(main, args)
    
    for count_parallel in range(len(results)):
        if count_parallel % 4 == 0:
            S_value_plus_max_array[int(count_parallel / 4)] = results[count_parallel][0]
            vpara_r_plus_max_array[int(count_parallel / 4)] = results[count_parallel][1]
        elif count_parallel % 4 == 1:
            S_value_plus_min_array[int(count_parallel / 4)] = results[count_parallel][0]
            vpara_r_plus_min_array[int(count_parallel / 4)] = results[count_parallel][1]
        elif count_parallel % 4 == 2:
            S_value_minus_max_array[int(count_parallel / 4)] = results[count_parallel][0]
            vpara_r_minus_max_array[int(count_parallel / 4)] = results[count_parallel][1]
        elif count_parallel % 4 == 3:
            S_value_minus_min_array[int(count_parallel / 4)] = results[count_parallel][0]
            vpara_r_minus_min_array[int(count_parallel / 4)] = results[count_parallel][1]

output_data_array = np.zeros((len(mlat_deg_array), 9))
output_data_array[:, 0] = mlat_deg_array
output_data_array[:, 1] = S_value_plus_max_array
output_data_array[:, 2] = vpara_r_plus_max_array
output_data_array[:, 3] = S_value_plus_min_array
output_data_array[:, 4] = vpara_r_plus_min_array
output_data_array[:, 5] = S_value_minus_max_array
output_data_array[:, 6] = vpara_r_minus_max_array
output_data_array[:, 7] = S_value_minus_min_array
output_data_array[:, 8] = vpara_r_minus_min_array

dir_path = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_limit_resonance'
os.makedirs(dir_path, exist_ok=True)
data_path = f'{dir_path}/data_{len(mlat_deg_array)}.csv'
np.savetxt(data_path, output_data_array, delimiter=',')
print(f'saved: {data_path}')