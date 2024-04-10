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
plt.rcParams["font.size"] = 35

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
mlat_deg_array = np.linspace(1E0, 30E0, 30)
mlat_rad_array = mlat_deg_array * np.pi / 180E0

detrap_point_psi = np.zeros_like(mlat_rad_array)
detrap_point_S = np.zeros_like(mlat_rad_array)
detrap_point_capital_theta = np.zeros_like(mlat_rad_array)


# detrap point
def initial_capital_theta_function(S_value, psi):
    capital_theta_2 = 5E-1 * (np.cos(psi) + np.sqrt(1E0 - S_value**2E0) - S_value * (psi + np.pi - np.arcsin(S_value)))
    if capital_theta_2 < 0E0:
        return 0E0
    else:
        return np.sqrt(capital_theta_2)

def initial_S_value_function(S_value, psi, mlat_rad):
    energy_ratio = energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)
    return S_value - energy_ratio * (1E0 + Gamma(mlat_rad)) * delta(mlat_rad) * (1E0 + initial_capital_theta_function(S_value, psi) / energy_ratio)**2E0

def W_value_function(S_value, capital_theta, mlat_rad):
    theta = 2E0 * trapping_frequency(mlat_rad) * capital_theta
    return (1E0 + Gamma(mlat_rad)) * (((1E0 + Gamma(mlat_rad) / 2E0 - delta_2(mlat_rad) / delta(mlat_rad)**2E0) * S_value - energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad) * (Gamma(mlat_rad)**2E0 - 3E0 * Gamma(mlat_rad) + 3E0) * delta(mlat_rad) * (1E0 + theta / wave_frequency)**2E0))

def initial_psi_function(psi, mlat_rad):
    # S_value
    diff_S = 1E-10
    S_value_old = 1E-2
    count_iteration = 0
    while True:
        diff_1 = initial_S_value_function(S_value_old, psi, mlat_rad)
        diff_2 = (initial_S_value_function(S_value_old + diff_S, psi, mlat_rad) - initial_S_value_function(S_value_old - diff_S, psi, mlat_rad)) / 2E0 / diff_S
        diff = diff_1 / diff_2
        if np.abs(diff) > 1E-4 / np.log(count_iteration + 2E0):
            diff = np.sign(diff) / np.log(count_iteration + 2E0) * 1E-4
        S_value_new = S_value_old - diff
        if np.abs(S_value_new - S_value_old) < 1E-5:
            break
        if S_value_new != S_value_new or S_value_new > 1E0:
            print('S_value_new is NaN')
            return np.nan, np.nan, np.nan
        if np.mod(count_iteration, 10000) == 9999:
            print(mlat_rad/np.pi*180, psi/np.pi, S_value_new, abs(S_value_new - S_value_old))
        S_value_old = S_value_new
        count_iteration += 1
    
    # capital_theta
    capital_theta = initial_capital_theta_function(S_value_new, psi)

    # W_value
    W_value = W_value_function(S_value_new, capital_theta, mlat_rad)

    return np.cos(psi) + np.sqrt(1E0 - S_value_new**2E0) + (psi + np.pi - np.arcsin(S_value_new)) * (np.sin(psi) + W_value - S_value_new), S_value_new, capital_theta

def iteration_detrap_point(mlat_rad):
    diff_psi = 1E-10
    psi_old = -2.9E0
    count_iteration_detrap_point = 0
    while True:
        diff_1 = initial_psi_function(psi_old, mlat_rad)[0]
        diff_2 = (initial_psi_function(psi_old + diff_psi, mlat_rad)[0] - initial_psi_function(psi_old - diff_psi, mlat_rad)[0]) / 2E0 / diff_psi
        diff = diff_1 / diff_2
        if np.abs(diff) > 1E-2:
            diff = np.sign(diff) * 1E-2
        psi_new = psi_old - diff
        if np.abs(psi_new - psi_old) < 1E-8:
            break
        if psi_new != psi_new:
            print('psi_new is NaN')
            return np.nan, np.nan, np.nan
        if count_iteration_detrap_point > 1000:
            print(mlat_rad/np.pi*180, psi_old, psi_new, abs(psi_new - psi_old))
            return np.nan, np.nan, np.nan
        psi_old = psi_new
        count_iteration_detrap_point += 1
    
    print(f'mlat_deg = {mlat_rad/np.pi*180E0}, psi_new = {psi_new/np.pi}')
    S_value, capital_theta = initial_psi_function(psi_new, mlat_rad)[1:]
    
    return psi_new, S_value, capital_theta

if __name__ == '__main__':
    results = []
    with Pool(number_parallel) as p:
        results = p.map(iteration_detrap_point, mlat_rad_array)
    
    for count_i in range(len(mlat_rad_array)):
        detrap_point_psi[count_i] = results[count_i][0]
        detrap_point_S[count_i] = results[count_i][1]
        detrap_point_capital_theta[count_i] = results[count_i][2]

output_data = np.zeros((len(mlat_deg_array), 5))
output_data[:, 0] = mlat_deg_array
output_data[:, 1] = mlat_rad_array
output_data[:, 2] = detrap_point_psi
output_data[:, 3] = detrap_point_S
output_data[:, 4] = detrap_point_capital_theta

# save data
dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/test_particle_simulation_detrapped_phase_edgedefinition_dfdt_0'
os.makedirs(dir_name, exist_ok=True)
file_path = f'{dir_name}/detrap_point_{len(mlat_deg_array)}.csv'
np.savetxt(file_path, output_data, delimiter=',')