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


# input parameters

number_parallel = os.cpu_count()

initial_S_value_min = 1E-1
initial_S_value_max = 1E0

separate_number_mesh = 10

grid_scale = 'linear'

if grid_scale == 'linear':
    initial_S_value_array = np.linspace(initial_S_value_min, initial_S_value_max, separate_number_mesh)
elif grid_scale == 'log':
    initial_S_value_array = np.logspace(np.log10(initial_S_value_min), np.log10(initial_S_value_max), separate_number_mesh)
else:
    print('Error: invalid grid_scale')
    quit()

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/auroral_electron_acceleration_mu_condition_edgedefinition/S_{initial_S_value_min:.2f}_{initial_S_value_max:.2f}_{separate_number_mesh}_{grid_scale}'
os.makedirs(dir_name, exist_ok=True)

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
    for count_i in range(separate_number_mesh):
        args.append(initial_S_value_array[count_i])
    
    results = []
    with Pool(number_parallel) as p:
        results = p.map(psi_max_detection, args)
    
    for count_i in range(separate_number_mesh):
        psi_max[count_i] = results[count_i]

initial_psi_array = np.zeros((separate_number_mesh, separate_number_mesh))
for count_i in range(separate_number_mesh):
    initial_psi_array[count_i, :] = np.linspace(psi_min_array[count_i], psi_max[count_i], separate_number_mesh)


def capital_theta_function(S_value, psi, sign):
    return sign * np.sqrt(5E-1 * (np.cos(psi) + np.sqrt(1E0 - S_value**2E0) - S_value * (psi + np.pi - np.arcsin(S_value))))    #[]

capital_theta_array = np.zeros((separate_number_mesh, separate_number_mesh, 2))
for count_i in range(separate_number_mesh):
    for count_j in range(separate_number_mesh):
        capital_theta_array[count_i, count_j, 0] = capital_theta_function(initial_S_value_array[count_i], initial_psi_array[count_i, count_j], 1E0)
        capital_theta_array[count_i, count_j, 1] = capital_theta_function(initial_S_value_array[count_i], initial_psi_array[count_i, count_j], - 1E0)


def integral_function(x, a, b):
    try:
        if a == -7:
            # -7 の場合の式
            return (-x*np.sin(x)**7*np.cos(b - 7*x)/128 - 7*x*np.sin(x)**6*np.sin(b - 7*x)*np.cos(x)/128 +
                    21*x*np.sin(x)**5*np.cos(x)**2*np.cos(b - 7*x)/128 + 35*x*np.sin(x)**4*np.sin(b - 7*x)*np.cos(x)**3/128 -
                    35*x*np.sin(x)**3*np.cos(x)**4*np.cos(b - 7*x)/128 - 21*x*np.sin(x)**2*np.sin(b - 7*x)*np.cos(x)**5/128 +
                    7*x*np.sin(x)*np.cos(x)**6*np.cos(b - 7*x)/128 + x*np.sin(b - 7*x)*np.cos(x)**7/128 -
                    np.sin(x)**7*np.sin(b - 7*x)/384 + np.sin(x)**6*np.cos(x)*np.cos(b - 7*x)/96 +
                    29*np.sin(x)**4*np.cos(x)**3*np.cos(b - 7*x)/384 + 77*np.sin(x)**3*np.sin(b - 7*x)*np.cos(x)**4/384 -
                    11*np.sin(x)**2*np.cos(x)**5*np.cos(b - 7*x)/40 - 119*np.sin(x)*np.sin(b - 7*x)*np.cos(x)**6/480 +
                    2381*np.cos(x)**7*np.cos(b - 7*x)/13440)
        elif a == -5:
            # -5 の場合の式
            return (7*x*np.sin(x)**7*np.cos(b - 5*x)/128 + 35*x*np.sin(x)**6*np.sin(b - 5*x)*np.cos(x)/128 -
                    63*x*np.sin(x)**5*np.cos(x)**2*np.cos(b - 5*x)/128 - 35*x*np.sin(x)**4*np.sin(b - 5*x)*np.cos(x)**3/128 -
                    35*x*np.sin(x)**3*np.cos(x)**4*np.cos(b - 5*x)/128 - 63*x*np.sin(x)**2*np.sin(b - 5*x)*np.cos(x)**5/128 +
                    35*x*np.sin(x)*np.cos(x)**6*np.cos(b - 5*x)/128 + 7*x*np.sin(b - 5*x)*np.cos(x)**7/128 +
                    35*np.sin(x)**7*np.sin(b - 5*x)/1152 - 7*np.sin(x)**6*np.cos(x)*np.cos(b - 5*x)/72 -
                    413*np.sin(x)**4*np.cos(x)**3*np.cos(b - 5*x)/1152 - 595*np.sin(x)**3*np.sin(b - 5*x)*np.cos(x)**4/1152 +
                    7*np.sin(x)**2*np.cos(x)**5*np.cos(b - 5*x)/40 - 7*np.sin(x)*np.sin(b - 5*x)*np.cos(x)**6/36 +
                    1313*np.cos(x)**7*np.cos(b - 5*x)/5760)
        elif a == -3:
            # -3 の場合の式
            return (-21*x*np.sin(x)**7*np.cos(b - 3*x)/128 - 63*x*np.sin(x)**6*np.sin(b - 3*x)*np.cos(x)/128 +
                    21*x*np.sin(x)**5*np.cos(x)**2*np.cos(b - 3*x)/128 - 105*x*np.sin(x)**4*np.sin(b - 3*x)*np.cos(x)**3/128 +
                    105*x*np.sin(x)**3*np.cos(x)**4*np.cos(b - 3*x)/128 - 21*x*np.sin(x)**2*np.sin(b - 3*x)*np.cos(x)**5/128 +
                    63*x*np.sin(x)*np.cos(x)**6*np.cos(b - 3*x)/128 + 21*x*np.sin(b - 3*x)*np.cos(x)**7/128 -
                    63*np.sin(x)**7*np.sin(b - 3*x)/128 + 21*np.sin(x)**6*np.cos(x)*np.cos(b - 3*x)/16 +
                    343*np.sin(x)**4*np.cos(x)**3*np.cos(b - 3*x)/128 + 231*np.sin(x)**3*np.sin(b - 3*x)*np.cos(x)**4/128 +
                    49*np.sin(x)**2*np.cos(x)**5*np.cos(b - 3*x)/40 + 119*np.sin(x)*np.sin(b - 3*x)*np.cos(x)**6/80 -
                    139*np.cos(x)**7*np.cos(b - 3*x)/640)
        elif a == -1:
            # -1 の場合の式
            return (35*x*np.sin(x)**7*np.cos(b - x)/128 + 35*x*np.sin(x)**6*np.sin(b - x)*np.cos(x)/128 +
                    105*x*np.sin(x)**5*np.cos(x)**2*np.cos(b - x)/128 + 105*x*np.sin(x)**4*np.sin(b - x)*np.cos(x)**3/128 +
                    105*x*np.sin(x)**3*np.cos(x)**4*np.cos(b - x)/128 + 105*x*np.sin(x)**2*np.sin(b - x)*np.cos(x)**5/128 +
                    35*x*np.sin(x)*np.cos(x)**6*np.cos(b - x)/128 + 35*x*np.sin(b - x)*np.cos(x)**7/128 -
                    35*np.sin(x)**7*np.sin(b - x)/384 + 35*np.sin(x)**6*np.cos(x)*np.cos(b - x)/96 +
                    385*np.sin(x)**4*np.cos(x)**3*np.cos(b - x)/384 + 175*np.sin(x)**3*np.sin(b - x)*np.cos(x)**4/384 +
                    7*np.sin(x)**2*np.cos(x)**5*np.cos(b - x)/8 + 49*np.sin(x)*np.sin(b - x)*np.cos(x)**6/96 +
                    83*np.cos(x)**7*np.cos(b - x)/384)

        elif a == 1:
            # 1 の場合の式
            return (-35*x*np.sin(x)**7*np.cos(b + x)/128 + 35*x*np.sin(x)**6*np.sin(b + x)*np.cos(x)/128 -
                    105*x*np.sin(x)**5*np.cos(x)**2*np.cos(b + x)/128 + 105*x*np.sin(x)**4*np.sin(b + x)*np.cos(x)**3/128 -
                    105*x*np.sin(x)**3*np.cos(x)**4*np.cos(b + x)/128 + 105*x*np.sin(x)**2*np.sin(b + x)*np.cos(x)**5/128 -
                    35*x*np.sin(x)*np.cos(x)**6*np.cos(b + x)/128 + 35*x*np.sin(b + x)*np.cos(x)**7/128 -
                    35*np.sin(x)**7*np.sin(b + x)/384 - 35*np.sin(x)**6*np.cos(x)*np.cos(b + x)/96 -
                    385*np.sin(x)**4*np.cos(x)**3*np.cos(b + x)/384 + 175*np.sin(x)**3*np.sin(b + x)*np.cos(x)**4/384 -
                    7*np.sin(x)**2*np.cos(x)**5*np.cos(b + x)/8 + 49*np.sin(x)*np.sin(b + x)*np.cos(x)**6/96 -
                    83*np.cos(x)**7*np.cos(b + x)/384)

        elif a == 3:
            # 3 の場合の式
            return (21*x*np.sin(x)**7*np.cos(b + 3*x)/128 - 63*x*np.sin(x)**6*np.sin(b + 3*x)*np.cos(x)/128 -
                    21*x*np.sin(x)**5*np.cos(x)**2*np.cos(b + 3*x)/128 - 105*x*np.sin(x)**4*np.sin(b + 3*x)*np.cos(x)**3/128 -
                    105*x*np.sin(x)**3*np.cos(x)**4*np.cos(b + 3*x)/128 - 21*x*np.sin(x)**2*np.sin(b + 3*x)*np.cos(x)**5/128 -
                    63*x*np.sin(x)*np.cos(x)**6*np.cos(b + 3*x)/128 + 21*x*np.sin(b + 3*x)*np.cos(x)**7/128 -
                    63*np.sin(x)**7*np.sin(b + 3*x)/128 - 21*np.sin(x)**6*np.cos(x)*np.cos(b + 3*x)/16 -
                    343*np.sin(x)**4*np.cos(x)**3*np.cos(b + 3*x)/128 + 231*np.sin(x)**3*np.sin(b + 3*x)*np.cos(x)**4/128 -
                    49*np.sin(x)**2*np.cos(x)**5*np.cos(b + 3*x)/40 + 119*np.sin(x)*np.sin(b + 3*x)*np.cos(x)**6/80 +
                    139*np.cos(x)**7*np.cos(b + 3*x)/640)
        elif a == 5:
            # 5 の場合の式
            return (-7*x*np.sin(x)**7*np.cos(b + 5*x)/128 + 35*x*np.sin(x)**6*np.sin(b + 5*x)*np.cos(x)/128 +
                    63*x*np.sin(x)**5*np.cos(x)**2*np.cos(b + 5*x)/128 - 35*x*np.sin(x)**4*np.sin(b + 5*x)*np.cos(x)**3/128 +
                    35*x*np.sin(x)**3*np.cos(x)**4*np.cos(b + 5*x)/128 - 63*x*np.sin(x)**2*np.sin(b + 5*x)*np.cos(x)**5/128 -
                    35*x*np.sin(x)*np.cos(x)**6*np.cos(b + 5*x)/128 + 7*x*np.sin(b + 5*x)*np.cos(x)**7/128 +
                    35*np.sin(x)**7*np.sin(b + 5*x)/1152 + 7*np.sin(x)**6*np.cos(x)*np.cos(b + 5*x)/72 +
                    413*np.sin(x)**4*np.cos(x)**3*np.cos(b + 5*x)/1152 - 595*np.sin(x)**3*np.sin(b + 5*x)*np.cos(x)**4/1152 -
                    7*np.sin(x)**2*np.cos(x)**5*np.cos(b + 5*x)/40 - 7*np.sin(x)*np.sin(b + 5*x)*np.cos(x)**6/36 -
                    1313*np.cos(x)**7*np.cos(b + 5*x)/5760)
        elif a == 7:
            # 7 の場合の式
            return (x*np.sin(x)**7*np.cos(b + 7*x)/128 - 7*x*np.sin(x)**6*np.sin(b + 7*x)*np.cos(x)/128 -
                    21*x*np.sin(x)**5*np.cos(x)**2*np.cos(b + 7*x)/128 + 35*x*np.sin(x)**4*np.sin(b + 7*x)*np.cos(x)**3/128 +
                    35*x*np.sin(x)**3*np.cos(x)**4*np.cos(b + 7*x)/128 - 21*x*np.sin(x)**2*np.sin(b + 7*x)*np.cos(x)**5/128 -
                     7*x*np.sin(x)*np.cos(x)**6*np.cos(b + 7*x)/128 + x*np.sin(b + 7*x)*np.cos(x)**7/128 -
                    np.sin(x)**7*np.sin(b + 7*x)/384 - np.sin(x)**6*np.cos(x)*np.cos(b + 7*x)/96 -
                    29*np.sin(x)**4*np.cos(x)**3*np.cos(b + 7*x)/384 + 77*np.sin(x)**3*np.sin(b + 7*x)*np.cos(x)**4/384 +
                    11*np.sin(x)**2*np.cos(x)**5*np.cos(b + 7*x)/40 - 119*np.sin(x)*np.sin(b + 7*x)*np.cos(x)**6/480 -
                    2381*np.cos(x)**7*np.cos(b + 7*x)/13440)

        else:
            # それ以外の場合の式
            return (-a**7*np.cos(x)**7*np.cos(a*x + b)/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) -
                    7*a**6*np.sin(x)*np.sin(a*x + b)*np.cos(x)**6/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    42*a**5*np.sin(x)**2*np.cos(x)**5*np.cos(a*x + b)/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    77*a**5*np.cos(x)**7*np.cos(a*x + b)/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    210*a**4*np.sin(x)**3*np.sin(a*x + b)*np.cos(x)**4/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    455*a**4*np.sin(x)*np.sin(a*x + b)*np.cos(x)**6/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) -
                    840*a**3*np.sin(x)**4*np.cos(x)**3*np.cos(a*x + b)/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) -
                    2100*a**3*np.sin(x)**2*np.cos(x)**5*np.cos(a*x + b)/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) -
                    1519*a**3*np.cos(x)**7*np.cos(a*x + b)/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) -
                    2520*a**2*np.sin(x)**5*np.sin(a*x + b)*np.cos(x)**2/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) -
                    7140*a**2*np.sin(x)**3*np.sin(a*x + b)*np.cos(x)**4/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) -
                    6433*a**2*np.sin(x)*np.sin(a*x + b)*np.cos(x)**6/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    5040*a*np.sin(x)**6*np.cos(x)*np.cos(a*x + b)/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    15960*a*np.sin(x)**4*np.cos(x)**3*np.cos(a*x + b)/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    17178*a*np.sin(x)**2*np.cos(x)**5*np.cos(a*x + b)/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    6483*a*np.cos(x)**7*np.cos(a*x + b)/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    5040*np.sin(x)**7*np.sin(a*x + b)/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    17640*np.sin(x)**5*np.sin(a*x + b)*np.cos(x)**2/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    22050*np.sin(x)**3*np.sin(a*x + b)*np.cos(x)**4/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025) +
                    11025*np.sin(x)*np.sin(a*x + b)*np.cos(x)**6/(a**8 - 84*a**6 + 1974*a**4 - 12916*a**2 + 11025))
    
    except:
        return np.nan


def delta_K_detrap_to_critical_approx_function(psi, capital_theta, mlat_rad):
    coefficient = - energy_wave_potential(mlat_rad) / kperp_rhoi * wave_frequency * r_eq / Alfven_speed(0) * np.sqrt(2E0 * tau(mlat_rad) / (1E0 + tau(mlat_rad)))   #[J]

    if capital_theta > 0E0:
        mlat_critical_rad = 40E0 * np.pi / 180E0
    elif capital_theta <= 0E0:
        mlat_critical_rad = 60E0 * np.pi / 180E0
    
    xi_integral = capital_theta / (capital_theta + np.sqrt(energy_wave_phase_speed(mlat_rad) / 2E0 / energy_wave_potential(mlat_rad))) * kpara(mlat_rad) * r_eq * np.cos(mlat_rad) * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad))
    if xi_integral != xi_integral:
        return np.nan
    
    A_integral = psi - xi_integral * mlat_rad

    if mlat_critical_rad != mlat_critical_rad:
        return np.nan

    integral_value_critical = integral_function(mlat_critical_rad, xi_integral, A_integral)
    integral_value_mlat_rad = integral_function(mlat_rad, xi_integral, A_integral)
    if integral_value_critical == integral_value_critical and integral_value_mlat_rad == integral_value_mlat_rad:
        integral_value = integral_value_critical - integral_value_mlat_rad
        return coefficient * integral_value    #[J]
    else:
        return np.nan

def iteration_h_function(S_value, psi, capital_theta, mlat_rad):
    h_function = S_value
    h_function -= energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad) * (Gamma(mlat_rad) + magnetic_flux_density(mlat_rad) / (magnetic_flux_density(mlat_upper_limit_rad) - magnetic_flux_density(mlat_rad))) * delta(mlat_rad) * (1E0 + np.sqrt(2E0 * energy_wave_potential(mlat_rad) / energy_wave_phase_speed(mlat_rad)) * capital_theta)**2E0
    h_function -= magnetic_flux_density(mlat_rad) / (magnetic_flux_density(mlat_upper_limit_rad) - magnetic_flux_density(mlat_rad)) * delta(mlat_rad) * delta_K_detrap_to_critical_approx_function(psi, capital_theta, mlat_rad)
    return h_function


def iteration_Newton_method(S_value, psi, capital_theta):
    mlat_rad_old = np.pi / 4E0
    count_iteration = 0
    while True:
        h_value = iteration_h_function(S_value, psi, capital_theta, mlat_rad_old)
        h_value_diff = (iteration_h_function(S_value, psi, capital_theta, mlat_rad_old + diff_rad) - iteration_h_function(S_value, psi, capital_theta, mlat_rad_old - diff_rad)) / 2E0 / diff_rad
        diff = h_value / h_value_diff
        if np.abs(diff) > 1E-3:
            diff = np.sign(diff) * 1E-3
        elif diff != diff:
            print(r'Erorr!: diff != diff')
            quit()
        mlat_rad_new = mlat_rad_old - diff
        if np.abs(mlat_rad_new - mlat_rad_old) < 1E-6:
            return mlat_rad_new
        else:
            mlat_rad_old = mlat_rad_new
            count_iteration += 1
            if mlat_rad_new > mlat_upper_limit_rad:
                print('mlat_rad_new > mlat_upper_limit_rad')
                mlat_rad_old = 2E0 * mlat_upper_limit_rad - mlat_rad_old
            if mlat_rad_old < 0E0:
                print('mlat_rad_old < 0E0')
                mlat_rad_old = - mlat_rad_old
            if count_iteration > 10000:
                print('count_iteration > 10000')
                return np.nan

mlat_rad_array = np.zeros_like(capital_theta_array)

def iteration_Newton_method_array(args):
    count_i, count_j, count_k = args
    S_value = initial_S_value_array[count_i]
    psi = initial_psi_array[count_i, count_j]
    capital_theta = capital_theta_array[count_i, count_j, count_k]
    if capital_theta != capital_theta:
        mlat_rad_array[count_i, count_j, count_k] = np.nan
    else:
        mlat_rad_array[count_i, count_j, count_k] = iteration_Newton_method(S_value, psi, capital_theta)
    return


if __name__ == '__main__':
    args = []
    for count_i in range(separate_number_mesh):
        for count_j in range(separate_number_mesh):
            for count_k in range(2):
                args.append([count_i, count_j, count_k])
    
    results = []
    with Pool(number_parallel) as p:
        results = p.map(iteration_Newton_method_array, args)


def K_perp_eq_ionosphere_function(psi, capital_theta, mlat_rad):
    K_perp_eq = magnetic_flux_density(0) / (magnetic_flux_density(mlat_upper_limit_rad) - magnetic_flux_density(0)) * (energy_wave_phase_speed(mlat_rad) * (1E0 + np.sqrt(2E0 * energy_wave_potential(mlat_rad) / energy_wave_phase_speed(mlat_rad)) * capital_theta)**2E0 + delta_K_detrap_to_critical_approx_function(psi, capital_theta, mlat_rad))
    K_perp_ionosphere = K_perp_eq * magnetic_flux_density(mlat_upper_limit_rad) / magnetic_flux_density(0)
    return K_perp_eq, K_perp_ionosphere

K_perp_eq = np.zeros_like(capital_theta_array)
K_perp_ionosphere = np.zeros_like(capital_theta_array)
for count_i in range(separate_number_mesh):
    for count_j in range(separate_number_mesh):
        for count_k in range(2):
            K_perp_eq[count_i, count_j, count_k], K_perp_ionosphere[count_i, count_j, count_k] = K_perp_eq_ionosphere_function(initial_psi_array[count_i, count_j], capital_theta_array[count_i, count_j, count_k], mlat_rad_array[count_i, count_j, count_k])

print(K_perp_ionosphere)