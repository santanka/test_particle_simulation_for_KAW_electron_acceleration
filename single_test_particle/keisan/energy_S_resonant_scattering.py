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
plt.rcParams["font.size"] = 35

# constants
speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]
electric_constant = 8.8541878128E-12  #[F m-1]
magnetic_constant = 1.25663706212E-6  #[N A-2]

#planet condition
planet_radius   = 6371E3  #[m]
lshell_number   = 9E0
r_eq            = planet_radius * lshell_number #[m]

def d_mlat_d_z(mlat_rad):
    return 1E0 / r_eq / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)    #[rad/m]

#gradient function
diff_rad = 1E-5
def gradient_meter(function, mlat_rad):
    if mlat_rad < diff_rad:
        return 0E0
    else:
        return (function(mlat_rad + diff_rad) - function(mlat_rad - diff_rad)) / 2E0 / diff_rad * d_mlat_d_z(mlat_rad)    #[m^-1]
    
def gradient_mlat(function, mlat_rad, pitch_angle_rad):
    if mlat_rad < diff_rad:
        return 0E0
    else:
        return (function(pitch_angle_rad, mlat_rad + diff_rad) - function(pitch_angle_rad, mlat_rad - diff_rad)) / 2E0 / diff_rad #[rad^-1]
    
def gradient_mlat_psi(function, mu, mlat_rad, psi):
    if mlat_rad < diff_rad:
        return 0E0
    else:
        return (function(mu, mlat_rad + diff_rad, psi) - function(mu, mlat_rad - diff_rad, psi)) / 2E0 / diff_rad #[rad^-1]



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

mu_upper_limit = 1E3 * elementary_charge / magnetic_flux_density(0E0)    #[J T-1]


# wave parameters
kperp_rhoi = 2E0 * np.pi    #[rad]
wave_frequency = 2E0 * np.pi * 0.15    #[rad/s]

def wave_phase_speed(mlat_rad):
    return Alfven_speed(mlat_rad) * kperp_rhoi * np.sqrt((1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad)))    #[m/s]

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
    return 1E0 / kpara(mlat_rad) / magnetic_flux_density(mlat_rad) * gradient_meter(magnetic_flux_density, mlat_rad)    #[rad^-1]

def epsilon(mlat_rad):
    return delta(mlat_rad) * (3E0 - 4E0 * tau(mlat_rad) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad)))    #[rad^-1]


# find S*
def function_S_star(S_variable):
    return ((np.pi - np.arcsin(S_variable))**2E0 + 1E0) * S_variable - 2E0 * (np.pi - np.arcsin(S_variable))

diff_S = 1E-5
def gradient_S_star(S_variable):
    return (function_S_star(S_variable + diff_S) - function_S_star(S_variable - diff_S)) / 2E0 / diff_S

def Newton_method_function_S_star():
    initial_S_variable = 5E-1
    S_variable_before_update = initial_S_variable
    count_iteration = 0
    while True:
        diff = function_S_star(S_variable_before_update) / gradient_S_star(S_variable_before_update)
        S_variable_after_update = S_variable_before_update - diff
        if abs(S_variable_after_update - S_variable_before_update) < 1E-15:
            #print(count_iteration)
            break
        else:
            S_variable_before_update = S_variable_after_update
            count_iteration += 1
    return S_variable_after_update

S_star = Newton_method_function_S_star()
#print(S_star)


#上部
# energy upper limit

energy_perp_upper_limit = 1E3 * elementary_charge    #[J]
mu_upper_limit = energy_perp_upper_limit / magnetic_flux_density(0E0)    #[J T-1]

def Xi_S(S_variable):
    return np.sqrt(S_variable * (np.pi + np.arcsin(S_variable)) + np.sqrt(1E0 - S_variable**2E0) + 1E0)

def gradient_S(function_S, mlat_rad, S_variable):
    return (function_S(mlat_rad, S_variable + diff_S) - function_S(mlat_rad, S_variable - diff_S)) / 2E0 / diff_S

def energy_upper_limit(mlat_rad, S_variable):
    if ((0E0 <= S_variable) and (S_variable <= S_star)):
        return (np.sqrt(energy_wave_potential(mlat_rad)) * Xi_S(S_variable) + np.sqrt(energy_wave_phase_speed(mlat_rad)))**2E0
    elif (S_star < S_variable):
        return (np.sqrt(2E0 * np.pi * S_variable * energy_wave_potential(mlat_rad)) + np.sqrt(energy_wave_phase_speed(mlat_rad)))**2E0
    else:
        return np.nan

def function_upper_energy_S(mlat_rad, S_variable):
    S_variable = np.abs(S_variable)
    if ((0E0 <= S_variable) and (S_variable <= S_star)):
        return S_variable - magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad) - (delta(mlat_rad) + epsilon(mlat_rad)) * (Xi_S(S_variable) + np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))**2E0
    elif (S_star < S_variable):
        return S_variable - magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad) - (delta(mlat_rad) + epsilon(mlat_rad)) * (np.sqrt(2E0 * np.pi * S_variable) + np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))**2E0
    else:
        return 0E0
    
def solve_upper_energy_S(mlat_rad):
    S_min = magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad)
    S_variable_array = np.linspace(S_min, S_star, 100)
    function_upper_energy_S_array = np.zeros(len(S_variable_array))
    for count_i in range(len(S_variable_array)):
        function_upper_energy_S_array[count_i] = function_upper_energy_S(mlat_rad, S_variable_array[count_i])
    max_function_upper_energy_S = np.max(function_upper_energy_S_array)
    max_S = S_variable_array[np.argmax(function_upper_energy_S_array)]
    min_function_upper_energy_S = np.min(function_upper_energy_S_array)
    min_S = S_variable_array[np.argmin(function_upper_energy_S_array)]

    S_solution_under_S_star = np.nan
    upper_energy_limit_under_S_star = np.nan

    if max_function_upper_energy_S == 0E0:
        S_solution_under_S_star = max_S
    
    elif min_function_upper_energy_S == 0E0:
        S_solution_under_S_star = min_S
    
    elif function_upper_energy_S(mlat_rad, S_star) == 0E0:
        S_solution_under_S_star = S_star
    
    elif function_upper_energy_S(mlat_rad, S_min) == 0E0:
        S_solution_under_S_star = S_min

    elif max_function_upper_energy_S * min_function_upper_energy_S < 0E0:
        #Newton法で解を求める
        initial_S_variable = (max_S + min_S) / 2E0
        S_variable_before_update = initial_S_variable
        count_iteration = 0
        while True:
            diff = function_upper_energy_S(mlat_rad, S_variable_before_update) / gradient_S(function_upper_energy_S, mlat_rad, S_variable_before_update)
            if np.abs(diff) > 1E-2:
                diff = np.sign(diff) * 1E-2
            S_variable_after_update = S_variable_before_update - diff
            if abs(S_variable_after_update - S_variable_before_update) < 1E-7:
                break
            elif S_variable_after_update != S_variable_after_update:
                print('Error!')
                quit()
            else:
                S_variable_before_update = S_variable_after_update
                count_iteration += 1
        S_solution_under_S_star = S_variable_after_update
    
    if S_solution_under_S_star == S_solution_under_S_star:
        upper_energy_limit_under_S_star = energy_upper_limit(mlat_rad, S_solution_under_S_star)
    
    S_solution_upper_S_star = np.nan
    upper_energy_limit_upper_S_star = np.nan

    #S_star以上で解があるかどうかを調べる
    coefficient_a = 1E0 - (delta(mlat_rad) + epsilon(mlat_rad)) * 2E0 * np.pi
    coefficient_b = - 2E0 * (delta(mlat_rad) + epsilon(mlat_rad)) * np.sqrt(2E0 * np.pi * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))
    coefficient_c = - magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad) - (delta(mlat_rad) + epsilon(mlat_rad)) * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)

    #coefficient_a * sqrt(S)**2 + coefficient_b * sqrt(S) + coefficient_c = 0
    discriminant = coefficient_b**2E0 - 4E0 * coefficient_a * coefficient_c
    if discriminant >= 0E0:
        S_solution_plus = (- coefficient_b + np.sqrt(discriminant)) / 2E0 / coefficient_a
        S_solution_minus = (- coefficient_b - np.sqrt(discriminant)) / 2E0 / coefficient_a
        if S_solution_plus < 0E0 or S_solution_plus**2E0 < S_star:
            S_solution_plus = np.nan
        if S_solution_minus < 0E0 or S_solution_minus**2E0 < S_star:
            S_solution_minus = np.nan
        S_solution_plus = S_solution_plus**2E0
        S_solution_minus = S_solution_minus**2E0
        S_solution = np.nanmin([S_solution_plus, S_solution_minus])
        if S_solution == S_solution:
            upper_energy_limit_upper_S_star = energy_upper_limit(mlat_rad, S_solution)
            S_solution_upper_S_star = S_solution
    
    if S_solution_under_S_star != S_solution_under_S_star and S_solution_upper_S_star != S_solution_upper_S_star:
        if function_upper_energy_S(mlat_rad, S_min) <= 0E0:
            return np.inf, np.inf
        else:
            return magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad), 0E0
    
    elif S_solution_under_S_star == S_solution_under_S_star and S_solution_upper_S_star != S_solution_upper_S_star:
        if function_upper_energy_S(mlat_rad, S_min) <= 0E0:
            return S_solution_under_S_star, upper_energy_limit_under_S_star
        else:
            return np.inf, np.inf
    
    elif S_solution_under_S_star != S_solution_under_S_star and S_solution_upper_S_star == S_solution_upper_S_star:
        if function_upper_energy_S(mlat_rad, S_star) <= 0E0:
            return S_solution_upper_S_star, upper_energy_limit_upper_S_star
        else:
            return np.inf, np.inf
    
    elif S_solution_under_S_star == S_solution_under_S_star and S_solution_upper_S_star == S_solution_upper_S_star:
        if function_upper_energy_S(mlat_rad, S_min) <= 0E0:
            return S_solution_under_S_star, upper_energy_limit_under_S_star
        elif function_upper_energy_S(mlat_rad, S_min) > 0E0:
            return S_solution_upper_S_star, upper_energy_limit_upper_S_star


# energy lower limit

def energy_lower_limit(mlat_rad, S_variable):
    if ((0E0 <= S_variable) and (S_variable <= S_star) and (Xi_S(S_variable) < np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))):
        return (np.sqrt(energy_wave_potential(mlat_rad)) * Xi_S(S_variable) - np.sqrt(energy_wave_phase_speed(mlat_rad)))**2E0
    elif ((S_star < S_variable) and (S_variable < 1E0 / 2E0 / np.pi * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))):
        return (np.sqrt(2E0 * np.pi * S_variable * energy_wave_potential(mlat_rad)) - np.sqrt(energy_wave_phase_speed(mlat_rad)))**2E0
    elif (((0E0 <= S_variable) and (S_variable <= S_star) and (Xi_S(S_variable) >= np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))) or ((S_star < S_variable) and (S_variable >= 1E0 / 2E0 / np.pi * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))):
        return 0E0
    else:
        return np.nan

def function_lower_energy_S(mlat_rad, S_variable):
    if ((0E0 <= S_variable) and (S_variable <= S_star) and (Xi_S(S_variable) < np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))):
        return S_variable - magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad) - (delta(mlat_rad) + epsilon(mlat_rad)) * (Xi_S(S_variable) - np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))**2E0
    elif ((S_star < S_variable) and (S_variable < 1E0 / 2E0 / np.pi * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))):
        return S_variable - magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad) - (delta(mlat_rad) + epsilon(mlat_rad)) * (np.sqrt(2E0 * np.pi * S_variable) - np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))**2E0
    elif (((0E0 <= S_variable) and (S_variable <= S_star) and (Xi_S(S_variable) >= np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))) or ((S_star < S_variable) and (S_variable >= 1E0 / 2E0 / np.pi * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))):
        return S_variable - magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad)
    else:
        return np.nan
    
def solve_lower_energy_S(mlat_rad):
    S_min = magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad)
    S_variable_array = np.linspace(S_min, S_star, 100)
    function_lower_energy_S_array = np.zeros(len(S_variable_array))
    for count_i in range(len(S_variable_array)):
        function_lower_energy_S_array[count_i] = function_lower_energy_S(mlat_rad, S_variable_array[count_i])
    max_function_lower_energy_S = np.max(function_lower_energy_S_array)
    max_S = S_variable_array[np.argmax(function_lower_energy_S_array)]
    min_function_lower_energy_S = np.min(function_lower_energy_S_array)
    min_S = S_variable_array[np.argmin(function_lower_energy_S_array)]

    S_solution_under_S_star = np.nan
    lower_energy_limit_under_S_star = np.nan
    S_solution = np.nan
    lower_energy_limit = np.nan

    if max_function_lower_energy_S == 0E0:
        if (Xi_S(max_S) >= np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))):
            S_solution = S_min
        else:
            S_solution = max_S
        lower_energy_limit = energy_lower_limit(mlat_rad, S_solution)
    
    elif min_function_lower_energy_S == 0E0:
        if (Xi_S(min_S) >= np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))):
            S_solution = S_min
        else:
            S_solution = min_S
        lower_energy_limit = energy_lower_limit(mlat_rad, S_solution)
    
    elif function_lower_energy_S(mlat_rad, S_star) == 0E0:
        if (Xi_S(S_star) >= np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))):
            S_solution = S_min
        else:
            S_solution = S_star
        lower_energy_limit = energy_lower_limit(mlat_rad, S_solution)
    
    elif function_lower_energy_S(mlat_rad, S_min) == 0E0:
        S_solution = S_min
        lower_energy_limit = energy_lower_limit(mlat_rad, S_solution)
    
    elif max_function_lower_energy_S * min_function_lower_energy_S < 0E0:
        #Newton法で解を求める
        initial_S_variable = (max_S + min_S) / 2E0
        S_variable_before_update = initial_S_variable
        count_iteration = 0
        while True:
            diff = function_lower_energy_S(mlat_rad, S_variable_before_update) / gradient_S(function_lower_energy_S, mlat_rad, S_variable_before_update)
            if np.abs(diff) > 1E-2:
                diff = np.sign(diff) * 1E-2
            S_variable_after_update = S_variable_before_update - diff
            if abs(S_variable_after_update - S_variable_before_update) < 1E-7:
                break
            elif S_variable_after_update != S_variable_after_update:
                print('Error!')
                quit()
            else:
                S_variable_before_update = S_variable_after_update
                count_iteration += 1
        lower_energy_limit = energy_lower_limit(mlat_rad, S_variable_after_update)
        S_solution = S_variable_after_update
    
    S_solution_under_S_star = S_solution
    lower_energy_limit_under_S_star = lower_energy_limit

    S_solution_upper_S_star = np.nan
    lower_energy_limit_upper_S_star = np.nan
    
    #S_star以上で解があるかどうかを調べる
    coefficient_a = 1E0 - (delta(mlat_rad) + epsilon(mlat_rad)) * 2E0 * np.pi
    coefficient_b = + 2E0 * (delta(mlat_rad) + epsilon(mlat_rad)) * np.sqrt(2E0 * np.pi * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))
    coefficient_c = - magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad) - (delta(mlat_rad) + epsilon(mlat_rad)) * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)
    
    #coefficient_a * sqrt(S)**2 + coefficient_b * sqrt(S) + coefficient_c = 0
    discriminant = coefficient_b**2E0 - 4E0 * coefficient_a * coefficient_c
    if discriminant >= 0E0:
        S_solution_plus = (- coefficient_b + np.sqrt(discriminant)) / 2E0 / coefficient_a
        S_solution_minus = (- coefficient_b - np.sqrt(discriminant)) / 2E0 / coefficient_a
        if S_solution_plus < 0E0 or S_solution_plus**2E0 < S_star:
            S_solution_plus = np.nan
        if S_solution_minus < 0E0 or S_solution_minus**2E0 < S_star:
            S_solution_minus = np.nan
        S_solution_plus = S_solution_plus**2E0
        S_solution_minus = S_solution_minus**2E0
        S_solution = np.nanmin([S_solution_plus, S_solution_minus])
        if S_solution == S_solution:
            lower_energy_limit = energy_lower_limit(mlat_rad, S_solution)
            S_solution_upper_S_star = S_solution
            lower_energy_limit_upper_S_star = lower_energy_limit
    
    if S_solution_under_S_star != S_solution_under_S_star and S_solution_upper_S_star != S_solution_upper_S_star:
        if function_lower_energy_S(mlat_rad, S_min) >= 0E0:
            return S_min, 0E0
        else:
            return np.nan, np.nan
    
    elif S_solution_under_S_star == S_solution_under_S_star and S_solution_upper_S_star != S_solution_upper_S_star:
        if function_lower_energy_S(mlat_rad, S_min) <= 0E0:
            return S_solution_under_S_star, lower_energy_limit_under_S_star
        else:
            return S_min, 0E0
    
    elif S_solution_under_S_star != S_solution_under_S_star and S_solution_upper_S_star == S_solution_upper_S_star:
        if function_lower_energy_S(mlat_rad, S_star) <= 0E0:
            return S_solution_upper_S_star, lower_energy_limit_upper_S_star
        else:
            return S_min, 0E0
    
    elif S_solution_under_S_star == S_solution_under_S_star and S_solution_upper_S_star == S_solution_upper_S_star:
        if function_lower_energy_S(mlat_rad, S_min) <= 0E0:
            return S_solution_under_S_star, lower_energy_limit_under_S_star
        elif function_lower_energy_S(mlat_rad, S_min) > 0E0:
            return S_min, 0E0
        
#下部 (minus)
# energy upper limit

def energy_minus_upper_limit(mlat_rad, S_variable):
    if ((0E0 <= S_variable) and (S_variable <= S_star) and (Xi_S(S_variable) > np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))):
        return (np.sqrt(energy_wave_potential(mlat_rad)) * Xi_S(S_variable) - np.sqrt(energy_wave_phase_speed(mlat_rad)))**2E0
    elif ((S_star < S_variable) and (S_variable > 1E0 / 2E0 / np.pi * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))):
        return (np.sqrt(2E0 * np.pi * S_variable * energy_wave_potential(mlat_rad)) - np.sqrt(energy_wave_phase_speed(mlat_rad)))**2E0
    elif (((0E0 <= S_variable) and (S_variable <= S_star) and (Xi_S(S_variable) <= np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))) or ((S_star < S_variable) and (S_variable <= 1E0 / 2E0 / np.pi * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))):
        return 0E0
    else:
        #print(S_variable, Xi_S(S_variable), 2E0*np.pi*S_variable, np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))
        return np.nan

def function_minus_upper_energy_S(mlat_rad, S_variable):
    if ((0E0 <= S_variable) and (S_variable <= S_star) and (Xi_S(S_variable) > np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))):
        return S_variable - magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad) - (delta(mlat_rad) + epsilon(mlat_rad)) * (Xi_S(S_variable) - np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))**2E0
    elif ((S_star < S_variable) and (S_variable > 1E0 / 2E0 / np.pi * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))):
        return S_variable - magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad) - (delta(mlat_rad) + epsilon(mlat_rad)) * (np.sqrt(2E0 * np.pi * S_variable) - np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))**2E0
    elif (((0E0 <= S_variable) and (S_variable <= S_star) and (Xi_S(S_variable) <= np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))) or ((S_star < S_variable) and (S_variable <= 1E0 / 2E0 / np.pi * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))):
        return S_variable - magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad)
    else:
        return np.nan
    
def solve_minus_upper_energy_S(mlat_rad):
    S_min = magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad)
    S_variable_array = np.linspace(S_min, S_star, 100)
    function_minus_upper_energy_S_array = np.zeros(len(S_variable_array))
    for count_i in range(len(S_variable_array)):
        function_minus_upper_energy_S_array[count_i] = function_minus_upper_energy_S(mlat_rad, S_variable_array[count_i])
    max_function_minus_upper_energy_S = np.max(function_minus_upper_energy_S_array)
    max_S = S_variable_array[np.argmax(function_minus_upper_energy_S_array)]
    min_function_minus_upper_energy_S = np.min(function_minus_upper_energy_S_array)
    min_S = S_variable_array[np.argmin(function_minus_upper_energy_S_array)]

    S_solution_under_S_star = np.nan
    minus_upper_energy_limit_under_S_star = np.nan

    if max_function_minus_upper_energy_S == 0E0:
        if (Xi_S(max_S) <= np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))):
            S_solution_under_S_star = S_min
        else:
            S_solution_under_S_star = max_S
    
    elif min_function_minus_upper_energy_S == 0E0:
        if (Xi_S(min_S) <= np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))):
            S_solution_under_S_star = S_min
        else:
            S_solution_under_S_star = min_S
    
    elif function_minus_upper_energy_S(mlat_rad, S_star) == 0E0:
        if (Xi_S(S_star) <= np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))):
            S_solution_under_S_star = S_min
        else:
            S_solution_under_S_star = S_star
    
    elif function_minus_upper_energy_S(mlat_rad, S_min) == 0E0:
        S_solution_under_S_star = S_min
    
    elif max_function_minus_upper_energy_S * min_function_minus_upper_energy_S < 0E0:
        #Newton法で解を求める
        initial_S_variable = (max_S + min_S) / 2E0
        S_variable_before_update = initial_S_variable
        count_iteration = 0
        while True:
            diff = function_minus_upper_energy_S(mlat_rad, S_variable_before_update) / gradient_S(function_minus_upper_energy_S, mlat_rad, S_variable_before_update)
            if np.abs(diff) > 1E-2:
                diff = np.sign(diff) * 1E-2
            S_variable_after_update = S_variable_before_update - diff
            if abs(S_variable_after_update - S_variable_before_update) < 1E-7:
                break
            elif S_variable_after_update != S_variable_after_update:
                print('Error!')
                quit()
            else:
                S_variable_before_update = S_variable_after_update
                count_iteration += 1
        S_solution_under_S_star = S_variable_after_update

    minus_upper_energy_limit_under_S_star = energy_minus_upper_limit(mlat_rad, S_solution_under_S_star)
    #print(S_solution_under_S_star, minus_upper_energy_limit_under_S_star/elementary_charge)

    S_solution_upper_S_star = np.nan
    minus_upper_energy_limit_upper_S_star = np.nan

    #S_star以上で解があるかどうかを調べる
    coefficient_a = 1E0 - (delta(mlat_rad) + epsilon(mlat_rad)) * 2E0 * np.pi
    coefficient_b = + 2E0 * (delta(mlat_rad) + epsilon(mlat_rad)) * np.sqrt(2E0 * np.pi * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad))
    coefficient_c = - magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad) - (delta(mlat_rad) + epsilon(mlat_rad)) * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)
    #coefficient_a * sqrt(S)**2 + coefficient_b * sqrt(S) + coefficient_c = 0
    discriminant = coefficient_b**2E0 - 4E0 * coefficient_a * coefficient_c
    if discriminant >= 0E0:
        S_solution_plus = (- coefficient_b + np.sqrt(discriminant)) / 2E0 / coefficient_a
        S_solution_minus = (- coefficient_b - np.sqrt(discriminant)) / 2E0 / coefficient_a
        if S_solution_plus < 0E0 or S_solution_plus**2E0 < S_star:
            S_solution_plus = np.nan
        if S_solution_minus < 0E0 or S_solution_minus**2E0 < S_star:
            S_solution_minus = np.nan
        S_solution_plus = S_solution_plus**2E0
        S_solution_minus = S_solution_minus**2E0
        S_solution_upper_S_star = np.nanmin([S_solution_plus, S_solution_minus])
        if S_solution_upper_S_star == S_solution_upper_S_star:
            minus_upper_energy_limit_upper_S_star = energy_minus_upper_limit(mlat_rad, S_solution_upper_S_star)
    
    #print(S_solution_upper_S_star, minus_upper_energy_limit_upper_S_star/elementary_charge)
    
    if S_solution_under_S_star != S_solution_under_S_star and S_solution_upper_S_star != S_solution_upper_S_star:
        if function_minus_upper_energy_S(mlat_rad, S_min) <= 0E0:
            return np.inf, np.inf
        else:
            return magnetic_flux_density(mlat_rad) * mu_upper_limit / energy_wave_potential(mlat_rad) * delta(mlat_rad), 0E0
    
    elif S_solution_under_S_star == S_solution_under_S_star and S_solution_upper_S_star != S_solution_upper_S_star:
        if function_minus_upper_energy_S(mlat_rad, S_min) <= 0E0:
            return S_solution_under_S_star, minus_upper_energy_limit_under_S_star
        else:
            return np.inf, np.inf
    
    elif S_solution_under_S_star != S_solution_under_S_star and S_solution_upper_S_star == S_solution_upper_S_star:
        if function_minus_upper_energy_S(mlat_rad, S_star) <= 0E0:
            return S_solution_upper_S_star, minus_upper_energy_limit_upper_S_star
        else:
            return np.inf, np.inf
    
    elif S_solution_under_S_star == S_solution_under_S_star and S_solution_upper_S_star == S_solution_upper_S_star:
        if function_minus_upper_energy_S(mlat_rad, S_min) <= 0E0:
            return S_solution_under_S_star, minus_upper_energy_limit_under_S_star
        elif function_minus_upper_energy_S(mlat_rad, S_min) > 0E0:
            return S_solution_upper_S_star, minus_upper_energy_limit_upper_S_star
            
        
# energy lower limit
def energy_minus_lower_limit(mlat_rad, S_variable):
    return 0E0

# function_upper_energy_SをSについてplotする

#input_deg = 7E0
#input_rad = input_deg / 180E0 * np.pi
#S_variable_array = np.linspace(0E0, 1E1, 10000)
#function_upper_energy_S_array = np.zeros(len(S_variable_array))
#function_lower_energy_S_array = np.zeros(len(S_variable_array))
#function_minus_upper_energy_S_array = np.zeros(len(S_variable_array))
#energy_upper_energy_S_array = np.zeros(len(S_variable_array))
#energy_lower_energy_S_array = np.zeros(len(S_variable_array))
#energy_minus_upper_energy_S_array = np.zeros(len(S_variable_array))
#energy_S_array = np.zeros(len(S_variable_array))
#for count_i in range(len(S_variable_array)):
#    function_upper_energy_S_array[count_i] = function_upper_energy_S(input_rad, S_variable_array[count_i])
#    function_minus_upper_energy_S_array[count_i] = function_minus_upper_energy_S(input_rad, S_variable_array[count_i])
#    function_lower_energy_S_array[count_i] = function_lower_energy_S(input_rad, S_variable_array[count_i])
#    energy_upper_energy_S_array[count_i] = energy_upper_limit(input_rad, S_variable_array[count_i])
#    energy_lower_energy_S_array[count_i] = energy_lower_limit(input_rad, S_variable_array[count_i])
#    energy_minus_upper_energy_S_array[count_i] = energy_minus_upper_limit(input_rad, S_variable_array[count_i])
#    energy_S_array[count_i] = (S_variable_array[count_i] * energy_wave_potential(input_rad) - magnetic_flux_density(input_rad) * mu_upper_limit * delta(input_rad)) / (delta(input_rad) + epsilon(input_rad))
#
#fig = plt.figure(figsize=(14, 14), dpi=100)
#ax = fig.add_subplot(111)
##ax.plot(S_variable_array, function_upper_energy_S_array)
##ax.plot(S_variable_array, function_minus_upper_energy_S_array)
##ax.plot(S_variable_array, function_lower_energy_S_array)
#ax.plot(energy_S_array / elementary_charge, energy_upper_energy_S_array / elementary_charge, c='orange', linewidth=4E0)
#ax.plot(energy_S_array / elementary_charge, energy_lower_energy_S_array / elementary_charge, c='blue', linewidth=4E0)
#ax.plot(energy_S_array / elementary_charge, energy_minus_upper_energy_S_array / elementary_charge, c='red', linewidth=4E0)
#ax.plot(energy_S_array / elementary_charge, energy_S_array / elementary_charge, c='black', linewidth=4E0)
##ax.set_xlabel(r'S', fontsize=20)
#ax.minorticks_on()
#ax.grid(which='both', alpha=0.3)
#plt.tight_layout()
#plt.show()
#
#
#quit()

# calculation

mlat_deg_array = np.linspace(0E0, 50E0, 1000)
mlat_rad_array = mlat_deg_array / 180E0 * np.pi

S_limit_upper_energy = np.zeros(len(mlat_deg_array))
energy_upper_limit_S = np.zeros(len(mlat_deg_array))
S_limit_lower_energy = np.zeros(len(mlat_deg_array))
energy_lower_limit_S = np.zeros(len(mlat_deg_array))
S_limit_minus_upper_energy = np.zeros(len(mlat_deg_array))
energy_minus_upper_limit_S = np.zeros(len(mlat_deg_array))
energy_minus_lower_limit_S = np.zeros(len(mlat_deg_array))

for count_i in range(len(mlat_deg_array)):
    S_limit_upper_energy[count_i], energy_upper_limit_S[count_i] = solve_upper_energy_S(mlat_rad_array[count_i])
    if S_limit_upper_energy[count_i] != S_limit_upper_energy[count_i]:
        break
    elif S_limit_upper_energy[count_i] == np.inf:
        S_limit_upper_energy[count_i] = 1E100
        energy_upper_limit_S[count_i] = 1E100
    S_limit_lower_energy[count_i], energy_lower_limit_S[count_i] = solve_lower_energy_S(mlat_rad_array[count_i])

    S_limit_minus_upper_energy[count_i], energy_minus_upper_limit_S[count_i] = solve_minus_upper_energy_S(mlat_rad_array[count_i])
    #print(S_limit_minus_upper_energy[count_i], energy_minus_upper_limit_S[count_i]/elementary_charge)
    if S_limit_minus_upper_energy[count_i] != S_limit_minus_upper_energy[count_i]:
        break
    elif S_limit_minus_upper_energy[count_i] == np.inf:
        S_limit_minus_upper_energy[count_i] = 1E100
        energy_minus_upper_limit_S[count_i] = 1E100
    energy_minus_lower_limit_S[count_i] = energy_minus_lower_limit(mlat_rad_array[count_i], S_limit_minus_upper_energy)
    now = datetime.datetime.now()
    print(now, count_i, len(mlat_deg_array) - 1, mlat_deg_array[count_i], S_limit_upper_energy[count_i], energy_upper_limit_S[count_i]/elementary_charge, S_limit_lower_energy[count_i], energy_lower_limit_S[count_i]/elementary_charge, S_limit_minus_upper_energy[count_i], energy_minus_upper_limit_S[count_i]/elementary_charge, energy_minus_lower_limit_S[count_i]/elementary_charge)



fig = plt.figure(figsize=(14, 14), dpi=100)
ax = fig.add_subplot(111, xlabel=r'MLAT [deg]', ylabel=r'Energy [eV]', xlim=(0E0, 50E0), ylim=(1E1, 1E5), yscale='log')
ax.set_title(r'$K \mathrm{cos}^2 \alpha$')

# energy_upper_limit_Sとenergy_lower_limit_Sの間のエネルギー領域を塗りつぶす
ax.fill_between(mlat_deg_array, energy_lower_limit_S / elementary_charge, energy_upper_limit_S / elementary_charge, facecolor='yellow', alpha=0.2)
ax.fill_between(mlat_deg_array, energy_minus_lower_limit_S / elementary_charge, energy_minus_upper_limit_S / elementary_charge, facecolor='aqua', alpha=0.2)

ax.plot(mlat_deg_array, energy_upper_limit_S / elementary_charge, color='orange', linewidth=4E0, alpha=0.5, label=r'plus energy limit')
ax.plot(mlat_deg_array, energy_lower_limit_S / elementary_charge, color='orange', linewidth=4E0, alpha=0.5)
ax.plot(mlat_deg_array, energy_minus_upper_limit_S / elementary_charge, color='dodgerblue', linewidth=4E0, alpha=0.5, label=r'minus energy limit')
ax.plot(mlat_deg_array, energy_wave_phase_speed(mlat_rad_array) / elementary_charge, color='red', linewidth=4E0, alpha=0.5, label=r'$K_{\mathrm{ph \parallel}}$')
ax.plot(mlat_deg_array, energy_wave_potential(mlat_rad_array) * np.ones(len(mlat_rad_array)) / elementary_charge, color='green', linewidth=4E0, alpha=0.5, label=r'$K_{\mathrm{E}}$')

ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

ax.legend()

plt.tight_layout()
#plt.savefig('/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_under_1_range_Earth_L_9.png', dpi=100)
#plt.savefig('/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_under_1_range_Earth_L_9.pdf', dpi=100)
plt.show()
