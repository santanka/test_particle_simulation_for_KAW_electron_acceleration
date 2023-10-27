import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import datetime
import os

# constants
speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]
electric_constant = 8.8541878128E-12  #[F m-1]
magnetic_constant = 1.25663706212E-6  #[N A-2]

#planet condition
planet_radius   = 6378.1E3  #[m]
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

#上部
# energy upper limit

def Pi_S(S_variable):
    if np.abs(S_variable) > 1E0:
        return np.nan
    else:
        return np.sqrt(np.sqrt(1E0 - S_variable**2E0) - 5E-1 * S_variable * (np.pi - 2E0 * np.arcsin(S_variable)))

diff_S = 1E-5
def gradient_S(function_S, mlat_rad, S_variable):
    return (function_S(mlat_rad, S_variable + diff_S) - function_S(mlat_rad, S_variable - diff_S)) / 2E0 / diff_S

def function_upper_energy_S(mlat_rad, S_variable):
    return S_variable - (delta(mlat_rad) + epsilon(mlat_rad)) * (np.sqrt(2E0) * Pi_S(S_variable) + np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))**2E0

def Newton_method_function_upper_energy_S(mlat_rad):
    initial_S_variable = 0E0
    S_variable_before_update = initial_S_variable
    count_iteration = 0
    while True:
        diff = function_upper_energy_S(mlat_rad, S_variable_before_update) / gradient_S(function_upper_energy_S, mlat_rad, S_variable_before_update)
        S_variable_after_update = S_variable_before_update - diff
        if abs(S_variable_after_update - S_variable_before_update) < 1E-7:
            break
        else:
            S_variable_before_update = S_variable_after_update
            count_iteration += 1
            if S_variable_after_update > 1E0 or S_variable_after_update < -1E0:
                S_variable_after_update = np.nan
                break
            if count_iteration > 100000:
                S_variable_after_update = np.nan
                break
    return S_variable_after_update

def energy_upper_limit(mlat_rad, S_variable):
    return energy_wave_potential(mlat_rad) / (delta(mlat_rad) + epsilon(mlat_rad)) * S_variable    #[J]


# energy lower limit

def trapping_frequency(mlat_rad):
    return kpara(mlat_rad) * np.sqrt(energy_wave_potential(mlat_rad) / electron_mass) #[rad/s]

def function_lower_energy_S(mlat_rad, S_variable):
    return S_variable - (delta(mlat_rad) + epsilon(mlat_rad)) * (np.sqrt(2E0) * Pi_S(S_variable) - np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))**2E0


def Newton_method_function_lower_energy_S(mlat_rad):
    initial_S_variable = 0E0
    S_variable_before_update = initial_S_variable
    count_iteration = 0
    while True:
        diff = function_lower_energy_S(mlat_rad, S_variable_before_update) / gradient_S(function_lower_energy_S, mlat_rad, S_variable_before_update)
        S_variable_after_update = S_variable_before_update - diff
        if abs(S_variable_after_update - S_variable_before_update) < 1E-7:
            break
        else:
            S_variable_before_update = S_variable_after_update
            count_iteration += 1
            if S_variable_after_update > 1E0 or S_variable_after_update < -1E0:
                S_variable_after_update = np.nan
                break
    
    if Pi_S(S_variable_after_update) >= np.sqrt(energy_wave_phase_speed(mlat_rad) / 2E0 / energy_wave_potential(mlat_rad)):
        S_variable_after_update = 0E0
    
    return S_variable_after_update

def energy_lower_limit(mlat_rad, S_variable):
    return energy_wave_potential(mlat_rad) / (delta(mlat_rad) + epsilon(mlat_rad)) * S_variable    #[J]


# 下部
# energy upper limit

def function_minus_upper_energy_S(mlat_rad, S_variable):
    return S_variable - (delta(mlat_rad) + epsilon(mlat_rad)) * (np.sqrt(2E0) * Pi_S(S_variable) - np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)))**2E0

def Newton_method_function_minus_upper_energy_S(mlat_rad):
    initial_S_variable = 0E0
    S_variable_before_update = initial_S_variable
    count_iteration = 0
    while True:
        diff = function_minus_upper_energy_S(mlat_rad, S_variable_before_update) / gradient_S(function_minus_upper_energy_S, mlat_rad, S_variable_before_update)
        S_variable_after_update = S_variable_before_update - diff
        if abs(S_variable_after_update - S_variable_before_update) < 1E-7:
            break
        else:
            S_variable_before_update = S_variable_after_update
            count_iteration += 1
            if S_variable_after_update > 1E0 or S_variable_after_update < -1E0:
                S_variable_after_update = np.nan
                break
    
    if Pi_S(S_variable_after_update) <= np.sqrt(energy_wave_phase_speed(mlat_rad) / 2E0 / energy_wave_potential(mlat_rad)):
        S_variable_after_update = 0E0

    return S_variable_after_update

def energy_minus_upper_limit(mlat_rad, S_variable):
    return energy_wave_potential(mlat_rad) / (delta(mlat_rad) + epsilon(mlat_rad)) * S_variable    #[J]

# energy lower limit
def energy_minus_lower_limit(mlat_rad):
    return 0E0

# calculation

mlat_deg_array = np.linspace(0E0, 50E0, 10000)
mlat_rad_array = mlat_deg_array / 180E0 * np.pi

#delta_array = np.zeros(len(mlat_deg_array))
#epsilon_array = np.zeros(len(mlat_deg_array))
#
#for count_i in range(len(mlat_deg_array)):
#    delta_array[count_i] = delta(mlat_rad_array[count_i])
#    epsilon_array[count_i] = epsilon(mlat_rad_array[count_i])
#
##delta_arrayとepsilon_arrayをplot
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.serif'] = ['Computer Modern Roman']
#mpl.rcParams['mathtext.fontset'] = 'cm'
#plt.rcParams["font.size"] = 35
#
#fig = plt.figure(figsize=(14, 14), dpi=100)
#ax = fig.add_subplot(111, xlabel=r'MLAT [deg]', xlim=(0E0, 50E0), yscale='log')
#ax.plot(mlat_deg_array, delta_array, color='red', linewidth=4E0, alpha=0.5, label=r'$\delta$')
#ax.plot(mlat_deg_array, epsilon_array, color='blue', linewidth=4E0, alpha=0.5, label=r'$\varepsilon$')
#ax.minorticks_on()
#ax.grid(which='both', alpha=0.3)
#ax.legend()
#plt.tight_layout()
#plt.show()
#
#quit()


S_limit_upper_energy = np.zeros(len(mlat_deg_array))
energy_upper_limit_S = np.zeros(len(mlat_deg_array))
S_limit_lower_energy = np.zeros(len(mlat_deg_array))
energy_lower_limit_S = np.zeros(len(mlat_deg_array))
S_limit_minus_upper_energy = np.zeros(len(mlat_deg_array))
energy_minus_upper_limit_S = np.zeros(len(mlat_deg_array))
energy_minus_lower_limit_S = np.zeros(len(mlat_deg_array))

for count_i in range(len(mlat_deg_array)):
    S_limit_upper_energy[count_i] = Newton_method_function_upper_energy_S(mlat_rad_array[count_i])
    energy_upper_limit_S[count_i] = energy_upper_limit(mlat_rad_array[count_i], S_limit_upper_energy[count_i])
    
    #print(energy_upper_limit_S[count_i])
    if energy_upper_limit_S[count_i] != energy_upper_limit_S[count_i]:
        energy_lower_limit_S[count_i] = np.nan
        energy_minus_upper_limit_S[count_i] = np.nan
        energy_minus_lower_limit_S[count_i] = np.nan
    else:
        S_limit_lower_energy[count_i] = Newton_method_function_lower_energy_S(mlat_rad_array[count_i])
        S_limit_minus_upper_energy[count_i] = Newton_method_function_minus_upper_energy_S(mlat_rad_array[count_i])
        energy_lower_limit_S[count_i] = energy_lower_limit(mlat_rad_array[count_i], S_limit_lower_energy[count_i])
        energy_minus_upper_limit_S[count_i] = energy_minus_upper_limit(mlat_rad_array[count_i], S_limit_minus_upper_energy[count_i])
        energy_minus_lower_limit_S[count_i] = energy_minus_lower_limit(mlat_rad_array[count_i])
    now = datetime.datetime.now()
    print(now, count_i, len(mlat_deg_array)-1, energy_upper_limit_S[count_i]/elementary_charge, energy_lower_limit_S[count_i]/elementary_charge, energy_minus_upper_limit_S[count_i]/elementary_charge, energy_minus_lower_limit_S[count_i]/elementary_charge)


upper_energy_trapped_S1 = np.zeros(len(mlat_deg_array))
for count_i in range(len(mlat_deg_array)):
    upper_energy_trapped_S1[count_i] = energy_wave_potential(mlat_rad_array[count_i]) / (delta(mlat_rad_array[count_i]) + epsilon(mlat_rad_array[count_i]))

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

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
ax.plot(mlat_deg_array, upper_energy_trapped_S1 / elementary_charge, color='blue', linewidth=4E0, alpha=0.5, label=r'$S < 1$ range')

ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

ax.legend()

plt.tight_layout()
plt.savefig('/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_under_1_range_Earth_L_9.png', dpi=100)
plt.savefig('/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_under_1_range_Earth_L_9.pdf', dpi=100)
#plt.show()
