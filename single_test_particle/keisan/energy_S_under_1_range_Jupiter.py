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
planet_radius   = 7.1492E7  #[m]
lshell_number   = 5.91E0
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

# magnetic field

dipole_moment   = 4.2E-4 * (4E0 * np.pi * planet_radius**3E0) / magnetic_constant #[Am]
B0_eq           = (1E-7 * dipole_moment) / r_eq**3E0

def magnetic_flux_density(mlat_rad):
    return B0_eq / np.cos(mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)     #[T]

# background plasma parameters
oxygen_number_density_eq = 1.163E9    # [m^-3]
oxygen_number_density_scale_height = 1.2842    # [R_J]
oxygen_mass = 2.677950266103E-26   # [kg] # O+
oxygen_temperature = 5E1   # [eV]

proton_number_density_eq = 5.8E7    # [m^-3]
proton_mass = 1.672621898E-27    # [kg]
proton_temperature = 8.6E0    # [eV]

electron_mass = 9.10938356E-31    # [kg]
electron_temperature_eq = 5E0  # [eV]

def coordinate_FA(mlat_rad):
    return r_eq * (np.arcsinh(np.sqrt(3)*np.sin(mlat_rad)) / 2E0 / np.sqrt(3) + np.sin(mlat_rad) * np.sqrt(5E0 - 3E0 * np.cos(2E0 * mlat_rad)) / 2E0 / np.sqrt(2)) #[m]

def oxygen_number_density(mlat_rad):
    return oxygen_number_density_eq * np.exp(- (coordinate_FA(mlat_rad) / planet_radius / oxygen_number_density_scale_height)**2E0)    #[m^-3]

def proton_number_density(mlat_rad):
    return proton_number_density_eq     #[m^-3]

def number_density(mlat_rad):
    return oxygen_number_density(mlat_rad) + proton_number_density(mlat_rad)    #[m^-3]

def ion_temperature(mlat_rad):
    oxygen_pressure = oxygen_number_density(mlat_rad) * oxygen_temperature * elementary_charge    #[Pa]
    proton_pressure = proton_number_density(mlat_rad) * proton_temperature * elementary_charge    #[Pa]
    return (oxygen_pressure + proton_pressure) / (oxygen_number_density(mlat_rad) + proton_number_density(mlat_rad)) / elementary_charge    #[eV]

def tau(mlat_rad):
    return ion_temperature(mlat_rad) / electron_temperature_eq

def ion_mass(mlat_rad):
    return (oxygen_mass * oxygen_number_density(mlat_rad) + proton_mass * proton_number_density(mlat_rad)) / number_density(mlat_rad)    #[kg]

def ion_thermal_speed(mlat_rad):
    return np.sqrt(2E0 * ion_temperature(mlat_rad) * elementary_charge / ion_mass(mlat_rad))    #[m/s]

def ion_gyrofrequency(mlat_rad):
    return elementary_charge * magnetic_flux_density(mlat_rad) / ion_mass(mlat_rad)    #[s^-1]

def ion_gyroradius(mlat_rad):
    return ion_thermal_speed(mlat_rad) / ion_gyrofrequency(mlat_rad)    #[m]

def electron_thermal_speed(mlat_rad):
    return np.sqrt(2E0 * electron_temperature_eq * elementary_charge / electron_mass)    #[m/s]



def Alfven_speed(mlat_rad):
    return magnetic_flux_density(mlat_rad) / np.sqrt(magnetic_constant * number_density(mlat_rad) * ion_mass(mlat_rad))    #[m/s]

def plasma_beta_ion(mlat_rad):
    return 2E0 * magnetic_constant * number_density(mlat_rad) * ion_temperature(mlat_rad) * elementary_charge / magnetic_flux_density(mlat_rad)**2E0  #[]

mu_upper_limit = 1E3 * elementary_charge / magnetic_flux_density(0E0)    #[J T-1]


# wave parameters
uperp_eq = 5.4E4    #[m/s]
#kperp_rhoi = ion_thermal_speed(0E0) / uperp_eq    #[rad]
epsilon_wave = 1E0 / 1000E0
kperp_rhoi = 2E0 * np.pi #[rad]

def kperp(mlat_rad):
    return kperp_rhoi / ion_gyroradius(mlat_rad)    #[rad]

def wave_length_perp(mlat_rad):
    return 2E0 * np.pi / kperp(mlat_rad)    #[m]

#kpara_eq = ion_gyrofrequency(0E0) / electron_thermal_speed(0E0)    #[rad m-1]
kpara_eq = epsilon_wave * kperp(0E0)    #[rad m-1]

wave_frequency = np.sqrt((1E0 + tau(0E0)) / (plasma_beta_ion(0E0) * (1E0 + tau(0E0)) + 2E0 * tau(0E0))) * kperp_rhoi * kpara_eq * Alfven_speed(0E0)    #[rad/s]


def kpara(mlat_rad):
    return wave_frequency / Alfven_speed(mlat_rad) / kperp_rhoi * np.sqrt(plasma_beta_ion(mlat_rad) + 2E0 * tau(mlat_rad) / (1E0 + tau(mlat_rad)))    #[rad/m]

def wave_length_para(mlat_rad):
    return 2E0 * np.pi / kpara(mlat_rad)    #[m]

def wave_phase_speed(mlat_rad):
    return wave_frequency / kpara(mlat_rad)    #[m/s]

electric_field_eq = 1E-3    #[V/m]

def wave_modified_potential(mlat_rad):
    return electric_field_eq * wave_length_para(0E0)    #[V]

def wave_scalar_potential(mlat_rad):
    return wave_modified_potential(mlat_rad) / (2E0 + 1E0 / tau(mlat_rad))    #[V]

def energy_wave_phase_speed(mlat_rad):
    return 5E-1 * electron_mass * wave_phase_speed(mlat_rad)**2E0 #[J]

def energy_wave_potential(mlat_rad):
    return elementary_charge * wave_modified_potential(mlat_rad)    #[J]

def diff_B0(mlat_rad):
    return gradient_meter(magnetic_flux_density, mlat_rad) / kpara(mlat_rad) / magnetic_flux_density(mlat_rad) #[rad^-1]

def diff_ion_number_density(mlat_rad):
    return gradient_meter(number_density, mlat_rad) / kpara(mlat_rad) / number_density(mlat_rad) #[rad^-1]

def diff_ion_temperature(mlat_rad):
    return gradient_meter(ion_temperature, mlat_rad) / kpara(mlat_rad) / ion_temperature(mlat_rad) #[rad^-1]

def diff_1_plus_tau(mlat_rad):
    return gradient_meter(tau, mlat_rad) / kpara(mlat_rad) / (1E0 + tau(mlat_rad)) #[rad^-1]

def delta(mlat_rad):
    return diff_B0(mlat_rad)    #[rad^-1]

def yy(mlat_rad):
    return plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]

def epsilon(mlat_rad):
    return (1E0 + 2E0 * yy(mlat_rad)) * diff_B0(mlat_rad) - (1E0 + yy(mlat_rad)) * diff_ion_number_density(mlat_rad) - yy(mlat_rad) * diff_ion_temperature(mlat_rad) - yy(mlat_rad) / plasma_beta_ion(mlat_rad) / (1E0 + tau(mlat_rad)) * diff_1_plus_tau(mlat_rad)    #[rad^-1]

#must reconsider the setting
print(wave_frequency)
print(uperp_eq)
print(Alfven_speed(0E0))
print(ion_thermal_speed(0E0))
print(electron_thermal_speed(0E0))
print(plasma_beta_ion(0E0))
print(ion_gyrofrequency(0E0))
print(wave_length_para(0E0))
print(wave_length_perp(0E0))
print(kpara(0E0)/kperp(0E0))
print(wave_scalar_potential(0E0))
print(wave_modified_potential(0E0))
#print(kperp_rhoi)
#quit()

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


# calculation

mlat_deg_array = np.linspace(0E0, 50E0, 1000)
mlat_rad_array = mlat_deg_array / 180E0 * np.pi

S_limit_upper_energy = np.zeros(len(mlat_deg_array))
energy_upper_limit_S = np.zeros(len(mlat_deg_array))
S_limit_lower_energy = np.zeros(len(mlat_deg_array))
energy_lower_limit_S = np.zeros(len(mlat_deg_array))

for count_i in range(len(mlat_deg_array)):
    S_limit_upper_energy[count_i] = Newton_method_function_upper_energy_S(mlat_rad_array[count_i])
    energy_upper_limit_S[count_i] = energy_upper_limit(mlat_rad_array[count_i], S_limit_upper_energy[count_i])
    S_limit_lower_energy[count_i] = Newton_method_function_lower_energy_S(mlat_rad_array[count_i])
    #print(energy_upper_limit_S[count_i])
    if energy_upper_limit_S[count_i] != energy_upper_limit_S[count_i]:
        energy_lower_limit_S[count_i] = np.nan
    else:
        energy_lower_limit_S[count_i] = energy_lower_limit(mlat_rad_array[count_i], S_limit_lower_energy[count_i])
    now = datetime.datetime.now()
    #print(count_i+1, len(mlat_deg_array), now, mlat_deg_array[count_i], S_limit_lower_energy[count_i], energy_lower_limit_S[count_i] / elementary_charge, function_lower_energy_S_border(mlat_rad_array[count_i], S_limit_lower_energy[count_i]))

upper_energy_trapped_S1 = np.zeros(len(mlat_deg_array))
for count_i in range(len(mlat_deg_array)):
    upper_energy_trapped_S1[count_i] = energy_wave_potential(mlat_rad_array[count_i]) / (delta(mlat_rad_array[count_i]) + epsilon(mlat_rad_array[count_i]))

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

fig = plt.figure(figsize=(14, 14), dpi=100)
ax = fig.add_subplot(111, xlabel=r'MLAT [deg]', ylabel=r'Energy [eV]', xlim=(0E0, 50E0), ylim=(1E0, 1E5), yscale='log')
ax.set_title(r'$K \mathrm{cos}^2 \alpha$')

# energy_upper_limit_Sとenergy_lower_limit_Sの間のエネルギー領域を塗りつぶす
ax.fill_between(mlat_deg_array, energy_lower_limit_S / elementary_charge, energy_upper_limit_S / elementary_charge, facecolor='yellow', alpha=0.2)

ax.plot(mlat_deg_array, energy_upper_limit_S / elementary_charge, color='orange', linewidth=4E0, alpha=0.5, label=r'energy limit')
ax.plot(mlat_deg_array, energy_lower_limit_S / elementary_charge, color='orange', linewidth=4E0, alpha=0.5)
ax.plot(mlat_deg_array, energy_wave_phase_speed(mlat_rad_array) / elementary_charge, color='red', linewidth=4E0, alpha=0.5, label=r'$K_{\mathrm{ph \parallel}}$')
ax.plot(mlat_deg_array, energy_wave_potential(mlat_rad_array) * np.ones(len(mlat_rad_array)) / elementary_charge, color='green', linewidth=4E0, alpha=0.5, label=r'$K_{\mathrm{E}}$')
ax.plot(mlat_deg_array, upper_energy_trapped_S1 / elementary_charge, color='blue', linewidth=4E0, alpha=0.5, label=r'$S < 1$ range')

ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

ax.legend()

plt.tight_layout()

plt.savefig('/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_under_1_range_Jupiter_L_5.91.png', dpi=100)
#plt.show()
