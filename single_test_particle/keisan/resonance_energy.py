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

def wave_phase_speed_para(mlat_rad):
    return Alfven_speed(mlat_rad) * kperp_rhoi * np.sqrt((1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))) * np.sign(mlat_rad)    #[m/s]

def kpara(mlat_rad):
    if mlat_rad == 0E0:
        return np.nan
    else:
        return wave_frequency / wave_phase_speed_para(mlat_rad)    #[rad/m]

def energy_wave_phase_speed_para(mlat_rad, vperp):
    vpara = wave_phase_speed_para(mlat_rad)    #[m/s]
    Lorentz_factor = 1E0 / np.sqrt(1E0 - (vpara**2E0 + vperp**2E0) / speed_of_light**2E0)
    return electron_mass * speed_of_light**2E0 * (Lorentz_factor - 1E0)    #[J]

def kperp(mlat_rad):
    vthi = np.sqrt(ion_temperature(mlat_rad) * elementary_charge / ion_mass)    #[m/s]
    omega_i = elementary_charge * magnetic_flux_density(mlat_rad) / ion_mass    #[rad/s]
    rhoi = vthi / omega_i    #[m]
    return kperp_rhoi / rhoi    #[rad/m]

def wave_phase_speed_perp(mlat_rad):
    return wave_frequency / kperp(mlat_rad)    #[m/s]

def Landau_resonance_energy(mlat_rad):
    if mlat_rad == 0E0:
        return np.nan
    else:
        Vph = wave_phase_speed_para(mlat_rad)    #[m/s]
        energy = electron_mass / 2E0 * Vph**2E0 / elementary_charge    #[eV]
        return energy

def cyclotron_resonance_energy(mlat_rad, pitch_angle):
    if mlat_rad == 0E0:
        return np.nan, np.nan
    else:
        omega_e = elementary_charge * magnetic_flux_density(mlat_rad) / electron_mass    #[rad/s]
        kpara_def = kpara(mlat_rad)    #[rad/m]
        #2次方程式の解の公式
        coefficient_a = wave_frequency**2E0 - kpara_def**2E0 * speed_of_light**2E0 * np.cos(pitch_angle)**2E0
        coefficient_b = -2E0 * omega_e * wave_frequency
        coefficient_c = omega_e**2E0 + kpara_def**2E0 * speed_of_light**2E0 * np.cos(pitch_angle)**2E0
        Lorentz_factor_plus = (-coefficient_b + np.sqrt(coefficient_b**2E0 - 4E0 * coefficient_a * coefficient_c)) / 2E0 / coefficient_a
        Lorentz_factor_minus = (-coefficient_b - np.sqrt(coefficient_b**2E0 - 4E0 * coefficient_a * coefficient_c)) / 2E0 / coefficient_a
        if Lorentz_factor_plus > 1E0 and Lorentz_factor_minus > 1E0:
            Lorentz_factor = min(Lorentz_factor_plus, Lorentz_factor_minus)
        elif Lorentz_factor_plus > 1E0 and Lorentz_factor_minus < 1E0:
            Lorentz_factor = Lorentz_factor_plus
        elif Lorentz_factor_plus < 1E0 and Lorentz_factor_minus > 1E0:
            Lorentz_factor = Lorentz_factor_minus
        else:
            Lorentz_factor = np.nan
        return electron_mass * speed_of_light**2E0 * (Lorentz_factor - 1E0) / elementary_charge, Lorentz_factor    #[eV]

fig = plt.figure(figsize=(20, 20), dpi=100)
ax = fig.add_subplot(111, xlabel=r'$\mathrm{MLAT}$ [deg]', ylabel=r'$\mathrm{Energy}$ [eV]', xlim=(0E0, mlat_upper_limit_deg), yscale='log')

mlat_rad_array = np.linspace(0E0, mlat_upper_limit_rad, 1000)
energy_cyclotron_resonance_array = np.zeros(len(mlat_rad_array))
Lorentz_factor_cyclotron_resonance_array = np.zeros(len(mlat_rad_array))
energy_Landau_resonance_array = np.zeros(len(mlat_rad_array))
for count_i in range(len(mlat_rad_array)):
    mlat_rad = mlat_rad_array[count_i]
    energy_cyclotron_resonance_array[count_i], Lorentz_factor_cyclotron_resonance_array[count_i] = cyclotron_resonance_energy(mlat_rad, 0)
    energy_Landau_resonance_array[count_i] = Landau_resonance_energy(mlat_rad)

ax.plot(mlat_rad_array * 180E0 / np.pi, energy_cyclotron_resonance_array, color='b', lw=4, label=r'$\mathrm{Cyclotron}$')
ax.plot(mlat_rad_array * 180E0 / np.pi, energy_Landau_resonance_array, color='r', lw=4, label=r'$\mathrm{Landau}$')

ax.legend(loc='upper left', fontsize=30)
#y軸のメモリの設定: 10^0, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6...
ax.set_yticks([10**i for i in range(14)])
ax.minorticks_on()
ax.grid(which='both', alpha=0.3)
fig.tight_layout()

fig_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/resonance_energy'
fig.savefig(fig_name + '.png')
fig.savefig(fig_name + '.pdf')

plt.close()
fig = plt.figure(figsize=(20, 20), dpi=100)
ax = fig.add_subplot(111, xlabel=r'$\mathrm{MLAT}$ [deg]', ylabel=r'$\mathrm{Lorentz}$ $\mathrm{factor}$', xlim=(0E0, mlat_upper_limit_deg), yscale='log')
ax.plot(mlat_rad_array * 180E0 / np.pi, Lorentz_factor_cyclotron_resonance_array, color='b', lw=4, label=r'$\mathrm{Cyclotron}$')
ax.minorticks_on()
ax.grid(which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(fig_name + '_Lorentz_factor.png')