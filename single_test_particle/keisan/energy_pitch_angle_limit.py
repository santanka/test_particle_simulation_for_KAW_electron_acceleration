import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# magnetic latitude
mlat_max_deg = 50.0   # [deg]
mlat_min_deg = 0.0    # [deg]
mlat_deg = np.linspace(mlat_min_deg, mlat_max_deg, 10000)   # [deg]
mlat_rad = mlat_deg * np.pi / 180.0   # [rad]


# constants
speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]
electric_constant = 8.8541878128E-12  #[F m-1]
magnetic_constant = 1.25663706212E-6  #[N A-2]


# background plasma parameters
number_density = 1E6    # [m^-3]
ion_mass = 1.672621898E-27   # [kg]
ion_temperature = 1E3   # [eV]
electron_mass = 9.10938356E-31    # [kg]
electron_temperature = 1E2  # [eV]

# magnetic field
planet_radius   = 6371E3  #[m]
lshell_number   = 9E0
r_eq            = planet_radius * lshell_number #[m]
dipole_moment   = 7.75E22 #[Am]
B0_eq           = (1E-7 * dipole_moment) / r_eq**3E0

b0 = B0_eq / np.cos(mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)     #[T]
db0_dz = 3E0 * np.sin(mlat_rad) * (5E0 * np.sin(mlat_rad)**2E0 + 3E0) / np.cos(mlat_rad)**8E0 / (3E0 * np.sin(mlat_rad)**2E0 + 1E0) / r_eq * B0_eq   #[T/m]


Alfven_speed = b0 / np.sqrt(magnetic_constant * number_density * ion_mass)    #[m/s]
plasma_beta_ion = 2E0 * magnetic_constant * number_density * ion_temperature * elementary_charge / b0**2E0  #[]

# wave parameters
kperp_rhoi = 2E0 * np.pi    #[rad]
wave_phase_speed = Alfven_speed * kperp_rhoi * np.sqrt((ion_temperature + electron_temperature) / (plasma_beta_ion * (ion_temperature + electron_temperature) + 2E0 * ion_temperature))    #[m/s]
wave_phase = 2E0 * np.pi * 0.5    #[rad/s]
kpara = wave_phase / wave_phase_speed    #[rad/m]
wave_modified_potential = 1E-3 / kpara[0]  #[V]
#wave_scalar_potential = 600E0   #[V]
#wave_modified_potential = wave_scalar_potential * (2E0 + electron_temperature / ion_temperature)    #[V]

energy_wave_phase_speed = 5E-1 * electron_mass * wave_phase_speed**2E0  #[J]
energy_wave_potential = elementary_charge * wave_modified_potential     #[J]

delta  = 1E0 / kpara / b0 * db0_dz    #[rad^-1]
epsilon = delta * (3E0 - 4E0 * ion_temperature / (plasma_beta_ion * (ion_temperature + electron_temperature) + 2E0 * ion_temperature))    #[rad^-1]


# solution
Ke_plus_Kphpara = energy_wave_potential + energy_wave_phase_speed
Ke_minus_Kphpara = energy_wave_potential - energy_wave_phase_speed
Ke_plus_3Kphpara = energy_wave_potential + 3E0 * energy_wave_phase_speed

cos_2_pitch_angle_limit = delta / epsilon * Ke_plus_Kphpara * Ke_plus_3Kphpara / (8E0 / np.pi / epsilon * energy_wave_potential * energy_wave_phase_speed - Ke_plus_Kphpara * Ke_plus_3Kphpara)
cos_2_pitch_angle_limit[cos_2_pitch_angle_limit < 0E0] = 0E0
cos_2_pitch_angle_limit[cos_2_pitch_angle_limit > 1E0] = np.nan
cos_pitch_angle_limit = np.sqrt(cos_2_pitch_angle_limit)
pitch_angle_limit_rad = np.arccos(cos_pitch_angle_limit)   #[rad]
pitch_angle_limit_deg = pitch_angle_limit_rad * 180.0 / np.pi   #[deg]
energy_limit_eV = ((np.pi / 2E0 * epsilon - 1E0) * cos_pitch_angle_limit**2E0 + np.pi / 2E0 * delta)**(-2E0) * ((np.pi / 2E0 * epsilon * Ke_plus_Kphpara - Ke_minus_Kphpara) * cos_pitch_angle_limit**2E0 + np.pi / 2E0 * delta * Ke_plus_Kphpara) / elementary_charge    #[eV]
energy_limit_eV[energy_limit_eV < 0E0] = 0E0

equatorial_pitch_angle_limit_rad = np.arcsin(np.sqrt(B0_eq / b0) * np.sin(pitch_angle_limit_rad))   #[rad]
equatorial_pitch_angle_limit_deg = equatorial_pitch_angle_limit_rad * 180.0 / np.pi   #[deg]

# plot

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

fig = plt.figure(figsize=(14, 14), dpi=100, tight_layout=True)
ax1 = fig.add_subplot(211, xlabel=r'MLAT [deg]', ylabel=r'Pitch Angle [deg]')
ax1.plot(mlat_deg, equatorial_pitch_angle_limit_deg, color='b', linewidth='4', label=r'Equatorial Pitch Angle Limit')
ax1.plot(mlat_deg, pitch_angle_limit_deg, color='r', linewidth='4', label=r'Pitch Angle Limit')
ax1.minorticks_on()
ax1.grid(which='both', alpha=0.3)
ax1.legend(loc='upper right', fontsize=20)

ax2 = fig.add_subplot(212, xlabel=r'MLAT [deg]', ylabel=r'Energy [eV]')
ax2.plot(mlat_deg, energy_limit_eV, color='b', linewidth='4', label=r'Energy Limit')
ylim = ax2.get_ylim()
ax2.plot(mlat_deg, energy_wave_phase_speed / elementary_charge, color='r', linewidth='4', label=r'Wave Phase Speed')
ax2.plot(mlat_deg, energy_wave_potential*np.ones(len(mlat_deg)) / elementary_charge, color='g', linewidth='4', label=r'Wave Potential')
ax2.set_ylim(ylim)
ax2.minorticks_on()
ax2.grid(which='both', alpha=0.3)
ax2.legend(loc='upper right', fontsize=20)

plt.show()