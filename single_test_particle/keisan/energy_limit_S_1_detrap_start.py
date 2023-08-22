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
wave_phase = 2E0 * np.pi * 0.15    #[rad/s]
kpara = wave_phase / wave_phase_speed    #[rad/m]
wave_modified_potential = 1E-3 / kpara[0]  #[V]
wave_scalar_potential = 2000E0   #[V]
wave_modified_potential = wave_scalar_potential * (2E0 + electron_temperature / ion_temperature)    #[V]

energy_wave_phase_speed = 5E-1 * electron_mass * wave_phase_speed**2E0  #[J]
energy_wave_potential = elementary_charge * wave_modified_potential     #[J]

delta  = 1E0 / kpara / b0 * db0_dz    #[rad^-1]
epsilon = delta * (3E0 - 4E0 * ion_temperature / (plasma_beta_ion * (ion_temperature + electron_temperature) + 2E0 * ion_temperature))    #[rad^-1]

energy_detrap_start = (energy_wave_potential - epsilon * energy_wave_phase_speed) / delta    #[J]

cos_pitch_angle_detrap_start = np.zeros(len(mlat_deg))    #[]

for count_i in range (len(mlat_deg)):
    cos_pitch_angle_detrap_start[count_i] = energy_wave_phase_speed[count_i] / (energy_wave_potential - epsilon[count_i] * energy_wave_phase_speed[count_i]) * delta[count_i]    #[]
    if cos_pitch_angle_detrap_start[count_i] < 0E0:
        cos_pitch_angle_detrap_start[count_i] = np.nan
        energy_detrap_start[count_i] = np.nan
    elif cos_pitch_angle_detrap_start[count_i] > 1E0:
        cos_pitch_angle_detrap_start[count_i] = np.nan
        energy_detrap_start[count_i] = np.nan
    else:
        cos_pitch_angle_detrap_start[count_i] = np.sqrt(cos_pitch_angle_detrap_start[count_i])    #[]

pitch_angle_detrap_start_rad = np.arccos(cos_pitch_angle_detrap_start)    #[rad]
pitch_angle_detrap_start_deg = pitch_angle_detrap_start_rad * 180E0 / np.pi    #[deg]

energy_detrap_start_eV = energy_detrap_start / elementary_charge    #[eV]

equatorial_pitch_angle_detrap_start_rad = np.arcsin(np.sqrt(B0_eq / b0) * np.sin(pitch_angle_detrap_start_rad))    #[rad]
equatorial_pitch_angle_detrap_start_deg = equatorial_pitch_angle_detrap_start_rad * 180E0 / np.pi    #[deg]


# plot

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

#fig = plt.figure(figsize=(14, 14))
#ax = fig.add_subplot(111, xlabel=r'MLAT [deg]', xlim=(0, 50))
#ax.plot(mlat_deg, epsilon, color='blue', linewidth=4, label=r'$\epsilon$')
#ax.plot(mlat_deg, delta, color='red', linewidth=4, label=r'$\delta$')
#ax.minorticks_on()
#ax.grid(which='both', alpha=0.3)
#ax.legend(loc='upper right', fontsize=30)
#
#plt.tight_layout()
#
#plt.show()
#
#
#quit()


fig = plt.figure(figsize=(14, 21))
ax1 = fig.add_subplot(211, xlabel=r'MLAT [deg]', ylabel=r'Pitch angle [deg]', xlim=(0, 50), ylim=(0, 90))
ax1.plot(mlat_deg, pitch_angle_detrap_start_deg, color='blue', linewidth=4, label=r'detrap start')
ax1.plot(mlat_deg, equatorial_pitch_angle_detrap_start_deg, color='red', linewidth=4, label=r'equatorial detrap start')
ax1.minorticks_on()
ax1.grid(which='both', alpha=0.3)
ax1.legend(loc='upper right', fontsize=30)

ax2 = fig.add_subplot(212, xlabel=r'MLAT [deg]', ylabel=r'Energy [eV]', xlim=(0, 50), yscale='log')
ax2.plot(mlat_deg, energy_detrap_start_eV, color='blue', linewidth=4, label=r'detrap start')
ax2.plot(mlat_deg, energy_wave_phase_speed / elementary_charge, color='red', linewidth=4, label=r'wave phase speed')
ax2.plot(mlat_deg, energy_wave_potential * np.ones(len(mlat_deg)) / elementary_charge, color='green', linewidth=4, label=r'wave potential')
ax2_ylim = ax2.get_ylim()
ax2.set_ylim(ax2_ylim[0], 1E5)
ax2.minorticks_on()
ax2.grid(which='both', alpha=0.3)
ax2.legend(loc='upper right', fontsize=30)

plt.tight_layout()

plt.show()