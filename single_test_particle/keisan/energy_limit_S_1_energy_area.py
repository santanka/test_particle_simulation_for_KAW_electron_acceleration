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

energy_wave_phase_speed_eV = energy_wave_phase_speed / elementary_charge    #[eV]
energy_wave_potential_eV = energy_wave_potential / elementary_charge    #[eV]

delta  = 1E0 / kpara / b0 * db0_dz    #[rad^-1]
epsilon = delta * (3E0 - 4E0 * ion_temperature / (plasma_beta_ion * (ion_temperature + electron_temperature) + 2E0 * ion_temperature))    #[rad^-1]

# S=1 energy limit
energy_upper_limit_eV = np.zeros(len(mlat_deg))
energy_lower_limit_eV = np.zeros(len(mlat_deg))

for count_i in range(len(mlat_deg)):
    if epsilon[count_i] >= 0E0:
        energy_upper_limit_eV[count_i] = energy_wave_potential_eV / (delta[count_i])
        energy_lower_limit_eV[count_i] = energy_wave_potential_eV / (delta[count_i] + epsilon[count_i])
    elif epsilon[count_i] < 0E0 and epsilon[count_i] > -delta[count_i]:
        energy_upper_limit_eV[count_i] = np.nan
        energy_lower_limit_eV[count_i] = energy_wave_potential_eV / (delta[count_i] + epsilon[count_i])
    else:
        energy_upper_limit_eV[count_i] = np.nan
        energy_lower_limit_eV[count_i] = np.nan


# detrapped phase energy

energy_detrapped = 1E0 / delta * ((1E0 - epsilon * (np.pi/2E0 - 1E0)) * energy_wave_potential_eV + 2E0 * epsilon * np.sqrt(energy_wave_phase_speed_eV * energy_wave_potential_eV) * np.sqrt(np.pi/2E0 - 1E0) - epsilon * energy_wave_phase_speed_eV)    #[eV]



# trapped energy range
trapping_frequency = kpara *np.sqrt(energy_wave_potential / electron_mass) #[rad/s]

def Pi_S(ss):
    return np.sqrt(np.sqrt(1E0 - ss**2E0) - 5E-1 * ss * (np.pi - 2E0 * np.arcsin(ss)))

ss_min = energy_wave_phase_speed_eV / energy_wave_potential_eV * (delta + epsilon)  #[]

Pi_min = Pi_S(ss_min)   #[]

lower_energy_trapped = np.zeros(len(mlat_deg))
upper_energy_trapped = (np.sqrt(2E0*energy_wave_potential) * Pi_min + np.sqrt(energy_wave_phase_speed))**2E0     #[J]
upper_energy_trapped_eV = upper_energy_trapped / elementary_charge    #[eV]

for count_i in range(len(mlat_deg)):
    if trapping_frequency[count_i] * Pi_min[count_i] > wave_phase * 0.5E0:
        lower_energy_trapped[count_i] = 0E0
    else:
        lower_energy_trapped[count_i] = (np.sqrt(2E0*energy_wave_potential) * Pi_min[count_i] - np.sqrt(energy_wave_phase_speed[count_i]))**2E0    #[J]

lower_energy_trapped_eV = lower_energy_trapped / elementary_charge    #[eV]

# S<1 energy range
upper_energy_trapped_S1 = energy_wave_potential_eV / (delta + epsilon)    #[eV]


# 断熱不変量によるエネルギーの制限
K_initial_max = 1E3     #[eV]

K_upper_limit_adiabatic = np.zeros(len(mlat_deg))
K_lower_limit_adiabatic = np.zeros(len(mlat_deg))

for count_i in range(len(mlat_deg)):
    if epsilon[count_i] >= 0E0 or epsilon[count_i] <= -delta[count_i]:
        K_upper_limit_adiabatic[count_i] = (b0[count_i] / b0[0] * epsilon[count_i] * K_initial_max + energy_wave_potential_eV) / (delta[count_i] + epsilon[count_i])
        K_lower_limit_adiabatic[count_i] = 0E0
    else:
        K_upper_limit_adiabatic[count_i] = np.inf
        K_lower_limit_adiabatic[count_i] = (b0[count_i] / b0[0] * epsilon[count_i] * K_initial_max + energy_wave_potential_eV) / (delta[count_i] + epsilon[count_i])



# plot

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

fig = plt.figure(figsize=(14, 28))
ax1 = fig.add_subplot(211, xlabel=r'MLAT [deg]', ylabel=r'Energy [eV]', xlim=(0E0, 50E0), yscale='log')

# energy_lower_limit_eV < energy_detrapped < K_upper_limit_adiabatic < energy_upper_limit_eVとなる、energy_lower_limit_eVとK_upper_limit_adiabaticの間の領域を塗りつぶす
ax1.fill_between(mlat_deg, energy_lower_limit_eV, K_upper_limit_adiabatic, where=(energy_lower_limit_eV < energy_detrapped) & (energy_detrapped < K_upper_limit_adiabatic) & (K_upper_limit_adiabatic < energy_upper_limit_eV), facecolor='yellow', alpha=0.3)
# energy_lower_limit_eV < energy_detrapped < energy_upper_limit_eV < K_upper_limit_adiabaticとなる、energy_lower_limit_eVとenergy_upper_limit_eVの間の領域を塗りつぶす
ax1.fill_between(mlat_deg, energy_lower_limit_eV, energy_upper_limit_eV, where=(energy_lower_limit_eV < energy_detrapped) & (energy_detrapped < energy_upper_limit_eV) & (energy_upper_limit_eV < K_upper_limit_adiabatic), facecolor='yellow', alpha=0.3)

ax1_ylim = ax1.get_ylim()

ax1.plot(mlat_deg, energy_upper_limit_eV, color='blue', linewidth=4, label=r'$S = 1$ range', alpha=0.6)
ax1.plot(mlat_deg, energy_lower_limit_eV, color='blue', linewidth=4, alpha=0.6)
ax1.plot(mlat_deg, energy_wave_phase_speed_eV, color='red', linewidth=4, label=r'wave phase speed', alpha=0.6)
ax1.plot(mlat_deg, energy_wave_potential_eV * np.ones(len(mlat_deg)), color='green', linewidth=4, label=r'wave potential', alpha=0.6)
ax1.plot(mlat_deg, energy_detrapped, color='orange', linewidth=4, label=r'detrapped phase', alpha=0.6)
ax1.plot(mlat_deg, K_upper_limit_adiabatic, color='purple', linewidth=4, label=r'adiabatic invariant', alpha=0.6)
ax1.plot(mlat_deg, K_lower_limit_adiabatic, color='purple', linewidth=4, alpha=0.6)
ax1.set_title(r'detrapped phase energy $K_{\mathrm{f}}$')
ax1.set_ylim(1E1, 1E5)
#ax1.set_ylim(ax1_ylim)
ax1.minorticks_on()
ax1.grid(which='both', alpha=0.3)
ax1.legend(loc='lower left')

ax2 = fig.add_subplot(212, xlabel=r'MLAT [deg]', ylabel=r'Energy [eV]', xlim=(0E0, 50E0), yscale='log')

# upper_energy_trapped_eV < upper_energy_trapped_S1 且つ lower_energy_trapped_eV < upper_energy_trapped_eVとなる領域を塗りつぶす
ax2.fill_between(mlat_deg, lower_energy_trapped_eV, upper_energy_trapped_eV, where=(upper_energy_trapped_eV < upper_energy_trapped_S1) & (lower_energy_trapped_eV < upper_energy_trapped_eV), facecolor='yellow', alpha=0.3)
# upper_energy_trapped_S1 < upper_energy_trapped_eV 且つ lower_energy_trapped_eV < upper_energy_trapped_S1となる領域を塗りつぶす
ax2.fill_between(mlat_deg, lower_energy_trapped_eV, upper_energy_trapped_S1, where=(upper_energy_trapped_S1 < upper_energy_trapped_eV) & (lower_energy_trapped_eV < upper_energy_trapped_S1), facecolor='yellow', alpha=0.3)


ax2.plot(mlat_deg, upper_energy_trapped_S1, color='blue', linewidth=4, label=r'$S < 1$ range', alpha=0.6)
ax2.plot(mlat_deg, energy_wave_phase_speed_eV, color='red', linewidth=4, label=r'wave phase speed', alpha=0.6)
ax2.plot(mlat_deg, energy_wave_potential_eV * np.ones(len(mlat_deg)), color='green', linewidth=4, label=r'wave potential', alpha=0.6)
ax2.plot(mlat_deg, lower_energy_trapped_eV, color='orange', linewidth=4, label=r'trapped energy range', alpha=0.6)
ax2.plot(mlat_deg, upper_energy_trapped_eV, color='orange', linewidth=4, alpha=0.6)
ax2.set_ylim(1E1, 1E5)
ax2.set_title(r'trapped phase energy $K \mathrm{cos}^{2} \alpha$')
ax2.minorticks_on()
ax2.grid(which='both', alpha=0.3)
ax2.legend(loc='lower left')

plt.tight_layout()

plt.show()




quit()
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111, xlabel=r'MLAT [deg]', ylabel=r'Energy [eV]', xlim=(0E0, 50E0), yscale='log')
ax.plot(mlat_deg, energy_upper_limit_eV, color='blue', linewidth=4, label=r'$S = 1$ range')
ax.plot(mlat_deg, energy_lower_limit_eV, color='blue', linewidth=4)
ax.plot(mlat_deg, energy_wave_phase_speed_eV, color='red', linewidth=4, label=r'wave phase speed')
ax.plot(mlat_deg, lower_energy_trapped_eV, color='red', linewidth=4, label=r'trapped energy range', linestyle='--')
ax.plot(mlat_deg, upper_energy_trapped_eV, color='red', linewidth=4, linestyle='--')
ax.plot(mlat_deg, energy_wave_potential_eV * np.ones(len(mlat_deg)), color='green', linewidth=4, label=r'wave potential')
ax.plot(mlat_deg, energy_detrapped, color='orange', linewidth=4, label=r'detrapped energy')
ax_ylim = ax.get_ylim()
ax.set_ylim(1E1, 1E5)
ax.minorticks_on()
ax.grid(which='both',alpha=0.3)
ax.legend()

plt.tight_layout()

plt.show()