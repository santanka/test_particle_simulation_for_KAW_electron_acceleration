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

# magnetic field
planet_radius   = 7.1492E7  #[m]
lshell_number   = 5.91E0
r_eq            = planet_radius * lshell_number #[m]
dipole_moment   = 4.2E-4 * (4E0 * np.pi * planet_radius**3E0) / magnetic_constant #[Am]
B0_eq           = (1E-7 * dipole_moment) / r_eq**3E0

b0 = B0_eq / np.cos(mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)     #[T]
db0_dz = 3E0 * np.sin(mlat_rad) * (5E0 * np.sin(mlat_rad)**2E0 + 3E0) / np.cos(mlat_rad)**8E0 / (3E0 * np.sin(mlat_rad)**2E0 + 1E0) / r_eq * B0_eq   #[T/m]

coordinate_FA = r_eq * (np.arcsinh(np.sqrt(3)*np.sin(mlat_rad)) / 2E0 / np.sqrt(3) + np.sin(mlat_rad) * np.sqrt(5E0 - 3E0 * np.cos(2E0 * mlat_rad)) / 2E0 / np.sqrt(2)) #[m]

# background plasma parameters
oxygen_number_density_eq = 1.163E9    # [m^-3]
oxygen_number_density_scale_height = 1.2842    # [R_J]
oxygen_number_density = oxygen_number_density_eq * np.exp(- (coordinate_FA / oxygen_number_density_scale_height / planet_radius)**2E0)    #[m^-3]

oxygen_mass = 2.677950266103E-26   # [kg] # O+
oxygen_temperature = 5E1   # [eV]

proton_number_density_eq = 5.8E7    # [m^-3]
proton_number_density = proton_number_density_eq * np.ones(mlat_rad.shape)    #[m^-3]

proton_mass = 1.672621898E-27    # [kg]
proton_temperature = 8.6E0    # [eV]

ion_number_density = oxygen_number_density + proton_number_density    #[m^-3]
ion_mass_density = oxygen_number_density * oxygen_mass + proton_number_density * proton_mass    #[kg m^-3]
ion_mass = ion_mass_density / ion_number_density    #[kg]

ion_pressure = (proton_number_density * proton_temperature + oxygen_number_density * oxygen_temperature) * elementary_charge    #[N m^-2]
ion_temperature = ion_pressure / ion_number_density / elementary_charge    #[eV]

electron_mass = 9.10938356E-31    # [kg]
electron_temperature = 5E0  # [eV]

tau = ion_temperature / electron_temperature    #[]

Alfven_speed = b0 / np.sqrt(magnetic_constant * ion_mass_density)    #[m/s]
#relativistic_Alfven_speed = Alfven_speed / np.sqrt(speed_of_light**2E0 + Alfven_speed**2E0) * speed_of_light   #[m/s]

plasma_beta_ion = 2E0 * magnetic_constant * ion_pressure / b0**2E0  #[]

ion_gyrofrequency = elementary_charge * b0 / ion_mass    #[rad/s]
ion_thermal_speed = np.sqrt(2E0 * ion_temperature * elementary_charge / ion_mass)    #[m/s]
ion_gyroradius = ion_thermal_speed / ion_gyrofrequency    #[m]

electron_gyrofrequency = elementary_charge * b0 / electron_mass    #[rad/s]
electron_thermal_speed = np.sqrt(2E0 * electron_temperature * elementary_charge / electron_mass)    #[m/s]
electron_gyroradius = electron_thermal_speed / electron_gyrofrequency    #[m]


# wave parameters

uperp_eq = 5.4E4 #[m/s]
kperp_rhoi = ion_thermal_speed[0] / uperp_eq    #[rad]

kperp = kperp_rhoi / ion_gyroradius    #[rad m^-1]
wavelength_perp = 2E0 * np.pi / kperp    #[m]

kpara_eq = ion_gyrofrequency[0] / electron_thermal_speed    #[rad m^-1]

wave_frequency = np.sqrt((1E0 + tau[0]) / (plasma_beta_ion[0] * (1E0 + tau[0]) + 2E0 * tau[0])) * kperp_rhoi * kpara_eq * Alfven_speed[0]    #[rad s^-1]

kpara = wave_frequency / Alfven_speed / kperp_rhoi * np.sqrt(plasma_beta_ion + 2E0 * tau / (1E0 + tau))    #[rad m^-1]
wavelength_para = 2E0 * np.pi / kpara    #[m]

wave_phase_speed_parallel = wave_frequency / kpara    #[m/s]

electric_field_eq = 1E-3 #[V/m]
wave_modified_potential = electric_field_eq / kpara[0]     #[V]
#wave_modified_potential = 1E1    #[V]

energy_wave_phase_speed = 5E-1 * electron_mass * wave_phase_speed_parallel**2E0 / elementary_charge    #[eV]
energy_wave_potential = 5E-1 * elementary_charge * wave_modified_potential / elementary_charge    #[eV]


# 微分
def diff(f):
    diff_f = np.zeros(f.shape)
    for count_i in range(len(coordinate_FA)):
        if count_i == 0:
            diff_f[count_i] = 0E0
        elif count_i == len(coordinate_FA) - 1:
            diff_f[count_i] = (f[count_i] - f[count_i - 1]) / (coordinate_FA[count_i] - coordinate_FA[count_i - 1])
        else:
            diff_f[count_i] = ((f[count_i + 1] - f[count_i]) / (coordinate_FA[count_i + 1] - coordinate_FA[count_i]) + (f[count_i] - f[count_i - 1]) / (coordinate_FA[count_i] - coordinate_FA[count_i - 1])) / 2E0
    return diff_f

diff_b0 = diff(b0) / kpara / b0
diff_ion_number_density = diff(ion_number_density) / kpara / ion_number_density
diff_ion_temperature = diff(ion_temperature) / kpara / ion_temperature
diff_1_plus_tau = diff(1E0 + tau) / kpara / (1E0 + tau)

yy = plasma_beta_ion * (1E0 + tau) / (plasma_beta_ion * (1E0 + tau) + 2E0 * tau)

delta = diff_b0
epsilon = (1E0 + 2E0 * yy) * diff_b0 - (1E0 + yy) * diff_ion_number_density - yy * diff_ion_temperature - yy / plasma_beta_ion / (1E0 + tau) * diff_1_plus_tau


# S=1 energy limit
energy_upper_limit_eV = np.zeros(len(mlat_deg))
energy_lower_limit_eV = np.zeros(len(mlat_deg))

for count_i in range(len(mlat_deg)):
    if epsilon[count_i] >= 0E0:
        energy_upper_limit_eV[count_i] = energy_wave_potential / delta[count_i]
        energy_lower_limit_eV[count_i] = energy_wave_potential / (delta[count_i] + epsilon[count_i])
    elif epsilon[count_i] < 0E0 and epsilon[count_i] >= -delta[count_i]:
        energy_upper_limit_eV[count_i] = energy_wave_potential / (delta[count_i] + epsilon[count_i])
        energy_lower_limit_eV[count_i] = energy_wave_potential / delta[count_i]
    elif epsilon[count_i] < -delta[count_i]:
        energy_upper_limit_eV[count_i] = np.inf
        energy_lower_limit_eV[count_i] = energy_wave_potential / delta[count_i]

# detrapped phase energy

energy_detrapped = 1E0 / delta * ((1E0 - epsilon * (np.pi/2E0 - 1E0)) * energy_wave_potential + 2E0 * epsilon * np.sqrt(energy_wave_phase_speed * energy_wave_potential) * np.sqrt(np.pi/2E0 - 1E0) - epsilon * energy_wave_phase_speed)    #[eV]


# trapped energy range
trapping_frequency = kpara *np.sqrt(energy_wave_potential * elementary_charge / electron_mass) #[rad/s]

def Pi_S(ss):
    return np.sqrt(np.sqrt(1E0 - ss**2E0) - 5E-1 * ss * (np.pi - 2E0 * np.arcsin(ss)))

ss_min = energy_wave_phase_speed / energy_wave_potential * (delta + epsilon)  #[]

Pi_min = Pi_S(ss_min)   #[]

lower_energy_trapped = np.zeros(len(mlat_deg))
upper_energy_trapped = (np.sqrt(2E0*energy_wave_potential) * Pi_min + np.sqrt(energy_wave_phase_speed))**2E0     #[eV]

for count_i in range(len(mlat_deg)):
    if trapping_frequency[count_i] * Pi_min[count_i] > wave_frequency * 0.5E0:
        lower_energy_trapped[count_i] = 0E0
    else:
        lower_energy_trapped[count_i] = (np.sqrt(2E0*energy_wave_potential) * Pi_min[count_i] - np.sqrt(energy_wave_phase_speed[count_i]))**2E0    #[eV]


# S<1 energy range
upper_energy_trapped_S1 = energy_wave_potential / (delta + epsilon)    #[eV]


# 断熱不変量によるエネルギー制限

K_initial_max = 1E3     #[eV]
K_upper_limit_adiabatic = np.zeros(len(mlat_deg))
K_lower_limit_adiabatic = np.zeros(len(mlat_deg))

for count_i in range(len(mlat_deg)):
    if epsilon[count_i] >= 0E0 or epsilon[count_i] <= -delta[count_i]:
        K_upper_limit_adiabatic[count_i] = (b0[count_i] / b0[0] * epsilon[count_i] * K_initial_max + energy_wave_potential) / (delta[count_i] + epsilon[count_i])
        K_lower_limit_adiabatic[count_i] = 0E0
    else:
        K_upper_limit_adiabatic[count_i] = np.inf
        K_lower_limit_adiabatic[count_i] = (b0[count_i] / b0[0] * epsilon[count_i] * K_initial_max + energy_wave_potential) / (delta[count_i] + epsilon[count_i])


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

ax1.plot(mlat_deg, energy_upper_limit_eV, color='blue', linewidth=4, label=r'$S = 1$ range', alpha=0.6)
ax1.plot(mlat_deg, energy_lower_limit_eV, color='blue', linewidth=4, alpha=0.6)
ax1.plot(mlat_deg, energy_wave_phase_speed, color='red', linewidth=4, label=r'wave phase speed', alpha=0.6)
ax1.plot(mlat_deg, energy_wave_potential * np.ones(len(mlat_deg)), color='green', linewidth=4, label=r'wave potential', alpha=0.6)
ax1.plot(mlat_deg, energy_detrapped, color='orange', linewidth=4, label=r'detrapped phase', alpha=0.6)
ax1.plot(mlat_deg, K_upper_limit_adiabatic, color='purple', linewidth=4, label=r'adiabatic invariant', alpha=0.6)
ax1.plot(mlat_deg, K_lower_limit_adiabatic, color='purple', linewidth=4, alpha=0.6)
ax1.set_title(r'detrapped phase energy $K_{\mathrm{f}}$')
ax1.set_ylim(1E0, 1E5)
ax1.minorticks_on()
ax1.grid(which='both', alpha=0.3)
ax1.legend(loc='upper left')

ax2 = fig.add_subplot(212, xlabel=r'MLAT [deg]', ylabel=r'Energy [eV]', xlim=(0E0, 50E0), yscale='log')

# upper_energy_trapped < upper_energy_trapped_S1 且つ lower_energy_trapped < upper_energy_trappedとなる領域を塗りつぶす
ax2.fill_between(mlat_deg, lower_energy_trapped, upper_energy_trapped, where=(upper_energy_trapped < upper_energy_trapped_S1) & (lower_energy_trapped < upper_energy_trapped), facecolor='yellow', alpha=0.3)
# upper_energy_trapped_S1 < upper_energy_trapped 且つ lower_energy_trapped < upper_energy_trapped_S1となる領域を塗りつぶす
ax2.fill_between(mlat_deg, lower_energy_trapped, upper_energy_trapped_S1, where=(upper_energy_trapped_S1 < upper_energy_trapped) & (lower_energy_trapped < upper_energy_trapped_S1), facecolor='yellow', alpha=0.3)


ax2.plot(mlat_deg, upper_energy_trapped_S1, color='blue', linewidth=4, label=r'$S < 1$ range', alpha=0.6)
ax2.plot(mlat_deg, energy_wave_phase_speed, color='red', linewidth=4, label=r'wave phase speed', alpha=0.6)
ax2.plot(mlat_deg, energy_wave_potential * np.ones(len(mlat_deg)), color='green', linewidth=4, label=r'wave potential', alpha=0.6)
ax2.plot(mlat_deg, lower_energy_trapped, color='orange', linewidth=4, label=r'trapped energy range', alpha=0.6)
ax2.plot(mlat_deg, upper_energy_trapped, color='orange', linewidth=4, alpha=0.6)
ax2.set_ylim(1E0, 1E5)
ax2.set_title(r'trapped phase energy $K \mathrm{cos}^{2} \alpha$')
ax2.minorticks_on()
ax2.grid(which='both', alpha=0.3)
ax2.legend(loc='upper left')

plt.tight_layout()

plt.show()