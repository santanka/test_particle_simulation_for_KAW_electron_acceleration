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
plt.rcParams["font.size"] = 40


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
    return Alfven_speed(mlat_rad) * kperp_rhoi * np.sqrt((1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad)))    #[m/s]

def kpara(mlat_rad):
    return wave_frequency / wave_phase_speed(mlat_rad)    #[rad/m]

def kperp(mlat_rad):
    vthi = np.sqrt(ion_temperature(mlat_rad) * elementary_charge / ion_mass)    #[m/s]
    omega_i = elementary_charge * magnetic_flux_density(mlat_rad) / ion_mass    #[rad/s]
    rhoi = vthi / omega_i    #[m]
    return kperp_rhoi / rhoi    #[rad/m]

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

def electron_inertial_length(mlat_rad):
    return speed_of_light / np.sqrt(number_density(mlat_rad) * elementary_charge**2E0 / electron_mass / electric_constant)    #[m]


mlat_rad_array = np.linspace(0E0, mlat_upper_limit_rad, 1000)
mlat_deg_array = mlat_rad_array * 180E0 / np.pi
wavelength_para_array = 2E0 * np.pi / kpara(mlat_rad_array)
wavelength_perp_array = 2E0 * np.pi / kperp(mlat_rad_array)
delta_scale_array = 1E0 / delta(mlat_rad_array) / kpara(mlat_rad_array)
electron_inertial_length_array = electron_inertial_length(mlat_rad_array) * np.ones(len(mlat_rad_array))

beta_ion_array = plasma_beta_ion(mlat_rad_array)
Gamma_array = Gamma(mlat_rad_array)
delta_array = delta(mlat_rad_array)
kpara_L_array = delta_scale_array * kpara(mlat_rad_array)
kpara_over_kperp_array = kpara(mlat_rad_array) / kperp(mlat_rad_array)
wave_frequency_over_omega_i_array = wave_frequency / elementary_charge / magnetic_flux_density(mlat_rad_array) * ion_mass

# plot
fig = plt.figure(figsize=(28, 14), dpi=100)

ax_1 = fig.add_subplot(121, xlabel=r'$\mathrm{MLAT}$ [$\mathrm{deg}$]', ylabel=r'Length [$\mathrm{m}$]', xlim=(0, mlat_upper_limit_deg), yscale='log')
ax_1.plot(mlat_deg_array, wavelength_para_array, color='orange', lw=4, label=r'$\lambda_{\parallel}$')
ax_1.plot(mlat_deg_array, wavelength_perp_array, color='green', lw=4, label=r'$\lambda_{\perp}=\rho_{\mathrm{i}}$')
ax_1.plot(mlat_deg_array, electron_inertial_length_array, color='blue', lw=4, label=r'$d_{\mathrm{e}}$')
ax_1.plot(mlat_deg_array, delta_scale_array, color='k', lw=4, label=r'$L$')
#ax_1.plot(mlat_deg_array, np.ones_like(mlat_rad_array) * planet_radius, color='red', lw=4, label=r'$R_{\mathrm{E}}$')
ax_1.legend()
ax_1.minorticks_on()
ax_1.grid(which='both', alpha=0.3)
ax_1.text(-0.15, 1.0, r'(a)', transform=ax_1.transAxes, fontsize=35)

ax_2 = fig.add_subplot(122, xlabel=r'$\mathrm{MLAT}$ [$\mathrm{deg}$]', ylabel=r'Ratio', xlim=(0, mlat_upper_limit_deg), yscale='log')
ax_2.plot(mlat_deg_array, beta_ion_array, color='orange', lw=4, label=r'$\beta_{\mathrm{i}}$')
ax_2.hlines(electron_mass / ion_mass, 0E0, mlat_upper_limit_deg, color='dimgrey', lw=4, label=r'$m_{\mathrm{e}} / m_{\mathrm{i}}$', linestyles='dashed')
ax_2.plot(mlat_deg_array, delta_array, color='blue', lw=4, label=r'$\delta$')
ax_2.plot(mlat_deg_array, Gamma_array, color='green', lw=4, label=r'$\Gamma$')
ax_2.plot(mlat_deg_array, kpara_L_array, color='k', lw=4, label=r'$k_{\parallel}L$')
ax_2.plot(mlat_deg_array, kpara_over_kperp_array, color='purple', lw=4, label=r'$k_{\parallel}/k_{\perp}$')
ax_2.plot(mlat_deg_array, wave_frequency_over_omega_i_array, color='red', lw=4, label=r'$\omega/\Omega_{\mathrm{i}}$')
ax_2.legend(loc='upper right')
ax_2.minorticks_on()
ax_2.grid(which='both', alpha=0.3)
ax_2.set_ylim(1E-5, 1E2)
ax_2.text(-0.15, 1.0, r'(b)', transform=ax_2.transAxes, fontsize=35)


fig.tight_layout()
#fig_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/length_scale'
#fig.savefig(fig_name + '.png')
#fig.savefig(fig_name + '.pdf')
#plt.close()
plt.show()
