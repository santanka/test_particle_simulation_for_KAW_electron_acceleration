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
    return 3E0 / kpara(mlat_rad) / r_eq * np.sin(mlat_rad) * (3E0 + 5E0 * np.sin(mlat_rad)**2E0) / np.cos(mlat_rad)**2E0 / (1E0 + 3E0 * np.sin(mlat_rad)**2E0)**1.5E0    #[rad]
    #grad_magnetic_flux_density = (magnetic_flux_density(mlat_rad + diff_rad) - magnetic_flux_density(mlat_rad - diff_rad)) / 2E0 / diff_rad * d_mlat_d_z(mlat_rad)    #[T/m]    
    #return 1E0 / kpara(mlat_rad) / magnetic_flux_density(mlat_rad) * grad_magnetic_flux_density    #[rad^-1]

def delta_2(mlat_rad):
    delta_plus = delta(mlat_rad + diff_rad) * kpara(mlat_rad + diff_rad) * magnetic_flux_density(mlat_rad + diff_rad)    #[rad]
    delta_minus = delta(mlat_rad - diff_rad) * kpara(mlat_rad - diff_rad) * magnetic_flux_density(mlat_rad - diff_rad)    #[rad]
    return (delta_plus - delta_minus) / 2E0 / diff_rad / kpara(mlat_rad)**2E0 / magnetic_flux_density(mlat_rad) * d_mlat_d_z(mlat_rad)    #[rad^-2]

def Gamma(mlat_rad):
    return 1E0 + 2E0 * plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]

def electron_inertial_length(mlat_rad):
    return speed_of_light / np.sqrt(number_density(mlat_rad) * elementary_charge**2E0 / electron_mass / electric_constant)    #[m]

def dkpara_dz_kpara2(mlat_rad):
    return -(1E0 + (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))) * delta(mlat_rad)

def d2kpara_dz2_kpara3(mlat_rad):
    c_1 = (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))
    return ((1E0 + c_1)**2E0 + (1E0 + c_1) - 2E0 * plasma_beta_ion(mlat_rad) * c_1**2E0) * delta(mlat_rad)**2E0 - (1E0 + c_1) * delta_2(mlat_rad)


mlat_rad_array = np.linspace(0E0, mlat_upper_limit_rad, 1000)
mlat_deg_array = mlat_rad_array * 180E0 / np.pi
wavelength_para_array = 2E0 * np.pi / kpara(mlat_rad_array)
wavelength_perp_array = 2E0 * np.pi / kperp(mlat_rad_array)
delta_scale_array = 1E0 / delta(mlat_rad_array) / kpara(mlat_rad_array)
electron_inertial_length_array = electron_inertial_length(mlat_rad_array) * np.ones(len(mlat_rad_array))

beta_ion_array = plasma_beta_ion(mlat_rad_array)
Gamma_array = Gamma(mlat_rad_array)
delta_array = delta(mlat_rad_array)
delta_2_array = delta_2(mlat_rad_array)
kpara_L_array = delta_scale_array * kpara(mlat_rad_array)
kpara_over_kperp_array = kpara(mlat_rad_array) / kperp(mlat_rad_array)
wave_frequency_over_omega_i_array = wave_frequency / elementary_charge / magnetic_flux_density(mlat_rad_array) * ion_mass

#dkpara_dz_kpara2_array = np.abs(dkpara_dz_kpara2(mlat_rad_array))
#d2kpara_dz2_kpara3_array = np.abs(d2kpara_dz2_kpara3(mlat_rad_array))
#
#upper_limit_energy_eV = 2E0 * energy_wave_potential(mlat_rad_array) / elementary_charge * np.ones_like(mlat_rad_array)
#lower_limit_energy_eV_1 = 2E0 * energy_wave_potential(mlat_rad_array) / elementary_charge * (1E0 - np.sqrt(energy_wave_phase_speed(mlat_rad_array) / energy_wave_potential(mlat_rad_array)) * (np.sqrt(2E0 + energy_wave_phase_speed(mlat_rad_array) / energy_wave_potential(mlat_rad_array)) - np.sqrt(energy_wave_phase_speed(mlat_rad_array) / energy_wave_potential(mlat_rad_array))))
#lower_limit_energy_eV_2 = 2E0 * energy_wave_potential(mlat_rad_array) / elementary_charge * (1E0 - 0.96561E0 * np.sqrt(energy_wave_phase_speed(mlat_rad_array) / energy_wave_potential(mlat_rad_array)))
#lower_limit_energy_eV_3 = 2E0 * energy_wave_potential(mlat_rad_array) / elementary_charge * (1E0 - 0.732215E0 * np.sqrt(energy_wave_phase_speed(mlat_rad_array) / energy_wave_potential(mlat_rad_array)))

def W_1_function(Theta, mlat_rad, mu):
    return ((Theta + np.sqrt(energy_wave_phase_speed(mlat_rad) / 2E0 / energy_wave_potential(mlat_rad)))**2E0 * (Gamma(mlat_rad) - 1E0) * (5E0 - 3E0 * Gamma(mlat_rad)) / (1E0 + Gamma(mlat_rad)) + 5E-1 * magnetic_flux_density(mlat_rad) * mu / energy_wave_potential(mlat_rad)) * delta(mlat_rad)

def W_2_function(Theta, mlat_rad, mu):
    return - (2E0 * (Theta + np.sqrt(energy_wave_phase_speed(mlat_rad) / 2E0 / energy_wave_potential(mlat_rad)))**2E0 + magnetic_flux_density(mlat_rad) * mu / energy_wave_potential(mlat_rad) / (1E0 + Gamma(mlat_rad))) * delta_2(mlat_rad) / delta(mlat_rad)

def W_function_Theta_input(Theta, mlat_rad, mu):
    return W_1_function(Theta, mlat_rad, mu) + W_2_function(Theta, mlat_rad, mu)

def sigma_function(mu, Theta, mlat_rad):
    return (2E0 * np.sqrt(energy_wave_phase_speed(mlat_rad) / 2E0 / energy_wave_potential(mlat_rad)) * (Theta + np.sqrt(energy_wave_phase_speed(mlat_rad) / 2E0 / energy_wave_potential(mlat_rad))) + magnetic_flux_density(mlat_rad) * mu / energy_wave_potential(mlat_rad)) * delta(mlat_rad)

mu_input = 1E2 * elementary_charge / magnetic_flux_density(0E0)


fig = plt.figure(figsize=(14, 14), dpi=100)
#ax = fig.add_subplot(111, xlabel=r'$\mathrm{MLAT}$ [$\mathrm{deg}$]', ylabel=r'$\Delta K_{\mathrm{after}}$ [eV]', xlim=(0, 40), ylim=(1E2, 1E4))
#ax.plot(mlat_deg_array, upper_limit_energy_eV, color='orange', lw=4, label=r'upper limit')
#ax.plot(mlat_deg_array, lower_limit_energy_eV_1, color='blue', lw=4, label=r'lower limit 1')
#ax.plot(mlat_deg_array, lower_limit_energy_eV_2, color='green', lw=4, label=r'lower limit 2')
#ax.plot(mlat_deg_array, lower_limit_energy_eV_3, color='red', lw=4, label=r'lower limit 3')
#ax = fig.add_subplot(111, xlabel=r'$\mathrm{MLAT}$ [$\mathrm{deg}$]', title=r'$K_{\perp} (\lambda = 0) = 100 \, \mathrm{eV}$, $\Theta = -1$')
#ax.plot(mlat_deg_array, W_1_function(-1E0, mlat_rad_array, mu_input), color='orange', lw=4, label=r'$W_{1}$', alpha=0.7)
#ax.plot(mlat_deg_array, W_2_function(-1E0, mlat_rad_array, mu_input), color='blue', lw=4, label=r'$W_{2}$', alpha=0.7)
#ax.plot(mlat_deg_array, W_function_Theta_input(-1E0, mlat_rad_array, mu_input), color='green', lw=4, label=r'$W$', alpha=0.7)
#ax.plot(mlat_deg_array, delta(mlat_rad_array), color='red', lw=4, label=r'$\delta_{1}$', alpha=0.7)
#ax.plot(mlat_deg_array, - delta_2(mlat_rad_array) / delta(mlat_rad_array), color='purple', lw=4, label=r'$ - \delta_{2} / \delta_{1}$', alpha=0.7)
#ax.set_xlim(0, 50)
#ax.set_ylim(-5, 5)
#ax.legend()
#ax.minorticks_on()
#ax.grid(which='both', alpha=0.3)
#plt.tight_layout()
#fig_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/W_function_scale'
#fig.savefig(fig_name + '.png')
#fig.savefig(fig_name + '.pdf')
#plt.close()

ax = fig.add_subplot(111, xlabel=r'$\mathrm{MLAT}$ [$\mathrm{deg}$]', xlim=(0, 40), ylim=(-1E1, 1E1))
ax.plot(mlat_deg_array, sigma_function(mu_input, -1E0, mlat_rad_array), color='orange', lw=4, label=r'$\sigma(\Theta = -1)$', alpha=0.7)
ax.plot(mlat_deg_array, sigma_function(mu_input, -1E1, mlat_rad_array), color='blue', lw=4, label=r'$\sigma(\Theta = -10)$', alpha=0.7)
ax.plot(mlat_deg_array, delta(mlat_rad_array), color='red', lw=4, label=r'$\delta_{1}$', alpha=0.7)
ax.set_yscale('symlog', linthresh=1E-2)
ax.legend()
ax.minorticks_on()
ax.grid(which='both', alpha=0.3)
plt.tight_layout()
plt.show()
#fig_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/sigma_function_scale'
#fig.savefig(fig_name + '.png')
#fig.savefig(fig_name + '.pdf')
#plt.close()

quit()

# plot
#fig = plt.figure(figsize=(28, 14), dpi=100)
#
#ax_1 = fig.add_subplot(121, xlabel=r'$\mathrm{MLAT}$ [$\mathrm{deg}$]', ylabel=r'Length [$\mathrm{m}$]', xlim=(0, mlat_upper_limit_deg), yscale='log')
#ax_1.plot(mlat_deg_array, wavelength_para_array, color='orange', lw=4, label=r'$\lambda_{\parallel}$')
#ax_1.plot(mlat_deg_array, wavelength_perp_array, color='green', lw=4, label=r'$\lambda_{\perp}=\rho_{\mathrm{i}}$')
#ax_1.plot(mlat_deg_array, electron_inertial_length_array, color='blue', lw=4, label=r'$d_{\mathrm{e}}$')
#ax_1.plot(mlat_deg_array, delta_scale_array, color='k', lw=4, label=r'$L$')
##ax_1.plot(mlat_deg_array, np.ones_like(mlat_rad_array) * planet_radius, color='red', lw=4, label=r'$R_{\mathrm{E}}$')
#ax_1.legend()
#ax_1.minorticks_on()
#ax_1.grid(which='both', alpha=0.3)
#ax_1.text(-0.15, 1.0, r'(a)', transform=ax_1.transAxes, fontsize=35)
#
#ax_2 = fig.add_subplot(122, xlabel=r'$\mathrm{MLAT}$ [$\mathrm{deg}$]', ylabel=r'Ratio', xlim=(0, mlat_upper_limit_deg), yscale='log')
#ax_2.plot(mlat_deg_array, beta_ion_array, color='orange', lw=4, label=r'$\beta_{\mathrm{i}}$')
#ax_2.hlines(electron_mass / ion_mass, 0E0, mlat_upper_limit_deg, color='dimgrey', lw=4, label=r'$m_{\mathrm{e}} / m_{\mathrm{i}}$', linestyles='dashed')
#ax_2.plot(mlat_deg_array, delta_array, color='blue', lw=4, label=r'$\delta_{1}$')
#ax_2.plot(mlat_deg_array, delta_2_array, color='deepskyblue', lw=4, label=r'$\delta_{2}$')
#ax_2.plot(mlat_deg_array, Gamma_array, color='green', lw=4, label=r'$\Gamma$')
#ax_2.plot(mlat_deg_array, kpara_L_array, color='k', lw=4, label=r'$k_{\parallel}L$')
#ax_2.plot(mlat_deg_array, kpara_over_kperp_array, color='purple', lw=4, label=r'$k_{\parallel}/k_{\perp}$')
#ax_2.plot(mlat_deg_array, wave_frequency_over_omega_i_array, color='red', lw=4, label=r'$\omega/\Omega_{\mathrm{i}}$')
#ax_2.legend(loc='upper right')
#ax_2.minorticks_on()
#ax_2.grid(which='both', alpha=0.3)
#ax_2.set_ylim(1E-5, 1E2)
#ax_2.text(-0.15, 1.0, r'(b)', transform=ax_2.transAxes, fontsize=35)

#ax_3 = fig.add_subplot(133, xlabel=r'$\mathrm{MLAT}$ [$\mathrm{deg}$]', ylabel=r'Scale', xlim=(0, mlat_upper_limit_deg), yscale='log', ylim=(1E-4, 1E4))
#ax_3.plot(mlat_deg_array, np.ones_like(mlat_rad_array), color='orange', lw=4, label=r'$1$')
#ax_3.plot(mlat_deg_array, np.abs(dkpara_dz_kpara2_array), color='green', lw=4, label=r'$\left| \frac{1}{k_{\parallel}^{2}} \frac{\mathrm{d} k_{\parallel}}{\mathrm{d} z} \right|$')
#ax_3.plot(mlat_deg_array, d2kpara_dz2_kpara3_array, color='blue', lw=4, label=r'$\frac{1}{k_{\parallel}^{3}} \frac{\mathrm{d}^{2} k_{\parallel}}{\mathrm{d} z^{2}}$')
#ax_3.legend()
#ax_3.minorticks_on()
#ax_3.grid(which='both', alpha=0.3)
#ax_3.text(-0.15, 1.0, r'(c)', transform=ax_3.transAxes, fontsize=35)


fig.tight_layout()
fig_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/length_scale'
fig.savefig(fig_name + '.png')
fig.savefig(fig_name + '.pdf')
plt.close()
#plt.show()
