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

def wave_phase_speed(mlat_rad):
    return Alfven_speed(mlat_rad) * kperp_rhoi * np.sqrt((1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))) * np.sign(mlat_rad)    #[m/s]

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
    grad_magnetic_flux_density = (magnetic_flux_density(mlat_rad + diff_rad) - magnetic_flux_density(mlat_rad - diff_rad)) / 2E0 / diff_rad * d_mlat_d_z(mlat_rad)    #[T/m]
    return 1E0 / kpara(mlat_rad) / magnetic_flux_density(mlat_rad) * grad_magnetic_flux_density    #[rad^-1]

def Gamma(mlat_rad):
    return 1E0 + 2E0 * plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]

def trapping_frequency(mlat_rad):
    return np.abs(kpara(mlat_rad)) * np.sqrt(energy_wave_potential(mlat_rad) / electron_mass)   #[rad/s]


# input parameters
#theta_initial = np.linspace(- wave_frequency, 1E1 * wave_frequency, 1000)    #[rad]
theta_2_omega_t_initial = np.linspace(- 5E-1, 1E0, 100)    #[rad]

wave_phase_parameter_array = np.linspace(-1E0, 1E0, 5)    #[]
color_list = ['darkorchid', 'yellowgreen', 'darkgrey', 'darkturquoise', 'lightpink']

# iteration function
def mlat_integration(mlat, wave_phase_parameter):
    return (- 1E0 / 7E0 * (np.sin(mlat_upper_limit_rad)**7E0 - np.sin(mlat)**7E0) + 3E0 / 5E0 * (np.sin(mlat_upper_limit_rad)**5E0 - np.sin(mlat)**5E0) + (np.cos(mlat_upper_limit_rad)**2E0 * np.sin(mlat_upper_limit_rad) - np.cos(mlat)**2E0 * np.sin(mlat))) * wave_phase_parameter #[]

def iteration_function(mlat, theta_2_omega_t, wave_phase_parameter):
    theta_i = theta_2_omega_t * 2E0 * trapping_frequency(mlat)
    value = energy_wave_phase_speed(mlat) / energy_wave_potential(mlat) * (1E0 + Gamma(mlat)) * delta(mlat) * (1E0 + theta_i / wave_frequency)**2E0
    value = value + magnetic_flux_density(mlat) / (magnetic_flux_density(mlat_upper_limit_rad) - magnetic_flux_density(mlat)) * delta(mlat) * (energy_wave_phase_speed(mlat) / energy_wave_potential(mlat) * (1E0 + theta_i / wave_frequency)**2E0 + 1E0 / kperp_rhoi * wave_frequency * r_eq / Alfven_speed(0E0) * np.sqrt(2E0 * tau(mlat) / (1E0 + tau(mlat))) * mlat_integration(mlat, wave_phase_parameter))
    value = value - 1E0
    return value

diff_rad = 1E-6 #[rad]
def gradient_iteration_function(mlat, theta_2_omega_t, wave_phase_parameter):
    theta_i = theta_2_omega_t * 2E0 * trapping_frequency(mlat)
    return (iteration_function(mlat + diff_rad, theta_i, wave_phase_parameter) - iteration_function(mlat - diff_rad, theta_i, wave_phase_parameter)) / 2E0 / diff_rad

def result_Kperp_eq(mlat, theta_2_omega_t, wave_phase_parameter):
    theta_i = theta_2_omega_t * 2E0 * trapping_frequency(mlat)
    value_1 = energy_wave_potential(mlat) * magnetic_flux_density(0E0) / (magnetic_flux_density(mlat_upper_limit_rad) - magnetic_flux_density(mlat))
    value_2 = energy_wave_phase_speed(mlat) / energy_wave_potential(mlat) * (1E0 + theta_i / wave_frequency)**2E0 + 1E0 / kperp_rhoi * wave_frequency * r_eq / Alfven_speed(0E0) * np.sqrt(2E0 * tau(mlat) / (1E0 + tau(mlat))) * mlat_integration(mlat, wave_phase_parameter)
    return value_1 * value_2

def result_Kperp_upper_limit(Kperp_eq):
    return Kperp_eq * magnetic_flux_density(mlat_upper_limit_rad) / magnetic_flux_density(0E0)


def Newton_method(theta_2_omega_t, wave_phase_parameter):
    initial_mlat_value = np.pi / 8E0
    mlat_rad_before = initial_mlat_value
    count_iteration = 0
    while True:
        diff = iteration_function(mlat_rad_before, theta_2_omega_t, wave_phase_parameter) / gradient_iteration_function(mlat_rad_before, theta_2_omega_t, wave_phase_parameter)
        if abs(diff) > 1E-2:
            diff = np.sign(diff) * 1E-2
        mlat_rad_after = mlat_rad_before - diff
        if abs(mlat_rad_after - mlat_rad_before) < 1E-10:
            break
        else:
            mlat_rad_before = mlat_rad_after
            count_iteration += 1
            if count_iteration > 1000:
                #print('count_iteration > 1000')
                mlat_rad_after = np.nan
                break
    
    if np.isnan(mlat_rad_after):
        return np.nan, np.nan, np.nan
    else:
        return mlat_rad_after, result_Kperp_eq(mlat_rad_after, theta_2_omega_t, wave_phase_parameter), result_Kperp_upper_limit(result_Kperp_eq(mlat_rad_after, theta_2_omega_t, wave_phase_parameter))


# main
mlat_initial = np.ones(len(theta_2_omega_t_initial)) * np.nan
Kperp_eq_array = np.ones(len(theta_2_omega_t_initial)) * np.nan
Kperp_upper_limit_array = np.ones(len(theta_2_omega_t_initial)) * np.nan

def main(args):
    count_i, count_j = args
    theta_2_omega_t_i = theta_2_omega_t_initial[count_i]
    wave_phase_parameter = wave_phase_parameter_array[count_j]
    mlat_rad, Kperp_eq, Kperp_upper_limit = Newton_method(theta_2_omega_t_i, wave_phase_parameter)
    mlat_initial[count_i] = mlat_rad
    Kperp_eq_array[count_i] = Kperp_eq
    Kperp_upper_limit_array[count_i] = Kperp_upper_limit
    if count_i % 100 == 0:
        print(f'count_i = {count_i}, count_j = {count_j}, {theta_2_omega_t_i:.2f}, {mlat_rad:.2f}, {Kperp_eq / elementary_charge:.2f}, {Kperp_upper_limit / elementary_charge:.2f}')
    return theta_2_omega_t_i, mlat_rad, Kperp_eq, Kperp_upper_limit, count_j

# plot
fig = plt.figure(figsize=(20, 20), dpi=100)
ax = fig.add_subplot(111, xlabel=r'$K_{\perp} (\lambda = 0)$ [eV]', ylabel=r'$K_{\perp} (\lambda = \lambda_{\mathrm{ionosphere}})$ [eV]', xscale='log', yscale='log')

vmin = theta_2_omega_t_initial.min()
vmax = theta_2_omega_t_initial.max()
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
cmap.set_array([])
cmap.set_clim(vmin, vmax)

#mainを非同期処理
if __name__ == '__main__':
    num_processes = 16

    args = np.array(np.meshgrid(np.arange(len(theta_2_omega_t_initial)), np.arange(len(wave_phase_parameter_array)))).T.reshape(-1, 2)

    with Pool(num_processes) as p:
        results = p.map(main, args)
    
    for result in results:
        theta_2_omega_t_i, mlat_rad, Kperp_eq, Kperp_upper_limit, count_j = result
        ax.scatter(Kperp_eq / elementary_charge, Kperp_upper_limit / elementary_charge, marker='o', s=100, edgecolor=cmap.to_rgba(theta_2_omega_t_i), color=color_list[count_j], zorder=3, linewidth=3)

edgecolor = 'black'

for count_j in range(len(wave_phase_parameter_array)):
    wave_phase_parameter = wave_phase_parameter_array[count_j]
    theta_i_0 = 0E0
    mlat_rad_0, Kperp_eq_0, Kperp_upper_limit_0 = Newton_method(theta_i_0, wave_phase_parameter)
    ax.scatter(Kperp_eq_0 / elementary_charge, Kperp_upper_limit_0 / elementary_charge, marker='*', s=1000, color=color_list[count_j], edgecolor=edgecolor, label=r'$\theta_{\mathrm{i}} = 0$, $\Psi = $' + f'{wave_phase_parameter:.1f}', zorder=100)

#theta_i_0 = 0E0
#mlat_rad_0, Kperp_eq_0, Kperp_upper_limit_0 = Newton_method(theta_i_0)
#ax.scatter(Kperp_eq_0 / elementary_charge, Kperp_upper_limit_0 / elementary_charge, marker='*', s=1000, color='white', edgecolor='black', label=r'$\theta_{\mathrm{i}} = 0$', zorder=100)
ax.legend()

cbar = fig.colorbar(cmap, ax=ax)
cbar.set_label(r'$\theta_{\mathrm{i}} / 2 \omega_{\mathrm{t}}$')

ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

#ax.set_xlim(1E0, 2.1E1)
#ax.set_ylim(1E3, 2.1E4)

fig.tight_layout()

fig.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/auroral_electron_acceleration_mu_condition.png')
fig.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/auroral_electron_acceleration_mu_condition.pdf')
plt.close()