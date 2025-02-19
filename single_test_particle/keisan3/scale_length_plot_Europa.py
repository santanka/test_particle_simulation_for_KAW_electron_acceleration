import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import os

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 25

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan3'

# constants
speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]
electric_constant = 8.8541878128E-12  #[F m-1]
magnetic_constant = 1.25663706212E-6  #[N A-2]
proton_mass = 1.672621898E-27   # [kg]
electron_mass = 9.10938356E-31    # [kg]

# magnetic flux density: dipole model
# number density: centrifugal scale height model

L_number = 9.65   # Europa's L number (mean) (Bagenal et al., 2015)
Radius_Jupiter = 71492E3   # [m] (equatorial radius)

Time_rotation_Jupiter = 9.9258E0 * 3600E0   # [s]
Omega_Jupiter = 2E0 * np.pi / Time_rotation_Jupiter   # [rad/s]

# centrifugal scale height model (Thomas et al., 2004)
ion_temperature = 88E0   # [eV] (Bagenal et al., 2015)
electron_temperature = 20E0   # [eV] (Bagenal et al., 2015)
ion_charge_number = 1.4E0   # (Bagenal et al., 2015)
ion_mass_number = 18E0   # (Bagenal et al., 2015)
electron_number_density_eq = 158E6   # [m-3] (Bagenal et al., 2015)
ion_number_density_eq = electron_number_density_eq / ion_charge_number   # [m-3]

def centrifugal_scale_height():
    return np.sqrt(2E0 * (ion_temperature + ion_charge_number * electron_temperature) * elementary_charge / 3E0 / ion_mass_number / proton_mass / Omega_Jupiter**2E0)   # [m]

H_centrifugal = centrifugal_scale_height()  # [m]
print(f'centrifugal scale height: {H_centrifugal / Radius_Jupiter} R_J')

# 自転軸の角度
alpha_rot_deg_list = [-9.6, 0.0, 9.6]   # [deg] (Thomas et al., 2004)
alpha_rot_rad_list = [np.deg2rad(alpha_rot_deg) for alpha_rot_deg in alpha_rot_deg_list]

def centrifugal_equator_mlat_rad(alpha_rot_rad):
    tan_lambda_0 = 2E0 / 3E0 * np.tan(alpha_rot_rad) / (1E0 + np.sqrt(1E0 + 8E0 / 9E0 * np.tan(alpha_rot_rad) ** 2))
    lambda_0_rad = np.arctan(tan_lambda_0)
    return lambda_0_rad

centrifugal_equator_mlat_rad_list = [centrifugal_equator_mlat_rad(alpha_rot_rad) for alpha_rot_rad in alpha_rot_rad_list]

centrifugal_equator_mlat_deg_list = [np.rad2deg(centrifugal_equator_mlat_rad) for centrifugal_equator_mlat_rad in centrifugal_equator_mlat_rad_list]


# 磁気緯度
mlat_deg_min = 0E0
mlat_deg_max = 15E0
mlat_deg_array = np.linspace(mlat_deg_min+0.2, mlat_deg_max, 10000)    # [deg]
mlat_rad_array = np.deg2rad(mlat_deg_array)   # [rad]


# distance from the magnetic equator along the magnetic field line
def distance_from_magnetic_equator(mlat_rad):
    base = np.arcsin(np.sqrt(3E0) * np.sin(mlat_rad)) / 2E0 / np.sqrt(3E0) + np.sqrt(5E0 - 3E0 * np.cos(2E0 * mlat_rad)) * np.sin(mlat_rad) / 2E0 / np.sqrt(2E0)
    return base * Radius_Jupiter * L_number # [m]

distance_from_magnetic_equator_array = distance_from_magnetic_equator(mlat_rad_array)


# height from the centrifugal equator
def height_from_centrifugal_equator_function(mlat_rad, alpha_rot_rad):
    lambda_0_rad = centrifugal_equator_mlat_rad(alpha_rot_rad)
    return Radius_Jupiter * L_number * np.cos(mlat_rad)**2E0 * np.sin(mlat_rad - lambda_0_rad)  # [m]

def distance_from_center_to_centrifugal_equator_location(mlat_rad, alpha_rot_rad):
    lambda_0_rad = centrifugal_equator_mlat_rad(alpha_rot_rad)
    return Radius_Jupiter * L_number * np.cos(mlat_rad)**2E0 * np.cos(mlat_rad - lambda_0_rad)  # [m]


# number density gradient scale length
def electron_number_density_function(mlat_rad, alpha_rot_rad):
    height_from_centrifugal_equator = height_from_centrifugal_equator_function(mlat_rad, alpha_rot_rad)
    return electron_number_density_eq * np.exp(-height_from_centrifugal_equator**2E0 / H_centrifugal**2E0)   # [m-3]

def number_density_gradient_scale_length_function(mlat_rad, alpha_rot_rad):
    lambda_0_rad = centrifugal_equator_mlat_rad(alpha_rot_rad)
    return - H_centrifugal**2E0 / 2E0 / Radius_Jupiter / L_number * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0) / np.cos(mlat_rad)**2E0 / (np.sin(mlat_rad - lambda_0_rad) * (np.cos(mlat_rad) * np.cos(mlat_rad - lambda_0_rad) - 2E0 * np.sin(mlat_rad) * np.sin(mlat_rad - lambda_0_rad)))   # [m]

height_from_centrifugal_equator_array = np.zeros((len(mlat_rad_array), len(alpha_rot_rad_list)))
distance_from_center_to_centrifugal_equator_location_array = np.zeros((len(mlat_rad_array), len(alpha_rot_rad_list)))
electron_number_density_array = np.zeros((len(mlat_rad_array), len(alpha_rot_rad_list)))
number_density_gradient_scale_length_array = np.zeros((len(mlat_rad_array), len(alpha_rot_rad_list)))

for count_i, count_j in np.ndindex(electron_number_density_array.shape):
    height_from_centrifugal_equator_array[count_i, count_j] = height_from_centrifugal_equator_function(mlat_rad_array[count_i], alpha_rot_rad_list[count_j])
    distance_from_center_to_centrifugal_equator_location_array[count_i, count_j] = distance_from_center_to_centrifugal_equator_location(mlat_rad_array[count_i], alpha_rot_rad_list[count_j])
    electron_number_density_array[count_i, count_j] = electron_number_density_function(mlat_rad_array[count_i], alpha_rot_rad_list[count_j])
    number_density_gradient_scale_length_array[count_i, count_j] = number_density_gradient_scale_length_function(mlat_rad_array[count_i], alpha_rot_rad_list[count_j])



# magnetic field strength
g01_Schmidt = 410993.4E-9  # [T] (Connerney et al., 2021)
g11_Schmidt = -71305.9E-9  # [T] (Connerney et al., 2021)
h11_Schmidt = 20958.4E-9  # [T] (Connerney et al., 2021)

def dipole_moment_function():
    return 4E0 * np.pi * Radius_Jupiter**3E0 / magnetic_constant * np.sqrt(g01_Schmidt**2E0 + g11_Schmidt**2E0 + h11_Schmidt**2E0)   # [A m2]

dipole_moment = dipole_moment_function()   # [A m2]
print(f'dipole moment: {dipole_moment} A m2')

def magnetic_flux_density(mlat_rad):
    return magnetic_constant * dipole_moment / 4E0 / np.pi / (Radius_Jupiter * L_number)**3E0 * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0) / np.cos(mlat_rad)**6E0   # [T]

magnetic_flux_density_array = magnetic_flux_density(mlat_rad_array)

def magnetic_field_gradient_scale_length_function(mlat_rad):
    return Radius_Jupiter * L_number * np.cos(mlat_rad)**2E0 * (1E0 + 3E0 * np.sin(mlat_rad)**2E0)**(3E0 / 2E0) / 3E0 / np.sin(mlat_rad) / (3E0 + 5E0 * np.sin(mlat_rad)**2E0)   # [m]

magnetic_field_gradient_scale_length_array = magnetic_field_gradient_scale_length_function(mlat_rad_array)


# wave length of the whistler mode chorus wave
def electron_cyclotron_frequency(mlat_rad):
    return elementary_charge * magnetic_flux_density(mlat_rad) / electron_mass # [rad/s]

def electron_plasma_frequency(mlat_rad, alpha_rot_rad):
    return elementary_charge / np.sqrt(electron_mass * electric_constant) * np.sqrt(electron_number_density_function(mlat_rad, alpha_rot_rad))   # [rad/s]

def whistler_mode_chorus_wave_length(mlat_rad, alpha_rot_rad, wave_frequency):
    return 2E0 * np.pi * speed_of_light / np.sqrt(wave_frequency**2E0 + wave_frequency * electron_plasma_frequency(mlat_rad, alpha_rot_rad)**2E0 / (electron_cyclotron_frequency(mlat_rad) - wave_frequency))   # [m]

# Lambda in Omura et al. (2009)
def Lambda_value_function(mlat_rad, alpha_rot_rad, wave_frequency):
    return 1E0 - (electron_cyclotron_frequency(mlat_rad) - wave_frequency) / electron_cyclotron_frequency(mlat_rad) * magnetic_field_gradient_scale_length_function(mlat_rad) / number_density_gradient_scale_length_function(mlat_rad, alpha_rot_rad)   # []

frequency_at_equator_list = np.asarray([0.25E0, 0.75E0]) * electron_cyclotron_frequency(0E0)   # [rad/s]

whistler_mode_chorus_wave_length_array = np.zeros((len(mlat_rad_array), len(alpha_rot_rad_list), len(frequency_at_equator_list)))
Lambda_value_array = np.zeros((len(mlat_rad_array), len(alpha_rot_rad_list), len(frequency_at_equator_list)))
for count_i, count_j, count_k in np.ndindex(whistler_mode_chorus_wave_length_array.shape):
    whistler_mode_chorus_wave_length_array[count_i, count_j, count_k] = whistler_mode_chorus_wave_length(mlat_rad_array[count_i], alpha_rot_rad_list[count_j], frequency_at_equator_list[count_k])
    Lambda_value_array[count_i, count_j, count_k] = Lambda_value_function(mlat_rad_array[count_i], alpha_rot_rad_list[count_j], frequency_at_equator_list[count_k])


# plot
def plot_each_alpha_rot(fig, axes, alpha_rot_index):
    ax0 = fig.add_subplot(axes[0], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$\alpha_{\mathrm{rot}} = %.1f$ [deg]' % alpha_rot_deg_list[alpha_rot_index] + '\n' + r'Ratio')
    ax1 = fig.add_subplot(axes[1], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'Length [m]', yscale='log')
    ax2 = fig.add_subplot(axes[2], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$\Lambda$')

    ax0.plot(mlat_deg_array, electron_number_density_array[:, alpha_rot_index] / electron_number_density_eq, label=r'$n_{\mathrm{e}} / n_{\mathrm{e, ceq}}$', color=r'orange', linewidth=4, alpha=0.6)
    ax0.plot(mlat_deg_array, magnetic_flux_density_array / magnetic_flux_density(0E0), label=r'$B / B_{\mathrm{eq}}$', color=r'purple', linewidth=4, alpha=0.6)
    ax0.axvline(centrifugal_equator_mlat_deg_list[alpha_rot_index], color='red', linestyle='--', linewidth=4, alpha=0.3)
    ax0.legend()

    ax1.plot(mlat_deg_array, np.abs(number_density_gradient_scale_length_array[:, alpha_rot_index]), label=r'$| L_{\mathrm{n}} |$', color=r'orange', linewidth=4, alpha=0.6)
    ax1.plot(mlat_deg_array, magnetic_field_gradient_scale_length_array, label=r'$L_{\mathrm{B}}$', color=r'purple', linewidth=4, alpha=0.6)
    ax1.plot(mlat_deg_array, whistler_mode_chorus_wave_length_array[:, alpha_rot_index, 0], label=r'$\Lambda_{%.2f}$' % (frequency_at_equator_list[0] / electron_cyclotron_frequency(0E0)), color=r'blue', linewidth=4, alpha=0.6)
    ax1.plot(mlat_deg_array, whistler_mode_chorus_wave_length_array[:, alpha_rot_index, 1], label=r'$\Lambda_{%.2f}$' % (frequency_at_equator_list[1] / electron_cyclotron_frequency(0E0)), color=r'green', linewidth=4, alpha=0.6)
    ax1.axvline(centrifugal_equator_mlat_deg_list[alpha_rot_index], color='red', linestyle='--', linewidth=4, alpha=0.3)
    ax1.legend(loc='lower right')

    ax2.plot(mlat_deg_array, Lambda_value_array[:, alpha_rot_index, 0], label=r'$\Lambda_{%.2f}$' % (frequency_at_equator_list[0] / electron_cyclotron_frequency(0E0)), color=r'blue', linewidth=4, alpha=0.6)
    ax2.plot(mlat_deg_array, Lambda_value_array[:, alpha_rot_index, 1], label=r'$\Lambda_{%.2f}$' % (frequency_at_equator_list[1] / electron_cyclotron_frequency(0E0)), color=r'green', linewidth=4, alpha=0.6)
    ax2.axvline(centrifugal_equator_mlat_deg_list[alpha_rot_index], color='red', linestyle='--', linewidth=4, alpha=0.3)
    ax2.legend()

    axes = [ax0, ax1, ax2]
    ax_number = 1
    for ax in axes:
        ax.minorticks_on()
        ax.grid(which='both', alpha=0.3)
        ax.set_xlim([mlat_deg_min, mlat_deg_max])
        ax.text(-0.35, 0.95, f'({chr(97 + alpha_rot_index)} - {ax_number})', transform=ax.transAxes)
        ax_number += 1
        
    return fig, axes

def main_plot():
    fig = plt.figure(figsize=(20, 20), dpi=100)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    for alpha_rot_index in range(len(alpha_rot_deg_list)):
        fig, axes = plot_each_alpha_rot(fig, [gs[alpha_rot_index, 0], gs[alpha_rot_index, 1], gs[alpha_rot_index, 2]], alpha_rot_index)
    
    fig.tight_layout()

    return fig

fig = main_plot()

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

fig.savefig(f'{dir_name}/scale_length_plot_Europa.png')
fig.savefig(f'{dir_name}/scale_length_plot_Europa.pdf')
plt.close(fig)
    