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

L_number = 5.91  
Radius_Jupiter = 71492E3   # [m] (equatorial radius)

Time_rotation_Jupiter = 9.9258E0 * 3600E0   # [s]
Omega_Jupiter = 2E0 * np.pi / Time_rotation_Jupiter   # [rad/s]

# 自転軸の角度
alpha_rot_deg_list = [-9.3, 0.0, 9.3]   # [deg] (Connerney et al., 2020)
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


# centrifugal scale height model (Phipps et al., 2018)
ion_temperature = 81.4E0   # [eV] (Phipps et al., 2018)
electron_temperature = 5E0   # [eV] (Bagenal, 1994)
tau = ion_temperature / electron_temperature
ion_charge_number = 1E0   #
ion_mass_number = 24.4E0   # (Phipps et al., 2018)
electron_number_density_eq = 3350E6   # [m-3] (Phipps et al., 2018)

def centrifugal_scale_height():
    #return np.sqrt(2E0 * (ion_temperature + ion_charge_number * electron_temperature) * elementary_charge / 3E0 / ion_mass_number / proton_mass / Omega_Jupiter**2E0)   # [m]
    return np.sqrt(2E0 * (ion_temperature) * elementary_charge / 3E0 / ion_mass_number / proton_mass / Omega_Jupiter**2E0)   # [m]

H_centrifugal = centrifugal_scale_height()   # [m]
print(f'centrifugal scale height: {H_centrifugal / Radius_Jupiter} R_J')


# height from the centrifugal equator
def height_from_centrifugal_equator_function(mlat_rad, alpha_rot_rad):
    lambda_0_rad = centrifugal_equator_mlat_rad(alpha_rot_rad)
    #return Radius_Jupiter * L_number * np.cos(mlat_rad)**2E0 * np.sin(mlat_rad - lambda_0_rad)  # [m]
    return distance_from_magnetic_equator(mlat_rad) - distance_from_magnetic_equator(lambda_0_rad)  # [m] # こちらの方が正確

def distance_from_center_to_centrifugal_equator_location(mlat_rad, alpha_rot_rad):
    lambda_0_rad = centrifugal_equator_mlat_rad(alpha_rot_rad)
    return Radius_Jupiter * L_number * np.cos(mlat_rad)**2E0 * np.cos(mlat_rad - lambda_0_rad)  # [m]

# number density gradient scale length
def electron_number_density_function(mlat_rad, alpha_rot_rad):
    height_from_centrifugal_equator = height_from_centrifugal_equator_function(mlat_rad, alpha_rot_rad)
    return electron_number_density_eq * np.exp(-height_from_centrifugal_equator**2E0 / H_centrifugal**2E0)   # [m-3]

def number_density_gradient_scale_length_function(mlat_rad, alpha_rot_rad):
    #lambda_0_rad = centrifugal_equator_mlat_rad(alpha_rot_rad)
    #return - H_centrifugal**2E0 / 2E0 / Radius_Jupiter / L_number * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0) / np.cos(mlat_rad)**2E0 / (np.sin(mlat_rad - lambda_0_rad) * (np.cos(mlat_rad) * np.cos(mlat_rad - lambda_0_rad) - 2E0 * np.sin(mlat_rad) * np.sin(mlat_rad - lambda_0_rad)))   # [m]
    return - H_centrifugal**2E0 / 2E0 / height_from_centrifugal_equator_function(mlat_rad, alpha_rot_rad)

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


# Alfven speed
def Alfven_speed_function(mlat_rad, alpha_rot_rad):
    #return magnetic_flux_density(mlat_rad) / np.sqrt(magnetic_constant * electron_number_density_function(mlat_rad, alpha_rot_rad) / ion_charge_number * ion_mass_number * proton_mass)   # [m s-1]
    v_A_base = magnetic_flux_density(mlat_rad) / np.sqrt(magnetic_constant * electron_number_density_function(mlat_rad, alpha_rot_rad) / ion_charge_number * ion_mass_number * proton_mass)   # [m s-1]
    return v_A_base / np.sqrt(1 + v_A_base**2E0 / speed_of_light**2E0)

# ion Larmor radius
def ion_Larmor_radius_function(mlat_rad):
    return np.sqrt(2E0 * ion_mass_number * proton_mass * ion_temperature * elementary_charge) / elementary_charge / ion_charge_number / magnetic_flux_density(mlat_rad)   # [m]

# ion plasma beta
def ion_plasma_beta_function(mlat_rad, alpha_rot_rad):
    return 2E0 * magnetic_constant * electron_number_density_function(mlat_rad, alpha_rot_rad) / ion_charge_number * ion_temperature * elementary_charge / magnetic_flux_density(mlat_rad)**2E0   # []

# perpendicular KAW wavelength
kperp_rho_i = 2E0 * np.pi
def perpendicular_KAW_wavelength_function(mlat_rad):
    return 2E0 * np.pi * ion_Larmor_radius_function(mlat_rad) / kperp_rho_i   # [m]

# parallel KAW wavelength
wave_frequency = 2.5E0   # [Hz]
wave_anguler_frequency = wave_frequency * 2E0 * np.pi   # [rad s-1]

def parallel_KAW_wavelength_function(mlat_rad, alpha_rot_rad):
    return (2E0 * np.pi)**2E0 * Alfven_speed_function(mlat_rad, alpha_rot_rad) / wave_anguler_frequency * np.sqrt((1E0 + tau) / (ion_plasma_beta_function(mlat_rad, alpha_rot_rad) * (1E0 + tau) + 2E0 * tau))   # [m]

# KAW pitch angle coefficient
def KAW_pitch_angle_coefficient_function(mlat_rad, alpha_rot_rad):
    return 1E0 + 2E0 * ion_plasma_beta_function(mlat_rad, alpha_rot_rad) * (1E0 + tau) / (ion_plasma_beta_function(mlat_rad, alpha_rot_rad) * (1E0 + tau) + 2E0 * tau)   # []


ion_Larmor_radius_array = ion_Larmor_radius_function(magnetic_flux_density_array)
perpendicular_KAW_wavelength_array = perpendicular_KAW_wavelength_function(magnetic_flux_density_array)
Alfven_speed_array = np.zeros_like(electron_number_density_array)
ion_plasma_beta_array = np.zeros_like(electron_number_density_array)
parallel_KAW_wavelength_array = np.zeros_like(electron_number_density_array)
KAW_pitch_angle_coefficient_array = np.zeros_like(electron_number_density_array)
for count_i in range(len(alpha_rot_rad_list)):
    Alfven_speed_array[:, count_i] = Alfven_speed_function(mlat_rad_array, alpha_rot_rad_list[count_i])
    ion_plasma_beta_array[:, count_i] = ion_plasma_beta_function(mlat_rad_array, alpha_rot_rad_list[count_i])
    parallel_KAW_wavelength_array[:, count_i] = parallel_KAW_wavelength_function(mlat_rad_array, alpha_rot_rad_list[count_i])
    KAW_pitch_angle_coefficient_array[:, count_i] = KAW_pitch_angle_coefficient_function(mlat_rad_array, alpha_rot_rad_list[count_i])


# plot
def plot_each_alpha_rot(fig, axes, alpha_rot_index):
    ax0 = fig.add_subplot(axes[0], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$\alpha_{\mathrm{rot}} = %.1f$ [deg]' % alpha_rot_deg_list[alpha_rot_index] + '\n' + r'Ratio')
    ax1 = fig.add_subplot(axes[1], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'Length [m]', yscale='log')
    ax2 = fig.add_subplot(axes[2], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$\mathrm{max} (|S_{\mathrm{n}} / S_{\mathrm{B}}|)$')

    alpha_rot_rad = alpha_rot_rad_list[alpha_rot_index]
    lambda_0 = centrifugal_equator_mlat_rad(alpha_rot_rad)

    ax0.plot(mlat_deg_array, electron_number_density_array[:, alpha_rot_index] / electron_number_density_function(lambda_0, alpha_rot_rad), label=r'$n_{0} / n_{0} (\lambda_{\mathrm{ceq}})$', color=r'orange', linewidth=4, alpha=0.6)
    ax0.plot(mlat_deg_array, magnetic_flux_density_array / magnetic_flux_density(0E0), label=r'$B_{0} / B_{0} (0)$', color=r'purple', linewidth=4, alpha=0.6)
    ax0.plot(mlat_deg_array, Alfven_speed_array[:, alpha_rot_index] / Alfven_speed_function(lambda_0, alpha_rot_rad), label=r'$v_{\mathrm{A}} / v_{\mathrm{A}} (\lambda_{\mathrm{ceq}})$', color=r'blue', linewidth=4, alpha=0.6)
    ax0.plot(mlat_deg_array, ion_plasma_beta_array[:, alpha_rot_index] / ion_plasma_beta_function(lambda_0, alpha_rot_rad), label=r'$\beta_{\mathrm{i}} / \beta_{\mathrm{i}} (\lambda_{\mathrm{ceq}})$', color=r'green', linewidth=4, alpha=0.6)
    ax0.axvline(centrifugal_equator_mlat_deg_list[alpha_rot_index], color='red', linestyle='--', linewidth=4, alpha=0.3)
    ax0.legend()
    ax0_ylim = ax0.get_ylim()
    if ax0_ylim[0] < 0E0:
        ax0.set_ylim([0E0, ax0_ylim[1]])
    ax0.set_title(r'$v_{\mathrm{A, ceq}} = %.2g$ c, $\beta_{\mathrm{i, ceq}} = %.2g$' % (Alfven_speed_function(lambda_0, alpha_rot_rad) / speed_of_light, ion_plasma_beta_function(lambda_0, alpha_rot_rad)))

    ax1.plot(mlat_deg_array, np.abs(number_density_gradient_scale_length_array[:, alpha_rot_index]), label=r'$| L_{\mathrm{n}} |$', color=r'orange', linewidth=4, alpha=0.6)
    ax1.plot(mlat_deg_array, magnetic_field_gradient_scale_length_array, label=r'$L_{\mathrm{B}}$', color=r'purple', linewidth=4, alpha=0.6)
    #ax1.plot(mlat_deg_array, perpendicular_KAW_wavelength_array, label=r'$L_{\mathrm{w} \perp}$', color=r'blue', linewidth=4, alpha=0.6)
    #ax1.plot(mlat_deg_array, parallel_KAW_wavelength_array[:, alpha_rot_index], label=r'$L_{\mathrm{w} \parallel}$', color=r'green', linewidth=4, alpha=0.6)
    ax1.axvline(centrifugal_equator_mlat_deg_list[alpha_rot_index], color='red', linestyle='--', linewidth=4, alpha=0.3)
    ax1.legend()

    ax2.plot(mlat_deg_array, np.abs(magnetic_field_gradient_scale_length_array / number_density_gradient_scale_length_array[:, alpha_rot_index] / 2E0), color=r'orange', linewidth=4, alpha=0.6)
    ax2.axvline(centrifugal_equator_mlat_deg_list[alpha_rot_index], color='red', linestyle='--', linewidth=4, alpha=0.3)

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

#fig.savefig(f'{dir_name}/scale_length_plot_Io.png') #_except_wavelength.png')
#fig.savefig(f'{dir_name}/scale_length_plot_Io.pdf') #_except_wavelength.pdf')
fig.savefig(f'{dir_name}/scale_length_plot_Io_except_wavelength.png')
fig.savefig(f'{dir_name}/scale_length_plot_Io_except_wavelength.pdf')

plt.close(fig)