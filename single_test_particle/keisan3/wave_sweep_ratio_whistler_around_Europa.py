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
    #return np.sqrt(2E0 * (ion_temperature + ion_charge_number * electron_temperature) * elementary_charge / 3E0 / ion_mass_number / proton_mass / Omega_Jupiter**2E0)   # [m]
    return np.sqrt(2E0 * (ion_temperature) * elementary_charge / 3E0 / ion_mass_number / proton_mass / Omega_Jupiter**2E0)   # [m]

H_centrifugal = centrifugal_scale_height()  # [m]
print(f'centrifugal scale height: {H_centrifugal / Radius_Jupiter} R_J')

# 自転軸の角度
alpha_rot_deg_list = [-9.3, 0.0, 9.3]   # [deg] (Connerney et al., 2020)
alpha_rot_rad_list = [np.deg2rad(alpha_rot_deg) for alpha_rot_deg in alpha_rot_deg_list]

def centrifugal_equator_mlat_rad(alpha_rot_rad):
    tan_lambda_0 = 2E0 / 3E0 * np.tan(alpha_rot_rad) / (1E0 + np.sqrt(1E0 + 8E0 / 9E0 * np.tan(alpha_rot_rad) ** 2))
    lambda_0_rad = np.arctan(tan_lambda_0)
    return lambda_0_rad

centrifugal_equator_mlat_rad_list = [centrifugal_equator_mlat_rad(alpha_rot_rad) for alpha_rot_rad in alpha_rot_rad_list]

centrifugal_equator_mlat_deg_list = [np.rad2deg(centrifugal_equator_mlat_rad) for centrifugal_equator_mlat_rad in centrifugal_equator_mlat_rad_list]

print(f'centrifugal equator mlat: {centrifugal_equator_mlat_deg_list} [deg]')
#quit()


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



# wave sweep ratio
# particle parameter
typical_vperp = 0.706 * speed_of_light
typical_Bwave = 4.7E-4 * magnetic_flux_density(0E0)
typical_inhomogeneity_factor = -0.4E0

wave_frequency_array = np.asarray([0.25E0, 0.75E0]) * electron_cyclotron_frequency(0E0)   # [rad/s]

def chi_parameter(mlat_rad, alpha_rot_rad, wave_frequency):
    wave_number = whistler_mode_chorus_wave_length(mlat_rad, alpha_rot_rad, wave_frequency)
    return np.sqrt(1E0 - (wave_frequency / speed_of_light / wave_number)**2E0)

def xi_parameter(mlat_rad, alpha_rot_rad, wave_frequency):
    return np.sqrt(wave_frequency * (electron_cyclotron_frequency(mlat_rad) - wave_frequency) / electron_plasma_frequency(mlat_rad, alpha_rot_rad)**2E0)

def wave_phase_speed(mlat_rad, alpha_rot_rad, wave_frequency):
    return speed_of_light * chi_parameter(mlat_rad, alpha_rot_rad, wave_frequency) * xi_parameter(mlat_rad, alpha_rot_rad, wave_frequency)

def wave_group_velocity(mlat_rad, alpha_rot_rad, wave_frequency):
    return speed_of_light * xi_parameter(mlat_rad, alpha_rot_rad, wave_frequency) / chi_parameter(mlat_rad, alpha_rot_rad, wave_frequency) / (xi_parameter(mlat_rad, alpha_rot_rad, wave_frequency)**2E0 + electron_cyclotron_frequency(mlat_rad) / 2E0 / (electron_cyclotron_frequency(mlat_rad) - wave_frequency))

def resonance_velocity_function(mlat_rad, alpha_rot_rad, wave_frequency, vperp):
    cyclotron_wave_ratio = electron_cyclotron_frequency(mlat_rad) / wave_frequency
    phase_speed_ratio = wave_phase_speed(mlat_rad, alpha_rot_rad, wave_frequency) / speed_of_light
    vperp_ratio = vperp / speed_of_light
    V_res = wave_phase_speed(mlat_rad, alpha_rot_rad, wave_frequency) / (1E0 + cyclotron_wave_ratio**2E0 * phase_speed_ratio**2E0) * (1E0 - np.sqrt(1E0 - (1E0 + cyclotron_wave_ratio**2E0 * phase_speed_ratio**2E0) * (1E0 - cyclotron_wave_ratio**2E0 * (1E0 - vperp_ratio**2E0))))
    return V_res

def gamma(vpara, vperp):
    return (1E0 - (vpara**2E0 + vperp**2E0) / speed_of_light**2E0)**(-5E-1)

def s_0_function(chi, xi, vperp):
    return chi * vperp / xi / speed_of_light

def s_1_function(resonance_velocity, wave_group_velocity, vperp):
    gamma_s_1 = gamma(resonance_velocity, vperp)
    return gamma_s_1 * (1E0 - resonance_velocity / wave_group_velocity)**2E0

def modified_s_2_function(chi, xi, resonance_velocity, wave_phase_speed, wave_group_velocity, vperp, mlat_rad, alpha_rot_rad, wave_frequency):
    gamma_s_2 = gamma(resonance_velocity, vperp)
    cyclotron_frequency = electron_cyclotron_frequency(mlat_rad)
    magnetic_field_scale_length = magnetic_field_gradient_scale_length_function(mlat_rad)
    number_density_scale_length = number_density_gradient_scale_length_function(mlat_rad, alpha_rot_rad)
    return 1E0 / 2E0 / xi / chi * (gamma_s_2 * wave_frequency * speed_of_light / magnetic_field_scale_length * (vperp / speed_of_light)**2E0 - (2E0 + chi**2E0 * (cyclotron_frequency - gamma_s_2 * wave_frequency) / (cyclotron_frequency - wave_frequency)) * cyclotron_frequency * resonance_velocity * wave_phase_speed / speed_of_light / magnetic_field_scale_length + chi**2E0 * (cyclotron_frequency - gamma_s_2 * wave_frequency) / number_density_scale_length * resonance_velocity * wave_phase_speed / speed_of_light)

def wave_sweep_rate_function(inhomogeneity_factor, alpha_rot_rad, mlat_rad, wave_frequency, Bwave, vperp):
    resonance_velocity = resonance_velocity_function(mlat_rad, alpha_rot_rad, wave_frequency, vperp)
    phase_speed = wave_phase_speed(mlat_rad, alpha_rot_rad, wave_frequency)
    group_velocity = wave_group_velocity(mlat_rad, alpha_rot_rad, wave_frequency)
    chi = chi_parameter(mlat_rad, alpha_rot_rad, wave_frequency)
    xi = xi_parameter(mlat_rad, alpha_rot_rad, wave_frequency)
    s_0 = s_0_function(chi, xi, vperp)
    s_1 = s_1_function(resonance_velocity, group_velocity, vperp)
    modified_s_2 = modified_s_2_function(chi, xi, resonance_velocity, phase_speed, group_velocity, vperp, mlat_rad, alpha_rot_rad, wave_frequency)
    wave_sweep_rate = - s_0 / s_1 * inhomogeneity_factor * wave_frequency * elementary_charge * Bwave / electron_mass - modified_s_2 / s_1
    return wave_sweep_rate, s_0, s_1, modified_s_2


wave_sweep_rate_array = np.zeros((len(mlat_rad_array), len(alpha_rot_rad_list), len(wave_frequency_array)))
s_0_array = np.zeros((len(mlat_rad_array), len(alpha_rot_rad_list), len(wave_frequency_array)))
s_1_array = np.zeros((len(mlat_rad_array), len(alpha_rot_rad_list), len(wave_frequency_array)))
modified_s_2_array = np.zeros((len(mlat_rad_array), len(alpha_rot_rad_list), len(wave_frequency_array)))

for count_mlat, count_alpha, count_wave in np.ndindex(wave_sweep_rate_array.shape):
    save_wave_sweep_rate, save_s_0, save_s_1, save_modified_s_2 = wave_sweep_rate_function(typical_inhomogeneity_factor, alpha_rot_rad_list[count_alpha], mlat_rad_array[count_mlat], wave_frequency_array[count_wave], typical_Bwave, typical_vperp)
    wave_sweep_rate_array[count_mlat, count_alpha, count_wave] = save_wave_sweep_rate
    s_0_array[count_mlat, count_alpha, count_wave] = save_s_0
    s_1_array[count_mlat, count_alpha, count_wave] = save_s_1
    modified_s_2_array[count_mlat, count_alpha, count_wave] = save_modified_s_2

# plot
def plot_each_alpha_rot(fig, axes, alpha_rot_index):
    ax0 = fig.add_subplot(axes[0], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$\alpha_{\mathrm{rot}} = %.1f$ [deg]' % alpha_rot_deg_list[alpha_rot_index] + '\n' + r'$1 / \Omega_{\mathrm{e}} \, \partial \omega / \partial t$ [$1 / \mathrm{s}$]')
    
    alpha_rot_rad = alpha_rot_rad_list[alpha_rot_index]
    lambda_0 = centrifugal_equator_mlat_rad(alpha_rot_rad)

    ax0.plot(mlat_deg_array, wave_sweep_rate_array[:, alpha_rot_index, 0] / electron_cyclotron_frequency(mlat_rad_array), label=r'$\omega_{%.2f}$' % (wave_frequency_array[0] / electron_cyclotron_frequency(0E0)), color=r'blue', linewidth=4, alpha=0.6)
    ax0.plot(mlat_deg_array, wave_sweep_rate_array[:, alpha_rot_index, 1] / electron_cyclotron_frequency(mlat_rad_array), label=r'$\omega_{%.2f}$' % (wave_frequency_array[1] / electron_cyclotron_frequency(0E0)), color=r'green', linewidth=4, alpha=0.6)
    ax0.legend()

    ax0.minorticks_on()
    ax0.grid(which='both', alpha=0.3)
    ax0.set_xlim([mlat_deg_min, mlat_deg_max])
    ax0.text(-0.15, 0.95, f'({alpha_rot_index+1})', transform=ax0.transAxes)

    axes = [ax0]

    return fig, axes

def main_plot():
    fig = plt.figure(figsize=(10, 30), dpi=100)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
    
    for alpha_rot_index in range(len(alpha_rot_deg_list)):
        fig, axes = plot_each_alpha_rot(fig, [gs[alpha_rot_index]], alpha_rot_index)
    
    fig.tight_layout()

    return fig

fig = main_plot()

fig.savefig(f'test.png')
plt.close(fig)