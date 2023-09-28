import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import datetime
import os
from multiprocessing import Pool


# constants
speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]
electric_constant = 8.8541878128E-12  #[F m-1]
magnetic_constant = 1.25663706212E-6  #[N A-2]

#planet condition
planet_radius   = 7.1492E7  #[m]
lshell_number   = 5.91E0
r_eq            = planet_radius * lshell_number #[m]

def d_mlat_d_z(mlat_rad):
    return 1E0 / r_eq / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)    #[rad/m]

#gradient function
diff_rad = 1E-5
def gradient_meter(function, mlat_rad):
    if mlat_rad < diff_rad:
        return 0E0
    else:
        return (function(mlat_rad + diff_rad) - function(mlat_rad - diff_rad)) / 2E0 / diff_rad * d_mlat_d_z(mlat_rad)    #[m^-1]
    
def gradient_mlat(function, mlat_rad, pitch_angle_rad):
    if mlat_rad < diff_rad:
        return 0E0
    else:
        return (function(pitch_angle_rad, mlat_rad + diff_rad) - function(pitch_angle_rad, mlat_rad - diff_rad)) / 2E0 / diff_rad #[rad^-1]
    
def gradient_mlat_psi(function, mu, mlat_rad, psi):
    if mlat_rad < diff_rad:
        return 0E0
    else:
        return (function(mu, mlat_rad + diff_rad, psi) - function(mu, mlat_rad - diff_rad, psi)) / 2E0 / diff_rad #[rad^-1]

# magnetic field

dipole_moment   = 4.2E-4 * (4E0 * np.pi * planet_radius**3E0) / magnetic_constant #[Am]
B0_eq           = (1E-7 * dipole_moment) / r_eq**3E0

def magnetic_flux_density(mlat_rad):
    return B0_eq / np.cos(mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)     #[T]


# background plasma parameters
oxygen_number_density_eq = 1.163E9    # [m^-3]
oxygen_number_density_scale_height = 1.2842    # [R_J]
oxygen_mass = 2.677950266103E-26   # [kg] # O+
oxygen_temperature = 5E1   # [eV]

proton_number_density_eq = 5.8E7    # [m^-3]
proton_mass = 1.672621898E-27    # [kg]
proton_temperature = 8.6E0    # [eV]

electron_mass = 9.10938356E-31    # [kg]
electron_temperature_eq = 5E0  # [eV]

def coordinate_FA(mlat_rad):
    return r_eq * (np.arcsinh(np.sqrt(3)*np.sin(mlat_rad)) / 2E0 / np.sqrt(3) + np.sin(mlat_rad) * np.sqrt(5E0 - 3E0 * np.cos(2E0 * mlat_rad)) / 2E0 / np.sqrt(2)) #[m]

def oxygen_number_density(mlat_rad):
    return oxygen_number_density_eq * np.exp(- (coordinate_FA(mlat_rad) / planet_radius / oxygen_number_density_scale_height)**2E0)    #[m^-3]

def proton_number_density(mlat_rad):
    return proton_number_density_eq     #[m^-3]

def number_density(mlat_rad):
    return oxygen_number_density(mlat_rad) + proton_number_density(mlat_rad)    #[m^-3]

def ion_temperature(mlat_rad):
    oxygen_pressure = oxygen_number_density(mlat_rad) * oxygen_temperature * elementary_charge    #[Pa]
    proton_pressure = proton_number_density(mlat_rad) * proton_temperature * elementary_charge    #[Pa]
    return (oxygen_pressure + proton_pressure) / (oxygen_number_density(mlat_rad) + proton_number_density(mlat_rad)) / elementary_charge    #[eV]

def tau(mlat_rad):
    return ion_temperature(mlat_rad) / electron_temperature_eq

def ion_mass(mlat_rad):
    return (oxygen_mass * oxygen_number_density(mlat_rad) + proton_mass * proton_number_density(mlat_rad)) / number_density(mlat_rad)    #[kg]

def ion_thermal_speed(mlat_rad):
    return np.sqrt(2E0 * ion_temperature(mlat_rad) * elementary_charge / ion_mass(mlat_rad))    #[m/s]

def ion_gyrofrequency(mlat_rad):
    return elementary_charge * magnetic_flux_density(mlat_rad) / ion_mass(mlat_rad)    #[s^-1]

def ion_gyroradius(mlat_rad):
    return ion_thermal_speed(mlat_rad) / ion_gyrofrequency(mlat_rad)    #[m]

def electron_thermal_speed(mlat_rad):
    return np.sqrt(2E0 * electron_temperature_eq * elementary_charge / electron_mass)    #[m/s]


def Alfven_speed(mlat_rad):
    return magnetic_flux_density(mlat_rad) / np.sqrt(magnetic_constant * number_density(mlat_rad) * ion_mass(mlat_rad))    #[m/s]

def plasma_beta_ion(mlat_rad):
    return 2E0 * magnetic_constant * number_density(mlat_rad) * ion_temperature(mlat_rad) * elementary_charge / magnetic_flux_density(mlat_rad)**2E0  #[]

mu_upper_limit = 1E3 * elementary_charge / magnetic_flux_density(0E0)    #[J T-1]


# wave parameters
uperp_eq = 5.4E4    #[m/s]
epsilon_wave = 1E0 / 1000E0
kperp_rhoi = 2E0 * np.pi #[rad]

def kperp(mlat_rad):
    return kperp_rhoi / ion_gyroradius(mlat_rad)    #[rad]

def wave_length_perp(mlat_rad):
    return 2E0 * np.pi / kperp(mlat_rad)    #[m]

#kpara_eq = ion_gyrofrequency(0E0) / electron_thermal_speed(0E0)    #[rad m-1]
kpara_eq = epsilon_wave * kperp(0E0)    #[rad m-1]

wave_frequency = np.sqrt((1E0 + tau(0E0)) / (plasma_beta_ion(0E0) * (1E0 + tau(0E0)) + 2E0 * tau(0E0))) * kperp_rhoi * kpara_eq * Alfven_speed(0E0)    #[rad/s]

def kpara(mlat_rad):
    return wave_frequency / Alfven_speed(mlat_rad) / kperp_rhoi * np.sqrt(plasma_beta_ion(mlat_rad) + 2E0 * tau(mlat_rad) / (1E0 + tau(mlat_rad)))    #[rad/m]

def wave_length_para(mlat_rad):
    return 2E0 * np.pi / kpara(mlat_rad)    #[m]

def wave_phase_speed(mlat_rad):
    return wave_frequency / kpara(mlat_rad)    #[m/s]

electric_field_eq = 1E-3    #[V/m]

def wave_modified_potential(mlat_rad):
    return electric_field_eq * wave_length_para(0E0)    #[V]

def wave_scalar_potential(mlat_rad):
    return wave_modified_potential(mlat_rad) / (2E0 + 1E0 / tau(mlat_rad))    #[V]

def energy_wave_phase_speed(mlat_rad):
    return 5E-1 * electron_mass * wave_phase_speed(mlat_rad)**2E0 #[J]

def energy_wave_potential(mlat_rad):
    return elementary_charge * wave_modified_potential(mlat_rad)    #[J]

def diff_B0(mlat_rad):
    return gradient_meter(magnetic_flux_density, mlat_rad) / kpara(mlat_rad) / magnetic_flux_density(mlat_rad) #[rad^-1]

def diff_ion_number_density(mlat_rad):
    return gradient_meter(number_density, mlat_rad) / kpara(mlat_rad) / number_density(mlat_rad) #[rad^-1]

def diff_ion_temperature(mlat_rad):
    return gradient_meter(ion_temperature, mlat_rad) / kpara(mlat_rad) / ion_temperature(mlat_rad) #[rad^-1]

def diff_1_plus_tau(mlat_rad):
    return gradient_meter(tau, mlat_rad) / kpara(mlat_rad) / (1E0 + tau(mlat_rad)) #[rad^-1]

def delta(mlat_rad):
    return diff_B0(mlat_rad)    #[rad^-1]

def yy(mlat_rad):
    return plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]

def epsilon(mlat_rad):
    return (1E0 + 2E0 * yy(mlat_rad)) * diff_B0(mlat_rad) - (1E0 + yy(mlat_rad)) * diff_ion_number_density(mlat_rad) - yy(mlat_rad) * diff_ion_temperature(mlat_rad) - yy(mlat_rad) / plasma_beta_ion(mlat_rad) / (1E0 + tau(mlat_rad)) * diff_1_plus_tau(mlat_rad)    #[rad^-1]


# energy trajectory
number_pitch_angle = 30
initial_pitch_angle_deg = np.linspace(0E0, 89E0, number_pitch_angle)
initial_pitch_angle_rad = initial_pitch_angle_deg / 180E0 * np.pi

def function_0(pitch_angle_rad, mlat_rad):
    return (energy_wave_potential(mlat_rad) - epsilon(mlat_rad) * energy_wave_phase_speed(mlat_rad)) * np.cos(pitch_angle_rad)**2E0 - delta(mlat_rad) * energy_wave_phase_speed(mlat_rad)

def Newton_method_function_0(pitch_angle_rad):
    initial_mlat_rad = np.pi / 4E0
    mlat_rad_before_update = initial_mlat_rad
    count_iteration = 0
    while True:
        diff = function_0(pitch_angle_rad, mlat_rad_before_update) / gradient_mlat(function_0, mlat_rad_before_update, pitch_angle_rad)
        if abs(diff) > 1E-1:
            diff = np.sign(diff)
        mlat_rad_after_update = mlat_rad_before_update - diff
        if abs(mlat_rad_after_update - mlat_rad_before_update) < 1E-10:
            #print(count_iteration, mlat_rad_after_update, function_0(pitch_angle_rad, mlat_rad_after_update))
            break
        else:
            mlat_rad_before_update = mlat_rad_after_update
            if mlat_rad_after_update < 0E0 or mlat_rad_after_update > np.pi / 2E0:
                mlat_rad_before_update = np.mod(mlat_rad_before_update, np.pi / 2E0)
            count_iteration += 1
            #if np.mod(count_iteration, 100) == 0:
                #print(count_iteration, mlat_rad_after_update)
                #print(function_0(pitch_angle_rad, mlat_rad_after_update), gradient_mlat(function_0, mlat_rad_after_update, pitch_angle_rad))
    
    return mlat_rad_after_update

def function_1(mu, mlat_rad, psi):
    return delta(mlat_rad) * mu * magnetic_flux_density(mlat_rad) - energy_wave_potential(mlat_rad) + (delta(mlat_rad) + epsilon(mlat_rad)) * (np.sqrt(energy_wave_phase_speed(mlat_rad)) - np.sqrt(np.cos(psi) - psi - np.pi / 2E0) * np.sqrt(energy_wave_potential(mlat_rad)))**2E0

def Newton_method_function_1(mu_const, mlat_rad_ini, psi):
    mlat_rad_before_update = mlat_rad_ini
    count_iteration_1 = 0
    while True:
        diff = function_1(mu_const, mlat_rad_before_update, psi) / gradient_mlat_psi(function_1, mu_const, mlat_rad_before_update, psi)
        if abs(diff) > 1E-1:
            diff = np.sign(diff)
        mlat_rad_after_update = mlat_rad_before_update - diff
        if abs(mlat_rad_after_update - mlat_rad_before_update) < 1E-10:
            break
        else:
            mlat_rad_before_update = mlat_rad_after_update
            if mlat_rad_after_update < 0E0 or mlat_rad_after_update > np.pi / 2E0:
                mlat_rad_before_update = np.mod(mlat_rad_before_update, np.pi / 2E0)
            count_iteration_1 += 1
            if count_iteration_1 > 100000:
                return np.nan
    
    return mlat_rad_after_update

def energy_result(mlat_rad, psi):
    return 1E0 / delta(mlat_rad) * (energy_wave_potential(mlat_rad) - epsilon(mlat_rad) * (np.sqrt(energy_wave_phase_speed(mlat_rad)) - np.sqrt(np.cos(psi) - psi - np.pi / 2E0) * np.sqrt(energy_wave_potential(mlat_rad)))**2E0)

wave_phase = np.linspace(- np.pi, - np.pi / 2E0, 1000)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

fig = plt.figure(figsize=(14, 14), dpi=100)
ax = fig.add_subplot(111, xlabel=r'MLAT [deg]', ylabel=r'Energy [eV]', xlim=(0E0, 50E0), ylim=(1E0, 1E5), yscale='log')

# turboカラーマップを取得
cmap_color = cm.get_cmap('cool')
colors = [cmap_color(i) for i in np.linspace(0, 1, number_pitch_angle)]

#for count_i in range(number_pitch_angle):
#    mlat_rad_initial = Newton_method_function_0(initial_pitch_angle_rad[count_i])
#    mu_const = 1E0 / delta(mlat_rad_initial) / magnetic_flux_density(mlat_rad_initial) * (energy_wave_potential(mlat_rad_initial) - (delta(mlat_rad_initial) + epsilon(mlat_rad_initial)) * energy_wave_phase_speed(mlat_rad_initial))
#    
#    if mu_const > mu_upper_limit:
#        continue
#    
#    mlat_rad_trajectory = np.zeros(len(wave_phase))
#    energy_result_trajectory = np.zeros(len(wave_phase))
#    for count_j in range(len(wave_phase)):
#        mlat_rad_trajectory[count_j] = Newton_method_function_1(mu_const, mlat_rad_initial, wave_phase[count_j])
#        energy_result_trajectory[count_j] = energy_result(mlat_rad_trajectory[count_j], wave_phase[count_j])
#    mlat_deg_trajectory = mlat_rad_trajectory / np.pi * 180E0
#    energy_result_eV_trajectory = energy_result_trajectory / elementary_charge
#    ax.plot(mlat_deg_trajectory, energy_result_eV_trajectory, linewidth=4, alpha=0.5, color=colors[count_i])
#
#    #時刻を取得
#    now = datetime.datetime.now()
#    print(count_i, now)

def main(args):
    count_i = args

    mlat_rad_initial = Newton_method_function_0(initial_pitch_angle_rad[count_i])
    mu_const = 1E0 / delta(mlat_rad_initial) / magnetic_flux_density(mlat_rad_initial) * (energy_wave_potential(mlat_rad_initial) - (delta(mlat_rad_initial) + epsilon(mlat_rad_initial)) * energy_wave_phase_speed(mlat_rad_initial))
    
    if mu_const > mu_upper_limit:
        return None  # Noneを返して、結果リストでフィルタリングする
    
    mlat_rad_trajectory = np.zeros(len(wave_phase))
    energy_result_trajectory = np.zeros(len(wave_phase))
    for count_j in range(len(wave_phase)):
        mlat_rad_trajectory[count_j] = Newton_method_function_1(mu_const, mlat_rad_initial, wave_phase[count_j])
        energy_result_trajectory[count_j] = energy_result(mlat_rad_trajectory[count_j], wave_phase[count_j])
    mlat_deg_trajectory = mlat_rad_trajectory / np.pi * 180E0
    energy_result_eV_trajectory = energy_result_trajectory / elementary_charge

    #時刻を取得
    now = datetime.datetime.now()
    print(count_i, now)
    
    # mlat_deg_trajectoryとenergy_result_eV_trajectoryのペアを返す
    return (mlat_deg_trajectory, energy_result_eV_trajectory)

if __name__ == '__main__':
    num_processes = 16
    # count_iに渡す引数のリスト
    args = range(number_pitch_angle)
    # 並列処理
    with Pool(num_processes) as p:
        results = p.map(main, args)
    
    # プロットは並列処理の後で実行
    for count_i, result in enumerate(results):
        if result is not None:  # Noneの結果をフィルタリング
            mlat_deg_trajectory, energy_result_eV_trajectory = result
            ax.plot(mlat_deg_trajectory, energy_result_eV_trajectory, linewidth=4, alpha=0.5, color=colors[count_i])



mlat_deg_array = np.linspace(0E0, 50E0, 1000)
mlat_rad_array = mlat_deg_array / 180E0 * np.pi
energy_wave_phase_speed_eV = energy_wave_phase_speed(mlat_rad_array) / elementary_charge
energy_wave_potential_eV = energy_wave_potential(mlat_rad_array) / elementary_charge * np.ones(len(mlat_deg_array))

energy_S_1_upper_limit_eV = np.zeros(len(mlat_deg_array))
energy_S_1_lower_limit_eV = np.zeros(len(mlat_deg_array))
for count_i in range(len(mlat_deg_array)):
    energy_S_1_upper_limit_eV[count_i] = energy_wave_potential_eV[count_i] / delta(mlat_rad_array[count_i])
    energy_S_1_lower_limit_eV[count_i] = energy_wave_potential_eV[count_i] / (delta(mlat_rad_array[count_i]) + epsilon(mlat_rad_array[count_i]))

ax.plot(mlat_deg_array, energy_wave_phase_speed_eV, color='red', linewidth=4, label=r'$K_{\mathrm{ph \parallel}}$', alpha=0.6)
ax.plot(mlat_deg_array, energy_wave_potential_eV, color='green', linewidth=4, label=r'$K_{\mathrm{E}}$', alpha=0.6)
ax.plot(mlat_deg_array, energy_S_1_upper_limit_eV, color='blue', linewidth=4, label=r'$S = 1$ range', alpha=0.6)
ax.plot(mlat_deg_array, energy_S_1_lower_limit_eV, color='blue', linewidth=4, alpha=0.6)

ax.set_xlim(26E0, 30E0)
ax.set_ylim(6E3, 1E4)

ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

ax.legend()

#カラーバーの設定
norm = plt.Normalize(vmin=0, vmax=90)
sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=np.linspace(0, 90, 10))
cbar.set_label(r'$\alpha_{\mathrm{detrap, initial}}$ [deg]')


plt.tight_layout()

plt.savefig('/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_1_trajectory_Jupiter_L_5.91_enlarged.png', dpi=100)
plt.show()