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

mu_upper_limit = 1E3 * elementary_charge / magnetic_flux_density(0E0)    #[J T-1]


# wave parameters
kperp_rhoi = 2E0 * np.pi    #[rad]
wave_frequency = 2E0 * np.pi * 0.15    #[rad/s]

def wave_phase_speed(mlat_rad):
    return Alfven_speed(mlat_rad) * kperp_rhoi * np.sqrt((1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad)))    #[m/s]

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
    return 1E0 / kpara(mlat_rad) / magnetic_flux_density(mlat_rad) * gradient_meter(magnetic_flux_density, mlat_rad)    #[rad^-1]

def epsilon(mlat_rad):
    return delta(mlat_rad) * (3E0 - 4E0 * tau(mlat_rad) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad)))    #[rad^-1]


# energy trajectory
number_pitch_angle = 30
initial_pitch_angle_deg = np.linspace(1E0, 89E0, number_pitch_angle)
initial_pitch_angle_rad = initial_pitch_angle_deg / 180E0 * np.pi

def function_0(pitch_angle_rad, mlat_rad):
    return (energy_wave_potential(mlat_rad) - epsilon(mlat_rad) * energy_wave_phase_speed(mlat_rad)) * np.cos(pitch_angle_rad)**2E0 - delta(mlat_rad) * energy_wave_phase_speed(mlat_rad)

def Newton_method_function_0(pitch_angle_rad):
    initial_mlat_rad = np.pi / 2E0
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
            #    print(count_iteration, mlat_rad_after_update)
    
    return mlat_rad_after_update

def function_1(mu, mlat_rad, psi):
    return delta(mlat_rad) * mu * magnetic_flux_density(mlat_rad) - energy_wave_potential(mlat_rad) + (delta(mlat_rad) + epsilon(mlat_rad)) * (np.sqrt(energy_wave_phase_speed(mlat_rad)) - np.sqrt(np.cos(psi) - psi - np.pi / 2E0) * np.sqrt(energy_wave_potential(mlat_rad)))**2E0  #initial_wave_phase = - np.pi / 2E0
    #return delta(mlat_rad) * mu * magnetic_flux_density(mlat_rad) - energy_wave_potential(mlat_rad) + (delta(mlat_rad) + epsilon(mlat_rad)) * (np.sqrt(energy_wave_phase_speed(mlat_rad)) - np.sqrt(np.cos(psi) - 2E0 * psi - 1E0) * np.sqrt(energy_wave_potential(mlat_rad)))**2E0 #initial_wave_phase = 0E0

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
    return 1E0 / delta(mlat_rad) * (energy_wave_potential(mlat_rad) - epsilon(mlat_rad) * (np.sqrt(energy_wave_phase_speed(mlat_rad)) - np.sqrt(np.cos(psi) - psi - np.pi / 2E0) * np.sqrt(energy_wave_potential(mlat_rad)))**2E0)  #initial_wave_phase = - np.pi / 2E0
    #return 1E0 / delta(mlat_rad) * (energy_wave_potential(mlat_rad) - epsilon(mlat_rad) * (np.sqrt(energy_wave_phase_speed(mlat_rad)) - np.sqrt(np.cos(psi) - 2E0 * psi - 1E0) * np.sqrt(energy_wave_potential(mlat_rad)))**2E0)  #initial_wave_phase = 0E0


wave_phase_number = 10000
initial_wave_phase = -np.pi/2E0
#initial_wave_phase = 0E0
wave_phase = np.linspace(- np.pi, initial_wave_phase, wave_phase_number)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

fig = plt.figure(figsize=(14, 14), dpi=100)
ax = fig.add_subplot(111, xlabel=r'MLAT [deg]', ylabel=r'Kinetic energy $K$ [eV]', yscale='log')

# turboカラーマップを取得
cmap_color = cm.cool
#colors = [cmap_color(i) for i in np.linspace(0, 1, number_pitch_angle)]
colors = [cmap_color(i) for i in np.linspace(0, 1, number_pitch_angle)]

def main(args):
    count_i = args

    mlat_rad_initial = Newton_method_function_0(initial_pitch_angle_rad[count_i])
    mu_const = 1E0 / delta(mlat_rad_initial) / magnetic_flux_density(mlat_rad_initial) * (energy_wave_potential(mlat_rad_initial) - (delta(mlat_rad_initial) + epsilon(mlat_rad_initial)) * energy_wave_phase_speed(mlat_rad_initial))
    
    if mu_const > mu_upper_limit:
        return None  # Noneを返して、結果リストでフィルタリングする
    
    mlat_rad_trajectory = np.array([])
    energy_result_trajectory = np.array([])
    wave_phase_trajectory = np.array([])
    for count_j in range(len(wave_phase)):
        if count_j == 0:
            new_mlat_rad = Newton_method_function_1(mu_const, mlat_rad_initial, wave_phase[count_j])
        else:
            new_mlat_rad = Newton_method_function_1(mu_const, mlat_rad_trajectory[-1], wave_phase[count_j])
        if new_mlat_rad != new_mlat_rad:
            break
        mlat_rad_trajectory = np.append(mlat_rad_trajectory, new_mlat_rad)
        wave_phase_trajectory = np.append(wave_phase_trajectory, wave_phase[count_j])
        energy_result_trajectory = np.append(energy_result_trajectory, energy_result(new_mlat_rad, wave_phase[count_j]))
    mlat_deg_trajectory = mlat_rad_trajectory / np.pi * 180E0
    energy_result_eV_trajectory = energy_result_trajectory / elementary_charge

    #時刻を取得
    now = datetime.datetime.now()
    print(count_i, now)
    
    # mlat_deg_trajectoryとenergy_result_eV_trajectoryのペアを返す
    return (mlat_deg_trajectory, energy_result_eV_trajectory, wave_phase_trajectory)

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
            mlat_deg_trajectory, energy_result_eV_trajectory, wave_phase_trajectory = result
            #ax.plot(mlat_deg_trajectory, energy_result_eV_trajectory, linewidth=4, alpha=0.5, color=colors[count_i])
            ax.scatter(mlat_deg_trajectory, energy_result_eV_trajectory, c=wave_phase_trajectory/np.pi, alpha=0.3, cmap=cmap_color, vmin=-1, vmax=initial_wave_phase/np.pi, s=1)

xlim_enlarged = ax.get_xlim()
print(xlim_enlarged)
ylim_enlarged = ax.get_ylim()
print(ylim_enlarged)

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

ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

ax.legend()

#カラーバーの設定
#norm = plt.Normalize(vmin=0, vmax=90)
norm = plt.Normalize(vmin=-1, vmax=initial_wave_phase/np.pi)
sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
sm.set_array([])
#cbar = plt.colorbar(sm, ticks=np.linspace(0, 90, 10))
#cbar.set_label(r'$\alpha_{\mathrm{detrap, initial}}$ [deg]')
cbar = plt.colorbar(sm, ticks=np.linspace(-1, initial_wave_phase/np.pi, 6))
cbar.set_label(r'$\psi$ [$\pi$ rad]')


ax.set_xlim(xlim_enlarged)
ax.set_ylim(ylim_enlarged)
plt.tight_layout()

plt.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_1_trajectory_Earth_L_9_enlarged_initialpsi_{(initial_wave_phase/np.pi):.1f}.png', dpi=100)
plt.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_1_trajectory_Earth_L_9_enlarged_initialpsi_{(initial_wave_phase/np.pi):.1f}.pdf', dpi=100)

ax.set_xlim(0E0, 50E0)
ax.set_ylim(1E1, 1E5)
plt.tight_layout()

plt.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_1_trajectory_Earth_L_9_initialpsi_{(initial_wave_phase/np.pi):.1f}.png', dpi=100)
plt.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_1_trajectory_Earth_L_9_initialpsi_{(initial_wave_phase/np.pi):.1f}.pdf', dpi=100)
plt.show()