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
lshell_number = 9E0
r_eq = planet_radius * lshell_number

def d_mlat_d_z(mlat_rad):
    return 1E0 / r_eq / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)    #[rad/m]

#gradient function
diff_rad = 1E-5
def gradient_meter(function, mlat_rad):
    if mlat_rad < diff_rad:
        return 0E0
    else:
        return (function(mlat_rad + diff_rad) - function(mlat_rad - diff_rad)) / 2E0 / diff_rad * d_mlat_d_z(mlat_rad)    #[m^-1]

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

mlat_deg_array = np.linspace(0E0, 50E0, 1000)
mlat_rad_array = mlat_deg_array / 180E0 * np.pi

def energy_perpendicular(mlat_rad):
    return (energy_wave_potential(mlat_rad) - (delta(mlat_rad) + epsilon(mlat_rad)) * energy_wave_phase_speed(mlat_rad)) / delta(mlat_rad)    #[J]

def energy_parallel(mlat_rad, wave_phase):
    return (np.sqrt(energy_wave_phase_speed(mlat_rad)) - np.sqrt(energy_wave_potential(mlat_rad) * (np.cos(wave_phase) - wave_phase - np.pi / 2E0)))**2E0    #[J]

wave_phase_array = np.linspace(-np.pi/2E0, -np.pi, 1000)
#print(wave_phase_array)
#quit()

energy_perp_array = np.zeros(len(mlat_rad_array))


fig = plt.figure(figsize=(14, 14), dpi=100)
ax = fig.add_subplot(111, xlabel=r'MLAT [deg]', ylabel=r'Kinetic energy $K$ [eV]', yscale='log')

cmap_color = cm.get_cmap('cool')

def main(args):
    count_i = args

    mlat_deg = mlat_deg_array[count_i]
    mlat_rad = mlat_rad_array[count_i]

    energy_perp = energy_perpendicular(mlat_rad)
    energy_perp_array[count_i] = energy_perp
    if energy_perp > mu_upper_limit * magnetic_flux_density(mlat_rad) or energy_perp < 0E0:
    #if energy_perp < 0E0:
        return
    
    energy_para_array = np.zeros(len(wave_phase_array))
    for count_j in range(len(wave_phase_array)):
        energy_para_array[count_j] = energy_parallel(mlat_rad, wave_phase_array[count_j])
    
    energy_total_array = energy_perp + energy_para_array

    initial_pitch_angle_deg = np.arcsin(np.sqrt(energy_perp / energy_total_array[0])) * 180E0 / np.pi

    #ax.plot(mlat_deg*np.ones(len(energy_total_array)), energy_total_array / elementary_charge, linewidth=4, alpha=0.5, color=cmap_color(initial_pitch_angle_deg / 90E0))
    wave_phase_array_modify = wave_phase_array/np.pi
    ax.scatter(mlat_deg*np.ones(len(energy_total_array)), energy_total_array / elementary_charge, c=wave_phase_array_modify, alpha=0.5, cmap=cmap_color, vmin=-1E0, vmax=-1E0/2E0, s=1)

    now = datetime.datetime.now()
    #print(now, count_i, mlat_deg, initial_pitch_angle_deg, energy_perp / elementary_charge, energy_para_array[0] / elementary_charge)

    return

#if __name__ == '__main__':
#    num_process = 16
#    
#    with Pool(num_process) as p:
#        results = []
#        for count_i in range(len(mlat_deg_array)):
#            result = p.apply_async(main, args=[count_i])
#            results.append(result)
#        for result in results:
#            result.get()

for count_i in range(len(mlat_deg_array)):
    main(count_i)

xlim_enlarged = ax.get_xlim()
ylim_enlarged = ax.get_ylim()
if ylim_enlarged[0] < 1E1:
    ylim_enlarged = (1E1, ylim_enlarged[1])


energy_wave_phase_speed_array = np.zeros(len(mlat_rad_array))
energy_wave_potential_array = np.zeros(len(mlat_rad_array))
energy_S_1_upper_limit_array = np.zeros(len(mlat_rad_array))
energy_S_1_lower_limit_array = np.zeros(len(mlat_rad_array))

for count_i in range(len(mlat_rad_array)):
    energy_wave_phase_speed_array[count_i] = energy_wave_phase_speed(mlat_rad_array[count_i])
    energy_wave_potential_array[count_i] = energy_wave_potential(mlat_rad_array[count_i])
    energy_S_1_upper_limit_array[count_i] = energy_wave_potential(mlat_rad_array[count_i]) / delta(mlat_rad_array[count_i])
    energy_S_1_lower_limit_array[count_i] = energy_wave_potential(mlat_rad_array[count_i]) / (delta(mlat_rad_array[count_i]) + epsilon(mlat_rad_array[count_i]))
    #print(mlat_deg_array[count_i], np.sqrt(energy_wave_potential_array[count_i] / energy_wave_phase_speed_array[count_i]))

ax.plot(mlat_deg_array, energy_wave_phase_speed_array / elementary_charge, linewidth=4, color='red', alpha=0.6, label=r'$K_{\mathrm{ph \parallel}}$')
ax.plot(mlat_deg_array, energy_wave_potential_array / elementary_charge, linewidth=4, color='green', alpha=0.6, label=r'$K_{\mathrm{E}}$')
ax.plot(mlat_deg_array, energy_S_1_upper_limit_array / elementary_charge, linewidth=4, color='blue', alpha=0.6, label=r'$S = 1$ range')
ax.plot(mlat_deg_array, energy_S_1_lower_limit_array / elementary_charge, linewidth=4, color='blue', alpha=0.6)
ax.plot(mlat_deg_array, energy_perp_array / elementary_charge, linewidth=4, color='orange', alpha=0.6, label=r'$K_{\perp}$')

ax.set_xlim(xlim_enlarged)
ax.set_ylim(ylim_enlarged)

ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

ax.legend()

#カラーバー
#norm = plt.Normalize(vmin=0, vmax=90)
#sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
#sm.set_array([])
#cbar = plt.colorbar(sm, ticks=np.linspace(0, 90, 10))
#cbar.set_label(r'$\alpha_{\mathrm{detrap, initial}}$ [deg]')

norm = plt.Normalize(vmin=-1E0, vmax=-1E0/2E0)
sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=np.linspace(-1E0, -1E0/2E0, 6))
cbar.set_label(r'$\psi \, [\pi \, \mathrm{rad}]$')

plt.tight_layout()
plt.savefig('/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_1_trajectory_nonrelation_Kpara_Earth_L_9.png', dpi=100)
plt.savefig('/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_S_1_trajectory_nonrelation_Kpara_Earth_L_9.pdf', dpi=100)
#plt.show()