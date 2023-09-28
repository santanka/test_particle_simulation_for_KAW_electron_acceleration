import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

wave_scalar_potential   = 2000E0     #[V]
initial_wavephase       = 0E0       #[deg]
gradient_parameter      = 2E0       #[]
wave_threshold          = 5E0       #[deg]

wavekind = [r'EparaBpara', r'Epara']
switch_delta_Epara      = 1E0
switch_delta_Eperp_perp = 0E0
switch_delta_Eperp_phi  = 0E0
switch_delta_Bpara      = [1E0, 0E0]
switch_delta_Bperp      = 0E0

switch_wave_packet      = 0E0

wave_frequency = 2E0 * np.pi * 0.15    #[rad/s]
kperp_rhoi = 2E0 * np.pi

particle_file_number    = r'20-102'
data_limit_under        = 0
data_limit_upper        =100000

channel = 1
#1:energy & equatorial pitch angle

rad2deg = 180E0 / np.pi
deg2rad = np.pi / 180E0

planet_radius   = 6371E3  #[m]
lshell_number   = 9E0
r_eq            = planet_radius * lshell_number #[m]
dipole_moment   = 7.75E22 #[Am]
B0_eq           = (1E-7 * dipole_moment) / r_eq**3E0

number_density_ion = 1E0    #[cm-3]
temperature_ion = 1E3   #[eV]
temperature_electron = 1E2  #[eV]

dir_name_1 = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave/0.15Hz/results_particle_{str(int(wave_scalar_potential))}V' \
    + f'_gradient_{int(gradient_parameter)}_threshold_{int(wave_threshold)}_wavephase_{int(initial_wavephase)}_{wavekind[0]}'
file_name_particle_1  = f'{dir_name_1}/myrank000/particle_trajectory{particle_file_number}.dat'
file_name_wave_1      = f'{dir_name_1}/myrank000/potential_prof.dat'

dir_name_2 = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave/0.15Hz/results_particle_{str(int(wave_scalar_potential))}V' \
    + f'_gradient_{int(gradient_parameter)}_threshold_{int(wave_threshold)}_wavephase_{int(initial_wavephase)}_{wavekind[1]}'
file_name_particle_2  = f'{dir_name_2}/myrank000/particle_trajectory{particle_file_number}.dat'
file_name_wave_2      = f'{dir_name_2}/myrank000/potential_prof.dat'

data_particle_1 = np.genfromtxt(file_name_particle_1, unpack=True)
data_particle_1 = data_particle_1[:, data_limit_under:data_limit_upper]
data_wave_1     = np.genfromtxt(file_name_wave_1, unpack=True)

data_particle_2 = np.genfromtxt(file_name_particle_2, unpack=True)
data_particle_2 = data_particle_2[:, data_limit_under:data_limit_upper]
data_wave_2     = np.genfromtxt(file_name_wave_2, unpack=True)

speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]

mass_electron   = 9.10938356E-28    #[g]
mass_ion        = 1.672621898E-24   #[g]

pressure_ion        = number_density_ion * temperature_ion * elementary_charge * 1E7    #cgs
pressure_electron   = number_density_ion * temperature_electron * elementary_charge * 1E7   #cgs

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 45

def z_position_m_to_mlat_rad(z_position):
    array_length = len(z_position)
    mlat = np.zeros(array_length)
    for count_i in range(array_length):
        mlat_old = 1E0
        for count_j in range(1000000):
            if (count_j == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_position[count_i]))
            ff = r_eq * (np.arcsinh(np.sqrt(3E0)*np.sin(mlat_old)) / 2E0 / np.sqrt(3) + np.sin(mlat_old) * np.sqrt(5E0-3E0*np.cos(2E0 * mlat_old)) \
                / 2E0 / np.sqrt(2E0)) - z_position[count_i]
            gg = r_eq * np.cos(mlat_old) * np.sqrt(1E0 + 3E0 * np.sin(mlat_old)**2E0)
            mlat_new = mlat_old - ff/gg

            if (abs(mlat_new - mlat_old) <= 1E-5):
                break

            mlat_old = mlat_new
        mlat[count_i] = mlat_new
    return mlat

if channel == 1:
    dp_1_time = data_particle_1[1, :]   #[s]
    dp_2_time = data_particle_2[1, :]   #[s]

    dp_1_energy = data_particle_1[6, :]    #[eV]
    dp_2_energy = data_particle_2[6, :]    #[eV]

    dp_1_pitchangle_eq = data_particle_1[7, :]  #[deg]
    dp_2_pitchangle_eq = data_particle_2[7, :]  #[deg]

    fig = plt.figure(figsize=(24, 12), dpi=100, tight_layout=True)

    ax1 = fig.add_subplot(121, xlabel=r'time [s]', ylabel=r'energy [eV]')
    ax1.plot(dp_2_time, dp_2_energy, color='blue', linewidth=4, label=r'only $\delta E_{\parallel}$', alpha=0.3)
    #ax1.plot(dp_1_time, dp_1_energy, color='red', linewidth=4, label=r'$\delta E_{\parallel}$ \& $\delta B_{\parallel}$', alpha=0.3)
    ax1.minorticks_on()
    ax1.grid(which='both', alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(122, xlabel=r'time [s]', ylabel=r'equatorial pitch angle [deg]')
    ax2.plot(dp_2_time, dp_2_pitchangle_eq, color='blue', linewidth=4, label=r'only $\delta E_{\parallel}$', alpha=0.3)
    #ax2.plot(dp_1_time, dp_1_pitchangle_eq, color='red', linewidth=4, label=r'$\delta E_{\parallel}$ \& $\delta B_{\parallel}$', alpha=0.3)
    ax2.minorticks_on()
    ax2.grid(which='both', alpha=0.3)
    ax2.legend()

    plt.tight_layout()


plt.show()