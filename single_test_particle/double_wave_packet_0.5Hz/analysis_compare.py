import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

wave_scalar_potential   = 600E0     #[V]
initial_wavephase       = 0E0       #[deg]
gradient_parameter      = 2E0       #[]
wave_threshold          = 5E0       #[deg]

wavekind_a              = r'EparaBpara'
wavekind_b              = r'Epara'

switch_delta_Epara_a      = 1E0
switch_delta_Eperp_perp_a = 0E0
switch_delta_Eperp_phi_a  = 0E0
switch_delta_Bpara_a      = 1E0
switch_delta_Bperp_a      = 0E0

switch_delta_Epara_b      = 1E0
switch_delta_Eperp_perp_b = 0E0
switch_delta_Eperp_phi_b  = 0E0
switch_delta_Bpara_b      = 0E0
switch_delta_Bperp_b      = 0E0

switch_wave_packet = 1E0

particle_file_number    = r'20-102'
data_limit_under        = 0
data_limit_upper        = 110000

channel = 3
#1:trajectory, 2:energy, 3:wavephase on particle vs. wave phase speed

rad2deg = 180E0/np.pi
deg2rad = np.pi/180E0

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


dir_name_a = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave_packet/results_particle_{str(int(wave_scalar_potential))}V' \
    + f'_gradient_{int(gradient_parameter)}_threshold_{int(wave_threshold)}_wavephase_{int(initial_wavephase)}_{wavekind_a}'
file_name_particle_a  = f'{dir_name_a}/myrank000/particle_trajectory{particle_file_number}.dat'
file_name_wave_a      = f'{dir_name_a}/myrank000/potential_prof.dat'

dir_name_b = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave_packet/results_particle_{str(int(wave_scalar_potential))}V' \
    + f'_gradient_{int(gradient_parameter)}_threshold_{int(wave_threshold)}_wavephase_{int(initial_wavephase)}_{wavekind_b}'
file_name_particle_b  = f'{dir_name_b}/myrank000/particle_trajectory{particle_file_number}.dat'
file_name_wave_b      = f'{dir_name_b}/myrank000/potential_prof.dat'

data_particle_a   = np.genfromtxt(file_name_particle_a, unpack=True)
data_particle_a   = data_particle_a[:, data_limit_under:data_limit_upper]
data_wave_a       = np.genfromtxt(file_name_wave_a, unpack=True)

data_particle_b   = np.genfromtxt(file_name_particle_b, unpack=True)
data_particle_b   = data_particle_b[:, data_limit_under:data_limit_upper]

speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]

mass_electron   = 9.10938356E-28    #[g]
mass_ion        = 1.672621898E-24   #[g]

pressure_ion        = number_density_ion * temperature_ion * elementary_charge * 1E7    #cgs
pressure_electron   = number_density_ion * temperature_electron * elementary_charge * 1E7   #cgs

#data_particle
dp_time_a             = data_particle_a[1, :]   #[s]
dp_z_position_a       = data_particle_a[2, :]   #[m]
dp_u_para_a           = data_particle_a[3, :]   #[m s-1]
dp_u_perp_a           = data_particle_a[4, :]   #[m s-1]
dp_u_phase_a          = data_particle_a[5, :]   #[rad]
dp_energy_a           = data_particle_a[6, :]   #[eV]
dp_pitchangle_eq_a    = data_particle_a[7, :]   #[deg]
dp_wavephase_1_a      = data_particle_a[8, :]   #[rad]
dp_wavephase_2_a      = data_particle_a[9, :]   #[rad]

dp_gamma_a    = np.sqrt(1E0 + (dp_u_para_a**2E0 + dp_u_perp_a**2E0) / speed_of_light**2E0)
dp_v_para_a   = dp_u_para_a / dp_gamma_a
dp_v_perp_a   = dp_u_perp_a / dp_gamma_a

dp_time_b             = data_particle_b[1, :]   #[s]
dp_z_position_b       = data_particle_b[2, :]   #[m]
dp_u_para_b           = data_particle_b[3, :]   #[m s-1]
dp_u_perp_b           = data_particle_b[4, :]   #[m s-1]
dp_u_phase_b          = data_particle_b[5, :]   #[rad]
dp_energy_b           = data_particle_b[6, :]   #[eV]
dp_pitchangle_eq_b    = data_particle_b[7, :]   #[deg]
dp_wavephase_1_b      = data_particle_b[8, :]   #[rad]
dp_wavephase_2_b      = data_particle_b[9, :]   #[rad]

dp_gamma_b    = np.sqrt(1E0 + (dp_u_para_b**2E0 + dp_u_perp_b**2E0) / speed_of_light**2E0)
dp_v_para_b   = dp_u_para_b / dp_gamma_b
dp_v_perp_b   = dp_u_perp_b / dp_gamma_b

#data_wave
dw_z_position         = data_wave_a[0, :]   #[/RE]

dw_wavenumber_para_1  = data_wave_a[1, :]   #[rad/m]
dw_wavenumber_perp_1  = data_wave_a[2, :]   #[rad/m]
dw_wave_frequency_1   = data_wave_a[3, :]   #[rad/s]
dw_wave_phasespeed_1  = data_wave_a[4, :]   #[m/s]
dw_wave_potential_1   = data_wave_a[5, :]   #[V]
dw_wave_Epara_1       = data_wave_a[6, :]   #[V/m]
dw_wave_Eperpperp_1   = data_wave_a[7, :]   #[V/m]
dw_wave_Eperpphi_1    = data_wave_a[8, :]   #[V/m]
dw_wave_Bpara_1       = data_wave_a[9, :]   #[T]
dw_wave_Bperp_1       = data_wave_a[10, :]  #[T]
dw_wave_phase_1       = data_wave_a[11, :]  #[rad]

dw_wavenumber_para_2  = data_wave_a[12, :]   #[rad/m]
dw_wavenumber_perp_2  = data_wave_a[13, :]   #[rad/m]
dw_wave_frequency_2   = data_wave_a[14, :]   #[rad/s]
dw_wave_phasespeed_2  = data_wave_a[15, :]   #[m/s]
dw_wave_potential_2   = data_wave_a[16, :]   #[V]
dw_wave_Epara_2       = data_wave_a[17, :]   #[V/m]
dw_wave_Eperpperp_2   = data_wave_a[18, :]   #[V/m]
dw_wave_Eperpphi_2    = data_wave_a[19, :]   #[V/m]
dw_wave_Bpara_2       = data_wave_a[20, :]   #[T]
dw_wave_Bperp_2       = data_wave_a[21, :]   #[T]
dw_wave_phase_2       = data_wave_a[22, :]   #[rad]

dw_alfven_speed           = data_wave_a[23, :]  #[m s-1]
dw_ion_Larmor_radius      = data_wave_a[24, :]  #[m]
dw_beta_ion               = data_wave_a[25, :]  #[]
dw_magnetic_flux_density  = data_wave_a[26, :]  #[T]
dw_temperature_ion        = data_wave_a[27, :]  #[eV]
dw_temperature_electron   = data_wave_a[28, :]  #[eV]
dw_number_density         = data_wave_a[29, :]  #[m-3]


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 25

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

def get_major_wave_component(position, component_1, component_2):
    array_length = len(position)
    component_major = np.zeros(array_length)
    for count_i in range(array_length):
        if (position[count_i] >= 0E0):
            component_major[count_i] = component_1[count_i]
        elif (position[count_i] < 0E0):
            component_major[count_i] = component_2[count_i]
    return component_major

if (channel == 1):
    dp_mlat_deg_a = z_position_m_to_mlat_rad(dp_z_position_a) * rad2deg
    dp_mlat_deg_b = z_position_m_to_mlat_rad(dp_z_position_b) * rad2deg

    fig = plt.figure(figsize=(8, 7), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'MLAT [degree]', ylabel=r'$v_{\parallel}$ [c]')
    ax.plot(dp_mlat_deg_a, dp_v_para_a / speed_of_light, color='red', label=r'$\delta E_{\parallel}$ \& $\delta B_{\parallel}$')
    ax.plot(dp_mlat_deg_b, dp_v_para_b / speed_of_light, color='blue', label=r'only $\delta E_{\parallel}$')

    ax.autoscale()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    mlat_deg_wave = z_position_m_to_mlat_rad(dw_z_position*planet_radius) * rad2deg
    dw_wave_phasespeed_major = get_major_wave_component(mlat_deg_wave, dw_wave_phasespeed_1, dw_wave_phasespeed_2)
    ax.plot(mlat_deg_wave, dw_wave_phasespeed_major / speed_of_light, linestyle='-.', color='orange', linewidth='4', label=r'phase speed')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

if (channel == 2):
    fig = plt.figure(figsize=(8, 7), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'time [s]', ylabel=r'energy [eV]')
    ax.plot(dp_time_a, dp_energy_a, color='red', label=r'$\delta E_{\parallel}$ \& $\delta B_{\parallel}$')
    ax.plot(dp_time_b, dp_energy_b, color='blue', label=r'only $\delta E_{\parallel}$')
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

if (channel == 3):
    dp_mlat_rad_a = z_position_m_to_mlat_rad(dp_z_position_a)
    dp_mlat_rad_b = z_position_m_to_mlat_rad(dp_z_position_b)

    dp_b0_a = B0_eq / np.cos(dp_mlat_rad_a)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad_a)**2E0) * 1E4    #[G]
    dp_b0_b = B0_eq / np.cos(dp_mlat_rad_b)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad_b)**2E0) * 1E4    #[G]

    dp_kpara_a = np.sqrt(2E0 * np.pi * number_density_ion * mass_ion * pressure_ion) / dp_b0_a**2E0 * np.sqrt(4E0 * np.pi + dp_b0_a**2E0 / (pressure_ion + pressure_electron)) * np.sign(dp_mlat_rad_a)   #[rad/cm]
    dp_kpara_b = np.sqrt(2E0 * np.pi * number_density_ion * mass_ion * pressure_ion) / dp_b0_b**2E0 * np.sqrt(4E0 * np.pi + dp_b0_b**2E0 / (pressure_ion + pressure_electron)) * np.sign(dp_mlat_rad_b)   #[rad/cm]

    dp_wavefreq = 2E0 * np.pi / 2E0 #[rad/s]

    dp_phasespeed_a = dp_wavefreq / dp_kpara_a / 1E2 #[m/s]
    dp_phasespeed_b = dp_wavefreq / dp_kpara_b / 1E2 #[m/s]

    dp_theta_a = dp_v_para_a / dp_phasespeed_a - 1E0
    dp_theta_b = dp_v_para_b / dp_phasespeed_b - 1E0

    dp_wavephase_major_a = get_major_wave_component(dp_z_position_a, dp_wavephase_1_a, dp_wavephase_2_a)
    dp_wavephase_major_b = get_major_wave_component(dp_z_position_b, dp_wavephase_1_b, dp_wavephase_2_b)

    fig = plt.figure(figsize=(8, 7), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'wave phase $\psi$ [$\times \pi$ rad]', ylabel=r'$\frac{v_{\parallel}}{V_{\mathrm{ph} \parallel}}-1$')
    ax.plot(dp_wavephase_major_a / np.pi, dp_theta_a, color='red', label=r'$\delta E_{\parallel}$ \& $\delta B_{\parallel}$')
    ax.plot(dp_wavephase_major_b / np.pi, dp_theta_b, color='blue', label=r'only $\delta E_{\parallel}$')
    ax.set_xlim((initial_wavephase*deg2rad-8*np.pi) / np.pi -1, (initial_wavephase*deg2rad) / np.pi +1)
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

plt.show()
plt.close()