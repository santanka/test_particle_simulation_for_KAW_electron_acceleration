import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

wave_scalar_potential   = 600E0     #[V]
initial_wavephase       = 270E0       #[deg]
gradient_parameter      = 2E0       #[]
wave_threshold          = 5E0       #[deg]

wavekind                = r'EparaBpara'
switch_delta_Epara      = 1E0
switch_delta_Eperp_perp = 0E0
switch_delta_Eperp_phi  = 0E0
switch_delta_Bpara      = 1E0
switch_delta_Bperp      = 0E0

switch_wave_packet = 0E0

particle_file_number    = r'30-153'
data_limit_under        = 0
data_limit_upper        = 100000

channel = 12
#1:trajectory, 2:energy & equatorial pitch angle, 3:delta_Epara (t=0), 4:delta_Eperpperp (t=8pi/wave_freq), 5:delta_Eperpphi (t=8pi/wave_freq)
#6:delta_Bpara (t=8pi/wave_freq), 7:delta_Bperp (t=8pi/wave_freq), 8:wave frequency, 9:wavelength, 10:wavephase variation on particle
#11:wavephase on particle vs. wave phase speed, 12:wave parallel components' forces, 13:particle velocity

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


dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave/results_particle_{str(int(wave_scalar_potential))}V' \
    + f'_gradient_{int(gradient_parameter)}_threshold_{int(wave_threshold)}_wavephase_{int(initial_wavephase)}_{wavekind}'
file_name_particle  = f'{dir_name}/myrank000/particle_trajectory{particle_file_number}.dat'
file_name_wave      = f'{dir_name}/myrank000/potential_prof.dat'

print(file_name_particle)
print(file_name_wave)

data_particle   = np.genfromtxt(file_name_particle, unpack=True)
data_particle   = data_particle[:, data_limit_under:data_limit_upper]
data_wave       = np.genfromtxt(file_name_wave, unpack=True)

speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]

mass_electron   = 9.10938356E-28    #[g]
mass_ion        = 1.672621898E-24   #[g]

pressure_ion        = number_density_ion * temperature_ion * elementary_charge * 1E7    #cgs
pressure_electron   = number_density_ion * temperature_electron * elementary_charge * 1E7   #cgs


#data_wave
dw_z_position         = data_wave[0, :]   #[/RE]

dw_wavenumber_para_1  = data_wave[1, :]   #[rad/m]
dw_wavenumber_perp_1  = data_wave[2, :]   #[rad/m]
dw_wave_frequency_1   = data_wave[3, :]   #[rad/s]
dw_wave_phasespeed_1  = data_wave[4, :]   #[m/s]
dw_wave_potential_1   = data_wave[5, :]   #[V]
dw_wave_Epara_1       = data_wave[6, :]   #[V/m]
dw_wave_Eperpperp_1   = data_wave[7, :]   #[V/m]
dw_wave_Eperpphi_1    = data_wave[8, :]   #[V/m]
dw_wave_Bpara_1       = data_wave[9, :]   #[T]
dw_wave_Bperp_1       = data_wave[10, :]  #[T]
dw_wave_phase_1       = data_wave[11, :]  #[rad]

dw_wavenumber_para_2  = data_wave[12, :]   #[rad/m]
dw_wavenumber_perp_2  = data_wave[13, :]   #[rad/m]
dw_wave_frequency_2   = data_wave[14, :]   #[rad/s]
dw_wave_phasespeed_2  = data_wave[15, :]   #[m/s]
dw_wave_potential_2   = data_wave[16, :]   #[V]
dw_wave_Epara_2       = data_wave[17, :]   #[V/m]
dw_wave_Eperpperp_2   = data_wave[18, :]   #[V/m]
dw_wave_Eperpphi_2    = data_wave[19, :]   #[V/m]
dw_wave_Bpara_2       = data_wave[20, :]   #[T]
dw_wave_Bperp_2       = data_wave[21, :]   #[T]
dw_wave_phase_2       = data_wave[22, :]   #[rad]

dw_alfven_speed           = data_wave[23, :]  #[m s-1]
dw_ion_Larmor_radius      = data_wave[24, :]  #[m]
dw_beta_ion               = data_wave[25, :]  #[]
dw_magnetic_flux_density  = data_wave[26, :]  #[T]
dw_temperature_ion        = data_wave[27, :]  #[eV]
dw_temperature_electron   = data_wave[28, :]  #[eV]
dw_number_density         = data_wave[29, :]  #[m-3]


#data_particle
dp_time             = data_particle[1, :]   #[s]
dp_z_position       = data_particle[2, :]   #[m]
dp_u_para           = data_particle[3, :]   #[m s-1]
dp_u_perp           = data_particle[4, :]   #[m s-1]
dp_u_phase          = data_particle[5, :]   #[rad]
dp_energy           = data_particle[6, :]   #[eV]
dp_pitchangle_eq    = data_particle[7, :]   #[deg]
dp_wavephase_1      = data_particle[8, :]   #[rad]
dp_wavephase_2      = data_particle[9, :]   #[rad]

dp_gamma    = np.sqrt(1E0 + (dp_u_para**2E0 + dp_u_perp**2E0) / speed_of_light**2E0)
dp_v_para   = dp_u_para / dp_gamma
dp_v_perp   = dp_u_perp / dp_gamma


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 25

#ファイルの確認
def check_file_exists(filename):
    if os.path.isfile(filename):
        return True
    else:
        return False
    
#フォルダの作成
def mkdir(path_name):
    if (check_file_exists(path_name) == False):
        #ディレクトリの生成 (ディレクトリは要指定)
        try:
            os.makedirs(path_name)
        except FileExistsError:
            pass


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
    mlat_deg = z_position_m_to_mlat_rad(dp_z_position) * rad2deg
    fig = plt.figure(figsize=(14, 14), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'MLAT [degree]', ylabel=r'$v_{\parallel}/c$')
    mappable = ax.scatter(mlat_deg, dp_v_para/speed_of_light, c=dp_time, cmap='turbo', marker='.', lw=0)
    fig.colorbar(mappable=mappable, ax=ax, label=r'time [s]')
    ax.scatter(mlat_deg[0], dp_v_para[0]/speed_of_light, marker='o', color='r', label=r'start', zorder=3, s=200)
    ax.scatter(mlat_deg[-1], dp_v_para[-1]/speed_of_light, marker='D', color='r', label=r'end', zorder=3, s=200)
    
    ax.autoscale()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    mlat_deg_wave = z_position_m_to_mlat_rad(dw_z_position*planet_radius) * rad2deg
    dw_wave_phasespeed_major = get_major_wave_component(mlat_deg_wave, dw_wave_phasespeed_1, dw_wave_phasespeed_2)
    ax.plot(mlat_deg_wave, dw_wave_phasespeed_major/speed_of_light, linestyle='-.', color='orange', linewidth='4')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

    mkdir(f'{dir_name}/result_trajectory')
    fig.savefig(f'{dir_name}/result_trajectory/particle_trajectory{particle_file_number}.png')

if (channel == 2):
    fig = plt.figure(figsize=(24, 12), dpi=100, tight_layout=True)
    ax1 = fig.add_subplot(121, xlabel=r'time [s]', ylabel=r'energy [eV]')
    ax1.plot(dp_time, dp_energy)
    ax1.minorticks_on()
    ax1.grid(which='both', alpha=0.3)

    ax2 = fig.add_subplot(122, xlabel=r'time [s]', ylabel=r'equatorial pitch angle [degree]')
    ax2.plot(dp_time, dp_pitchangle_eq)
    ax2.minorticks_on()
    ax2.grid(which='both', alpha=0.3)
    fig.savefig(f'{dir_name}/result_energy_eqpitchangle/particle_trajectory{particle_file_number}.png')

def profile_plot_mlat(z_position_m, profile, profile_name):
    mlat_deg = z_position_m_to_mlat_rad(z_position_m) * rad2deg
    fig = plt.figure(figsize=(14, 14), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'MLAT [degree]', ylabel=f'{profile_name}')
    ax.plot(mlat_deg, profile)
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)

if (channel == 3):
    name = r'$\delta E_{\parallel}$ [mV/m]'
    profile_plot_mlat(dw_z_position*planet_radius, (dw_wave_Epara_1+dw_wave_Epara_2)*1E3, name)

if (channel == 4):
    name = r'$\delta E_{\perp \perp}$ [mV/m]'
    profile_plot_mlat(dw_z_position*planet_radius, (dw_wave_Eperpperp_1+dw_wave_Eperpperp_2)*1E3, name)

if (channel == 5):
    name = r'$\delta E_{\perp \phi}$ [mV/m]'
    profile_plot_mlat(dw_z_position*planet_radius, (dw_wave_Eperpphi_1+dw_wave_Eperpphi_2)*1E3, name)

if (channel == 6):
    name = r'$\delta B_{\parallel}$ [nT]'
    profile_plot_mlat(dw_z_position*planet_radius, (dw_wave_Bpara_1+dw_wave_Bpara_2)*1E9, name)

if (channel == 7):
    name = r'$\delta B_{\perp}$ [nT]'
    profile_plot_mlat(dw_z_position*planet_radius, (dw_wave_Bperp_1+dw_wave_Bperp_2)*1E9, name)


if (channel == 8):
    name = r'wave frequency [Hz]'
    dw_wave_frequency_major = get_major_wave_component(dw_z_position, dw_wave_frequency_1, dw_wave_frequency_2)
    profile_plot_mlat(dw_z_position*planet_radius, dw_wave_frequency_major / 2E0 / np.pi, name)

if (channel == 9):
    mlat_rad = z_position_m_to_mlat_rad(dw_z_position*planet_radius)
    mlat_deg = mlat_rad * rad2deg

    dw_wavenumber_para_major = get_major_wave_component(mlat_deg, dw_wavenumber_para_1, dw_wavenumber_para_2)
    dw_wavenumber_perp_major = get_major_wave_component(mlat_deg, dw_wavenumber_perp_1, dw_wavenumber_perp_2)
    dlog_dz = 3E0 * np.sin(mlat_rad) / r_eq / np.cos(mlat_rad) / np.log(10E0) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0) \
        * (1E0 / (1E0 + 3E0 * np.sin(mlat_rad)**2E0) + 2E0 / np.cos(mlat_rad)**2E0)

    fig = plt.figure(figsize=(14, 14), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'MLAT [degree]', ylabel=r'length [km]', yscale='log')
    ax.plot(mlat_deg, np.abs(2E0*np.pi / dw_wavenumber_para_major / 1E3), label=r'$\lambda_{\parallel} = \frac{2\pi}{k_{\parallel}}$', linewidth='4', c='blue')
    ax.plot(mlat_deg, np.abs(2E0*np.pi / dw_wavenumber_perp_major / 1E3), label=r'$\lambda_{\perp} = \frac{2\pi}{k_{\perp}}$', linewidth='4', c='green')
    ax.plot(mlat_deg, np.abs(1E0 / dlog_dz / 1E3), label=r'$\left( \frac{d \left( \rm{log}_{10} \frac{B_0}{B_E} \right)}{dz} \right)^{-1}$', linewidth='4', c='orange')
    ax.minorticks_on()
    ax.grid(which="both", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)


if (channel == 10):
    dp_wavephase_major = get_major_wave_component(dp_z_position, dp_wavephase_1, dp_wavephase_2)
    dp_wavephase_major = np.mod(dp_wavephase_major+np.pi, 2E0*np.pi) - np.pi
    fig = plt.figure(figsize=(14, 14), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'time [s]', ylabel=r'wave phase [rad]')
    ax.plot(dp_time, dp_wavephase_major)
    ax.minorticks_on()
    ax.grid(which="both", alpha=0.3)
    fig.savefig(f'{dir_name}/result_wavephase/particle_trajectory{particle_file_number}.png')

if (channel == 11):
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)
    b0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]
    dp_kpara = np.sqrt(2E0 * np.pi * number_density_ion * mass_ion * pressure_ion) / b0**2E0 * np.sqrt(4E0 * np.pi + b0**2E0 / (pressure_ion + pressure_electron)) * np.sign(dp_mlat_rad)   #[rad/cm]
    dp_wavefreq = 2E0 * np.pi / 2E0 #[rad/s]
    dp_phasespeed = dp_wavefreq / dp_kpara / 1E2    #[m s-1]
    
    dp_theta = dp_v_para / dp_phasespeed - 1E0

    dp_wavephase_major = get_major_wave_component(dp_z_position, dp_wavephase_1, dp_wavephase_2)
    dp_wavephase_major = np.mod(dp_wavephase_major+np.pi, 2E0*np.pi) - np.pi

    fig = plt.figure(figsize=(14, 14), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'wave phase $\psi$ [rad]', ylabel=r'$\frac{v_{\parallel}}{V_{R \parallel}}-1$')
    ax.plot(dp_wavephase_major, dp_theta)
    ax.scatter(dp_wavephase_major[0], dp_theta[0], marker='o', color='r', label='start', zorder=3, s=200)
    ax.scatter(dp_wavephase_major[-1], dp_theta[-1], marker='D', color='r', label='start', zorder=3, s=200)
    ax.minorticks_on()
    ax.grid(which="both", alpha=0.3)
    fig.savefig(f'{dir_name}/result_wavephase_phasespeed/particle_trajectory{particle_file_number}.png')

def make_h_function(array_size, wave_phase, kpara):
    h_function = np.ones(array_size)
    dh_dz = np.zeros(array_size)
    if (switch_wave_packet == 1E0):
        for count_i in range(array_size):
            if (wave_phase[count_i] >= initial_wavephase-8.*np.pi and wave_phase[count_i] <= initial_wavephase):
                h_function[count_i] = 5E-1 * (1E0 - np.cos(1E0/4E0 * (wave_phase[count_i] - initial_wavephase)))
                dh_dz[count_i] = kpara[count_i] / 8E0 * np.sin((wave_phase[count_i] - initial_wavephase) / 4E0)
            else:
                h_function[count_i] = 0E0
                dh_dz[count_i] == 0E0
    return h_function, dh_dz
        

if (channel == 12):
    array_size = len(dp_z_position)
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)

    dp_B0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]
    dp_kpara_1 = + np.sqrt(2E0 * np.pi * number_density_ion * mass_ion * pressure_ion) / dp_B0**2E0 * np.sqrt(4E0 * np.pi + dp_B0**2E0 / (pressure_ion + pressure_electron))    #[rad/cm]
    dp_kpara_2 = - np.sqrt(2E0 * np.pi * number_density_ion * mass_ion * pressure_ion) / dp_B0**2E0 * np.sqrt(4E0 * np.pi + dp_B0**2E0 / (pressure_ion + pressure_electron))    #[rad/cm]
    
    ion_Larmor_radius = (speed_of_light*1E2) * np.sqrt(2E0*mass_ion*pressure_ion/number_density_ion) / (elementary_charge/1E1*speed_of_light*1E2) / dp_B0   #[cm]

    dp_kperp_1 = 2E0 * np.pi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]
    dp_kperp_2 = 2E0 * np.pi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]

    dp_dB0_dz = 3E0 * np.sin(dp_mlat_rad) * (5E0 * np.sin(dp_mlat_rad)**2E0 + 3E0) / np.cos(dp_mlat_rad)**8E0 / (3E0 * np.sin(dp_mlat_rad)**2E0 + 1E0) / (r_eq*1E2) * (B0_eq*1E4)   #[G/cm]
    Alpha = 4E0 * np.pi * (1E0 + pressure_electron / pressure_ion) * (elementary_charge/1E1*speed_of_light*1E2) * number_density_ion * (wave_scalar_potential*1E8/(speed_of_light*1E2))

    g_function_1 = 5E-1 * (np.tanh(+ gradient_parameter * (dp_mlat_rad*rad2deg - wave_threshold/2E0)) + 1E0)
    g_function_2 = 5E-1 * (np.tanh(- gradient_parameter * (dp_mlat_rad*rad2deg + wave_threshold/2E0)) + 1E0)

    dg_dz_1 = + 90E0 * gradient_parameter / np.pi / np.cosh(+ gradient_parameter * (dp_mlat_rad*rad2deg - wave_threshold/2E0))**2E0 / (r_eq*1E2) / np.cos(dp_mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0)
    dg_dz_2 = - 90E0 * gradient_parameter / np.pi / np.cosh(- gradient_parameter * (dp_mlat_rad*rad2deg + wave_threshold/2E0))**2E0 / (r_eq*1E2) / np.cos(dp_mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0)

    h_function_1, dh_dz_1 = make_h_function(array_size, dp_wavephase_1, dp_kpara_1)
    h_function_2, dh_dz_2 = make_h_function(array_size, dp_wavephase_2, dp_kpara_2)

    dp_deltaBpara_1 = Alpha * g_function_1 * h_function_1 / dp_B0 * np.cos(dp_wavephase_1) * switch_delta_Bpara
    dp_deltaBpara_2 = Alpha * g_function_2 * h_function_2 / dp_B0 * np.cos(dp_wavephase_2) * switch_delta_Bpara
    dp_deltaBpara_sum = dp_deltaBpara_1 + dp_deltaBpara_2

    dp_deltaEpara_1 = (2E0 + pressure_electron / pressure_ion) * dp_kpara_1 * (wave_scalar_potential*1E8/(speed_of_light*1E2)) * g_function_1 * h_function_1 * np.sin(dp_wavephase_1) * switch_delta_Epara
    dp_deltaEpara_2 = (2E0 + pressure_electron / pressure_ion) * dp_kpara_2 * (wave_scalar_potential*1E8/(speed_of_light*1E2)) * g_function_2 * h_function_2 * np.sin(dp_wavephase_2) * switch_delta_Epara
    dp_deltaEpara_sum = dp_deltaEpara_1 + dp_deltaEpara_2

    dp_Larmor_radius = mass_electron * (dp_u_perp*1E2) * (speed_of_light*1E2) / (elementary_charge/1E1*speed_of_light*1E2) / (dp_B0 + dp_deltaBpara_sum)   #[cm]

    def make_Delta(kperp):
        Delta_real = np.zeros(array_size)
        Delta_imag = np.zeros(array_size)
        for count_i in range(array_size):
            if (kperp[count_i] * dp_Larmor_radius[count_i] * np.sin(dp_u_phase[count_i]) != 0E0):
                Delta_real[count_i] = (1E0 - np.cos(kperp[count_i] * dp_Larmor_radius[count_i] * np.sin(dp_u_phase[count_i]))) / (kperp[count_i] * dp_Larmor_radius[count_i] * np.sin(dp_u_phase[count_i]))**2E0
                Delta_imag[count_i] = (- kperp[count_i] * dp_Larmor_radius[count_i] * np.sin(dp_u_phase[count_i]) + np.sin(kperp[count_i] * dp_Larmor_radius[count_i] * np.sin(dp_u_phase[count_i]))) \
                    / (kperp[count_i] * dp_Larmor_radius[count_i] * np.sin(dp_u_phase[count_i]))**2E0
            elif (kperp[count_i] * dp_Larmor_radius[count_i] * np.sin(dp_u_phase[count_i]) == 0E0):
                Delta_real[count_i] = 5E-1
                Delta_imag[count_i] = 0E0
        return Delta_real, Delta_imag
    
    dp_Delta_real_1, dp_Delta_imag_1 = make_Delta(dp_kperp_1)
    dp_Delta_real_2, dp_Delta_imag_2 = make_Delta(dp_kperp_2)

    B0_function_1 = 2E0 * Alpha * (- 1E0 / dp_B0**2E0 * dp_dB0_dz * g_function_1 * h_function_1) * (dp_Delta_real_1 * np.cos(dp_wavephase_1) - dp_Delta_imag_1 * np.sin(dp_wavephase_1)) * switch_delta_Bpara
    B0_function_2 = 2E0 * Alpha * (- 1E0 / dp_B0**2E0 * dp_dB0_dz * g_function_2 * h_function_2) * (dp_Delta_real_2 * np.cos(dp_wavephase_2) - dp_Delta_imag_2 * np.sin(dp_wavephase_2)) * switch_delta_Bpara
    B0_function_sum = B0_function_1 + B0_function_2

    kpara_function_1 = 2E0 * Alpha * (- dp_kpara_1 * g_function_1 * h_function_1 / dp_B0) * (dp_Delta_real_1 * np.sin(dp_wavephase_1) - dp_Delta_imag_1 * np.cos(dp_wavephase_1)) * switch_delta_Bpara
    kpara_function_2 = 2E0 * Alpha * (- dp_kpara_2 * g_function_2 * h_function_2 / dp_B0) * (dp_Delta_real_2 * np.sin(dp_wavephase_2) - dp_Delta_imag_2 * np.cos(dp_wavephase_2)) * switch_delta_Bpara
    kpara_function_sum = kpara_function_1 + kpara_function_2

    dg_dz_function_1 = 2E0 * Alpha * (dg_dz_1 / dp_B0 * h_function_1) * (dp_Delta_real_1 * np.cos(dp_wavephase_1) - dp_Delta_imag_1 * np.sin(dp_wavephase_1)) * switch_delta_Bpara
    dg_dz_function_2 = 2E0 * Alpha * (dg_dz_2 / dp_B0 * h_function_2) * (dp_Delta_real_2 * np.cos(dp_wavephase_2) - dp_Delta_imag_2 * np.sin(dp_wavephase_2)) * switch_delta_Bpara
    dg_dz_function_sum = dg_dz_function_1 + dg_dz_function_2

    dh_dz_function_1 = 2E0 * Alpha * (dh_dz_1 / dp_B0 * g_function_1) * (dp_Delta_real_1 * np.cos(dp_wavephase_1) - dp_Delta_imag_1 * np.sin(dp_wavephase_1)) * switch_delta_Bpara
    dh_dz_function_2 = 2E0 * Alpha * (dh_dz_2 / dp_B0 * g_function_2) * (dp_Delta_real_2 * np.cos(dp_wavephase_2) - dp_Delta_imag_2 * np.sin(dp_wavephase_2)) * switch_delta_Bpara
    dh_dz_function_sum = dh_dz_function_1 + dh_dz_function_2

    Xi_function = B0_function_sum + kpara_function_sum + dg_dz_function_sum
    
    F_mirror_background = - mass_electron * (dp_u_perp*1E2)**2E0 / 2E0 / (dp_B0 + dp_deltaBpara_sum) / dp_gamma * dp_dB0_dz * 1E-5  #[N]
    F_mirror_wave_B0    = - mass_electron * (dp_u_perp*1E2)**2E0 / 2E0 / (dp_B0 + dp_deltaBpara_sum) / dp_gamma * B0_function_sum * 1E-5  #[N]
    F_mirror_wave_kpara = - mass_electron * (dp_u_perp*1E2)**2E0 / 2E0 / (dp_B0 + dp_deltaBpara_sum) / dp_gamma * kpara_function_sum * 1E-5  #[N]
    F_mirror_wave_dg_dz = - mass_electron * (dp_u_perp*1E2)**2E0 / 2E0 / (dp_B0 + dp_deltaBpara_sum) / dp_gamma * dg_dz_function_sum * 1E-5  #[N]
    F_mirror_wave_dh_dz = - mass_electron * (dp_u_perp*1E2)**2E0 / 2E0 / (dp_B0 + dp_deltaBpara_sum) / dp_gamma * dh_dz_function_sum * 1E-5  #[N]
    F_electric          = - (elementary_charge/1E1*speed_of_light*1E2) * dp_deltaEpara_sum * 1E-5   #[N]

    fig = plt.figure(figsize=(24, 12), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'time [s]', ylabel=r'Force [N]')
    ax.plot(dp_time, F_mirror_background, color='purple', alpha=0.5, label=r'$F_{B_0}$', lw=4)
    if (switch_delta_Bpara == 1E0):
        ax.plot(dp_time, F_mirror_wave_B0, color='red', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (B_0)$', lw=4)
        ax.plot(dp_time, F_mirror_wave_kpara, color='magenta', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (k_{\parallel})$', lw=4)
        ax.plot(dp_time, F_mirror_wave_dg_dz, color='orange', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (g)$', lw=4)
        if (switch_wave_packet == 1E0):
            ax.plot(dp_time, F_mirror_wave_dh_dz, color='green', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (h)$', lw=4)
    if (switch_delta_Epara == 1E0):
        ax.plot(dp_time, F_electric, color='b', alpha=0.5, label=r'$F_{\delta E_{\parallel}}$', lw=4)
    ax.minorticks_on()
    ax.grid(which="both", alpha=0.3)
    ax.legend()

    fig.savefig(f'{dir_name}/result_parallel_force/particle_trajectory{particle_file_number}.png')

if (channel == 13):
    fig = plt.figure(figsize=(24, 12), dpi=100, tight_layout=True)
    ax1 = fig.add_subplot(121, xlabel=r'time [s]', ylabel=r'parallel velocity [/c]')
    ax1.plot(dp_time, dp_v_para/speed_of_light)
    ax1.minorticks_on()
    ax1.grid(which='both', alpha=0.3)

    ax2 = fig.add_subplot(122, xlabel=r'time [s]', ylabel=r'perpendicular speed [/c]')
    ax2.plot(dp_time, dp_v_perp/speed_of_light)
    ax2.minorticks_on()
    ax2.grid(which='both', alpha=0.3)
    fig.savefig(f'{dir_name}/result_particle_velocity/particle_trajectory{particle_file_number}.png')

plt.show()