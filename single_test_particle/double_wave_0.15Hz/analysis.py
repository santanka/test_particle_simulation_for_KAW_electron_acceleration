import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

wave_scalar_potential   = 2000E0     #[V]
initial_wavephase       = 0E0       #[deg]
gradient_parameter      = 2E0       #[]
wave_threshold          = 5E0       #[deg]

wavekind                = r'EparaBpara'
switch_delta_Epara      = 1E0
switch_delta_Eperp_perp = 0E0
switch_delta_Eperp_phi  = 0E0
switch_delta_Bpara      = 1E0
switch_delta_Bperp      = 0E0

switch_wave_packet = 0E0

wave_frequency = 2E0 * np.pi * 0.15    #[rad/s]
kperp_rhoi = 2E0 * np.pi

particle_file_number    = r'20-102'
data_limit_under        = 0
data_limit_upper        =100000

channel = 24
#1:trajectory, 2:energy & equatorial pitch angle, 3:delta_Epara (t=8pi/wave_freq), 4:delta_Eperpperp (t=8pi/wave_freq), 5:delta_Eperpphi (t=8pi/wave_freq)
#6:delta_Bpara (t=8pi/wave_freq), 7:delta_Bperp (t=8pi/wave_freq), 8:wave frequency, 9:wavelength, 10:wavephase variation on particle
#11:wavephase on particle vs. wave phase speed, 12:wave parallel components' forces, 13:particle velocity, 14:plasma beta on particle
#15:energy (colored by time), 16:wavephase on particle vs. wave phase speed (colored by time), 17:wave parallel components' forces (simple 3 kinds of forces)
#18:wave parallel components' forces (simple 3 kinds of forces) & pitch angle & energy
#19:wave parallel components' forces (6 kinds of forces) & pitch angle & energy & plasma beta
#20:wave parallel components times particle parallel velocity (7 kinds) & pitch angle & energy & plasma beta & scalar potential
#21:wave phase & wave trapping frequency & delta_m & stable/unstable point time variation (old)
#22:psi-theta plot, 23:wave phase & wave trapping frequency & delta_m & stable/unstable point time variation, 24:delta_m & Gamma_tr

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


dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave/0.15Hz/results_particle_{str(int(wave_scalar_potential))}V' \
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
dp_v_para   = dp_u_para / dp_gamma  #[m s-1]
dp_v_perp   = dp_u_perp / dp_gamma  #[m s-1]


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

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
    ax = fig.add_subplot(111, xlabel=r'MLAT [degree]', ylabel=r'$v_{\parallel}$ [$\times$c]')
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
    mkdir(f'{dir_name}/result_energy_eqpitchangle')
    fig.savefig(f'{dir_name}/result_energy_eqpitchangle/particle_trajectory{particle_file_number}.png')

def profile_plot_mlat(z_position_m, profile, profile_name):
    mlat_deg = z_position_m_to_mlat_rad(z_position_m) * rad2deg
    fig = plt.figure(figsize=(7, 7), dpi=100, tight_layout=True)
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
    mkdir(f'{dir_name}/result_wavephase')
    fig.savefig(f'{dir_name}/result_wavephase/particle_trajectory{particle_file_number}.png')

if (channel == 11):
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)
    dp_B0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]

    dp_Alfven_speed = dp_B0 / np.sqrt(4E0 * np.pi * mass_ion * number_density_ion)  #[cm/s]
    dp_beta_ion = pressure_ion / (dp_B0**2E0 / 2E0 / 4E0 / np.pi)    #[]
    dp_kpara = wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron)) * np.sign(dp_mlat_rad)   #[rad/cm]
    dp_phasespeed = wave_frequency / dp_kpara / 1E2    #[m s-1]
    
    dp_theta = dp_v_para / dp_phasespeed - 1E0

    dp_wavephase_major = get_major_wave_component(dp_z_position, dp_wavephase_1, dp_wavephase_2)

    fig = plt.figure(figsize=(14, 14), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'wave phase $\psi$ [$\times \pi$ rad]', ylabel=r'$\frac{v_{\parallel}}{V_{R \parallel}}-1$')
    ax.plot(dp_wavephase_major/np.pi, dp_theta)
    ax.scatter(dp_wavephase_major[0]/np.pi, dp_theta[0], marker='o', color='r', label='start', zorder=3, s=200)
    ax.scatter(dp_wavephase_major[-1]/np.pi, dp_theta[-1], marker='D', color='r', label='start', zorder=3, s=200)
    ax.set_xlim((initial_wavephase*deg2rad-8*np.pi) / np.pi -1, (initial_wavephase*deg2rad) / np.pi +1)
    ax.minorticks_on()
    ax.grid(which="both", alpha=0.3)
    #mkdir(f'{dir_name}/result_wavephase_phasespeed')
    #fig.savefig(f'{dir_name}/result_wavephase_phasespeed/particle_trajectory{particle_file_number}.png')

def make_h_function(array_size, wave_phase, kpara):
    h_function = np.ones(array_size)
    dh_dz = np.zeros(array_size)
    if (switch_wave_packet == 1E0):
        for count_i in range(array_size):
            if (wave_phase[count_i] >= initial_wavephase*deg2rad-8.*np.pi and wave_phase[count_i] <= initial_wavephase*deg2rad):
                h_function[count_i] = 5E-1 * (1E0 - np.cos(1E0/4E0 * (wave_phase[count_i] - initial_wavephase*deg2rad)))
                dh_dz[count_i] = kpara[count_i] / 8E0 * np.sin((wave_phase[count_i] - initial_wavephase*deg2rad) / 4E0)
            else:
                h_function[count_i] = 0E0
                dh_dz[count_i] == 0E0
    return h_function, dh_dz
        

if (channel == 12):
    mkdir(f'{dir_name}/result_parallel_force')
    
    array_size = len(dp_z_position)
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)

    dp_B0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]
    dp_Alfven_speed = dp_B0 / np.sqrt(4E0 * np.pi * mass_ion * number_density_ion)  #[cm/s]
    dp_beta_ion = pressure_ion / (dp_B0**2E0 / 2E0 / 4E0 / np.pi)    #[]
    dp_kpara_1 = + wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    dp_kpara_2 = - wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]

    ion_Larmor_radius = (speed_of_light*1E2) * np.sqrt(2E0*mass_ion*pressure_ion/number_density_ion) / (elementary_charge/1E1*speed_of_light*1E2) / dp_B0   #[cm]

    dp_kperp_1 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]
    dp_kperp_2 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]

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

    #fig.savefig(f'{dir_name}/result_parallel_force/particle_trajectory{particle_file_number}.png')

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

if (channel == 14):
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)
    dp_b0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]
    dp_beta_ion = 8E0 * np.pi * pressure_ion / dp_b0**2E0

    fig = plt.figure(figsize=(14, 14), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'time [s]', ylabel=r'ion plasma $\beta$', yscale='log')
    ax.plot(dp_time, dp_beta_ion)
    ax.minorticks_on()
    ax.grid(which="both", alpha=0.3)
    mkdir(f'{dir_name}/result_plasma_beta_ion')
    fig.savefig(f'{dir_name}/result_plasma_beta_ion/particle_trajectory{particle_file_number}.png')

if (channel == 15):
    fig = plt.figure(figsize=(10, 10), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'time [s]', ylabel=r'energy [eV]')
    mappable = ax.scatter(dp_time, dp_energy, c=dp_time, cmap='turbo', marker='.', lw=0)
    fig.colorbar(mappable=mappable, ax=ax, label=r'time [s]')
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.set_axisbelow(True)

    #mkdir(f'{dir_name}/result_energy_color')
    #fig.savefig(f'{dir_name}/result_energy_color/particle_trajectory{particle_file_number}.png')

if (channel == 16):
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)
    b0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]
    
    dp_Alfven_speed = b0 / np.sqrt(4E0 * np.pi * mass_ion * number_density_ion)  #[cm/s]
    dp_beta_ion = pressure_ion / (b0**2E0 / 2E0 / 4E0 / np.pi)    #[]
    dp_kpara = wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron)) * np.sign(dp_mlat_rad)     #[rad/cm]
    dp_phasespeed = wave_frequency / dp_kpara / 1E2    #[m s-1]
    
    dp_theta = dp_v_para / dp_phasespeed - 1E0

    dp_wavephase_major = get_major_wave_component(dp_z_position, dp_wavephase_1, dp_wavephase_2)

    fig = plt.figure(figsize=(10, 7), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'wave phase $\psi$ [$\times \pi$ rad]', ylabel=r'$\frac{v_{\parallel}}{V_{\mathrm{ph} \parallel}}-1$')
    mappable = ax.scatter(dp_wavephase_major / np.pi, dp_theta, c=dp_time, cmap='turbo', marker='.', lw=0)
    fig.colorbar(mappable=mappable, ax=ax, label=r'time [s]')
    ax.scatter(dp_wavephase_major[0], dp_theta[0], marker='o', color='r', label='start', zorder=3, s=200)
    ax.scatter(dp_wavephase_major[-1], dp_theta[-1], marker='D', color='r', label='start', zorder=3, s=200)
    ax.set_xlim((initial_wavephase*deg2rad-8*np.pi) / np.pi -1, (initial_wavephase*deg2rad) / np.pi +1)
    ax.minorticks_on()
    ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True)

    #mkdir(f'{dir_name}/result_wavephase_phasespeed_color')
    #fig.savefig(f'{dir_name}/result_wavephase_phasespeed_color/particle_trajectory{particle_file_number}.png')

if (channel == 17):
    mkdir(f'{dir_name}/result_parallel_force_simple')
    
    array_size = len(dp_z_position)
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)

    dp_B0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]
    dp_Alfven_speed = dp_B0 / np.sqrt(4E0 * np.pi * mass_ion * number_density_ion)  #[cm/s]
    dp_beta_ion = pressure_ion / (dp_B0**2E0 / 2E0 / 4E0 / np.pi)    #[]
    dp_kpara_1 = + wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    dp_kpara_2 = - wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    
    ion_Larmor_radius = (speed_of_light*1E2) * np.sqrt(2E0*mass_ion*pressure_ion/number_density_ion) / (elementary_charge/1E1*speed_of_light*1E2) / dp_B0   #[cm]

    dp_kperp_1 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]
    dp_kperp_2 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]

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

    F_mirror_wave = F_mirror_wave_B0 + F_mirror_wave_kpara + F_mirror_wave_dg_dz + F_mirror_wave_dh_dz

    fig = plt.figure(figsize=(24, 12), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111, xlabel=r'time [s]', ylabel=r'Force [N]')
    ax.plot(dp_time, F_mirror_background, color='purple', alpha=0.5, label=r'$F_{B_0}$', lw=4)
    if (switch_delta_Bpara == 1E0):
        ax.plot(dp_time, F_mirror_wave, color='red', alpha=0.5, label=r'$F_{\delta B_{\parallel}}$', lw=4)
    if (switch_delta_Epara == 1E0):
        ax.plot(dp_time, F_electric, color='b', alpha=0.5, label=r'$F_{\delta E_{\parallel}}$', lw=4)
    ax.minorticks_on()
    ax.grid(which="both", alpha=0.3)
    ax.legend()

    fig.savefig(f'{dir_name}/result_parallel_force_simple/particle_trajectory{particle_file_number}.png')

if (channel == 18):
    mkdir(f'{dir_name}/result_parallel_force_simple_pitch_angle_energy')
    
    array_size = len(dp_z_position)
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)

    dp_B0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]
    dp_Alfven_speed = dp_B0 / np.sqrt(4E0 * np.pi * mass_ion * number_density_ion)  #[cm/s]
    dp_beta_ion = pressure_ion / (dp_B0**2E0 / 2E0 / 4E0 / np.pi)    #[]
    dp_kpara_1 = + wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    dp_kpara_2 = - wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    
    ion_Larmor_radius = (speed_of_light*1E2) * np.sqrt(2E0*mass_ion*pressure_ion/number_density_ion) / (elementary_charge/1E1*speed_of_light*1E2) / dp_B0   #[cm]

    dp_kperp_1 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]
    dp_kperp_2 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]

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

    F_mirror_wave = F_mirror_wave_B0 + F_mirror_wave_kpara + F_mirror_wave_dg_dz + F_mirror_wave_dh_dz

    dp_pitchangle = np.arctan(dp_v_perp/dp_v_para) * rad2deg    #[deg]
    dp_pitchangle = np.mod(dp_pitchangle, 180.)

    
    plt.rcParams["font.size"] = 40

    fig = plt.figure(figsize=(30, 20), dpi=100)
    fig.suptitle(str(wavekind) + r', initial energy = ' + str(int(dp_energy[0])) + r' [eV], pitch angle = ' + str(int(np.round(dp_pitchangle_eq[0]))) + r' [deg], grad = ' + str(int(gradient_parameter)) + r', wavephase @ 0 deg = ' + str(int(initial_wavephase)) + r' [deg]')
    
    gs = fig.add_gridspec(6, 1)

    ax1 = fig.add_subplot(gs[0, 0], xlabel=r'time [s]', ylabel=r'Energy [eV]')
    ax1.plot(dp_time, dp_energy, lw=4)
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.set_ticks_position('top')
    ax1.minorticks_on()
    ax1.grid(which="both", alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, ylabel=r'Pitch angle' '\n' r'[deg]')
    ax2.hlines(90, dp_time[0], dp_time[-1], color='k', lw=4, linestyles='dashed', alpha=0.5)
    ax2.plot(dp_time, dp_pitchangle, lw=4)
    ax2.minorticks_on()
    ax2.grid(which="both", alpha=0.3)
    ax2.tick_params(labelbottom=False, bottom=True)

    ax3 = fig.add_subplot(gs[2:4, 0], sharex=ax1, ylabel=r'Force [$\times 10^{-23}$ N]')
    ax3.plot(dp_time, F_mirror_background*1E23, color='purple', alpha=0.5, label=r'$F_{B_0}$', lw=4)
    if (switch_delta_Bpara == 1E0):
        ax3.plot(dp_time, F_mirror_wave*1E23, color='red', alpha=0.5, label=r'$F_{\delta B_{\parallel}}$', lw=4)
    if (switch_delta_Epara == 1E0):
        ax3.plot(dp_time, F_electric*1E23, color='b', alpha=0.5, label=r'$F_{\delta E_{\parallel}}$', lw=4)
    ax3.minorticks_on()
    ax3.grid(which="both", alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(gs[4:, 0], xlabel=r'time [s]', ylabel=r'Force [$\times 10^{-23}$ N]')
    if (switch_delta_Bpara == 1E0):
        ax4.plot(dp_time, F_mirror_wave*1E23, color='red', alpha=0, lw=4)
        ylim = ax4.get_ylim()
    ax4.plot(dp_time, F_mirror_background*1E23, color='purple', alpha=0.2, label=r'$F_{B_0}$', lw=4)
    if (switch_delta_Bpara == 1E0):
        ax4.plot(dp_time, F_mirror_wave*1E23, color='red', alpha=0.5, label=r'$F_{\delta B_{\parallel}}$', lw=4)
    if (switch_delta_Epara == 1E0):
        ax4.plot(dp_time, F_electric*1E23, color='b', alpha=0.2, label=r'$F_{\delta E_{\parallel}}$', lw=4)
    if (switch_delta_Bpara == 1E0):
        ax4.set_ylim(ylim)
    ax4.minorticks_on()
    ax4.grid(which="both", alpha=0.3)
    ax4.legend(loc='upper right')

    fig.subplots_adjust(hspace=0)

    fig.savefig(f'{dir_name}/result_parallel_force_simple_pitch_angle_energy/particle_trajectory{particle_file_number}.png')

if (channel == 19):
    mkdir(f'{dir_name}/result_parallel_force_pitch_angle_energy')
    
    array_size = len(dp_z_position)
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)

    dp_B0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]
    dp_Alfven_speed = dp_B0 / np.sqrt(4E0 * np.pi * mass_ion * number_density_ion)  #[cm/s]
    dp_beta_ion = pressure_ion / (dp_B0**2E0 / 2E0 / 4E0 / np.pi)    #[]
    dp_kpara_1 = + wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    dp_kpara_2 = - wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    
    ion_Larmor_radius = (speed_of_light*1E2) * np.sqrt(2E0*mass_ion*pressure_ion/number_density_ion) / (elementary_charge/1E1*speed_of_light*1E2) / dp_B0   #[cm]

    dp_kperp_1 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]
    dp_kperp_2 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]

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

    F_mirror_wave = F_mirror_wave_B0 + F_mirror_wave_kpara + F_mirror_wave_dg_dz + F_mirror_wave_dh_dz

    dp_pitchangle = np.arctan(dp_v_perp/dp_v_para) * rad2deg    #[deg]
    dp_pitchangle = np.mod(dp_pitchangle, 180.)

    dp_beta_ion = 8E0 * np.pi * pressure_ion / dp_B0**2E0

    
    plt.rcParams["font.size"] = 40

    fig = plt.figure(figsize=(30, 30), dpi=100)
    fig.suptitle(str(wavekind) + r', initial energy = ' + str(int(dp_energy[0])) + r' [eV], pitch angle = ' + str(int(np.round(dp_pitchangle_eq[0]))) + r' [deg], grad = ' + str(int(gradient_parameter)) + r', wavephase @ 0 deg = ' + str(int(initial_wavephase)) + r' [deg]')
    
    gs = fig.add_gridspec(8, 1)

    ax1 = fig.add_subplot(gs[0, 0], xlabel=r'time [s]', ylabel=r'Energy [eV]')
    ax1.plot(dp_time, dp_energy, lw=4)
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.set_ticks_position('top')
    ax1.minorticks_on()
    ax1.grid(which="both", alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, ylabel=r'Pitch angle' '\n' r'[deg]')
    ax2.hlines(90, dp_time[0], dp_time[-1], color='k', lw=4, linestyles='dashed', alpha=0.5)
    ax2.plot(dp_time, dp_pitchangle, lw=4)
    ax2.minorticks_on()
    ax2.grid(which="both", alpha=0.3)
    ax2.tick_params(labelbottom=False, bottom=True)

    ax5 = fig.add_subplot(gs[2:4, 0], sharex=ax1, ylabel=r'plasma $\beta_{\mathrm{i}}$', yscale='log')
    ax5.plot(dp_time, dp_beta_ion, lw=4)
    ax5.minorticks_on()
    ylim_ax5 = ax5.get_ylim()
    ax5.hlines(mass_electron/mass_ion, dp_time[0], dp_time[-1], color='dimgrey', lw=4, linestyles='dashed', alpha=0.5)
    ax5.set_ylim(ylim_ax5)
    ax5.grid(which="both", alpha=0.3)
    ax5.tick_params(labelbottom=False, bottom=True)

    ax3 = fig.add_subplot(gs[4:6, 0], sharex=ax1, ylabel=r'Force [$\times 10^{-23}$ N]')
    ax3.plot(dp_time, F_mirror_background*1E23, color='purple', alpha=0.5, label=r'$F_{B_0}$', lw=4)
    if (switch_delta_Bpara == 1E0):
        ax3.plot(dp_time, F_mirror_wave*1E23, color='firebrick', alpha=0.5, label=r'$F_{\delta B_{\parallel}}$', lw=4)
    if (switch_delta_Epara == 1E0):
        ax3.plot(dp_time, F_electric*1E23, color='b', alpha=0.5, label=r'$F_{\delta E_{\parallel}}$', lw=4)
    ax3.minorticks_on()
    ax3.grid(which="both", alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(gs[6:, 0], xlabel=r'time [s]', ylabel=r'Force [$\times 10^{-23}$ N]')
    if (switch_delta_Bpara == 1E0):
        ax4.plot(dp_time, F_mirror_wave*1E23, color='red', alpha=0, lw=4)
        ylim = ax4.get_ylim()
    ax4.plot(dp_time, F_mirror_background*1E23, color='purple', alpha=0.1, lw=4)
    if (switch_delta_Bpara == 1E0):
        ax4.plot(dp_time, F_mirror_wave_B0*1E23, color='red', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (B_0)$', lw=4)
        ax4.plot(dp_time, F_mirror_wave_kpara*1E23, color='magenta', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (k_{\parallel})$', lw=4)
        ax4.plot(dp_time, F_mirror_wave_dg_dz*1E23, color='orange', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (g)$', lw=4)
        if (switch_wave_packet == 1E0):
            ax4.plot(dp_time, F_mirror_wave_dh_dz*1E23, color='green', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (h)$', lw=4)
    if (switch_delta_Epara == 1E0):
        ax4.plot(dp_time, F_electric*1E23, color='b', alpha=0.1, lw=4)
    if (switch_delta_Bpara == 1E0):
        ax4.set_ylim(ylim)
    ax4.minorticks_on()
    ax4.grid(which="both", alpha=0.3)
    ax4.legend(loc='upper right')

    fig.subplots_adjust(hspace=0)

    fig.savefig(f'{dir_name}/result_parallel_force_pitch_angle_energy/particle_trajectory{particle_file_number}.png')

def step(x):
    return 1.0 * (x >= 0.0)

if (channel == 20):
    mkdir(f'{dir_name}/result_parallel_force_times_vpara_scalar_potential')
    
    array_size = len(dp_z_position)
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)

    dp_B0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]
    dp_Alfven_speed = dp_B0 / np.sqrt(4E0 * np.pi * mass_ion * number_density_ion)  #[cm/s]
    dp_beta_ion = pressure_ion / (dp_B0**2E0 / 2E0 / 4E0 / np.pi)    #[]
    dp_kpara_1 = + wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    dp_kpara_2 = - wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    
    ion_Larmor_radius = (speed_of_light*1E2) * np.sqrt(2E0*mass_ion*pressure_ion/number_density_ion) / (elementary_charge/1E1*speed_of_light*1E2) / dp_B0   #[cm]

    dp_kperp_1 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]
    dp_kperp_2 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]

    dp_dB0_dz = 3E0 * np.sin(dp_mlat_rad) * (5E0 * np.sin(dp_mlat_rad)**2E0 + 3E0) / np.cos(dp_mlat_rad)**8E0 / (3E0 * np.sin(dp_mlat_rad)**2E0 + 1E0) / (r_eq*1E2) * (B0_eq*1E4)   #[G/cm]
    Alpha = 4E0 * np.pi * (1E0 + pressure_electron / pressure_ion) * (elementary_charge/1E1*speed_of_light*1E2) * number_density_ion * (wave_scalar_potential*1E8/(speed_of_light*1E2))

    g_function_1 = 5E-1 * (np.tanh(+ gradient_parameter * (dp_mlat_rad*rad2deg - wave_threshold/2E0)) + 1E0)
    g_function_2 = 5E-1 * (np.tanh(- gradient_parameter * (dp_mlat_rad*rad2deg + wave_threshold/2E0)) + 1E0)

    step_function_1 = np.zeros(array_size)
    step_function_2 = np.zeros(array_size)
    for count_i in range(array_size):
        step_function_1[count_i] = step(dp_z_position[count_i])
        step_function_2[count_i] = step(- dp_z_position[count_i])

    dg_dz_1 = + 90E0 * gradient_parameter / np.pi / np.cosh(+ gradient_parameter * (dp_mlat_rad*rad2deg - wave_threshold/2E0))**2E0 / (r_eq*1E2) / np.cos(dp_mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0)
    dg_dz_2 = - 90E0 * gradient_parameter / np.pi / np.cosh(- gradient_parameter * (dp_mlat_rad*rad2deg + wave_threshold/2E0))**2E0 / (r_eq*1E2) / np.cos(dp_mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0)

    h_function_1, dh_dz_1 = make_h_function(array_size, dp_wavephase_1, dp_kpara_1)
    h_function_2, dh_dz_2 = make_h_function(array_size, dp_wavephase_2, dp_kpara_2)

    dp_scalar_potential = wave_scalar_potential * (g_function_1 * h_function_1 + g_function_2 * h_function_2)

    #neglecting the g-function
    dp_effective_scalar_potential = np.zeros(array_size)
    for count_i in range(array_size):
        if (switch_wave_packet == 0):
            dp_effective_scalar_potential[count_i] = 2E0 * (2E0 + pressure_electron / pressure_ion) * wave_scalar_potential
        elif (switch_wave_packet == 1):
            psi_pi_1 = np.floor(dp_wavephase_1[count_i] / np.pi)
            psi_pi_2 = np.floor(dp_wavephase_2[count_i] / np.pi)
            initial_wavephase_rad = initial_wavephase * deg2rad

            if (psi_pi_1 < initial_wavephase_rad / np.pi and psi_pi_1 >= initial_wavephase_rad / np.pi - 1E0):
                dp_effective_scalar_potential[count_i] = np.abs(1/30E0 * np.cos(initial_wavephase_rad) - 0.2 * np.cos((5*np.pi*psi_pi_1 - initial_wavephase_rad)/4E0) \
                    - 1/3E0 * np.cos((3*np.pi*psi_pi_1 + initial_wavephase_rad)/4E0) + 0.5 * (-1)**psi_pi_1)
            elif (psi_pi_1 >= initial_wavephase_rad / np.pi - 8E0 and psi_pi_1 < initial_wavephase_rad / np.pi - 1E0):
                dp_effective_scalar_potential[count_i] = np.abs(-0.4*np.cos(8/np.pi)*np.cos((5*np.pi*psi_pi_1 - initial_wavephase_rad + np.pi/2)/4E0) \
                    + 2/3E0 * np.cos(8/np.pi) * np.cos((3*np.pi*psi_pi_1 + initial_wavephase_rad - np.pi/2)/4E0) + (-1)**psi_pi_1)
            elif (psi_pi_1 >= initial_wavephase_rad / np.pi - 9E0 and psi_pi_1 < initial_wavephase_rad / np.pi - 8E0):
                dp_effective_scalar_potential[count_i] = np.abs(0.2 * np.cos((5*np.pi*psi_pi_1 - initial_wavephase_rad + 5*np.pi)/4E0) \
                    + 1/3E0 * np.cos((3*np.pi*psi_pi_1 + initial_wavephase_rad + 3*np.pi)/4E0) + 0.5 * (-1)**psi_pi_1 - 1/30E0 * np.cos(initial_wavephase_rad))
            
            dp_effective_scalar_potential[count_i] = np.abs(dp_effective_scalar_potential[count_i]) * step_function_1[count_i]

            if (psi_pi_2 < initial_wavephase_rad / np.pi and psi_pi_2 >= initial_wavephase_rad / np.pi - 1E0):
                dp_effective_scalar_potential[count_i] = dp_effective_scalar_potential[count_i] + np.abs(1/30E0 * np.cos(initial_wavephase_rad) - 0.2 * np.cos((5*np.pi*psi_pi_2 - initial_wavephase_rad)/4E0) \
                    - 1/3E0 * np.cos((3*np.pi*psi_pi_2 + initial_wavephase_rad)/4E0) + 0.5 * (-1)**psi_pi_2) * step_function_2[count_i]
            elif (psi_pi_2 >= initial_wavephase_rad / np.pi - 8E0 and psi_pi_2 < initial_wavephase_rad / np.pi - 1E0):
                dp_effective_scalar_potential[count_i] = dp_effective_scalar_potential[count_i] + np.abs(-0.4*np.cos(8/np.pi)*np.cos((5*np.pi*psi_pi_2 - initial_wavephase_rad + np.pi/2)/4E0) \
                    + 2/3E0 * np.cos(8/np.pi) * np.cos((3*np.pi*psi_pi_2 + initial_wavephase_rad - np.pi/2)/4E0) + (-1)**psi_pi_2) * step_function_2[count_i]
            elif (psi_pi_2 >= initial_wavephase_rad / np.pi - 9E0 and psi_pi_2 < initial_wavephase_rad / np.pi - 8E0):
                dp_effective_scalar_potential[count_i] = dp_effective_scalar_potential[count_i] + np.abs(0.2 * np.cos((5*np.pi*psi_pi_2 - initial_wavephase_rad + 5*np.pi)/4E0) \
                    + 1/3E0 * np.cos((3*np.pi*psi_pi_2 + initial_wavephase_rad + 3*np.pi)/4E0) + 0.5 * (-1)**psi_pi_2 - 1/30E0 * np.cos(initial_wavephase_rad)) * step_function_2[count_i]
            
            dp_effective_scalar_potential[count_i] = dp_effective_scalar_potential[count_i] * (2E0 + pressure_electron / pressure_ion) * wave_scalar_potential

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

    F_mirror_wave = F_mirror_wave_B0 + F_mirror_wave_kpara + F_mirror_wave_dg_dz + F_mirror_wave_dh_dz

    vpara_F_electric = dp_v_para * F_electric
    vpara_F_mirror_wave = dp_v_para * F_mirror_wave
    vpara_F_mirror_background = dp_v_para * F_mirror_background

    vpara_F_mirror_wave_B0 = dp_v_para * F_mirror_wave_B0
    vpara_F_mirror_wave_kpara = dp_v_para * F_mirror_wave_kpara
    vpara_F_mirror_wave_dg_dz = dp_v_para * F_mirror_wave_dg_dz
    vpara_F_mirror_wave_dh_dz = dp_v_para * F_mirror_wave_dh_dz

    dp_pitchangle = np.arctan(dp_v_perp/dp_v_para) * rad2deg    #[deg]
    dp_pitchangle = np.mod(dp_pitchangle, 180.)

    dp_energy_para = dp_energy * np.cos(dp_pitchangle*deg2rad)**2E0
    dp_energy_perp = dp_energy * np.sin(dp_pitchangle*deg2rad)**2E0

    dp_kpara = np.sqrt(2E0 * np.pi * number_density_ion * mass_ion * pressure_ion) / dp_B0**2E0 * np.sqrt(4E0 * np.pi + dp_B0**2E0 / (pressure_ion + pressure_electron)) * np.sign(dp_mlat_rad)   #[rad/cm]
    dp_wavefreq = 2E0 * np.pi / 2E0 #[rad/s]
    dp_phasespeed = dp_wavefreq / dp_kpara / 1E2    #[m s-1]
    dp_energy_para_waveframe = 0.5 * mass_electron*1E-3 * (dp_v_para - dp_phasespeed)**2E0 / elementary_charge #[eV]
    for count_i in range(array_size):
        if (h_function_1[count_i] == 0E0 and h_function_2[count_i] == 0E0):
            dp_energy_para_waveframe[count_i] = np.nan

    dp_beta_ion = 8E0 * np.pi * pressure_ion / dp_B0**2E0
    
    plt.rcParams["font.size"] = 30

    fig = plt.figure(figsize=(30, 20), dpi=100)
    fig.suptitle(str(wavekind) + r', initial energy = ' + str(int(dp_energy[0])) + r' [eV], pitch angle = ' + str(int(np.round(dp_pitchangle_eq[0]))) + r' [deg], grad = ' + str(int(gradient_parameter)) + r', wavephase @ 0 deg = ' + str(int(initial_wavephase)) + r' [deg]')
    
    gs = fig.add_gridspec(11, 1)

    ax1 = fig.add_subplot(gs[0:2, 0], xlabel=r'time [s]', ylabel=r'Energy')
    ax1.plot(dp_time, dp_energy, lw=4, color='orange', label=r'$K$ [eV]', alpha=0.5)
    ax1.plot(dp_time, dp_energy_para, lw=4, color='b', label=r'$K_{\parallel}$ [eV]', alpha=0.5)
    ax1.plot(dp_time, dp_energy_perp, lw=4, color='g', label=r'$K_{\perp}$ [eV]', alpha=0.5)
    ax1.plot(dp_time, dp_effective_scalar_potential, lw=4, color='r', label=r'$\phi_{\mathrm{eff}}$ [V]', alpha=0.5)
    ylim_ax1 = ax1.get_ylim()
    ax1.plot(dp_time, dp_energy_para_waveframe, lw=4, color='purple', label=r'$K_{\parallel \mathrm{wf}}$ [eV]', alpha=0.5)
    ax1.set_ylim(ylim_ax1)
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.set_ticks_position('top')
    ax1.minorticks_on()
    ax1.grid(which="both", alpha=0.3)
    ax1.legend(loc='upper right', fontsize=20)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1, ylabel=r'Pitch angle' '\n' r'[deg]')
    ax2.hlines(90, dp_time[0], dp_time[-1], color='k', lw=4, linestyles='dashed', alpha=0.5)
    ax2.plot(dp_time, dp_pitchangle, lw=4)
    ax2.minorticks_on()
    ax2.grid(which="both", alpha=0.3)
    ax2.tick_params(labelbottom=False, bottom=True)

    ax5 = fig.add_subplot(gs[3:5, 0], sharex=ax1, ylabel=r'plasma $\beta_{\mathrm{i}}$', yscale='log')
    ax5.plot(dp_time, dp_beta_ion, lw=4)
    ax5.minorticks_on()
    ylim_ax5 = ax5.get_ylim()
    ax5.hlines(mass_electron/mass_ion, dp_time[0], dp_time[-1], color='dimgrey', lw=4, linestyles='dashed', alpha=0.5)
    ax5.set_ylim(ylim_ax5)
    ax5.grid(which="both", alpha=0.3)
    ax5.tick_params(labelbottom=False, bottom=True)

    ax3 = fig.add_subplot(gs[5:8, 0], sharex=ax1, ylabel=r'$v_{\parallel} F_{\parallel}$ [eV/s]')
    ax3.plot(dp_time, vpara_F_mirror_background/elementary_charge, color='purple', alpha=0.5, label=r'$v_{\parallel} F_{B_0}$', lw=4)
    if (switch_delta_Bpara == 1E0):
        ax3.plot(dp_time, vpara_F_mirror_wave/elementary_charge, color='red', alpha=0.5, label=r'$v_{\parallel} F_{\delta B_{\parallel}}$', lw=4)
    if (switch_delta_Epara == 1E0):
        ax3.plot(dp_time, vpara_F_electric/elementary_charge, color='b', alpha=0.5, label=r'$v_{\parallel} F_{\delta E_{\parallel}}$', lw=4)
    ax3.minorticks_on()
    ax3.grid(which="both", alpha=0.3)
    ax3.legend(loc='upper right')

    ax4 = fig.add_subplot(gs[8:, 0], xlabel=r'time [s]', ylabel=r'$v_{\parallel} F_{\parallel}$ [eV/s]')
    if (switch_delta_Bpara == 1E0):
        ax4.plot(dp_time, vpara_F_mirror_wave/elementary_charge, color='red', alpha=0, lw=4)
        ylim = ax4.get_ylim()
    ax4.plot(dp_time, vpara_F_mirror_background/elementary_charge, color='purple', alpha=0.1, lw=4)
    if (switch_delta_Bpara == 1E0):
        ax4.plot(dp_time, vpara_F_mirror_wave_B0/elementary_charge, color='red', alpha=0.5, label=r'$v_{\parallel} F_{\delta B_{\parallel}} (B_0)$', lw=4)
        ax4.plot(dp_time, vpara_F_mirror_wave_kpara/elementary_charge, color='magenta', alpha=0.5, label=r'$v_{\parallel} F_{\delta B_{\parallel}} (k_{\parallel})$', lw=4)
        ax4.plot(dp_time, vpara_F_mirror_wave_dg_dz/elementary_charge, color='orange', alpha=0.5, label=r'$v_{\parallel} F_{\delta B_{\parallel}} (g)$', lw=4)
        if (switch_wave_packet == 1E0):
            ax4.plot(dp_time, vpara_F_mirror_wave_dh_dz/elementary_charge, color='green', alpha=0.5, label=r'$v_{\parallel} F_{\delta B_{\parallel}} (h)$', lw=4)
    if (switch_delta_Epara == 1E0):
        ax4.plot(dp_time, vpara_F_electric/elementary_charge, color='b', alpha=0.1, lw=4)
    if (switch_delta_Bpara == 1E0):
        ax4.set_ylim(ylim)
    ax4.minorticks_on()
    ax4.grid(which="both", alpha=0.3)
    ax4.legend(loc='upper right')
    fig.subplots_adjust(hspace=0)

    fig.savefig(f'{dir_name}/result_parallel_force_times_vpara_scalar_potential/particle_trajectory{particle_file_number}.png')

def normalize_angle_rad(angle):
    normalized_angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi
    return normalized_angle

def wavecheck(array_size, base, wavephase):
    if (switch_wave_packet == 0):
        return base
    elif (switch_wave_packet == 1):
        for count_i in range(array_size):
            if (wavephase[count_i] > initial_wavephase * deg2rad or wavephase[count_i] < initial_wavephase * deg2rad - 8E0 * np.pi):
                base[count_i] = np.nan
        return base


if (channel == 21 and switch_delta_Epara == 1E0):
    
    mkdir(f'{dir_name}/result_pendulum_theory_time_variation')

    # h(psi)に未対応

    array_size = len(dp_z_position)
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)

    dp_B0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]
    dp_Alfven_speed = dp_B0 / np.sqrt(4E0 * np.pi * mass_ion * number_density_ion)  #[cm/s]
    dp_beta_ion = pressure_ion / (dp_B0**2E0 / 2E0 / 4E0 / np.pi)    #[]
    dp_kpara_1 = + wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    dp_kpara_2 = - wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    
    ion_Larmor_radius = (speed_of_light*1E2) * np.sqrt(2E0*mass_ion*pressure_ion/number_density_ion) / (elementary_charge/1E1*speed_of_light*1E2) / dp_B0   #[cm]

    dp_kperp_1 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]
    dp_kperp_2 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]

    dp_dB0_dz = 3E0 * np.sin(dp_mlat_rad) * (5E0 * np.sin(dp_mlat_rad)**2E0 + 3E0) / np.cos(dp_mlat_rad)**8E0 / (3E0 * np.sin(dp_mlat_rad)**2E0 + 1E0) / (r_eq*1E2) * (B0_eq*1E4)   #[G/cm]

    dp_g_function_1 = 5E-1 * (np.tanh(+ gradient_parameter * (dp_mlat_rad*rad2deg - wave_threshold/2E0)) + 1E0)
    dp_g_function_2 = 5E-1 * (np.tanh(- gradient_parameter * (dp_mlat_rad*rad2deg + wave_threshold/2E0)) + 1E0)

    dp_dg_dz_1 = + 90E0 * gradient_parameter / np.pi / np.cosh(+ gradient_parameter * (dp_mlat_rad*rad2deg - wave_threshold/2E0))**2E0 / (r_eq*1E2) / np.cos(dp_mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0)
    dp_dg_dz_2 = - 90E0 * gradient_parameter / np.pi / np.cosh(- gradient_parameter * (dp_mlat_rad*rad2deg + wave_threshold/2E0))**2E0 / (r_eq*1E2) / np.cos(dp_mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0)

    dp_h_function_1, dp_dh_dz_1 = make_h_function(array_size, dp_wavephase_1, dp_kpara_1)
    dp_h_function_2, dp_dh_dz_2 = make_h_function(array_size, dp_wavephase_2, dp_kpara_2)

    dp_Phi = (2E0 + temperature_electron / temperature_ion) * (wave_scalar_potential*1E8/(speed_of_light*1E2)) * (dp_g_function_1 * dp_h_function_1 + dp_g_function_2 * dp_h_function_2)   #[statV]=[erg/statC]

    dp_Phi_B = 4E0 * np.pi * number_density_ion * (elementary_charge/1E1*speed_of_light*1E2) * (1E0 - temperature_ion / (2E0 * temperature_ion + temperature_electron)) * dp_Phi * switch_delta_Bpara    #[erg/cm3]=[dyn/cm^2]=[G^2]

    dp_d_Phi_B_dz = 4E0 * np.pi * number_density_ion * (elementary_charge * wave_scalar_potential * 1E7) * (1E0 + temperature_electron / temperature_ion) \
        * (dp_dg_dz_1 * dp_h_function_1 + dp_g_function_1 * dp_dh_dz_1 + dp_dg_dz_2 * dp_h_function_2 + dp_g_function_2 * dp_dh_dz_2) * switch_delta_Bpara   #[G^2/cm]

    dp_d_Phi_B_B0_dz = dp_d_Phi_B_dz / dp_B0 - dp_dB0_dz * dp_Phi_B / dp_B0**2E0    #[G/cm]

    dp_pitchangle = np.arctan(dp_v_perp/dp_v_para) * rad2deg    #[deg]
    dp_pitchangle = np.mod(dp_pitchangle, 180.)
    
    step_function_1 = np.zeros(array_size)
    step_function_2 = np.zeros(array_size)
    for count_i in range(array_size):
        step_function_1[count_i] = step(+ dp_z_position[count_i])
        step_function_2[count_i] = step(- dp_z_position[count_i])
    
    dp_kpara = (dp_kpara_1 * step_function_1 + dp_kpara_2 * step_function_2) #[rad/cm]
    dp_kperp = (dp_kperp_1 * step_function_1 + dp_kperp_2 * step_function_2) #[rad/cm]
    dp_wavephase = (dp_wavephase_1 * step_function_1 + dp_wavephase_2 * step_function_2) #[rad]

    dp_cyclotron_freq_electron = (elementary_charge/1E1*speed_of_light*1E2) * dp_B0 / mass_electron / (speed_of_light*1E2)  #[rad/s]


    dp_omega_t = dp_kpara * np.sqrt((elementary_charge * wave_scalar_potential * 1E7) / mass_electron)   #[rad/s]
    dp_omega_tr = dp_omega_t / np.sqrt(dp_gamma) * np.sqrt(1E0 - (dp_v_para/speed_of_light)**2E0)    #[rad/s]

    #dp_omega_mtr_nondeltaB = (dp_omega_tr**2E0 - (dp_kpara * dp_v_para*1E2) * (dp_kperp * dp_v_perp*1E2 * np.cos(dp_u_phase)) * (dp_omega_t / dp_kpara / (speed_of_light*1E2))**2E0)**(0.5)    #[rad/s]
    dp_omega_mtr_nondeltaB = dp_omega_tr    #[rad/s]

    #dp_S_nondeltaB = dp_omega_mtr_nondeltaB**(-2E0) * (((dp_kpara * dp_v_perp*1E2)**2E0 / 2E0 + 2E0 * (dp_kpara * dp_v_para*1E2)**2E0 * (1E0 - temperature_ion / (dp_beta_ion * (temperature_ion + temperature_electron) + 2E0 * temperature_ion)) \
    #                                                    + (dp_kpara * dp_v_para*1E2) * (dp_kperp * dp_v_perp*1E2 * np.cos(dp_u_phase))) / dp_kpara / dp_B0 * dp_dB0_dz + (dp_kperp * dp_v_perp * np.sin(dp_u_phase)) * dp_cyclotron_freq_electron / dp_gamma)  #[]
    dp_S_nondeltaB = ((dp_v_perp*1E2)**2E0 / 2E0 + 2E0 * (dp_v_para*1E2)**2E0 * (1E0 - temperature_ion / (dp_beta_ion * (temperature_ion + temperature_electron) + 2E0 * temperature_ion))) * (dp_kpara / dp_omega_mtr_nondeltaB)**2E0 / dp_kpara / dp_B0 * dp_dB0_dz   #[]
    
    dp_wavephase_stable_nondeltaB = normalize_angle_rad(np.arcsin(-dp_S_nondeltaB))   #[rad]
    dp_wavephase_unstable_nondeltaB = np.zeros(array_size)
    for count_i in range(array_size):
        if (dp_wavephase_stable_nondeltaB[count_i] >= 0E0):
            dp_wavephase_unstable_nondeltaB[count_i] = np.pi - dp_wavephase_stable_nondeltaB[count_i]
        elif (dp_wavephase_stable_nondeltaB[count_i] < 0E0):
            dp_wavephase_unstable_nondeltaB[count_i] = - np.pi - dp_wavephase_stable_nondeltaB[count_i]
        elif (dp_wavephase_stable_nondeltaB[count_i] != dp_wavephase_stable_nondeltaB[count_i]):
            dp_wavephase_unstable_nondeltaB[count_i] = np.nan

    dp_wavefreq = 2E0 * np.pi / 2E0 #[rad/s]
    dp_theta_nondeltaB = dp_kpara * dp_v_para*1E2 - dp_wavefreq     #[rad/s]

    dp_theta_limit_plus_nondeltaB = np.sqrt(1E0 / 2E0 * ((np.cos(dp_wavephase) - np.cos(dp_wavephase_unstable_nondeltaB)) - dp_S_nondeltaB * (dp_wavephase - dp_wavephase_unstable_nondeltaB)))    #[]
    dp_theta_limit_minus_nondeltaB = - dp_theta_limit_plus_nondeltaB    #[]

    #波があるときのみ
    dp_omega_t = np.abs(wavecheck(array_size, dp_omega_t, dp_wavephase))    #[rad/s]
    dp_omega_tr = np.abs(wavecheck(array_size, dp_omega_tr, dp_wavephase))    #[rad/s]
    dp_omega_mtr_nondeltaB = np.abs(wavecheck(array_size, dp_omega_mtr_nondeltaB, dp_wavephase))    #[rad/s]
    dp_S_nondeltaB = wavecheck(array_size, dp_S_nondeltaB, dp_wavephase)    #[]
    dp_wavephase_stable_nondeltaB = wavecheck(array_size, dp_wavephase_stable_nondeltaB, dp_wavephase)    #[rad]
    dp_wavephase_unstable_nondeltaB = wavecheck(array_size, dp_wavephase_unstable_nondeltaB, dp_wavephase)    #[rad]
    dp_theta_nondeltaB = wavecheck(array_size, dp_theta_nondeltaB, dp_wavephase)    #[rad/s]
    dp_theta_limit_plus_nondeltaB = wavecheck(array_size, dp_theta_limit_plus_nondeltaB, dp_wavephase)    #[]
    dp_theta_limit_minus_nondeltaB = wavecheck(array_size, dp_theta_limit_minus_nondeltaB, dp_wavephase)    #[]
    

    if (switch_delta_Bpara == 1E0):
        #A_deltaBpara = dp_omega_tr**2E0 - ((dp_kpara * dp_v_perp*1E2)**2E0 - (dp_kpara * dp_v_para*1E2) * (dp_kperp * dp_v_perp*1E2 * np.cos(dp_u_phase)) * dp_Phi_B / dp_B0**2E0) - (dp_kpara * dp_v_para*1E2) * (dp_kperp * dp_v_perp*1E2 * np.cos(dp_u_phase)) * (dp_omega_t / dp_kpara / (speed_of_light*1E2))**2E0    #[rad^2/s^2]
        #B_deltaBpara = ((dp_kpara * dp_v_perp*1E2)**2E0 - (dp_kpara * dp_v_para*1E2) * (dp_kperp * dp_v_perp*1E2 * np.cos(dp_u_phase))) / dp_kpara / dp_B0 * dp_d_Phi_B_B0_dz \
        #    + (dp_kpara * dp_v_para*1E2) * (2E0 * (dp_kpara * dp_v_para*1E2) * (1E0 - temperature_ion / (dp_beta_ion * (temperature_ion + temperature_electron) + 2E0 * temperature_ion)) + (dp_kperp * dp_v_perp*1E2) * np.cos(dp_u_phase)) / dp_kpara / dp_B0 * dp_dB0_dz * dp_Phi_B / dp_B0**2E0 \
        #        + 2E0 * (dp_kperp * dp_v_perp*1E2 * np.sin(dp_u_phase)) * dp_cyclotron_freq_electron / dp_gamma * dp_Phi_B / dp_B0**2E0     #[rad^2/s^2]
        #C_deltaBpara = 1E0 / 2E0 * dp_Phi_B / dp_B0**2E0 * (dp_omega_tr**2E0 - (dp_kpara * dp_v_para*1E2) * (dp_kperp * dp_v_perp*1E2 * np.cos(dp_u_phase)) * (dp_omega_t / dp_kpara / (speed_of_light*1E2))**2E0)    #[rad^2/s^2]
        #D_deltaBpara = 1E0 / 2E0 * (dp_Phi_B / dp_B0**2E0)**2E0 * (dp_kperp * dp_v_perp*1E2 * np.sin(dp_u_phase)) * dp_cyclotron_freq_electron / dp_gamma    #[rad^2/s^2]
        #E_deltaBpara = (1E0 / 2E0 * (dp_kpara * dp_v_perp*1E2)**2E0 + 2E0 * (dp_kpara * dp_v_para*1E2)**2E0 * (1E0 - temperature_ion / (dp_beta_ion * (temperature_ion + temperature_electron) + 2E0 * temperature_ion)) + 1E0 / 2E0 * (dp_kpara * dp_v_para*1E2) * (dp_kperp * dp_v_perp*1E2 * np.cos(dp_u_phase))) / dp_kpara / dp_B0 * dp_dB0_dz \
        #    + (dp_kperp * dp_v_perp*1E2 * np.sin(dp_u_phase)) * dp_cyclotron_freq_electron / dp_gamma + 1E0 / 2E0 * (dp_Phi_B / dp_B0**2E0)**2E0 * (dp_kperp * dp_v_perp*1E2 * np.sin(dp_u_phase)) * dp_cyclotron_freq_electron / dp_gamma    #[rad^2/s^2]
        
        A_deltaBpara = dp_omega_tr**2E0 - ((dp_kpara * dp_v_perp*1E2)**2E0 * dp_Phi_B / dp_B0**2E0)    #[rad^2/s^2]
        B_deltaBpara = ((dp_kpara * dp_v_perp*1E2)**2E0) / dp_kpara / dp_B0 * dp_d_Phi_B_B0_dz \
            + (dp_kpara * dp_v_para*1E2) * (2E0 * (dp_kpara * dp_v_para*1E2) * (1E0 - temperature_ion / (dp_beta_ion * (temperature_ion + temperature_electron) + 2E0 * temperature_ion))) / dp_kpara / dp_B0 * dp_dB0_dz * dp_Phi_B / dp_B0**2E0   #[rad^2/s^2]
        C_deltaBpara = 1E0 / 2E0 * dp_Phi_B / dp_B0**2E0 * (dp_omega_tr**2E0)    #[rad^2/s^2]
        D_deltaBpara = np.zeros(array_size)    #[rad^2/s^2]
        E_deltaBpara = (1E0 / 2E0 * (dp_kpara * dp_v_perp*1E2)**2E0 + 2E0 * (dp_kpara * dp_v_para*1E2)**2E0 * (1E0 - temperature_ion / (dp_beta_ion * (temperature_ion + temperature_electron) + 2E0 * temperature_ion))) / dp_kpara / dp_B0 * dp_dB0_dz    #[rad^2/s^2]
        
        dp_preomega_mtr = (A_deltaBpara**2E0 + B_deltaBpara**2E0)**(1E0/4E0)    #[rad/s]
        dp_preomega_mtr2 = (C_deltaBpara**2E0 + D_deltaBpara**2E0)**(1E0/4E0)    #[rad/s]
        dp_delta_AB = np.arctan2(B_deltaBpara, A_deltaBpara)    #[rad]
        dp_delta_CD = np.arctan2(D_deltaBpara, C_deltaBpara)    #[rad]

        dp_omega_mtr = dp_preomega_mtr / np.sqrt(1E0 + dp_Phi_B / dp_B0**2E0 * np.cos(dp_wavephase))    #[rad/s]
        
        dp_omega_mtr2 = dp_preomega_mtr2 / dp_preomega_mtr  #[]

        dp_S = E_deltaBpara / dp_preomega_mtr**2E0  #[]

        #A_deltaBpara * np.sin(dp_wavephase) + B_deltaBpara * np.cos(dp_wavephase) + C_deltaBpara * np.sin(dp_wavephase) * np.cos(dp_wavephase) + D_deltaBpara * np.cos(dp_wavephase)**2E0 + E_deltaBpara = 0 の解を求める
        dp_wavephase_unstable = np.zeros(array_size)
        dp_wavephase_stable = np.zeros(array_size)
        for count_i in range(array_size):

            if (dp_S[count_i] != dp_S[count_i] or dp_omega_mtr[count_i] != dp_omega_mtr[count_i]):
                dp_wavephase_unstable[count_i] = np.nan
                dp_wavephase_stable[count_i] = np.nan
                continue

            coefficients = [
                - B_deltaBpara[count_i] + D_deltaBpara[count_i] + E_deltaBpara[count_i],
                2E0 * (A_deltaBpara[count_i] - C_deltaBpara[count_i]),
                2E0 * (- D_deltaBpara[count_i] + E_deltaBpara[count_i]),
                2E0 * (A_deltaBpara[count_i] + C_deltaBpara[count_i]),
                B_deltaBpara[count_i] + D_deltaBpara[count_i] + E_deltaBpara[count_i]
            ]
            roots = np.roots(coefficients)
            #実数解のみを抽出
            real_roots = []
            for root in roots:
                if np.isreal(root):
                    real_roots.append(root.real)
            #実数解をarctanで変換
            try:
                dp_wavephase_2ndrespoint = 2E0 * np.arctan(real_roots)    #[rad] unstableとstableの両方が存在する。stableはunstableよりも0に近い。
                #より0から離れているものをunstableとして選択
                dp_wavephase_unstable[count_i] = dp_wavephase_2ndrespoint[np.argmax(np.abs(dp_wavephase_2ndrespoint))]    #[rad]
                dp_wavephase_stable[count_i] = dp_wavephase_2ndrespoint[np.argmin(np.abs(dp_wavephase_2ndrespoint))]    #[rad]
            except:
                dp_wavephase_unstable[count_i] = np.nan
                dp_wavephase_stable[count_i] = np.nan

        dp_theta = (dp_kpara * dp_v_para*1E2 - dp_wavefreq) / dp_omega_mtr   #[]

        dp_theta_limit_plus = np.sqrt(1E0 / 2E0 * (dp_omega_tr / dp_omega_mtr)**2E0 * (np.cos(dp_wavephase) - np.cos(dp_wavephase_unstable)) \
                                    - 1E0 / 2E0 * (dp_kpara * dp_v_perp*1E2 / dp_omega_mtr)**2E0 * (np.log((dp_B0**2E0 + dp_Phi_B * np.cos(dp_wavephase)) / (dp_B0**2E0 + dp_Phi_B * np.cos(dp_wavephase_unstable))) \
                                        + (1E0 - dp_Phi_B / dp_B0**2E0)**(-1E0 / 2E0) / dp_kpara / dp_B0 * (dp_Phi_B / dp_B0**2E0 * dp_d_Phi_B_B0_dz - 1E0 / 2E0 * dp_dB0_dz) * (np.arctan(np.sqrt((dp_B0**2E0 - dp_Phi_B) / (dp_B0**2E0 + dp_Phi_B)) * np.tan(dp_wavephase / 2E0)) - np.arctan(np.sqrt((dp_B0**2E0 - dp_Phi_B) / (dp_B0**2E0 + dp_Phi_B)) * np.tan(dp_wavephase_unstable / 2E0)))) \
                                            - ((dp_kpara * dp_v_para*1E2 / dp_omega_mtr)**2E0 * (1E0 - temperature_ion / (dp_beta_ion * (temperature_ion + temperature_electron) + 2E0 * temperature_ion)) / dp_kpara / dp_B0 * dp_dB0_dz \
                                                + (dp_kpara * dp_v_perp*1E2 / dp_omega_mtr)**2E0 / dp_kpara / dp_B0 * dp_Phi_B / dp_B0 * dp_d_Phi_B_B0_dz) * (dp_wavephase - dp_wavephase_unstable))   #[]
        dp_theta_limit_minus = - dp_theta_limit_plus   #[]

        #波があるときのみ
        dp_omega_mtr = np.abs(wavecheck(array_size, dp_omega_mtr, dp_wavephase))    #[rad/s]
        dp_omega_mtr2 = np.abs(wavecheck(array_size, dp_omega_mtr2, dp_wavephase))    #[rad/s]
        dp_delta_AB = wavecheck(array_size, dp_delta_AB, dp_wavephase)    #[rad]
        dp_delta_CD = wavecheck(array_size, dp_delta_CD, dp_wavephase)    #[rad]
        dp_S = wavecheck(array_size, dp_S, dp_wavephase)    #[]
        dp_wavephase_stable = wavecheck(array_size, dp_wavephase_stable, dp_wavephase)    #[rad]
        dp_wavephase_unstable = wavecheck(array_size, dp_wavephase_unstable, dp_wavephase)    #[rad]
        dp_theta = wavecheck(array_size, dp_theta, dp_wavephase)    #[]
        dp_theta_limit_plus = wavecheck(array_size, dp_theta_limit_plus, dp_wavephase)    #[]
        dp_theta_limit_minus = wavecheck(array_size, dp_theta_limit_minus, dp_wavephase)    #[]

    dp_wavephase = wavecheck(array_size, dp_wavephase, dp_wavephase)    #[rad]
    dp_wavephase = normalize_angle_rad(dp_wavephase)    #[rad]

    #plot
    plt.rcParams["font.size"] = 30
    
    fig = plt.figure(figsize=(20, 30), dpi=100)
    fig.suptitle(str(wavekind) + r', initial energy = ' + str(int(dp_energy[0])) + r' [eV], pitch angle = ' + str(int(np.round(dp_pitchangle_eq[0]))) + r' [deg],' '\n' r'grad = ' + str(int(gradient_parameter)) + r', wavephase @ 0 deg = ' + str(int(initial_wavephase)) + r' [deg]')

    gs = fig.add_gridspec(8, 1)

    ax1 = fig.add_subplot(gs[0, 0], xlabel=r'time [s]', ylabel=r'energy [eV]')
    ax1.plot(dp_time, dp_energy, lw=4, color='blue')
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.set_ticks_position('top')
    ax1.minorticks_on()
    ax1.grid(which="both", alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, ylabel=r'pitch angle' '\n' r'[deg]')
    ax2.hlines(90, dp_time[0], dp_time[-1], color='k', lw=4, linestyles='dashed', alpha=0.5)
    ax2.plot(dp_time, dp_pitchangle, lw=4, color='blue')
    ax2.minorticks_on()
    ax2.grid(which="both", alpha=0.3)
    ax2.tick_params(labelbottom=False, bottom=True)

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1, ylabel=r'$\beta_{\mathrm{i}}$', yscale='log')
    ax3.plot(dp_time, dp_beta_ion, lw=4, color='blue')
    ax3.minorticks_on()
    ylim_ax3 = ax3.get_ylim()
    ax3.hlines(mass_electron/mass_ion, dp_time[0], dp_time[-1], color='dimgrey', lw=4, linestyles='dashed', alpha=0.5)
    ax3.set_ylim(ylim_ax3)
    ax3.grid(which="both", alpha=0.3)
    ax3.tick_params(labelbottom=False, bottom=True)

    #ax4 = fig.add_subplot(gs[3, 0], sharex=ax1, ylabel=r'$\frac{\theta + k_{\parallel} v_{\perp} \cos \phi}{2 \omega_{\mathrm{mtr}}}$')
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1, ylabel=r'$\frac{\theta}{2 \omega_{\mathrm{mtr}}}$')
    if (switch_delta_Bpara == 1E0):
        #ax4.plot(dp_time, dp_theta + dp_kpara * dp_v_perp*1E2 * np.cos(dp_u_phase), lw=4, color='blue', alpha=0.5, label=r'include $delta B_{\parallel}$')
        ax4.plot(dp_time, dp_theta, lw=4, color='blue', alpha=0.5, label=r'$\theta / 2 \omega_{mtr}$')
        ylim_ax4 = ax4.get_ylim()
        ax4.plot(dp_time, dp_theta_limit_plus, lw=4, color='green', linestyle='-.', alpha=0.5, label=r'$\delta B_{\parallel}$ limit')
        ax4.plot(dp_time, dp_theta_limit_minus, lw=4, color='green', linestyle='-.', alpha=0.5)
    #ax4.plot(dp_time, dp_theta_nondeltaB + dp_kpara * dp_v_perp*1E2 * np.cos(dp_u_phase), lw=4, color='red', alpha=0.5, label=r'not include $delta B_{\parallel}$')
    if (switch_delta_Bpara == 0E0):
        ax4.plot(dp_time, dp_theta_nondeltaB, lw=4, color='red', alpha=0.5, label=r'$\theta / 2 \omega_{tr}$')
        ylim_ax4 = ax4.get_ylim()
    ax4.plot(dp_time, dp_theta_limit_plus_nondeltaB, lw=4, color='orange', linestyle='-.', alpha=0.5, label=r'not $\delta B_{\parallel}$ limit')
    ax4.plot(dp_time, dp_theta_limit_minus_nondeltaB, lw=4, color='orange', linestyle='-.', alpha=0.5)
    ax4.set_ylim(ylim_ax4)
    ax4.minorticks_on()
    ax4.grid(which="both", alpha=0.3)
    ax4.tick_params(labelbottom=False, bottom=True)
    ax4.legend(loc='upper right')

    ax5 = fig.add_subplot(gs[4, 0], sharex=ax1, ylabel=r'trapping' '\n' r'frequency' '\n' r'[$\pi$ rad/s]')
    #x5.plot(dp_time, dp_omega_tr / np.pi, lw=4, color='green', label=r'$\omega_{\mathrm{tr}}$', alpha=0.5)
    if (switch_delta_Bpara == 1E0):
        ax5.plot(dp_time, dp_omega_mtr / np.pi, lw=4, color='blue', label=r'$\omega_{\mathrm{mtr}}$', alpha=0.5)
        ax5.plot(dp_time, dp_omega_mtr*dp_omega_mtr2 / np.pi, lw=4, color='orange', label=r'$\omega_{\mathrm{mtr}} \times \omega_{\mathrm{mtr2}}$', alpha=0.5)
    ax5.plot(dp_time, dp_omega_mtr_nondeltaB / np.pi, lw=4, color='red', label=r'$\omega^{*}_{\mathrm{mtr}}$', alpha=0.5)
    ax5.minorticks_on()
    ax5.grid(which="both", alpha=0.3)
    ax5.tick_params(labelbottom=False, bottom=True)
    ax5.legend(loc='upper right')

    ax6 = fig.add_subplot(gs[6, 0], sharex=ax1, ylabel=r'[$\pi$ rad]')
    ax6.plot(dp_time, dp_wavephase / np.pi, lw=4, color='green', label=r'$\psi$', alpha=0.5)
    if (switch_delta_Bpara == 1E0):
        ax6.plot(dp_time, dp_delta_AB / np.pi, lw=4, color='blue', label=r'$\delta_{\mathrm{mtr}}$', alpha=0.25)
        ax6.plot(dp_time, dp_delta_CD / np.pi, lw=4, color='orange', label=r'$\delta_{\mathrm{mtr2}}$', alpha=0.25)
    ax6.minorticks_on()
    ax6.grid(which="both", alpha=0.3)
    ax6.tick_params(labelbottom=False, bottom=True)
    ax6.legend(loc='upper right')

    ax7 = fig.add_subplot(gs[5, 0], sharex=ax1, ylabel=r'S')
    if (switch_delta_Bpara == 1E0):
        ax7.plot(dp_time, dp_S, lw=4, color='blue', label=r'$S$', alpha=0.5)
    ax7.plot(dp_time, dp_S_nondeltaB, lw=4, color='red', label=r'$S^{*}$', alpha=0.5)
    ylim_ax7 = ax7.get_ylim()
    #ylim_ax7が-1.5E0~1.5E0の範囲外になったら修正
    if (ylim_ax7[0] < -1.5E0):
        ylim_ax7 = (-1.5E0, ylim_ax7[1])
    if (ylim_ax7[1] > 1.5E0):
        ylim_ax7 = (ylim_ax7[0], 1.5E0)
    ax7.set_ylim(ylim_ax7)
    ax7.minorticks_on()
    ax7.grid(which="both", alpha=0.3)
    ax7.tick_params(labelbottom=False, bottom=True)
    ax7.legend(loc='upper right')

    ax9 = fig.add_subplot(gs[7, 0], sharex=ax1, xlabel=r'time [s]', ylabel=r'[$\pi$ rad]')
    if (switch_delta_Bpara == 1E0):
        ax9.scatter(dp_time, dp_wavephase_stable / np.pi, color='blue', label=r'$\psi_{\mathrm{stable}/\mathrm{unstable}}$', alpha=0.1)
        ax9.scatter(dp_time, dp_wavephase_unstable / np.pi, color='blue', alpha=0.1)
    ax9.plot(dp_time, dp_wavephase_stable_nondeltaB / np.pi, color='red', label=r'$\psi^{*}_{\mathrm{stable}/\mathrm{unstable}}$', alpha=0.5, lw=4, linestyle='-.')
    ax9.plot(dp_time, dp_wavephase_unstable_nondeltaB / np.pi, color='red', alpha=0.5, linestyle='-.', lw=4)
    ylim_ax9 = ax9.get_ylim()
    ax9.plot(dp_time, dp_wavephase / np.pi, lw=4, color='green', label=r'$\psi$', alpha=0.3)
    ax9.set_ylim(ylim_ax9)
    ax9.minorticks_on()
    ax9.grid(which="both", alpha=0.3)
    ax9.legend(loc='upper right')

    fig.subplots_adjust(hspace=0, top=0.91, bottom=0.04, left=0.10, right=0.97)

    fig.savefig(f'{dir_name}/result_pendulum_theory_time_variation/particle_trajectory{particle_file_number}.png')

if ((channel == 22 or channel == 23 or channel == 24) and switch_delta_Epara == 1E0):

    array_size = len(dp_z_position)
    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)

    dp_B0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]

    dp_Alfven_speed = dp_B0 / np.sqrt(4E0 * np.pi * mass_ion * number_density_ion)  #[cm/s]
    dp_beta_ion = pressure_ion / (dp_B0**2E0 / 2E0 / 4E0 / np.pi)    #[]
    dp_kpara_1 = + wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    dp_kpara_2 = - wave_frequency / kperp_rhoi / dp_Alfven_speed * (dp_beta_ion + 2E0 * temperature_ion / (temperature_ion + temperature_electron))     #[rad/cm]
    
    ion_Larmor_radius = (speed_of_light*1E2) * np.sqrt(2E0*mass_ion*pressure_ion/number_density_ion) / (elementary_charge/1E1*speed_of_light*1E2) / dp_B0   #[cm]

    dp_kperp_1 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]
    dp_kperp_2 = kperp_rhoi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]

    dp_dB0_dz = 3E0 * np.sin(dp_mlat_rad) * (5E0 * np.sin(dp_mlat_rad)**2E0 + 3E0) / np.cos(dp_mlat_rad)**8E0 / (3E0 * np.sin(dp_mlat_rad)**2E0 + 1E0) / (r_eq*1E2) * (B0_eq*1E4)   #[G/cm]

    dp_g_function_1 = 5E-1 * (np.tanh(+ gradient_parameter * (dp_mlat_rad*rad2deg - wave_threshold/2E0)) + 1E0)
    dp_g_function_2 = 5E-1 * (np.tanh(- gradient_parameter * (dp_mlat_rad*rad2deg + wave_threshold/2E0)) + 1E0)

    dp_dg_dz_1 = + 90E0 * gradient_parameter / np.pi / np.cosh(+ gradient_parameter * (dp_mlat_rad*rad2deg - wave_threshold/2E0))**2E0 / (r_eq*1E2) / np.cos(dp_mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0)
    dp_dg_dz_2 = - 90E0 * gradient_parameter / np.pi / np.cosh(- gradient_parameter * (dp_mlat_rad*rad2deg + wave_threshold/2E0))**2E0 / (r_eq*1E2) / np.cos(dp_mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0)

    electron_cyclotron_freq = (elementary_charge/1E1*speed_of_light*1E2) * dp_B0 / mass_electron / (speed_of_light*1E2)   #[rad/s]
    dp_wavephase_1 = dp_wavephase_1 - dp_kperp_1 * dp_v_perp*1E2 / electron_cyclotron_freq * np.sin(dp_u_phase)    #[rad]
    dp_wavephase_2 = dp_wavephase_2 - dp_kperp_2 * dp_v_perp*1E2 / electron_cyclotron_freq * np.sin(dp_u_phase)    #[rad]

    dp_h_function_1, dp_dh_dz_1 = make_h_function(array_size, dp_wavephase_1, dp_kpara_1)
    dp_h_function_2, dp_dh_dz_2 = make_h_function(array_size, dp_wavephase_2, dp_kpara_2)

    dp_Phi = (2E0 + temperature_electron / temperature_ion) * (wave_scalar_potential*1E8/(speed_of_light*1E2)) * (dp_g_function_1 * dp_h_function_1 + dp_g_function_2 * dp_h_function_2)   #[statV]=[erg/statC]

    dp_Phi_B = 4E0 * np.pi * number_density_ion * (elementary_charge/1E1*speed_of_light*1E2) * (1E0 - temperature_ion / (2E0 * temperature_ion + temperature_electron)) * dp_Phi * switch_delta_Bpara    #[erg/cm3]=[dyn/cm^2]=[G^2]

    dp_d_Phi_B_dz = 4E0 * np.pi * number_density_ion * (elementary_charge * wave_scalar_potential * 1E7) * (1E0 + temperature_electron / temperature_ion) \
        * (dp_dg_dz_1 * dp_h_function_1 + dp_g_function_1 * dp_dh_dz_1 + dp_dg_dz_2 * dp_h_function_2 + dp_g_function_2 * dp_dh_dz_2) * switch_delta_Bpara   #[G^2/cm]

    dp_d_Phi_B_B0_dz = dp_d_Phi_B_dz / dp_B0 - dp_dB0_dz * dp_Phi_B / dp_B0**2E0    #[G/cm]

    dp_pitchangle = np.arctan(dp_v_perp/dp_v_para) * rad2deg    #[deg]
    dp_pitchangle = np.mod(dp_pitchangle, 180.)
    
    step_function_1 = np.zeros(array_size)
    step_function_2 = np.zeros(array_size)
    for count_i in range(array_size):
        step_function_1[count_i] = step(+ dp_z_position[count_i])
        step_function_2[count_i] = step(- dp_z_position[count_i])
    
    dp_kpara = (dp_kpara_1 * step_function_1 + dp_kpara_2 * step_function_2) #[rad/cm]
    dp_kperp = (dp_kperp_1 * step_function_1 + dp_kperp_2 * step_function_2) #[rad/cm]
    dp_wavephase = (dp_wavephase_1 * step_function_1 + dp_wavephase_2 * step_function_2) #[rad]

    dp_delta_Bpara = dp_Phi_B / dp_B0 * np.cos(dp_wavephase)    #[G]
    dp_mu = mass_electron * (dp_v_perp*1E2)**2E0 / 2E0 / (dp_B0 + dp_delta_Bpara)   #[erg/G]=[statC/cm^2]=[cm^2/s^2]

    dp_omega_tr = np.abs(dp_kpara) * np.sqrt((elementary_charge/1E1*speed_of_light*1E2) * dp_Phi / mass_electron / dp_gamma * (1E0 - (dp_v_para / speed_of_light)**2E0))    #[rad/s]
    dp_S = 1E0 / dp_kpara / ((elementary_charge/1E1*speed_of_light*1E2) * dp_Phi / mass_electron / dp_gamma * (1E0 - (dp_v_para / speed_of_light)**2E0)) \
        * (dp_mu/mass_electron + (1E0 + dp_beta_ion * (temperature_ion + temperature_electron) / (dp_beta_ion * (temperature_ion + temperature_electron) + 2E0 * temperature_ion)) * (dp_v_para*1E2)**2E0 / dp_B0) * dp_dB0_dz    #[]

    if (switch_delta_Bpara == 1):
        dp_Gamma_pre = speed_of_light**2E0 / (speed_of_light**2E0 - dp_v_para**2E0)
        dp_Gamma_tr = np.sqrt((1E0 - dp_mu / (elementary_charge/1E1*speed_of_light*1E2) / dp_Phi * dp_Phi_B / dp_B0 * dp_Gamma_pre * dp_gamma)**2E0 + 1E0 / dp_kpara**2E0 * (dp_mu / (elementary_charge/1E1*speed_of_light*1E2) / dp_Phi * dp_Gamma_pre * dp_gamma * dp_d_Phi_B_B0_dz)**2E0)    #[]
        dp_omega_tr = dp_omega_tr * np.sqrt(dp_Gamma_tr)    #[rad/s]
        dp_S = dp_S / dp_Gamma_tr    #[]

    dp_theta = dp_kpara * dp_v_para*1E2 - wave_frequency    #[rad/s]
    dp_theta = wavecheck(array_size, dp_theta, dp_wavephase)    #[rad/s]

    #dp_wavephaseの範囲を-pi~piにする
    dp_wavephase_mod = normalize_angle_rad(dp_wavephase)    #[rad]
    dp_wavephase_mod = wavecheck(array_size, dp_wavephase_mod, dp_wavephase)    #[rad]

    if (channel == 22):
        fig = plt.figure(figsize=(14, 14), dpi=100, tight_layout=True)
        ax = fig.add_subplot(111, xlabel=r'wave phase $\psi$ [$\times \pi$ rad]')
        if (switch_delta_Bpara == 0E0):
            ax.set_ylabel(r'$\frac{\theta}{2\omega_{\mathrm{tr}}}$')
        elif (switch_delta_Bpara == 1E0):
            ax.set_ylabel(r'$\frac{\theta}{2\omega_{\mathrm{mtr}}}$')
        mappable = ax.scatter(dp_wavephase_mod/np.pi, dp_theta/2E0/dp_omega_tr, c=dp_time, cmap='turbo', marker='.', lw=0)
        fig.colorbar(mappable=mappable, ax=ax, label=r'time [s]')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-3, 3)
        ax.minorticks_on()
        ax.grid(which="both", alpha=0.3)

    if (channel == 23):

        dp_pitchangle = np.arctan(dp_v_perp/dp_v_para) * rad2deg    #[deg]
        dp_pitchangle = np.mod(dp_pitchangle, 180.)

        if (switch_delta_Bpara == 1E0):
            dp_cos_delta_m = dp_kpara**2E0 / dp_omega_tr**2E0 * ((elementary_charge/1E1*speed_of_light*1E2) * dp_Phi / mass_electron / dp_gamma * (1E0 - (dp_v_para / speed_of_light)**2E0) - dp_mu / mass_electron * dp_Phi_B / dp_B0)    #[]
            dp_sin_delta_m = dp_kpara / dp_omega_tr**2E0 * dp_mu / mass_electron * dp_d_Phi_B_B0_dz    #[]
            dp_delta_m = np.arctan2(dp_sin_delta_m, dp_cos_delta_m)    #[rad]
            dp_wavephase_stable = np.arcsin(- dp_S) - dp_delta_m    #[rad]
            dp_wavephase_unstable = - np.pi - np.arcsin(- dp_S) - dp_delta_m    #[rad]
            dp_wavephase_unstable = normalize_angle_rad(dp_wavephase_unstable)    #[rad]

            dp_wavephase_stable_nondeltaB = np.arcsin(- dp_S * dp_Gamma_tr)    #[rad]
            dp_wavephase_unstable_nondeltaB = - np.pi - np.arcsin(- dp_S * dp_Gamma_tr)    #[rad]

        if (switch_delta_Bpara == 0E0):
            dp_wavephase_stable = np.arcsin(- dp_S)
            dp_wavephase_unstable = - np.pi - np.arcsin(- dp_S)

        fig = plt.figure(figsize=(20, 30), dpi=100)
        fig.suptitle(str(wavekind) + r', initial energy = ' + str(int(dp_energy[0])) + r' [eV], pitch angle = ' + str(int(np.round(dp_pitchangle_eq[0]))) + r' [deg],' '\n' r'grad = ' + str(int(gradient_parameter)) + r', wavephase @ 0 deg = ' + str(int(initial_wavephase)) + r' [deg]')
        gs = fig.add_gridspec(8, 1)

        ax1 = fig.add_subplot(gs[0, 0], xlabel=r'time [s]', ylabel=r'energy [eV]')
        ax1.plot(dp_time, dp_energy, lw=4, color='blue')
        ax1.xaxis.set_label_position('top')
        ax1.xaxis.set_ticks_position('top')
        ax1.minorticks_on()
        ax1.grid(which="both", alpha=0.3)

        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, ylabel=r'pitch angle' '\n' r'[deg]')
        ax2.hlines(90, dp_time[0], dp_time[-1], color='k', lw=4, linestyles='dashed', alpha=0.5)
        ax2.plot(dp_time, dp_pitchangle, lw=4, color='blue')
        ax2.minorticks_on()
        ax2.grid(which="both", alpha=0.3)

        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1, ylabel=r'$\beta_{\mathrm{i}}$', yscale='log')
        ax3.plot(dp_time, dp_beta_ion, lw=4, color='blue')
        ax3.minorticks_on()
        ylim_ax3 = ax3.get_ylim()
        ax3.hlines(mass_electron/mass_ion, dp_time[0], dp_time[-1], color='dimgrey', lw=4, linestyles='dashed', alpha=0.5)
        ax3.set_ylim(ylim_ax3)
        ax3.grid(which="both", alpha=0.3)
        ax3.tick_params(labelbottom=False, bottom=True)

        ax4 = fig.add_subplot(gs[3, 0], sharex=ax1, ylabel=r'$\frac{\theta}{2 \omega_{\mathrm{tr}}}$')
        if (switch_delta_Bpara == 1E0):
            ax4.plot(dp_time, dp_theta / 2E0 / dp_omega_tr, lw=4, color='blue', alpha=0.5, label=r'$\theta / 2 \omega_{\mathrm{mtr}}$')
            ax4.plot(dp_time, dp_theta / 2E0 / dp_omega_tr * np.sqrt(dp_Gamma_tr), lw=4, color='orange', alpha=0.5, label=r'$\theta / 2 \omega_{\mathrm{tr}}$')
        elif (switch_delta_Bpara == 0E0):
            ax4.plot(dp_time, dp_theta / 2E0 / dp_omega_tr, lw=4, color='blue', alpha=0.5, label=r'$\theta / 2 \omega_{\mathrm{tr}}$')
        ax4.set_ylim(-3E0, 3E0)
        ax4.minorticks_on()
        ax4.grid(which="both", alpha=0.3)
        ax4.tick_params(labelbottom=False, bottom=True)
        ax4.legend(loc='upper right')

        ax5 = fig.add_subplot(gs[4, 0], sharex=ax1, ylabel=r'trapping' '\n' r'frequency' '\n' r'[$\pi$ rad/s]')
        if (switch_delta_Bpara == 1E0):
            ax5.plot(dp_time, dp_omega_tr / np.pi, lw=4, color='blue', alpha=0.5, label=r'$\omega_{\mathrm{mtr}}$')
            ax5.plot(dp_time, dp_omega_tr / np.sqrt(dp_Gamma_tr) / np.pi, lw=4, color='orange', alpha=0.5, label=r'$\omega_{\mathrm{tr}}$')
        elif (switch_delta_Bpara == 0E0):
            ax5.plot(dp_time, dp_omega_tr / np.pi, lw=4, color='blue', alpha=0.5, label=r'$\omega_{\mathrm{tr}}$')
        ax5.minorticks_on()
        ax5.grid(which="both", alpha=0.3)
        ax5.tick_params(labelbottom=False, bottom=True)
        ax5.legend(loc='upper right')

        ax6 = fig.add_subplot(gs[5, 0], sharex=ax1, ylabel=r'S')
        if (switch_delta_Bpara == 1E0):
            ax6.plot(dp_time, dp_S, lw=4, color='blue', alpha=0.5, label=r'$S_{\mathrm{m}}$')
            ax6.plot(dp_time, dp_S * dp_Gamma_tr, lw=4, color='orange', alpha=0.5, label=r'$S$')
        elif (switch_delta_Bpara == 0E0):
            ax6.plot(dp_time, dp_S, lw=4, color='blue', alpha=0.5, label=r'$S$')
        ax6.set_ylim(0E0, 2E0)
        ax6.minorticks_on()
        ax6.grid(which="both", alpha=0.3)
        ax6.tick_params(labelbottom=False, bottom=True)
        ax6.legend(loc='upper right')

        ax7 = fig.add_subplot(gs[6, 0], sharex=ax1, ylabel=r'[$\pi$ rad]')
        ax7.plot(dp_time, dp_wavephase_mod / np.pi, lw=4, color='green', label=r'$\psi$', alpha=0.5)
        if (switch_delta_Bpara == 1E0):
            ax7.plot(dp_time, dp_delta_m / np.pi, lw=4, color='blue', label=r'$\delta_{\mathrm{m}}$', alpha=0.25)
        ax7.minorticks_on()
        ax7.grid(which="both", alpha=0.3)
        ax7.tick_params(labelbottom=False, bottom=True)
        ax7.legend(loc='upper right')

        ax8 = fig.add_subplot(gs[7, 0], sharex=ax1, xlabel=r'time [s]', ylabel=r'[$\pi$ rad]')
        if (switch_delta_Bpara == 1E0):
            ax8.scatter(dp_time, dp_wavephase_stable / np.pi, color='blue', label=r'$\psi_{\mathrm{m} \, \mathrm{stable}/\mathrm{unstable}}$', alpha=0.1)
            ax8.scatter(dp_time, dp_wavephase_unstable / np.pi, color='blue', alpha=0.1)
            ax8.scatter(dp_time, dp_wavephase_stable_nondeltaB / np.pi, color='orange', label=r'$\psi_{\mathrm{stable}/\mathrm{unstable}}$', alpha=0.1)
            ax8.scatter(dp_time, dp_wavephase_unstable_nondeltaB / np.pi, color='orange', alpha=0.1)
        elif (switch_delta_Bpara == 0E0):
            ax8.scatter(dp_time, dp_wavephase_stable / np.pi, color='blue', label=r'$\psi_{\mathrm{stable}/\mathrm{unstable}}$', alpha=0.1)
            ax8.scatter(dp_time, dp_wavephase_unstable / np.pi, color='blue', alpha=0.1)
        ylim_ax8 = ax8.get_ylim()
        ax8.plot(dp_time, dp_wavephase_mod / np.pi, lw=4, color='green', label=r'$\psi$', alpha=0.3)
        ax8.set_ylim(ylim_ax8)
        ax8.minorticks_on()
        ax8.grid(which="both", alpha=0.3)
        ax8.legend(loc='upper right')

        fig.subplots_adjust(hspace=0, top=0.91, bottom=0.04, left=0.10, right=0.97)

    
    if (channel == 24 and switch_delta_Bpara == 1E0):

        dp_pitchangle = np.arctan(dp_v_perp/dp_v_para) * rad2deg    #[deg]
        dp_pitchangle = np.mod(dp_pitchangle, 180.)

        if (switch_delta_Bpara == 1E0):
            dp_cos_delta_m = dp_kpara**2E0 / dp_omega_tr**2E0 * ((elementary_charge/1E1*speed_of_light*1E2) * dp_Phi / mass_electron / dp_gamma * (1E0 - (dp_v_para / speed_of_light)**2E0) - dp_mu / mass_electron * dp_Phi_B / dp_B0)    #[]
            dp_sin_delta_m = dp_kpara / dp_omega_tr**2E0 * dp_mu / mass_electron * dp_d_Phi_B_B0_dz    #[]
            dp_delta_m = np.arctan2(dp_sin_delta_m, dp_cos_delta_m)    #[rad]
            dp_wavephase_stable = np.arcsin(- dp_S) - dp_delta_m    #[rad]
            dp_wavephase_unstable = - np.pi - np.arcsin(- dp_S) - dp_delta_m    #[rad]
            dp_wavephase_unstable = normalize_angle_rad(dp_wavephase_unstable)    #[rad]

            dp_wavephase_stable_nondeltaB = np.arcsin(- dp_S * dp_Gamma_tr)    #[rad]
            dp_wavephase_unstable_nondeltaB = - np.pi - np.arcsin(- dp_S * dp_Gamma_tr)    #[rad]

        if (switch_delta_Bpara == 0E0):
            dp_wavephase_stable = np.arcsin(- dp_S)
            dp_wavephase_unstable = - np.pi - np.arcsin(- dp_S)

        fig = plt.figure(figsize=(20, 30), dpi=100)
        fig.suptitle(str(wavekind) + r', initial energy = ' + str(int(dp_energy[0])) + r' [eV], pitch angle = ' + str(int(np.round(dp_pitchangle_eq[0]))) + r' [deg],' '\n' r'grad = ' + str(int(gradient_parameter)) + r', wavephase @ 0 deg = ' + str(int(initial_wavephase)) + r' [deg]')
        
        ax1 = fig.add_subplot(211, xlabel=r'time [s]', ylabel=r'$\delta_{\mathrm{m}}$ [$\pi$ rad]')
        ax1.plot(dp_time, dp_delta_m / np.pi, lw=4, color='blue', alpha=0.5, label=r'$\delta_{\mathrm{m}}$')
        ax1.minorticks_on()
        ax1.grid(which="both", alpha=0.3)

        ax2 = fig.add_subplot(212, ylabel=r'$(\omega_{\mathrm{mtr}} / \omega_{\mathrm{tr}})^{2} = S / S_{\mathrm{m}}$')
        ax2.plot(dp_time, dp_Gamma_tr, lw=4, color='blue', alpha=0.5)
        ax2.minorticks_on()
        ax2.grid(which="both", alpha=0.3)

        fig.subplots_adjust()






plt.show()