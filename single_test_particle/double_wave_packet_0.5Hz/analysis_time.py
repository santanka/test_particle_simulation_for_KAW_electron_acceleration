import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count
import datetime

wave_scalar_potential   = 600E0     #[V]
initial_wavephase_list  = [0E0, 90E0, 180E0, 270E0]       #[deg]
gradient_parameter_list = [1E0, 2E0, 4E0]       #[]
wave_threshold          = 5E0       #[deg]

wavekind_list           = [r'EparaBpara', r'Epara', r'Bpara']
switch_delta_Epara_list = [1E0, 1E0, 0E0]
switch_delta_Eperp_perp = 0E0
switch_delta_Eperp_phi  = 0E0
switch_delta_Bpara_list = [1E0, 0E0, 1E0]
switch_delta_Bperp      = 0E0

switch_wave_packet = 1E0

data_limit_under        = 0
data_limit_upper        = 200000

initial_particle_number = 180
initial_particle_number_divide = 5

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

speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]
mass_electron   = 9.10938356E-28    #[g]
mass_ion        = 1.672621898E-24   #[g]
pressure_ion        = number_density_ion * temperature_ion * elementary_charge * 1E7    #cgs
pressure_electron   = number_density_ion * temperature_electron * elementary_charge * 1E7   #cgs

wavephaselist_number = len(initial_wavephase_list)
wavekindlist_number = len(wavekind_list)
gradientparameter_number = len(gradient_parameter_list)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 30

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
    return

#z[m] -> mlat[rad]
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


def main(count_grad, count_angle, count_kind, particle_file_number):
    
    wavekind = wavekind_list[count_kind]
    switch_delta_Epara = switch_delta_Epara_list[count_kind]
    switch_delta_Bpara = switch_delta_Bpara_list[count_kind]
    initial_wavephase = initial_wavephase_list[count_angle]
    gradient_parameter = gradient_parameter_list[count_grad]

    dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave_packet/results_particle_{str(int(wave_scalar_potential))}V' \
        + f'_gradient_{int(gradient_parameter)}_threshold_{int(wave_threshold)}_wavephase_{int(initial_wavephase)}_{wavekind}'
    file_name_particle  = f'{dir_name}/myrank000/particle_trajectory{particle_file_number}.dat'
    if (check_file_exists(file_name_particle) == False):
        return
    #file_name_wave      = f'{dir_name}/myrank000/potential_prof.dat'

    mkdir(f'{dir_name}/result_time_variation_particle')
    if (check_file_exists(f'{dir_name}/result_time_variation_particle/particle_trajectory{particle_file_number}.png') == True):
        return

    data_particle   = np.genfromtxt(file_name_particle, unpack=True)
    data_particle   = data_particle[:, data_limit_under:data_limit_upper]

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

    dp_mlat_rad = z_position_m_to_mlat_rad(dp_z_position)
    dp_mlat_deg = dp_mlat_rad * rad2deg

    dp_pitchangle = np.arctan(dp_v_perp/dp_v_para) * rad2deg    #[deg]
    dp_pitchangle = np.mod(dp_pitchangle, 180.)
    
    array_size = len(dp_z_position)

    dp_B0 = B0_eq / np.cos(dp_mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0) * 1E4     #[G]

    ion_Larmor_radius = (speed_of_light*1E2) * np.sqrt(2E0*mass_ion*pressure_ion/number_density_ion) / (elementary_charge/1E1*speed_of_light*1E2) / dp_B0   #[cm]

    dp_wavefreq = 2E0 * np.pi / 2E0 #[rad/s]
    dp_kpara_1 = + np.sqrt(2E0 * np.pi * number_density_ion * mass_ion * pressure_ion) / dp_B0**2E0 * np.sqrt(4E0 * np.pi + dp_B0**2E0 / (pressure_ion + pressure_electron))    #[rad/cm]
    dp_kpara_2 = - np.sqrt(2E0 * np.pi * number_density_ion * mass_ion * pressure_ion) / dp_B0**2E0 * np.sqrt(4E0 * np.pi + dp_B0**2E0 / (pressure_ion + pressure_electron))    #[rad/cm]
    dp_kperp_1 = 2E0 * np.pi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]
    dp_kperp_2 = 2E0 * np.pi / ion_Larmor_radius * np.ones(array_size) #[rad/cm]

    dp_phasespeed = get_major_wave_component(dp_z_position, dp_wavefreq/dp_kpara_1, dp_wavefreq/dp_kpara_2) * 1E-2

    dp_theta = dp_v_para / dp_phasespeed - 1E0

    dp_wavephase_major = get_major_wave_component(dp_z_position, dp_wavephase_1, dp_wavephase_2)
    dp_wavephase_major = np.mod(dp_wavephase_major+np.pi, 2E0*np.pi) - np.pi

    dp_dB0_dz = 3E0 * np.sin(dp_mlat_rad) * (5E0 * np.sin(dp_mlat_rad)**2E0 + 3E0) / np.cos(dp_mlat_rad)**8E0 / (3E0 * np.sin(dp_mlat_rad)**2E0 + 1E0) / (r_eq*1E2) * (B0_eq*1E4)   #[G/cm]
    Alpha = 4E0 * np.pi * (1E0 + pressure_electron / pressure_ion) * (elementary_charge/1E1*speed_of_light*1E2) * number_density_ion * (wave_scalar_potential*1E8/(speed_of_light*1E2))

    g_function_1 = 5E-1 * (np.tanh(+ gradient_parameter * (dp_mlat_rad*rad2deg - wave_threshold/2E0)) + 1E0)
    g_function_2 = 5E-1 * (np.tanh(- gradient_parameter * (dp_mlat_rad*rad2deg + wave_threshold/2E0)) + 1E0)

    dg_dz_1 = + 90E0 * gradient_parameter / np.pi / np.cosh(+ gradient_parameter * (dp_mlat_rad*rad2deg - wave_threshold/2E0))**2E0 / (r_eq*1E2) / np.cos(dp_mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0)
    dg_dz_2 = - 90E0 * gradient_parameter / np.pi / np.cosh(- gradient_parameter * (dp_mlat_rad*rad2deg + wave_threshold/2E0))**2E0 / (r_eq*1E2) / np.cos(dp_mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(dp_mlat_rad)**2E0)

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

    F_mirror_background = - mass_electron * (dp_u_perp*1E2)**2E0 / 2E0 / (dp_B0 + dp_deltaBpara_sum) / dp_gamma * dp_dB0_dz * 1E-5  #[N]
    F_mirror_wave_B0    = - mass_electron * (dp_u_perp*1E2)**2E0 / 2E0 / (dp_B0 + dp_deltaBpara_sum) / dp_gamma * B0_function_sum * 1E-5  #[N]
    F_mirror_wave_kpara = - mass_electron * (dp_u_perp*1E2)**2E0 / 2E0 / (dp_B0 + dp_deltaBpara_sum) / dp_gamma * kpara_function_sum * 1E-5  #[N]
    F_mirror_wave_dg_dz = - mass_electron * (dp_u_perp*1E2)**2E0 / 2E0 / (dp_B0 + dp_deltaBpara_sum) / dp_gamma * dg_dz_function_sum * 1E-5  #[N]
    F_mirror_wave_dh_dz = - mass_electron * (dp_u_perp*1E2)**2E0 / 2E0 / (dp_B0 + dp_deltaBpara_sum) / dp_gamma * dh_dz_function_sum * 1E-5  #[N]
    F_electric          = - (elementary_charge/1E1*speed_of_light*1E2) * dp_deltaEpara_sum * 1E-5   #[N]

    fig = plt.figure(figsize=(25, 45), dpi=100)
    fig.suptitle(str(wavekind) + r', initial energy = ' + str(int(dp_energy[0])) + r' [eV], pitch angle = ' + str(int(np.round(dp_pitchangle[0]))) + r' [deg], grad = ' + str(int(gradient_parameter)) + r', wavephase @ 0 deg = ' + str(int(initial_wavephase)) + r' [deg]')

    gs = fig.add_gridspec(10, 1)

    ax1 = fig.add_subplot(gs[0, 0], ylabel=r'MLAT [rad]', xlabel=r'time [s]')
    ax1.set_title(r'(a)', x=-0.075, y=0.95)
    ax1.plot(dp_time, dp_mlat_deg)
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.set_ticks_position('top')
    ax1.minorticks_on()
    ax1.grid(which='both', alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0], ylabel=r'Energy [eV]', sharex=ax1)
    ax2.set_title(r'(b)', x=-0.075, y=0.95)
    ax2.plot(dp_time, dp_energy)
    ax2.minorticks_on()
    ax2.grid(which='both', alpha=0.3)
    ax2.tick_params(labelbottom=False, bottom=True)

    ax3 = fig.add_subplot(gs[2, 0], ylabel=r'$v_{\parallel}$ [$\times$c]', sharex=ax1)
    ax3.set_title(r'(c)', x=-0.075, y=0.95)
    ax3.plot(dp_time, dp_v_para/speed_of_light)
    ax3.minorticks_on()
    ax3.grid(which='both', alpha=0.3)
    ax3.tick_params(labelbottom=False, bottom=True)

    ax4 = fig.add_subplot(gs[3, 0], ylabel=r'$v_{\perp}$ [$\times$c]', sharex=ax1)
    ax4.set_title(r'(d)', x=-0.075, y=0.95)
    ax4.plot(dp_time, dp_v_perp/speed_of_light)
    ax4.minorticks_on()
    ax4.grid(which='both', alpha=0.3)
    ax4.tick_params(labelbottom=False, bottom=True)

    ax5 = fig.add_subplot(gs[4, 0], ylabel=r'pitch angle [deg]', sharex=ax1)
    ax5.set_title(r'(e)', x=-0.075, y=0.95)
    ax5.plot(dp_time, dp_pitchangle)
    ax5.minorticks_on()
    ax5.grid(which='both', alpha=0.3)
    ax5.tick_params(labelbottom=False, bottom=True)

    ax6 = fig.add_subplot(gs[5, 0], ylabel=r'pitch angle (eq) [deg]', sharex=ax1)
    ax6.set_title(r'(f)', x=-0.075, y=0.95)
    ax6.plot(dp_time, dp_pitchangle_eq)
    ax6.minorticks_on()
    ax6.grid(which='both', alpha=0.3)
    ax6.tick_params(labelbottom=False, bottom=True)

    ax7 = fig.add_subplot(gs[6, 0], ylabel=r'$v_{\parallel}/V_{\mathrm{R} \parallel}-1$', sharex=ax1)
    ax7.set_title(r'(g)', x=-0.075, y=0.95)
    ax7.plot(dp_time, dp_theta)
    ax7.minorticks_on()
    ax7.grid(which='both', alpha=0.3)
    ax7.tick_params(labelbottom=False, bottom=True)

    ax8 = fig.add_subplot(gs[7, 0], ylabel=r'wave phase [$\times \pi$ rad]', sharex=ax1)
    ax8.set_title(r'(h)', x=-0.075, y=0.95)
    ax8.plot(dp_time, dp_wavephase_major/np.pi)
    ax8.minorticks_on()
    ax8.grid(which='both', alpha=0.3)
    ax8.tick_params(labelbottom=False, bottom=True)

    ax9 = fig.add_subplot(gs[8:, 0], ylabel=r'Force [$\times 10^{-22}$ N]', xlabel=r'time [s]')
    ax9.set_title(r'(i)', x=-0.075, y=0.95)
    ax9.plot(dp_time, F_mirror_background*1E22, color='purple', alpha=0.5, label=r'$F_{B_0}$', lw=4)
    if (switch_delta_Bpara == 1E0):
        ax9.plot(dp_time, F_mirror_wave_B0*1E22, color='red', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (B_0)$', lw=4)
        ax9.plot(dp_time, F_mirror_wave_kpara*1E22, color='magenta', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (k_{\parallel})$', lw=4)
        ax9.plot(dp_time, F_mirror_wave_dg_dz*1E22, color='orange', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (g)$', lw=4)
        if (switch_wave_packet == 1E0):
            ax9.plot(dp_time, F_mirror_wave_dh_dz*1E22, color='green', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (h)$', lw=4)
    if (switch_delta_Epara == 1E0):
        ax9.plot(dp_time, F_electric*1E22, color='b', alpha=0.5, label=r'$F_{\delta E_{\parallel}}$', lw=4)
    ax9.minorticks_on()
    ax9.grid(which="both", alpha=0.3)
    ax9.legend(loc=(1.002, 0.25))

    fig.subplots_adjust(top=0.94, bottom=0.025, left=0.08, right=0.87, hspace=0)
    fig.savefig(f'{dir_name}/result_time_variation_particle/particle_trajectory{particle_file_number}.png')
    plt.close()



def main_loop(args):
    count_grad, count_angle, count_kind, count_i = args
    count_file = int(np.floor(count_i/initial_particle_number_divide))
    particle_file_number = f'{str(count_file).zfill(2)}-{str(count_i+1).zfill(3)}'
    now = str(datetime.datetime.now())
    print(r'analysis time: gradient: ' + str(count_grad) + r', phase: ' + str(count_angle) + r', kind: ' + str(count_kind) + r', file number: ' + str(particle_file_number) + r'   ' + now)
    main(count_grad=count_grad, count_angle=count_angle, count_kind=count_kind, particle_file_number=particle_file_number)

#main_loop([0, 0, 101])

#並列処理
if __name__ == '__main__':
    # プロセス数
    num_processes = cpu_count()-1
    if (num_processes > 20):
        num_processes = 20
    print(r'num_processes: ' + str(num_processes))

    # 非同期処理の指定
    with Pool(processes=num_processes) as pool:
        results = []
        for count_grad in range(gradientparameter_number):
            for count_angle in range(wavephaselist_number):
                for count_kind in range(wavekindlist_number):
                    for count_i in range(initial_particle_number):
                        result = pool.apply_async(main_loop, [(count_grad, count_angle, count_kind, count_i)])
                        results.append(result)
        # 全ての非同期処理の終了を待機
        for result in results:
            result.get()
        print(r'finish')
        quit()
