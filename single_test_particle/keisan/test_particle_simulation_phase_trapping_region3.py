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

diff_rad = 1E-6 #[rad]


# wave parameters
kperp_rhoi = 2E0 * np.pi    #[rad]
wave_frequency = 2E0 * np.pi * 0.15    #[rad/s]

def wave_phase_speed(mlat_rad):
    return Alfven_speed(mlat_rad) * kperp_rhoi * np.sqrt((1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))) * np.sign(mlat_rad)    #[m/s]

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
    grad_magnetic_flux_density = (magnetic_flux_density(mlat_rad + diff_rad) - magnetic_flux_density(mlat_rad - diff_rad)) / 2E0 / diff_rad * d_mlat_d_z(mlat_rad)    #[T/m]
    return 1E0 / kpara(mlat_rad) / magnetic_flux_density(mlat_rad) * grad_magnetic_flux_density    #[rad^-1]

def Gamma(mlat_rad):
    return 1E0 + 2E0 * plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]


# energy trajectory
def Kperp_energy(mu, mlat_rad):
    return mu * magnetic_flux_density(mlat_rad) #[J]

def Kpara_energy(theta, mlat_rad):
    return (1E0 + theta / wave_frequency)**2E0 * energy_wave_phase_speed(mlat_rad) #[J]

def Ktotal_energy(mu, theta, mlat_rad):
    return Kperp_energy(mu, mlat_rad) + Kpara_energy(theta, mlat_rad) #[J]

def d_Ktotal_d_t(theta, psi, mlat_rad):
    return - energy_wave_potential(mlat_rad) * (theta + wave_frequency) * np.sin(psi)   #[J/s]

def trapping_frequency(mlat_rad):
    return np.abs(kpara(mlat_rad)) * np.sqrt(energy_wave_potential(mlat_rad) / electron_mass)   #[rad/s]

def S_value(mu, theta, mlat_rad):
    return (Kpara_energy(theta, mlat_rad) / energy_wave_potential(mlat_rad) * (1E0 + Gamma(mlat_rad)) + Kperp_energy(mu, mlat_rad) / energy_wave_potential(mlat_rad))  * delta(mlat_rad)    #[]

def d_psi_d_t(theta):
    return theta    #[rad/s]

def d_theta_d_t(mu, theta, psi, mlat_rad):
    return - trapping_frequency(mlat_rad)**2E0 * (np.sin(psi) + S_value(mu, theta, mlat_rad))    #[rad/s]

def d_mlat_rad_d_t(theta, mlat_rad):
    return (theta + wave_frequency) / kpara(mlat_rad) / r_eq / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)    #[rad/s]

def vpara(theta, mlat_rad):
    return (theta + wave_frequency) / kpara(mlat_rad)    #[m/s]

def Xi_dSdt(mlat_rad):
    return 2E0 * (1E0 + plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))) * delta(mlat_rad)    #[]

def d_Gamma_d_t(theta, mlat_rad):
    return - 8E0 * plasma_beta_ion(mlat_rad) * tau(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))**2E0 * (theta + wave_frequency) * delta(mlat_rad)    #[s^-1]

def d_alpha_d_t(mu, theta, psi, mlat_rad):
    pitch_angle_rad = np.arccos(np.sqrt(Kpara_energy(theta, mlat_rad) / Ktotal_energy(mu, theta, mlat_rad)))
    return 1E0 / 2E0 / np.cos(pitch_angle_rad) * (theta + wave_frequency) * (delta(mlat_rad) * np.sin(pitch_angle_rad) + energy_wave_potential(mlat_rad) / Ktotal_energy(mu, theta, mlat_rad) * np.sin(pitch_angle_rad) * np.sin(psi))    #[rad/s]

def d_delta_d_t(theta, mlat_rad):
    d_delta_d_z = (delta(mlat_rad + diff_rad) - delta(mlat_rad - diff_rad)) / 2E0 / diff_rad * d_mlat_d_z(mlat_rad)
    return (theta + wave_frequency) / kpara(mlat_rad) * d_delta_d_z    #[s^-1]

def d_S_d_t(mu, theta, psi, mlat_rad):
    pitch_angle_rad = np.arccos(np.sqrt(Kpara_energy(theta, mlat_rad) / Ktotal_energy(mu, theta, mlat_rad)))
    coef_dKdt = S_value(mu, theta, mlat_rad) / Ktotal_energy(mu, theta, mlat_rad) * d_Ktotal_d_t(theta, psi, mlat_rad)
    coef_dGammadt = S_value(mu, theta, mlat_rad) / (1E0 + Gamma(mlat_rad) * np.cos(pitch_angle_rad)**2E0) * d_Gamma_d_t(theta, mlat_rad) * np.cos(pitch_angle_rad)**2E0
    coef_dalphadt = S_value(mu, theta, mlat_rad) / (1E0 + Gamma(mlat_rad) * np.cos(pitch_angle_rad)**2E0) * d_alpha_d_t(mu, theta, psi, mlat_rad) * Gamma(mlat_rad) * np.sin(2E0 * pitch_angle_rad)
    coef_ddeltadt = S_value(mu, theta, mlat_rad) / delta(mlat_rad) * d_delta_d_t(theta, mlat_rad)
    return coef_dKdt + coef_dGammadt + coef_dalphadt + coef_ddeltadt    #[s^-1]

def W_value_dSdt(mu, theta, psi, mlat_rad):
    return - 1E0 / (theta + wave_frequency) / Xi_dSdt(mlat_rad) * d_S_d_t(mu, theta, psi, mlat_rad) - np.sin(psi)    #[]

def region_detection(mu, theta, psi, mlat_rad, time_now, time_end):
    S_value_now = S_value(mu, theta, mlat_rad)
    trapping_frequency_now = trapping_frequency(mlat_rad)
    if S_value_now <= 1E0 and S_value_now >= 0E0:
        function_phase_trapping = (theta / 2E0 / trapping_frequency_now)**2E0 - 5E-1 * (np.cos(psi) + np.sqrt(1E0 - S_value_now**2E0) - S_value_now * (psi + np.pi - np.arcsin(S_value_now)))
        function_saddle_point = psi + np.pi - np.arcsin(S_value_now)
    else:
        function_phase_trapping = np.nan
        function_saddle_point = np.nan
    function_resonant_scattering = (theta / 2E0 / trapping_frequency_now)**2E0 - 5E-1 * (np.cos(psi) + S_value_now * (np.pi - psi) + 1E0)

    #characterize the region
    if function_resonant_scattering > 0E0:
        region = 2 + time_now / time_end
    else:
        if S_value_now <= 1E0 and S_value_now >= 0E0:
            if function_phase_trapping <= 0E0 and function_saddle_point >= 0E0:
                region = 0 + time_now / time_end
            else:
                region = 1 + time_now / time_end
        else:
            region = 1 + time_now / time_end
    
    return function_phase_trapping, function_saddle_point, function_resonant_scattering, region




# runge-kutta method
dt = 1E-3
def RK4(mlat_rad_0, theta_0, psi_0, mu):
    # 1st step
    k1_mlat_rad = d_mlat_rad_d_t(theta_0, mlat_rad_0)
    k1_theta = d_theta_d_t(mu, theta_0, psi_0, mlat_rad_0)
    k1_psi = d_psi_d_t(theta_0)

    # 2nd step
    k2_mlat_rad = d_mlat_rad_d_t(theta_0 + k1_theta * dt / 2E0, mlat_rad_0 + k1_mlat_rad * dt / 2E0)
    k2_theta = d_theta_d_t(mu, theta_0 + k1_theta * dt / 2E0, psi_0 + k1_psi * dt / 2E0, mlat_rad_0 + k1_mlat_rad * dt / 2E0)
    k2_psi = d_psi_d_t(theta_0 + k1_theta * dt / 2E0)

    # 3rd step
    k3_mlat_rad = d_mlat_rad_d_t(theta_0 + k2_theta * dt / 2E0, mlat_rad_0 + k2_mlat_rad * dt / 2E0)
    k3_theta = d_theta_d_t(mu, theta_0 + k2_theta * dt / 2E0, psi_0 + k2_psi * dt / 2E0, mlat_rad_0 + k2_mlat_rad * dt / 2E0)
    k3_psi = d_psi_d_t(theta_0 + k2_theta * dt / 2E0)

    # 4th step
    k4_mlat_rad = d_mlat_rad_d_t(theta_0 + k3_theta * dt, mlat_rad_0 + k3_mlat_rad * dt)
    k4_theta = d_theta_d_t(mu, theta_0 + k3_theta * dt, psi_0 + k3_psi * dt, mlat_rad_0 + k3_mlat_rad * dt)
    k4_psi = d_psi_d_t(theta_0 + k3_theta * dt)

    # update
    mlat_rad_1 = mlat_rad_0 + dt * (k1_mlat_rad + 2E0 * k2_mlat_rad + 2E0 * k3_mlat_rad + k4_mlat_rad) / 6E0
    theta_1 = theta_0 + dt * (k1_theta + 2E0 * k2_theta + 2E0 * k3_theta + k4_theta) / 6E0
    psi_1 = psi_0 + dt * (k1_psi + 2E0 * k2_psi + 2E0 * k3_psi + k4_psi) / 6E0

    return mlat_rad_1, theta_1, psi_1


# initial condition
#initial_K_eV = 500E0 #[eV]
#initial_pitch_angle_deg = 89E0 #[deg]
#initial_mlat_deg = 1E0 #[deg]

initial_K_eV = np.linspace(1E2, 1E3, 10)
initial_pitch_angle_deg = np.linspace(5E0, 85E0, 17)
initial_mlat_deg = 1E0

initial_pitch_angle_rad = initial_pitch_angle_deg * np.pi / 180E0 #[rad]
initial_mlat_rad = initial_mlat_deg * np.pi / 180E0 #[rad]

INI_K_EV, INI_PITCH_ANGLE_RAD = np.meshgrid(initial_K_eV, initial_pitch_angle_rad)
INI_PITCH_ANGLE_DEG = INI_PITCH_ANGLE_RAD * 180E0 / np.pi

INITIAL_KPERP_EV = INI_K_EV * np.sin(INI_PITCH_ANGLE_RAD)**2E0 #[eV]
INITIAL_KPARA_EV = INI_K_EV * np.cos(INI_PITCH_ANGLE_RAD)**2E0 #[eV]

INITIAL_MU = INITIAL_KPERP_EV * elementary_charge / magnetic_flux_density(initial_mlat_rad) #[J/T]
INITIAL_THETA = kpara(initial_mlat_rad) * np.sqrt(2E0 * INITIAL_KPARA_EV * elementary_charge / electron_mass) - wave_frequency  #[rad/s]
initial_psi = -0.8 * np.pi #[rad]

dt = 1E-3
time_end = 2E1


# simulation
def main(args):
    initial_mlat_rad_main, initial_theta_main, initial_psi_main, initial_mu_main = args

    initial_K_eV_main = Ktotal_energy(initial_mu_main, initial_theta_main, initial_mlat_rad_main) / elementary_charge
    #initial_Kperp_eV_main = Kperp_energy(initial_mu_main, initial_mlat_rad_main) / elementary_charge
    initial_Kpara_eV_main = Kpara_energy(initial_theta_main, initial_mlat_rad_main) / elementary_charge
    initial_pitch_angle_deg_main = np.arccos(np.sqrt(initial_Kpara_eV_main / initial_K_eV_main)) * 180E0 / np.pi
    initial_mlat_deg_main = initial_mlat_rad_main * 180E0 / np.pi

    dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/phase_trapping_region3/test_particle_simulation_phase_trapping_initialpsi_{initial_psi_main/np.pi:.2f}_pi'
    if os.path.exists(dir_name) == False:
        os.makedirs(dir_name, exist_ok=True)
    fig_name = f'{dir_name}/test_particle_simulation_phase_trapping_initialK_{initial_K_eV_main:.1f}_alpha_{initial_pitch_angle_deg_main:.1f}_mlat_{initial_mlat_deg_main:.1f}'
    #if os.path.exists(fig_name + '.png') == True:
    #    if os.path.exists(fig_name + '.pdf') == True:
    #        return
    

    time = 0E0

    mlat_rad_array = np.array([initial_mlat_rad_main])
    theta_array = np.array([initial_theta_main])
    vpara_array = np.array([vpara(initial_theta_main, initial_mlat_rad_main)])
    psi_array = np.array([initial_psi_main])
    time_array = np.array([time])
    Kperp_energy_array = np.array([Kperp_energy(initial_mu_main, initial_mlat_rad_main)])
    Kpara_energy_array = np.array([Kpara_energy(initial_theta_main, initial_mlat_rad_main)])
    Ktotal_energy_array = np.array([Ktotal_energy(initial_mu_main, initial_theta_main, initial_mlat_rad_main)])
    d_Ktotal_d_t_array = np.array([d_Ktotal_d_t(initial_theta_main, initial_psi_main, initial_mlat_rad_main)])
    trapping_frequency_array = np.array([trapping_frequency(initial_mlat_rad_main)])
    S_value_array = np.array([S_value(initial_mu_main, initial_theta_main, initial_mlat_rad_main)])
    d_S_d_t_array = np.array([d_S_d_t(initial_mu_main, initial_theta_main, initial_psi_main, initial_mlat_rad_main)])

    mlat_rad_0 = initial_mlat_rad_main
    theta_0 = initial_theta_main
    psi_0 = initial_psi_main

    while time_array[-1] < time_end:
        mlat_rad_1, theta_1, psi_1 = RK4(mlat_rad_0, theta_0, psi_0, initial_mu_main)
        time += dt

        if mlat_rad_1 > mlat_upper_limit_rad or mlat_rad_1 <= 1E-1 * np.pi / 180E0:
            break
        
        mlat_rad_array = np.append(mlat_rad_array, mlat_rad_1)
        theta_array = np.append(theta_array, theta_1)
        vpara_array = np.append(vpara_array, vpara(theta_1, mlat_rad_1))
        psi_array = np.append(psi_array, psi_1)
        time_array = np.append(time_array, time)
        Kperp_energy_array = np.append(Kperp_energy_array, Kperp_energy(initial_mu_main, mlat_rad_1))
        Kpara_energy_array = np.append(Kpara_energy_array, Kpara_energy(theta_1, mlat_rad_1))
        Ktotal_energy_array = np.append(Ktotal_energy_array, Ktotal_energy(initial_mu_main, theta_1, mlat_rad_1))
        d_Ktotal_d_t_array = np.append(d_Ktotal_d_t_array, d_Ktotal_d_t(theta_1, psi_1, mlat_rad_1))
        trapping_frequency_array = np.append(trapping_frequency_array, trapping_frequency(mlat_rad_1))
        S_value_array = np.append(S_value_array, S_value(initial_mu_main, theta_1, mlat_rad_1))
        d_S_d_t_array = np.append(d_S_d_t_array, d_S_d_t(initial_mu_main, theta_1, psi_1, mlat_rad_1))

        if time_array[-1] > time_end:
            break
        
        if psi_1 != psi_1:
            print(r'Error!: psi is nan')
            quit()
        else:
            mlat_rad_0 = mlat_rad_1
            theta_0 = theta_1
            psi_0 = psi_1

    # psi_arrayをmodを取って-piからpiの範囲にする
    psi_array = psi_array + np.pi
    psi_array = np.mod(psi_array, 2E0 * np.pi)
    psi_array = psi_array - np.pi

    # phase trapping region detection
    function_phase_trapping_array = np.zeros_like(mlat_rad_array)
    function_saddle_point_array = np.zeros_like(mlat_rad_array)
    function_resonant_scattering_array = np.zeros_like(mlat_rad_array)
    region_array = np.zeros_like(mlat_rad_array)
    for count_i in range(len(mlat_rad_array)):
        function_phase_trapping_array[count_i], function_saddle_point_array[count_i], function_resonant_scattering_array[count_i], region_array[count_i] = region_detection(initial_mu_main, theta_array[count_i], psi_array[count_i], mlat_rad_array[count_i], time_array[count_i], time_array[-1])

    d2_S_d_t2_array = np.zeros_like(mlat_rad_array)
    for count_i in range(len(mlat_rad_array)):
        if count_i == 0 or count_i == len(mlat_rad_array) - 1:
            d2_S_d_t2_array[count_i] = np.nan
        else:
            d2_S_d_t2_array[count_i] = (d_S_d_t_array[count_i + 1] - d_S_d_t_array[count_i - 1]) / (time_array[count_i + 1] - time_array[count_i - 1])
    
    # plot
    fig = plt.figure(figsize=(40, 40), dpi=100)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.05])

    cmap_color = cm.turbo
    color_target = time_array
    vmin = np.min(color_target)
    vmax = np.max(color_target)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
    sm.set_array([])

    cbarax = fig.add_subplot(gs[3, :])
    cbar = fig.colorbar(sm, cax=cbarax, orientation='horizontal')
    cbar.set_label(r'Time [s]')

    ax_1_1 = fig.add_subplot(gs[0, 0], xlabel=r'MLAT [deg]', ylabel=r'$v_{\parallel} / c$')
    ax_1_2 = fig.add_subplot(gs[1, 0], xlabel=r'MLAT [deg]', ylabel=r'Kinetic energy $K$ [eV]', yscale='log')
    ax_1_3 = fig.add_subplot(gs[2, 0], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$', xlim=[-1, 1], ylim=[-3, 3])
    ax_2_1 = fig.add_subplot(gs[0, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$S$', yscale='log', xlim=[-1, 1])
    ax_2_2 = fig.add_subplot(gs[1, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\mathrm{d} S / \mathrm{d} t / \omega_{\mathrm{t}}$', xlim=[-1, 1], ylim=[-5, 5])
    ax_2_3 = fig.add_subplot(gs[2, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\mathrm{d}^2 S / \mathrm{d} t^2 / \omega^{2}_{\mathrm{t}}$', xlim=[-1, 1], ylim=[-5, 5])
    ax_3_1 = fig.add_subplot(gs[0, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'Trajectory characteristics', xlim=[-1, 1], ylim=[-0.1, 3.1])
    ax_3_2 = fig.add_subplot(gs[1, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'vs. orange line', xlim=[-1, 1])
    ax_3_3 = fig.add_subplot(gs[2, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'vs. purple line', xlim=[-1, 1], ylim=[-8, 3])

    ax_1_1.scatter(mlat_rad_array * 180E0 / np.pi, vpara_array / speed_of_light, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax)
    ax_1_1.scatter(mlat_rad_array[0] * 180E0 / np.pi, vpara_array[0] / speed_of_light, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
    ax_1_1.scatter(mlat_rad_array[-1] * 180E0 / np.pi, vpara_array[-1] / speed_of_light, c='orange', s=200, marker='D', edgecolors='k', zorder=1)

    ax_1_2.scatter(mlat_rad_array * 180E0 / np.pi, Ktotal_energy_array / elementary_charge, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax)
    ax_1_2.scatter(mlat_rad_array[0] * 180E0 / np.pi, Ktotal_energy_array[0] / elementary_charge, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
    ax_1_2.scatter(mlat_rad_array[-1] * 180E0 / np.pi, Ktotal_energy_array[-1] / elementary_charge, c='orange', s=200, marker='D', edgecolors='k', zorder=1)

    ax_1_3.scatter(psi_array / np.pi, theta_array / 2E0 / trapping_frequency_array, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax)
    ax_1_3.scatter(psi_array[0] / np.pi, theta_array[0] / 2E0 / trapping_frequency_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
    ax_1_3.scatter(psi_array[-1] / np.pi, theta_array[-1] / 2E0 / trapping_frequency_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=1)
    ax_1_3.axhline(y=0E0, c='k', linewidth=4E0, alpha=0.5)

    ax_2_1.scatter(psi_array / np.pi, S_value_array, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax)
    ax_2_1.scatter(psi_array[0] / np.pi, S_value_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
    ax_2_1.scatter(psi_array[-1] / np.pi, S_value_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=1)
    ax_2_1.axhline(y=1E0, c='k', linewidth=4E0, alpha=0.5)

    ax_2_2.scatter(psi_array / np.pi, d_S_d_t_array / trapping_frequency_array, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax)
    ax_2_2.scatter(psi_array[0] / np.pi, d_S_d_t_array[0] / trapping_frequency_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
    ax_2_2.scatter(psi_array[-1] / np.pi, d_S_d_t_array[-1] / trapping_frequency_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=1)
    ax_2_2.axhline(y=0E0, c='k', linewidth=4E0, alpha=0.5)
    ax_2_2.axhline(y=np.pi/2E0, c='k', linewidth=4E0, alpha=0.5)

    ax_2_3.scatter(psi_array / np.pi, d2_S_d_t2_array / trapping_frequency_array**2E0, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax)
    ax_2_3.axhline(y=0E0, c='k', linewidth=4E0, alpha=0.5)
    ax_2_3.axhline(y=(np.pi/2E0)**2E0, c='k', linewidth=4E0, alpha=0.5)

    ax_3_1.scatter(psi_array / np.pi, region_array, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax)
    ax_3_1.scatter(psi_array[0] / np.pi, region_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
    ax_3_1.scatter(psi_array[-1] / np.pi, region_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=1)
    ax_3_1.axhline(y=1E0, c='k', linewidth=4E0, alpha=0.5)
    ax_3_1.axhline(y=2E0, c='k', linewidth=4E0, alpha=0.5)

    ax_3_2.scatter(psi_array / np.pi, function_phase_trapping_array, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax)
    ax_3_2.scatter(psi_array[0] / np.pi, function_phase_trapping_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
    ax_3_2.scatter(psi_array[-1] / np.pi, function_phase_trapping_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=1)
    ax_3_2.axhline(y=0E0, c='k', linewidth=4E0, alpha=0.5)

    ax_3_3.scatter(psi_array / np.pi, function_resonant_scattering_array, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax)
    ax_3_3.scatter(psi_array[0] / np.pi, function_resonant_scattering_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
    ax_3_3.scatter(psi_array[-1] / np.pi, function_resonant_scattering_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=1)
    ax_3_3.axhline(y=0E0, c='k', linewidth=4E0, alpha=0.5)

    axes = [ax_1_1, ax_1_2, ax_1_3, ax_2_1, ax_2_2, ax_2_3, ax_3_1, ax_3_2, ax_3_3]

    mlat_deg_for_background = np.linspace(-mlat_upper_limit_deg, mlat_upper_limit_deg, 1000)
    mlat_rad_for_background = mlat_deg_for_background * np.pi / 180E0
    energy_wave_phase_speed_for_background = energy_wave_phase_speed(mlat_rad_for_background)
    energy_wave_potential_for_background = energy_wave_potential(mlat_rad_for_background)
    energy_perp_for_background = Kperp_energy(initial_mu_main, mlat_rad_for_background) * np.ones_like(mlat_rad_for_background)
    Vph_para_for_background = wave_phase_speed(mlat_rad_for_background)

    xlim_enlarged_1_1 = ax_1_1.get_xlim()
    ylim_enlarged_1_1 = ax_1_1.get_ylim()
    xlim_enlarged_1_2 = ax_1_2.get_xlim()
    ylim_enlarged_1_2 = ax_1_2.get_ylim()

    ax_1_1.plot(mlat_deg_for_background, Vph_para_for_background / speed_of_light, c='r', linewidth=4E0, label=r'$V_{\mathrm{ph \parallel}}$', alpha=0.6)

    ax_1_2.plot(mlat_deg_for_background, energy_wave_phase_speed_for_background / elementary_charge, c='r', linewidth=4E0, label=r'$K_{\mathrm{ph \parallel}}$', alpha=0.6)
    ax_1_2.plot(mlat_deg_for_background, energy_wave_potential_for_background / elementary_charge * np.ones(len(mlat_deg_for_background)), c='g', linewidth=4E0, label=r'$K_{\mathrm{E}}$', alpha=0.6)
    ax_1_2.plot(mlat_deg_for_background, energy_perp_for_background / elementary_charge, c='orange', linewidth=4E0, label=r'$K_{\perp}$', alpha=0.6)

    ax_1_1.set_xlim(xlim_enlarged_1_1)
    ax_1_1.set_ylim(ylim_enlarged_1_1)
    ax_1_2.set_xlim(xlim_enlarged_1_2)
    ax_1_2.set_ylim(ylim_enlarged_1_2)

    for ax in axes:
        ax.minorticks_on()
        ax.grid(which='both', alpha=0.3)
        #各図に(a)、(b)、(c)、(d)をつける
        ax.text(-0.15, 1.0, '(' + chr(97 + axes.index(ax)) + ')', transform=ax.transAxes, fontsize=40)

    ax_1_1.legend()
    ax_1_2.legend()

    ax_1_1.legend()
    ax_1_2.legend()

    fig.suptitle(r'$K(t=0)$ = ' + f'{initial_K_eV_main:.1f} eV, ' + r'$\alpha(t=0)$ = ' + f'{initial_pitch_angle_deg_main:.1f} deg, ' + r'$\lambda(t=0)$ = ' + f'{initial_mlat_deg_main:.1f} deg')

    fig.tight_layout()

    os.makedirs(dir_name, exist_ok=True)
    fig.savefig(f'{fig_name}.png')
    #fig.savefig(f'{fig_name}.pdf')
    plt.close()


    return

#main([initial_mlat_rad, INITIAL_THETA[0, 0], initial_psi, INITIAL_MU[0, 0]])

#quit()

if __name__ == '__main__':
    
    num_process = 16

    args = []
    for count_j in range(len(initial_K_eV)):
        for count_i in range(len(initial_pitch_angle_deg)):
            args.append([initial_mlat_rad, INITIAL_THETA[count_i, count_j], initial_psi, INITIAL_MU[count_i, count_j]])
    
    results = []
    with Pool(num_process) as pool:
        result = pool.map(main, args)
    
    print('finish!')



quit()
#以下、旧バージョン
gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 0.1])

cmap_color = cm.turbo
color_target = time_array
vmin = np.min(color_target)
vmax = np.max(color_target)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
sm.set_array([])

cbarax = fig.add_subplot(gs[4, :])
cbar = fig.colorbar(sm, cax=cbarax, orientation='horizontal')
cbar.set_label(r'Time [s]')


ax_1_1 = fig.add_subplot(gs[0:2, 0], xlabel=r'MLAT [deg]', ylabel=r'Kinetic energy $K$ [eV]', yscale='log', xlim=[-mlat_upper_limit_deg, mlat_upper_limit_deg], ylim=[1E1, 1E5])
ax_1_2 = fig.add_subplot(gs[2:4, 0], xlabel=r'MLAT [deg]', ylabel=r'Kinetic energy $K$ [eV]', yscale='log')
ax_2_1 = fig.add_subplot(gs[0, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$K$ [eV]', yscale='log', xlim=[-1, 1])
ax_2_2 = fig.add_subplot(gs[1, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$K_{\perp}$ [eV]', yscale='log', xlim=[-1, 1])
ax_2_3 = fig.add_subplot(gs[2, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$K_{\parallel}$ [eV]', yscale='log', xlim=[-1, 1])
ax_2_4 = fig.add_subplot(gs[3, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\mathrm{d} K / \mathrm{d} t$ [eV $\mathrm{s}^{-1}$]', xlim=[-1, 1])
ax_3_1 = fig.add_subplot(gs[0, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\omega_{\mathrm{t}}$ [rad $\mathrm{s}^{-1}$]', xlim=[-1, 1])
ax_3_2 = fig.add_subplot(gs[1, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$S$', yscale='log', xlim=[-1, 1])
ax_3_3 = fig.add_subplot(gs[2, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$', xlim=[-1, 1])
ax_3_4 = fig.add_subplot(gs[3, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'MLAT [deg]', xlim=[-1, 1])

ax_1_1.scatter(mlat_rad_array * 180E0 / np.pi, Ktotal_energy_array / elementary_charge, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax, alpha=0.3)
ax_1_1.scatter(mlat_rad_array[0] * 180E0 / np.pi, Ktotal_energy_array[0] / elementary_charge, c='black', s=50, marker='o')
ax_1_1.scatter(mlat_rad_array[-1] * 180E0 / np.pi, Ktotal_energy_array[-1] / elementary_charge, c='black', s=50, marker='D')

ax_1_2.scatter(mlat_rad_array * 180E0 / np.pi, Ktotal_energy_array / elementary_charge, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax, alpha=0.3)
ax_1_2.scatter(mlat_rad_array[0] * 180E0 / np.pi, Ktotal_energy_array[0] / elementary_charge, c='black', s=50, marker='o')
ax_1_2.scatter(mlat_rad_array[-1] * 180E0 / np.pi, Ktotal_energy_array[-1] / elementary_charge, c='black', s=50, marker='D')

ax_2_1.scatter(psi_array / np.pi, Ktotal_energy_array / elementary_charge, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax, alpha=0.3)
ax_2_1.scatter(psi_array[0] / np.pi, Ktotal_energy_array[0] / elementary_charge, c='black', s=50, marker='o')
ax_2_1.scatter(psi_array[-1] / np.pi, Ktotal_energy_array[-1] / elementary_charge, c='black', s=50, marker='D')

ax_2_2.scatter(psi_array / np.pi, Kperp_energy_array / elementary_charge, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax, alpha=0.3)
ax_2_2.scatter(psi_array[0] / np.pi, Kperp_energy_array[0] / elementary_charge, c='black', s=50, marker='o')
ax_2_2.scatter(psi_array[-1] / np.pi, Kperp_energy_array[-1] / elementary_charge, c='black', s=50, marker='D')

ax_2_3.scatter(psi_array / np.pi, Kpara_energy_array / elementary_charge, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax, alpha=0.3)
ax_2_3.scatter(psi_array[0] / np.pi, Kpara_energy_array[0] / elementary_charge, c='black', s=50, marker='o')
ax_2_3.scatter(psi_array[-1] / np.pi, Kpara_energy_array[-1] / elementary_charge, c='black', s=50, marker='D')

ax_2_4.scatter(psi_array / np.pi, d_Ktotal_d_t_array / elementary_charge, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax, alpha=0.3)
ax_2_4.scatter(psi_array[0] / np.pi, d_Ktotal_d_t_array[0] / elementary_charge, c='black', s=50, marker='o')
ax_2_4.scatter(psi_array[-1] / np.pi, d_Ktotal_d_t_array[-1] / elementary_charge, c='black', s=50, marker='D')

ax_3_1.scatter(psi_array / np.pi, trapping_frequency_array, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax, alpha=0.3)
ax_3_1.scatter(psi_array[0] / np.pi, trapping_frequency_array[0], c='black', s=50, marker='o')
ax_3_1.scatter(psi_array[-1] / np.pi, trapping_frequency_array[-1], c='black', s=50, marker='D')

ax_3_2.scatter(psi_array / np.pi, S_value_array, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax, alpha=0.3)
ax_3_2.scatter(psi_array[0] / np.pi, S_value_array[0], c='black', s=50, marker='o')
ax_3_2.scatter(psi_array[-1] / np.pi, S_value_array[-1], c='black', s=50, marker='D')

ax_3_3.scatter(psi_array / np.pi, theta_array / 2E0 / trapping_frequency_array, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax, alpha=0.3)
ax_3_3.scatter(psi_array[0] / np.pi, theta_array[0] / 2E0 / trapping_frequency_array[0], c='black', s=50, marker='o')
ax_3_3.scatter(psi_array[-1] / np.pi, theta_array[-1] / 2E0 / trapping_frequency_array[-1], c='black', s=50, marker='D')

ax_3_4.scatter(psi_array / np.pi, mlat_rad_array * 180E0 / np.pi, c=color_target, cmap=cmap_color, s=1E0, vmin=vmin, vmax=vmax, alpha=0.3)
ax_3_4.scatter(psi_array[0] / np.pi, mlat_rad_array[0] * 180E0 / np.pi, c='black', s=50, marker='o')
ax_3_4.scatter(psi_array[-1] / np.pi, mlat_rad_array[-1] * 180E0 / np.pi, c='black', s=50, marker='D')

axes = [ax_1_1, ax_1_2, ax_2_1, ax_2_2, ax_2_3, ax_2_4, ax_3_1, ax_3_2, ax_3_3, ax_3_4]


mlat_deg_for_background = np.linspace(-mlat_upper_limit_deg, mlat_upper_limit_deg, 1000)
mlat_rad_for_background = mlat_deg_for_background * np.pi / 180E0
energy_wave_phase_speed_for_background = energy_wave_phase_speed(mlat_rad_for_background)
energy_wave_potential_for_background = energy_wave_potential(mlat_rad_for_background)
energy_perp_for_background = Kperp_energy(initial_mu, mlat_rad_for_background)

ax_1_1.plot(mlat_deg_for_background, energy_wave_phase_speed_for_background / elementary_charge, c='r', linewidth=4E0, label=r'$K_{\mathrm{ph \parallel}}$', alpha=0.6)
ax_1_1.plot(mlat_deg_for_background, energy_wave_potential_for_background / elementary_charge * np.ones(len(mlat_deg_for_background)), c='g', linewidth=4E0, label=r'$K_{\mathrm{E}}$', alpha=0.6)
ax_1_1.plot(mlat_deg_for_background, energy_perp_for_background / elementary_charge, c='orange', linewidth=4E0, label=r'$K_{\perp}$', alpha=0.6)

xlim_enlarged = ax_1_2.get_xlim()
ylim_enlarged = ax_1_2.get_ylim()

ax_1_2.plot(mlat_deg_for_background, energy_wave_phase_speed_for_background / elementary_charge, c='r', linewidth=4E0, label=r'$K_{\mathrm{ph \parallel}}$', alpha=0.6)
ax_1_2.plot(mlat_deg_for_background, energy_wave_potential_for_background / elementary_charge * np.ones(len(mlat_deg_for_background)), c='g', linewidth=4E0, label=r'$K_{\mathrm{E}}$', alpha=0.6)
ax_1_2.plot(mlat_deg_for_background, energy_perp_for_background / elementary_charge, c='orange', linewidth=4E0, label=r'$K_{\perp}$', alpha=0.6)
ax_1_2.set_xlim(xlim_enlarged)
ax_1_2.set_ylim(ylim_enlarged)

ylim_Kpara = ax_2_3.get_ylim()
if ylim_Kpara[0] < 1E0:
    ylim_Kpara = [1E0, ylim_Kpara[1]]
ax_2_3.set_ylim(ylim_Kpara)

for ax in axes:
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)

ax_1_1.legend()
ax_1_2.legend()

fig.tight_layout()

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/test_particle_simulation_phase_trapping'
os.makedirs(dir_name, exist_ok=True)
fig.savefig(f'{dir_name}/test_particle_simulation_phase_trapping.png')
fig.savefig(f'{dir_name}/test_particle_simulation_phase_trapping.pdf')
plt.close()