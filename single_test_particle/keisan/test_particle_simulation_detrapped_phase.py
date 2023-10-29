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
    grad_magnetic_flux_density = (magnetic_flux_density(mlat_rad + diff_rad) - magnetic_flux_density(mlat_rad - diff_rad)) / 2E0 / diff_rad * d_mlat_d_z(mlat_rad)    #[T/m]
    return 1E0 / kpara(mlat_rad) / magnetic_flux_density(mlat_rad) * grad_magnetic_flux_density    #[rad^-1]

def Gamma(mlat_rad):
    return 1E0 + 2E0 * plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) / (plasma_beta_ion(mlat_rad) * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad))    #[]


# input parameters
array_size = 13

Kperp_eq_eV_base = 5E2 * np.ones(array_size) #[eV]
Kperp_eq_eV_array = np.array([1E1, 2E1, 5E1, 1E2, 2E2, 3E2, 4E2, 5E2, 6E2, 7E2, 8E2, 9E2, 1E3])
Kperp_eq_eV_vmin = 1E1
Kperp_eq_eV_vmax = 1E3

mu_base = Kperp_eq_eV_base * elementary_charge / magnetic_flux_density(0E0)
mu_array = Kperp_eq_eV_array * elementary_charge / magnetic_flux_density(0E0)

theta_initial_base = 0E0 * np.ones(array_size)    #[rad/s]
theta_initial_array = np.linspace(-wave_frequency, wave_frequency, array_size)
theta_initial_vmin = -wave_frequency / wave_frequency
theta_initial_vmax = wave_frequency / wave_frequency

psi_initial_base = - np.pi / 2E0 * np.ones(array_size)    #[rad]
psi_initial_array = np.linspace(-np.pi, 0E0, array_size)
psi_initial_vmin = -np.pi / np.pi
psi_initial_vmax = 0E0 / np.pi

psi_end = -np.pi

figure_name_suffix = [r'mu_change', r'theta_change', r'psi_change']


# initial position
def function_0(mu, theta_initial, mlat_rad):
    return ((1E0 + Gamma(mlat_rad)) * (1E0 + theta_initial / wave_frequency)**2E0 * energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad) + magnetic_flux_density(mlat_rad) * mu / energy_wave_potential(mlat_rad)) * delta(mlat_rad) - 1E0

def gradient_function_0(mu, theta_initial, mlat_rad):
    if mlat_rad < diff_rad:
        return 0E0
    else:
        return (function_0(mu, theta_initial, mlat_rad + diff_rad) - function_0(mu, theta_initial, mlat_rad - diff_rad)) / 2E0 / diff_rad

def Newton_method_function_0(mu, theta_initial):
    mlat_rad_before_update = np.pi / 2E0
    count_iteration = 0

    while True:
        diff = function_0(mu, theta_initial, mlat_rad_before_update) / gradient_function_0(mu, theta_initial, mlat_rad_before_update)
        if abs(diff) > 1E-1:
            diff = np.sign(diff) * 1E-1
        mlat_rad_after_update = mlat_rad_before_update - diff
        if abs(diff) < 1E-10:
            break
        else:
            mlat_rad_before_update = mlat_rad_after_update
            if mlat_rad_after_update < 0E0 or mlat_rad_after_update > np.pi / 2E0:
                mlat_rad_before_update = np.mod(mlat_rad_before_update, np.pi / 2E0)
            count_iteration += 1
            if count_iteration > 1000 or mlat_rad_after_update != mlat_rad_after_update:
                mlat_rad_after_update = np.nan
                break
    
    return mlat_rad_after_update


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


# main function

def particle_calculation(args):

    mu, theta_initial, psi_initial, count_i = args

    # initial position
    mlat_rad_0 = Newton_method_function_0(mu, theta_initial)
    theta_0 = theta_initial
    psi_0 = psi_initial

    # initial time
    time = 0E0

    # initial array
    mlat_rad_array = np.array([mlat_rad_0])
    theta_array = np.array([theta_0])
    psi_array = np.array([psi_0])
    time_array = np.array([time])
    Kperp_energy_array = np.array([Kperp_energy(mu, mlat_rad_0)])
    Kpara_energy_array = np.array([Kpara_energy(theta_0, mlat_rad_0)])
    Ktotal_energy_array = np.array([Ktotal_energy(mu, theta_0, mlat_rad_0)])
    d_Ktotal_d_t_array = np.array([d_Ktotal_d_t(theta_0, psi_0, mlat_rad_0)])
    trapping_frequency_array = np.array([trapping_frequency(mlat_rad_0)])
    S_value_array = np.array([S_value(mu, theta_0, mlat_rad_0)])

    # main loop
    while True:
        mlat_rad_1, theta_1, psi_1 = RK4(mlat_rad_0, theta_0, psi_0, mu)
        time += dt
        mlat_rad_array = np.append(mlat_rad_array, mlat_rad_1)
        theta_array = np.append(theta_array, theta_1)
        psi_array = np.append(psi_array, psi_1)
        time_array = np.append(time_array, time)
        Kperp_energy_array = np.append(Kperp_energy_array, Kperp_energy(mu, mlat_rad_1))
        Kpara_energy_array = np.append(Kpara_energy_array, Kpara_energy(theta_1, mlat_rad_1))
        Ktotal_energy_array = np.append(Ktotal_energy_array, Ktotal_energy(mu, theta_1, mlat_rad_1))
        d_Ktotal_d_t_array = np.append(d_Ktotal_d_t_array, d_Ktotal_d_t(theta_1, psi_1, mlat_rad_1))
        trapping_frequency_array = np.append(trapping_frequency_array, trapping_frequency(mlat_rad_1))
        S_value_array = np.append(S_value_array, S_value(mu, theta_1, mlat_rad_1))

        if psi_1 < psi_end or mlat_rad_1 < 0E0 or mlat_rad_1 > mlat_upper_limit_rad or time == dt*10:
            break

        else:
            if psi_1 != psi_1:
                print(r'Error: psi is nan')
                quit()
            else:
                mlat_rad_0 = mlat_rad_1
                theta_0 = theta_1
                psi_0 = psi_1
    
    return mlat_rad_array, theta_array, psi_array, time_array, Kperp_energy_array, Kpara_energy_array, Ktotal_energy_array, d_Ktotal_d_t_array, trapping_frequency_array, S_value_array, count_i


def main_1(index):
    # plot
    fig = plt.figure(figsize=(30, 30), dpi=100)
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 0.1])

    axes = []

    ax_1_1 = fig.add_subplot(gs[0:2, 0], xlabel=r'MLAT [deg]', ylabel=r'Kinetic energy $K$ [eV]', yscale='log', xlim=[0E0, mlat_upper_limit_deg], ylim=[1E1, 1E5])
    ax_1_2 = fig.add_subplot(gs[2:4, 0], xlabel=r'MLAT [deg]', ylabel=r'Kinetic energy $K$ [eV]', yscale='log')
    ax_2_1 = fig.add_subplot(gs[0, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$K$ [eV]', yscale='log')
    ax_2_2 = fig.add_subplot(gs[1, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$K_{\perp}$ [eV]', yscale='log')
    ax_2_3 = fig.add_subplot(gs[2, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$K_{\parallel}$ [eV]', yscale='log')
    ax_2_4 = fig.add_subplot(gs[3, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\mathrm{d} K / \mathrm{d} t$ [eV $\mathrm{s}^{-1}$]')
    ax_3_1 = fig.add_subplot(gs[0, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\omega_{\mathrm{t}}$ [rad $\mathrm{s}^{-1}$]')
    ax_3_2 = fig.add_subplot(gs[1, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$S$', yscale='log')
    ax_3_3 = fig.add_subplot(gs[2, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$')
    ax_3_4 = fig.add_subplot(gs[3, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'MLAT [deg]')

    axes.append(ax_1_1)
    axes.append(ax_1_2)
    axes.append(ax_2_1)
    axes.append(ax_2_2)
    axes.append(ax_2_3)
    axes.append(ax_2_4)
    axes.append(ax_3_1)
    axes.append(ax_3_2)
    axes.append(ax_3_3)
    axes.append(ax_3_4)

    # index = 0 -> mu change
    # index = 1 -> theta change
    # index = 2 -> psi change

    if index == 0:
        mu_reference = mu_array
        theta_initial_reference = theta_initial_base
        psi_initial_reference = psi_initial_base
        vmin = Kperp_eq_eV_vmin
        vmax = Kperp_eq_eV_vmax
        color_target = Kperp_eq_eV_array
        psi_max = 0E0 / np.pi
        psi_min = psi_end / np.pi
        fig.suptitle(r'$\theta_{\mathrm{i}} = 0$, $\psi_{\mathrm{i}} = -0.5 \pi$')

    elif index == 1:
        mu_reference = mu_base
        theta_initial_reference = theta_initial_array
        psi_initial_reference = psi_initial_base
        vmin = theta_initial_vmin
        vmax = theta_initial_vmax
        color_target = theta_initial_array
        psi_max = 0E0 / np.pi
        psi_min = psi_end / np.pi
        fig.suptitle(r'$K_{\perp}(\lambda = 0) = 500 \, \mathrm{eV}$, $\psi_{\mathrm{i}} = -0.5 \pi$')
    
    elif index == 2:
        mu_reference = mu_base
        theta_initial_reference = theta_initial_base
        psi_initial_reference = psi_initial_array
        vmin = psi_initial_vmin
        vmax = psi_initial_vmax
        color_target = psi_initial_array / np.pi
        psi_max = 0E0 / np.pi
        psi_min = psi_end / np.pi
        fig.suptitle(r'$K_{\perp}(\lambda = 0) = 500 \, \mathrm{eV}$, $\theta_{\mathrm{i}} = 0$')
    
    else:
        return
    
    for count_k in range(2, 10):
        axes[count_k].set_xlim([psi_min, psi_max])

    return mu_reference, theta_initial_reference, psi_initial_reference, fig, gs, axes, vmin, vmax, color_target

cmap_color = cm.turbo

for index in range(3):
    mu_reference, theta_initial_reference, psi_initial_reference, fig, gs, axes, vmin, vmax, color_target = main_1(index)

    mlat_deg_for_background = np.linspace(0E0, mlat_upper_limit_deg, 1000)
    mlat_rad_for_background = mlat_deg_for_background * np.pi / 180E0
    energy_wave_phase_speed_eV = energy_wave_phase_speed(mlat_rad_for_background) / elementary_charge
    energy_wave_potential_eV = energy_wave_potential(mlat_rad_for_background) / elementary_charge * np.ones(len(mlat_rad_for_background))

    energy_S_1_upper_limit_eV = np.zeros(len(mlat_rad_for_background))
    energy_S_1_lower_limit_eV = np.zeros(len(mlat_rad_for_background))
    for count_j in range(len(mlat_rad_for_background)):
        energy_S_1_upper_limit_eV[count_j] = energy_wave_potential_eV[count_j] / delta(mlat_rad_for_background[count_j])
        energy_S_1_lower_limit_eV[count_j] = energy_wave_potential_eV[count_j] / delta(mlat_rad_for_background[count_j]) / (1E0 + Gamma(mlat_rad_for_background[count_j]))
    
    if index == 1 or index == 2:
        mu_reference_mode = mu_reference[0]
        energy_perp_for_background_eV = mu_reference_mode * magnetic_flux_density(mlat_rad_for_background) / elementary_charge
    

    if __name__ == '__main__':

        if array_size > 16:
            num_process = 16
        else:
            num_process = array_size
        
        num_process = 1
    
        args = []
        for count_i in range(array_size):
            args.append([mu_reference[count_i], theta_initial_reference[count_i], psi_initial_reference[count_i], count_i])
        
        with Pool(num_process) as p:
            results = p.map(particle_calculation, args)
        
        for result in results:
            mlat_rad_array, theta_array, psi_array, time_array, Kperp_energy_array, Kpara_energy_array, Ktotal_energy_array, d_Ktotal_d_t_array, trapping_frequency_array, S_value_array, count_particle = result

            color_target_array = np.ones(len(mlat_rad_array)) * color_target[count_particle]

            # plot
            axes[0].scatter(mlat_rad_array * 180E0 / np.pi, Ktotal_energy_array / elementary_charge, c=color_target_array, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.3, s=1)
            axes[0].scatter(mlat_rad_array[0] * 180E0 / np.pi, Ktotal_energy_array[0] / elementary_charge, s=50, c='black')
            axes[0].scatter(mlat_rad_array[-1] * 180E0 / np.pi, Ktotal_energy_array[-1] / elementary_charge, s=50, c='black')

            axes[1].scatter(mlat_rad_array * 180E0 / np.pi, Ktotal_energy_array / elementary_charge, c=color_target_array, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.3, s=1)
            axes[1].scatter(mlat_rad_array[0] * 180E0 / np.pi, Ktotal_energy_array[0] / elementary_charge, s=50, c='black')
            axes[1].scatter(mlat_rad_array[-1] * 180E0 / np.pi, Ktotal_energy_array[-1] / elementary_charge, s=50, c='black')

            axes[2].scatter(psi_array / np.pi, Ktotal_energy_array / elementary_charge, c=color_target_array, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.3, s=1)
            axes[2].scatter(psi_array[0] / np.pi, Ktotal_energy_array[0] / elementary_charge, s=50, c='black')
            axes[2].scatter(psi_array[-1] / np.pi, Ktotal_energy_array[-1] / elementary_charge, s=50, c='black')

            axes[3].scatter(psi_array / np.pi, Kperp_energy_array / elementary_charge, c=color_target_array, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.3, s=1)
            axes[3].scatter(psi_array[0] / np.pi, Kperp_energy_array[0] / elementary_charge, s=50, c='black')
            axes[3].scatter(psi_array[-1] / np.pi, Kperp_energy_array[-1] / elementary_charge, s=50, c='black')

            axes[4].scatter(psi_array / np.pi, Kpara_energy_array / elementary_charge, c=color_target_array, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.3, s=1)
            axes[4].scatter(psi_array[0] / np.pi, Kpara_energy_array[0] / elementary_charge, s=50, c='black')
            axes[4].scatter(psi_array[-1] / np.pi, Kpara_energy_array[-1] / elementary_charge, s=50, c='black')

            axes[5].scatter(psi_array / np.pi, d_Ktotal_d_t_array / elementary_charge, c=color_target_array, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.3, s=1)
            axes[5].scatter(psi_array[0] / np.pi, d_Ktotal_d_t_array[0] / elementary_charge, s=50, c='black')
            axes[5].scatter(psi_array[-1] / np.pi, d_Ktotal_d_t_array[-1] / elementary_charge, s=50, c='black')

            axes[6].scatter(psi_array / np.pi, trapping_frequency_array, c=color_target_array, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.3, s=1)
            axes[6].scatter(psi_array[0] / np.pi, trapping_frequency_array[0], s=50, c='black')
            axes[6].scatter(psi_array[-1] / np.pi, trapping_frequency_array[-1], s=50, c='black')

            axes[7].scatter(psi_array / np.pi, S_value_array, c=color_target_array, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.3, s=1)
            axes[7].scatter(psi_array[0] / np.pi, S_value_array[0], s=50, c='black')
            axes[7].scatter(psi_array[-1] / np.pi, S_value_array[-1], s=50, c='black')

            axes[8].scatter(psi_array / np.pi, theta_array / 2E0 / trapping_frequency_array, c=color_target_array, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.3, s=1)
            axes[8].scatter(psi_array[0] / np.pi, theta_array[0] / 2E0 / trapping_frequency_array[0], s=50, c='black')
            axes[8].scatter(psi_array[-1] / np.pi, theta_array[-1] / 2E0 / trapping_frequency_array[-1], s=50, c='black')

            axes[9].scatter(psi_array / np.pi, mlat_rad_array * 180E0 / np.pi, c=color_target_array, cmap=cmap_color, vmin=vmin, vmax=vmax, alpha=0.3, s=1)
            axes[9].scatter(psi_array[0] / np.pi, mlat_rad_array[0] * 180E0 / np.pi, s=50, c='black')
            axes[9].scatter(psi_array[-1] / np.pi, mlat_rad_array[-1] * 180E0 / np.pi, s=50, c='black')
        
    axes[0].plot(mlat_deg_for_background, energy_wave_phase_speed_eV, c='red', linewidth=4, label=r'$K_{\mathrm{ph \parallel}}$', alpha=0.6)
    axes[0].plot(mlat_deg_for_background, energy_wave_potential_eV, c='green', linewidth=4, label=r'$K_{\mathrm{E}}$', alpha=0.6)
    axes[0].plot(mlat_deg_for_background, energy_S_1_upper_limit_eV, c='blue', linewidth=4, label=r'$S = 1$ range', alpha=0.6)
    axes[0].plot(mlat_deg_for_background, energy_S_1_lower_limit_eV, c='blue', linewidth=4, alpha=0.6)
    if index == 1 or index == 2:
        axes[0].plot(mlat_deg_for_background, energy_perp_for_background_eV, c='orange', linewidth=4, label=r'$K_{\perp}$', alpha=0.6)

    xlim_enlarged = axes[1].get_xlim()
    ylim_enlarged = axes[1].get_ylim()
    axes[1].plot(mlat_deg_for_background, energy_wave_phase_speed_eV, c='red', linewidth=4, label=r'$K_{\mathrm{ph \parallel}}$', alpha=0.6)
    axes[1].plot(mlat_deg_for_background, energy_wave_potential_eV, c='green', linewidth=4, label=r'$K_{\mathrm{E}}$', alpha=0.6)
    axes[1].plot(mlat_deg_for_background, energy_S_1_upper_limit_eV, c='blue', linewidth=4, label=r'$S = 1$ range', alpha=0.6)
    axes[1].plot(mlat_deg_for_background, energy_S_1_lower_limit_eV, c='blue', linewidth=4, alpha=0.6)
    if index == 1 or index == 2:
        axes[1].plot(mlat_deg_for_background, energy_perp_for_background_eV, c='orange', linewidth=4, label=r'$K_{\perp}$', alpha=0.6)
    axes[1].set_xlim(xlim_enlarged)
    axes[1].set_ylim(ylim_enlarged)

    ylim_Kpara = axes[4].get_ylim()
    if ylim_Kpara[0] < 1E0:
        ylim_Kpara = [1E0, ylim_Kpara[1]]
    axes[4].set_ylim(ylim_Kpara)

    for count_k in range(10):
        axes[count_k].minorticks_on()
        axes[count_k].grid(which='both', alpha=0.3)

    axes[0].legend()
    axes[1].legend()

    fig.tight_layout()
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
    sm.set_array([])

    cbarax = fig.add_subplot(gs[-1, :])
    cbar = fig.colorbar(sm, cax=cbarax, orientation='horizontal', aspect=100)

    if index == 0:
        cbarax.set_xlabel(r'$K_{\perp}(\lambda = 0) \, [\mathrm{eV}]$')
    elif index == 1:
        cbarax.set_xlabel(r'$\theta_{\mathrm{i}} \, [\omega]$')
    elif index == 2:
        cbarax.set_xlabel(r'$\psi_{\mathrm{i}} \, [\pi \, \mathrm{rad}]$')
    
    fig.tight_layout()
    
    dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/test_particle_simulation_detrapped_phase/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig.savefig(dir_name + f'figure_{figure_name_suffix[index]}_end_psi_{(psi_end/np.pi):.2f}_pi.png')