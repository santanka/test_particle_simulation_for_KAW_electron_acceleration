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


# initial energy

energy_perp_upper_limit = 1E0   #[eV]
mu_upper_limit = energy_perp_upper_limit * elementary_charge / magnetic_flux_density(0E0)    #[J T-1]



def function_pitch_angle(mlat_rad, mu):
    return delta(mlat_rad) * magnetic_flux_density(mlat_rad) * mu + (delta(mlat_rad) + epsilon(mlat_rad)) * energy_wave_phase_speed(mlat_rad) - energy_wave_potential(mlat_rad)

def Newton_method_function_pitch_angle():
    before_update_mlat_rad = np.pi / 2E0
    while True:
        diff = function_pitch_angle(before_update_mlat_rad, mu_upper_limit) / (function_pitch_angle(before_update_mlat_rad + diff_rad, mu_upper_limit) - function_pitch_angle(before_update_mlat_rad - diff_rad, mu_upper_limit)) * 2E0 * diff_rad
        if abs(diff) > 1E-1:
            diff = np.sign(diff)
        after_update_mlat_rad = before_update_mlat_rad - diff
        if abs(after_update_mlat_rad - before_update_mlat_rad) < 1E-10:
            break
        else:
            before_update_mlat_rad = after_update_mlat_rad
            if after_update_mlat_rad < 0E0 or after_update_mlat_rad > np.pi / 2E0:
                quit()
    pitch_angle = np.arctan(np.sqrt(magnetic_flux_density(after_update_mlat_rad) * mu_upper_limit / energy_wave_phase_speed(after_update_mlat_rad)))    #[rad]
    return pitch_angle

initial_pitch_angle_rad = Newton_method_function_pitch_angle()
initial_pitch_angle_deg = initial_pitch_angle_rad * 180E0 / np.pi   #[deg]

#initial_pitch_angle_deg = 68E0    #[deg]
#initial_pitch_angle_rad = initial_pitch_angle_deg / 180E0 * np.pi   #[rad]

trajectory_number = 11
intial_psi_array = np.linspace(- np.pi, 0E0, trajectory_number)
theta_per_2_trapping_freq_array = np.linspace(- np.sqrt(0.75*np.pi + 0.5), np.sqrt(0.75*np.pi + 0.5), trajectory_number)

end_psi = - np.pi * 3E0

def function_0(pitch_angle_rad, mlat_rad):
    return delta(mlat_rad) * energy_wave_phase_speed(mlat_rad) + (epsilon(mlat_rad) * energy_wave_phase_speed(mlat_rad) - energy_wave_potential(mlat_rad)) * np.cos(pitch_angle_rad)**2E0

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
            break
        else:
            mlat_rad_before_update = mlat_rad_after_update
            if mlat_rad_after_update < 0E0 or mlat_rad_after_update > np.pi / 2E0:
                mlat_rad_before_update = np.mod(mlat_rad_before_update, np.pi / 2E0)
            count_iteration += 1
            if count_iteration > 1000 or mlat_rad_after_update != mlat_rad_after_update:
                mlat_rad_after_update = np.nan
                print('Error: Newton method')
                quit()
    
    return mlat_rad_after_update

def mu_adibatic(pitch_angle_rad, mlat_rad):
    return energy_wave_phase_speed(mlat_rad) / magnetic_flux_density(mlat_rad) * np.tan(pitch_angle_rad)**2E0    #[J T-1]

initial_mlat_rad = Newton_method_function_0(initial_pitch_angle_rad)
mu_init = mu_adibatic(initial_pitch_angle_rad, initial_mlat_rad)
#if mu_init > mu_upper_limit:
#    print('Error: mu')
#    quit()


# energy trajectory
def trapping_frequency(mlat_rad):
    return np.abs(kpara(mlat_rad)) * np.sqrt(energy_wave_potential(mlat_rad) / electron_mass)   #[rad/s]

def S_value(energy, mlat_rad, pitch_angle_rad):
    return energy / energy_wave_potential(mlat_rad) * (delta(mlat_rad) + epsilon(mlat_rad) * np.cos(pitch_angle_rad)**2E0)    #[]

def kinetic_energy_and_pitch_angle_rad(mu, mlat_rad, theta):
    kinetic_energy = mu * magnetic_flux_density(mlat_rad) + 5E-1 * electron_mass * ((theta + wave_frequency) / kpara(mlat_rad))**2E0    #[J]
    pitch_angle_rad = np.arcsin(np.sqrt(mu * magnetic_flux_density(mlat_rad) / kinetic_energy))    #[rad]
    return kinetic_energy, pitch_angle_rad

def d_psi_d_t(theta):
    return theta # [rad/s]

def d_theta_d_t(mlat_rad, psi, theta, mu):
    kinetic_energy, pitch_angle_rad = kinetic_energy_and_pitch_angle_rad(mu, mlat_rad, theta)
    return - trapping_frequency(mlat_rad)**2E0 * (np.sin(psi) + S_value(kinetic_energy, mlat_rad, pitch_angle_rad)) #[rad/s^2]

def d_mlat_rad_d_t(theta, mlat_rad):
    return (theta + wave_frequency) / kpara(mlat_rad) / r_eq / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)    #[rad/s]


# runge-kutta method
dt = 1E-3
def RK4(mlat_rad_0, theta_0, psi_0, mu):
    # 1st step
    k1_mlat_rad = d_mlat_rad_d_t(theta_0, mlat_rad_0)
    k1_theta = d_theta_d_t(mlat_rad_0, psi_0, theta_0, mu)
    k1_psi = d_psi_d_t(theta_0)

    # 2nd step
    k2_mlat_rad = d_mlat_rad_d_t(theta_0 + k1_theta * dt / 2E0, mlat_rad_0 + k1_mlat_rad * dt / 2E0)
    k2_theta = d_theta_d_t(mlat_rad_0 + k1_mlat_rad * dt / 2E0, psi_0 + k1_psi * dt / 2E0, theta_0 + k1_theta * dt / 2E0, mu)
    k2_psi = d_psi_d_t(theta_0 + k1_theta * dt / 2E0)

    # 3rd step
    k3_mlat_rad = d_mlat_rad_d_t(theta_0 + k2_theta * dt / 2E0, mlat_rad_0 + k2_mlat_rad * dt / 2E0)
    k3_theta = d_theta_d_t(mlat_rad_0 + k2_mlat_rad * dt / 2E0, psi_0 + k2_psi * dt / 2E0, theta_0 + k2_theta * dt / 2E0, mu)
    k3_psi = d_psi_d_t(theta_0 + k2_theta * dt / 2E0)

    # 4th step
    k4_mlat_rad = d_mlat_rad_d_t(theta_0 + k3_theta * dt, mlat_rad_0 + k3_mlat_rad * dt)
    k4_theta = d_theta_d_t(mlat_rad_0 + k3_mlat_rad * dt, psi_0 + k3_psi * dt, theta_0 + k3_theta * dt, mu)
    k4_psi = d_psi_d_t(theta_0 + k3_theta * dt)

    # update
    mlat_rad_1 = mlat_rad_0 + dt * (k1_mlat_rad + 2E0 * k2_mlat_rad + 2E0 * k3_mlat_rad + k4_mlat_rad) / 6E0
    theta_1 = theta_0 + dt * (k1_theta + 2E0 * k2_theta + 2E0 * k3_theta + k4_theta) / 6E0
    psi_1 = psi_0 + dt * (k1_psi + 2E0 * k2_psi + 2E0 * k3_psi + k4_psi) / 6E0

    return mlat_rad_1, theta_1, psi_1



# plot

fig_1 = plt.figure(figsize=(14, 14), dpi=100)
ax_1 = fig_1.add_subplot(111, xlabel=r'MLAT [deg]', ylabel=r'Energy [$\mathrm{eV}$]', yscale='log')

fig_2 = plt.figure(figsize=(14, 28), dpi=100)
ax_2_1 = fig_2.add_subplot(411, xlabel=r'$\psi$ [$\pi$ $\mathrm{rad}$]', ylabel=r'$K$ [$\mathrm{eV}$]', xlim=(end_psi/np.pi, 0), yscale='log')
ax_2_2 = fig_2.add_subplot(412, xlabel=r'$\psi$ [$\pi$ $\mathrm{rad}$]', ylabel=r'$K_{\perp}$ [$\mathrm{eV}$]', xlim=(end_psi/np.pi, 0), yscale='log')
ax_2_3 = fig_2.add_subplot(413, xlabel=r'$\psi$ [$\pi$ $\mathrm{rad}$]', ylabel=r'$K_{\parallel}$ [$\mathrm{eV}$]', xlim=(end_psi/np.pi, 0), yscale='log')
#ax_2_4 = fig_2.add_subplot(414, xlabel=r'$\psi$ [$\pi$ $\mathrm{rad}$]', ylabel=r'$\alpha_{\mathrm{eq}}$ [deg]', xlim=(-1, 0))
ax_2_4 = fig_2.add_subplot(414, xlabel=r'$\psi$ [$\pi$ $\mathrm{rad}$]', ylabel=r'MLAT [deg]', xlim=(end_psi/np.pi, 0))

fig_3 = plt.figure(figsize=(14, 28), dpi=100)
ax_3_1 = fig_3.add_subplot(411, xlabel=r'$\psi$ [$\pi$ $\mathrm{rad}$]', ylabel=r'$\omega_{\mathrm{t}}$ [$\pi \, \mathrm{rad/s}$]', xlim=(end_psi/np.pi, 0))
ax_3_2 = fig_3.add_subplot(412, xlabel=r'$\psi$ [$\pi$ $\mathrm{rad}$]', ylabel=r'$\mathrm{S}$', xlim=(end_psi/np.pi, 0))
ax_3_3 = fig_3.add_subplot(413, xlabel=r'$\psi$ [$\pi$ $\mathrm{rad}$]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$', xlim=(end_psi/np.pi, 0))
ax_3_4 = fig_3.add_subplot(414, xlabel=r'$\psi$ [$\pi$ $\mathrm{rad}$]', ylabel=r'MLAT [deg]', xlim=(end_psi/np.pi, 0))

cmap_color = cm.turbo


def main(args):
    count_i = args

    mlat_rad_initial = initial_mlat_rad
    initial_mu = mu_init
    initial_theta = 0E0
    initial_energy, initial_pitch_angle_rad = kinetic_energy_and_pitch_angle_rad(initial_mu, mlat_rad_initial, initial_theta)
    initial_S = S_value(initial_energy, mlat_rad_initial, initial_pitch_angle_rad)
    initial_psi = intial_psi_array[count_i]

    mlat_rad_array_RK4 = np.array([mlat_rad_initial])
    theta_array_RK4 = np.array([initial_theta])
    psi_array_RK4 = np.array([initial_psi])
    time_array_RK4 = np.array([0E0])

    mlat_old = mlat_rad_initial
    theta_old = initial_theta
    psi_old = initial_psi
    time_old = 0E0

    now = datetime.datetime.now()
    print(now, r'calculation start', initial_pitch_angle_rad * 180E0 / np.pi, mlat_rad_initial * 180E0 / np.pi, theta_old, psi_old / np.pi, time_old, initial_S, initial_psi / np.pi)
    
    count_iteration = 0

    while True:
        count_iteration += 1
        mlat_new, theta_new, psi_new = RK4(mlat_old, theta_old, psi_old, initial_mu)
        time_new = time_old + dt
        mlat_rad_array_RK4 = np.append(mlat_rad_array_RK4, mlat_new)
        theta_array_RK4 = np.append(theta_array_RK4, theta_new)
        psi_array_RK4 = np.append(psi_array_RK4, psi_new)
        time_array_RK4 = np.append(time_array_RK4, time_new)

        if psi_new < end_psi or mlat_new < 0E0 or mlat_new > mlat_upper_limit_rad:
            break
        
        else:
            if psi_new != psi_new:
                print('Error: NaN')
                quit()
            mlat_old = mlat_new
            theta_old = theta_new
            psi_old = psi_new
            time_old = time_new
    
    mlat_deg_array_RK4 = mlat_rad_array_RK4 * 180E0 / np.pi
    energy_array_RK4 = np.zeros(len(mlat_rad_array_RK4))
    pitch_angle_rad_array_RK4 = np.zeros(len(mlat_rad_array_RK4))
    S_RK4 = np.zeros(len(mlat_rad_array_RK4))
    trapping_frequency_array_RK4 = np.zeros(len(mlat_rad_array_RK4))
    for count_j in range(len(mlat_rad_array_RK4)):
        energy_array_RK4[count_j], pitch_angle_rad_array_RK4[count_j] = kinetic_energy_and_pitch_angle_rad(initial_mu, mlat_rad_array_RK4[count_j], theta_array_RK4[count_j])
        S_RK4[count_j] = S_value(energy_array_RK4[count_j], mlat_rad_array_RK4[count_j], pitch_angle_rad_array_RK4[count_j])
        trapping_frequency_array_RK4[count_j] = trapping_frequency(mlat_rad_array_RK4[count_j])

    now = datetime.datetime.now()
    print(now, r'calculation finish', initial_pitch_angle_rad * 180E0 / np.pi, mlat_rad_initial * 180E0 / np.pi, energy_array_RK4[0] / elementary_charge, energy_array_RK4[-1] / elementary_charge, psi_array_RK4[0] / np.pi)

    return mlat_deg_array_RK4, energy_array_RK4, psi_array_RK4, time_array_RK4, pitch_angle_rad_array_RK4, initial_mu, theta_array_RK4, S_RK4, trapping_frequency_array_RK4

#ファイル作成(縦: trajectory_number, 横: 9)
data = np.zeros((trajectory_number, 9))

mlat_deg_array = np.linspace(0E0, mlat_upper_limit_deg, 1000)
mlat_rad_array = mlat_deg_array / 180E0 * np.pi

if __name__ == '__main__':
    num_process = 16

    args = range(trajectory_number)
    
    with Pool(num_process) as p:
        results = p.map(main, args)
    
    for result in results:
        mlat_deg_array_RK4 = result[0]
        energy_array_RK4 = result[1]
        psi_array_RK4 = result[2]
        time_array_RK4 = result[3]
        pitch_angle_rad_array_RK4 = result[4]
        initial_mu = result[5]
        theta_array_RK4 = result[6]
        S_RK4 = result[7]
        trapping_frequency_array_RK4 = result[8]

        pitch_angle_rad_array_RK4_eq = np.arcsin(np.sqrt(magnetic_flux_density(0E0) / magnetic_flux_density(mlat_deg_array_RK4 * np.pi / 180E0) * np.sin(pitch_angle_rad_array_RK4)**2E0))

        psi_color = psi_array_RK4[0] / np.pi * np.ones(len(mlat_deg_array_RK4))
        #print(psi_array_RK4[0] / np.pi, psi_array_RK4[-1] / np.pi, psi_color[0])

        ax_1.scatter(mlat_deg_array_RK4, energy_array_RK4 / elementary_charge, c=psi_color, alpha=0.3, cmap=cmap_color, vmin=-1E0, vmax=0E0, s=1)
        ax_1.scatter(mlat_deg_array_RK4[-1], energy_array_RK4[-1] / elementary_charge, c='black', s=50)

        ax_2_1.scatter(psi_array_RK4 / np.pi, energy_array_RK4 / elementary_charge, c=psi_color, alpha=0.3, cmap=cmap_color, vmin=-1E0, vmax=0E0, s=1)
        ax_2_1.scatter(psi_array_RK4[0] / np.pi, energy_array_RK4[0] / elementary_charge, c='black', s=50)
        ax_2_1.scatter(psi_array_RK4[-1] / np.pi, energy_array_RK4[-1] / elementary_charge, c='black', s=50)
        ax_2_2.scatter(psi_array_RK4 / np.pi, energy_array_RK4 * np.sin(pitch_angle_rad_array_RK4)**2E0 / elementary_charge, c=psi_color, alpha=0.3, cmap=cmap_color, vmin=-1E0, vmax=0E0, s=1)
        ax_2_2.scatter(psi_array_RK4[0] / np.pi, energy_array_RK4[0] * np.sin(pitch_angle_rad_array_RK4[0])**2E0 / elementary_charge, c='black', s=50)
        ax_2_2.scatter(psi_array_RK4[-1] / np.pi, energy_array_RK4[-1] * np.sin(pitch_angle_rad_array_RK4[-1])**2E0 / elementary_charge, c='black', s=50)
        ax_2_3.scatter(psi_array_RK4 / np.pi, energy_array_RK4 * np.cos(pitch_angle_rad_array_RK4)**2E0 / elementary_charge, c=psi_color, alpha=0.3, cmap=cmap_color, vmin=-1E0, vmax=0E0, s=1)
        ax_2_3.scatter(psi_array_RK4[0] / np.pi, energy_array_RK4[0] * np.cos(pitch_angle_rad_array_RK4[0])**2E0 / elementary_charge, c='black', s=50)
        ax_2_3.scatter(psi_array_RK4[-1] / np.pi, energy_array_RK4[-1] * np.cos(pitch_angle_rad_array_RK4[-1])**2E0 / elementary_charge, c='black', s=50)
        #ax_2_4.scatter(psi_array_RK4 / np.pi, pitch_angle_rad_array_RK4_eq / np.pi * 180E0, c=psi_color, alpha=0.3, cmap=cmap_color, vmin=-1E0, vmax=0E0, s=1)
        #ax_2_4.scatter(psi_array_RK4[0] / np.pi, pitch_angle_rad_array_RK4_eq[0] / np.pi * 180E0, c='black', s=0)
        #ax_2_4.scatter(psi_array_RK4[-1] / np.pi, pitch_angle_rad_array_RK4_eq[-1] / np.pi * 180E0, c='black', s=50)
        ax_2_4.scatter(psi_array_RK4 / np.pi, mlat_deg_array_RK4, c=psi_color, alpha=0.3, cmap=cmap_color, vmin=-1E0, vmax=0E0, s=1)
        ax_2_4.scatter(psi_array_RK4[0] / np.pi, mlat_deg_array_RK4[0], c='black', s=50)
        ax_2_4.scatter(psi_array_RK4[-1] / np.pi, mlat_deg_array_RK4[-1], c='black', s=50)

        ax_3_1.scatter(psi_array_RK4 / np.pi, trapping_frequency_array_RK4 / np.pi, c=psi_color, alpha=0.3, cmap=cmap_color, vmin=-1E0, vmax=0E0, s=1)
        ax_3_1.scatter(psi_array_RK4[0] / np.pi, trapping_frequency_array_RK4[0] / np.pi, c='black', s=50)
        ax_3_1.scatter(psi_array_RK4[-1] / np.pi, trapping_frequency_array_RK4[-1] / np.pi, c='black', s=50)
        ax_3_2.scatter(psi_array_RK4 / np.pi, S_RK4, c=psi_color, alpha=0.3, cmap=cmap_color, vmin=-1E0, vmax=0E0, s=1)
        ax_3_2.scatter(psi_array_RK4[0] / np.pi, S_RK4[0], c='black', s=50)
        ax_3_2.scatter(psi_array_RK4[-1] / np.pi, S_RK4[-1], c='black', s=50)
        ax_3_3.scatter(psi_array_RK4 / np.pi, theta_array_RK4 / trapping_frequency_array_RK4 / 2E0, c=psi_color, alpha=0.3, cmap=cmap_color, vmin=-1E0, vmax=0E0, s=1)
        ax_3_3.scatter(psi_array_RK4[0] / np.pi, theta_array_RK4[0] / trapping_frequency_array_RK4[0] / 2E0, c='black', s=50)
        ax_3_3.scatter(psi_array_RK4[-1] / np.pi, theta_array_RK4[-1] / trapping_frequency_array_RK4[-1] / 2E0, c='black', s=50)
        ax_3_4.scatter(psi_array_RK4 / np.pi, mlat_deg_array_RK4, c=psi_color, alpha=0.3, cmap=cmap_color, vmin=-1E0, vmax=0E0, s=1)
        ax_3_4.scatter(psi_array_RK4[0] / np.pi, mlat_deg_array_RK4[0], c='black', s=50)
        ax_3_4.scatter(psi_array_RK4[-1] / np.pi, mlat_deg_array_RK4[-1], c='black', s=50)
        
        
        #ax_1.scatter(mlat_deg_array_RK4, energy_array_RK4 / elementary_charge, c=time_array_RK4, alpha=0.5, cmap=cm.turbo, vmin=0E0, vmax=1.5E0, s=1)
        #ax_1_psi.plot(time_array_RK4, psi_array_RK4 / np.pi, linewidth=4, alpha=0.5, color='black')
        #dataに値を入れる
        data[results.index(result), 0] = mlat_deg_array_RK4[0]
        data[results.index(result), 1] = energy_array_RK4[0] / elementary_charge
        data[results.index(result), 2] = psi_array_RK4[0] / np.pi
        data[results.index(result), 3] = pitch_angle_rad_array_RK4[0] / np.pi * 180E0
        data[results.index(result), 4] = mlat_deg_array_RK4[-1]
        data[results.index(result), 5] = energy_array_RK4[-1] / elementary_charge
        data[results.index(result), 6] = psi_array_RK4[-1] / np.pi
        data[results.index(result), 7] = pitch_angle_rad_array_RK4[-1] / np.pi * 180E0
        data[results.index(result), 8] = time_array_RK4[-1]

ylim_ax_2_4 = ax_2_4.get_ylim()

#loss_cone_pitch_angle_eq_array_deg = np.arcsin(np.sqrt(magnetic_flux_density(0E0) / magnetic_flux_density(mlat_upper_limit_rad))) * 180E0 / np.pi
#ax_2_4.axhline(y=loss_cone_pitch_angle_eq_array_deg, color='black', linestyle='--', linewidth=4, alpha=0.5)

ax_1.set_title(r'$K_{\perp}(\lambda \, = \, 0) \, = \,$' + '{:.2f}'.format(initial_mu * magnetic_flux_density(0E0) / elementary_charge) + r' [$\mathrm{eV}$]')

xlim_enlarged = ax_1.get_xlim()
ylim_enlarged = ax_1.get_ylim()

energy_perp_array = initial_mu * magnetic_flux_density(mlat_rad_array) / elementary_charge
ax_1.plot(mlat_deg_array, energy_perp_array, color='orange', linewidth=4, label=r'$K_{\perp}$', alpha=0.6)


energy_wave_phase_speed_eV = energy_wave_phase_speed(mlat_rad_array) / elementary_charge
energy_wave_potential_eV = energy_wave_potential(mlat_rad_array) / elementary_charge * np.ones(len(mlat_deg_array))

energy_S_1_upper_limit_eV = np.zeros(len(mlat_deg_array))
energy_S_1_lower_limit_eV = np.zeros(len(mlat_deg_array))
for count_i in range(len(mlat_deg_array)):
    energy_S_1_upper_limit_eV[count_i] = energy_wave_potential_eV[count_i] / delta(mlat_rad_array[count_i])
    energy_S_1_lower_limit_eV[count_i] = energy_wave_potential_eV[count_i] / (delta(mlat_rad_array[count_i]) + epsilon(mlat_rad_array[count_i]))

ax_1.plot(mlat_deg_array, energy_wave_phase_speed_eV, color='red', linewidth=4, label=r'$K_{\mathrm{ph \parallel}}$', alpha=0.6)
ax_1.plot(mlat_deg_array, energy_wave_potential_eV, color='green', linewidth=4, label=r'$K_{\mathrm{E}}$', alpha=0.6)
ax_1.plot(mlat_deg_array, energy_S_1_upper_limit_eV, color='blue', linewidth=4, label=r'$S = 1$ range', alpha=0.6)
ax_1.plot(mlat_deg_array, energy_S_1_lower_limit_eV, color='blue', linewidth=4, alpha=0.6)

ax_1.minorticks_on()
ax_1.grid(which='both', alpha=0.3)

ax_1.legend()

#ax_1_psi.minorticks_on()
#ax_1_psi.grid(which='both', alpha=0.3)

norm = mpl.colors.Normalize(vmin=-1E0, vmax=0E0)
sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
sm.set_array([])
cbar_1 = fig_1.colorbar(sm, ticks=np.linspace(-1, 0, 11))
cbar_1.set_label(r'$\psi_{\mathrm{i}}$ [$\pi$ rad]')

ax_1.set_xlim(xlim_enlarged)
ax_1.set_ylim(ylim_enlarged)
fig_1.tight_layout()

fig_2.suptitle(r'$K_{\perp}(\lambda \, = \, 0) \, = \,$' + '{:.2f}'.format(initial_mu * magnetic_flux_density(0E0) / elementary_charge) + r' [$\mathrm{eV}$]')
ax_2_1.minorticks_on()
ax_2_1.grid(which='both', alpha=0.3)
ax_2_2.minorticks_on()
ax_2_2.grid(which='both', alpha=0.3)
ax_2_3.minorticks_on()
ax_2_3.grid(which='both', alpha=0.3)
ax_2_4.minorticks_on()
ax_2_4.grid(which='both', alpha=0.3)
ylim_ax_2_3 = ax_2_3.get_ylim()
if ylim_ax_2_3[0] < 1E0:
    ax_2_3.set_ylim(1E0, ylim_ax_2_3[1])
#ax_2_4.set_ylim(ylim_ax_2_4)

#cbar_2 = fig_2.colorbar(sm, ticks=np.linspace(-1, 0, 11))
#cbar_2.set_label(r'$\psi_{\mathrm{i}}$ [$\pi$ rad]')
fig_2.tight_layout()

fig_3.suptitle(r'$K_{\perp}(\lambda \, = \, 0) \, = \,$' + '{:.2f}'.format(initial_mu * magnetic_flux_density(0E0) / elementary_charge) + r' [$\mathrm{eV}$]')
ax_3_1.minorticks_on()
ax_3_1.grid(which='both', alpha=0.3)
ax_3_2.minorticks_on()
ax_3_2.grid(which='both', alpha=0.3)
ax_3_3.minorticks_on()
ax_3_3.grid(which='both', alpha=0.3)
ax_3_4.minorticks_on()
ax_3_4.grid(which='both', alpha=0.3)

fig_3.tight_layout()


#フォルダがなければ作成
if not os.path.exists(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_trajectory_S_not_constant_initialpsi_constant/end_psi_{(end_psi/np.pi):.2f}'):
    os.makedirs(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_trajectory_S_not_constant_initialpsi_constant/end_psi_{(end_psi/np.pi):.2f}')

fig_1.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_trajectory_S_not_constant_initialpsi_constant/end_psi_{(end_psi/np.pi):.2f}/energy_trajectory_S_not_constant_initialpsi_constant_{initial_pitch_angle_deg:.2f}_Earth_L_9_enlarged.png', dpi=100)
fig_1.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_trajectory_S_not_constant_initialpsi_constant/end_psi_{(end_psi/np.pi):.2f}/energy_trajectory_S_not_constant_initialpsi_constant_{initial_pitch_angle_deg:.2f}_Earth_L_9_enlarged.pdf', dpi=100)

ax_1.set_xlim(0E0, mlat_upper_limit_deg)
ax_1.set_ylim(1E1, 1E5)
fig_1.tight_layout()

fig_1.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_trajectory_S_not_constant_initialpsi_constant/end_psi_{(end_psi/np.pi):.2f}/energy_trajectory_S_not_constant_initialpsi_constant_{initial_pitch_angle_deg:.2f}_Earth_L_9.png', dpi=100)
fig_1.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_trajectory_S_not_constant_initialpsi_constant/end_psi_{(end_psi/np.pi):.2f}/energy_trajectory_S_not_constant_initialpsi_constant_{initial_pitch_angle_deg:.2f}_Earth_L_9.pdf', dpi=100)

fig_2.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_trajectory_S_not_constant_initialpsi_constant/end_psi_{(end_psi/np.pi):.2f}/energy_trajectory_S_not_constant_initialpsi_constant_{initial_pitch_angle_deg:.2f}_Earth_L_9_psi_vs_energy.png', dpi=100)
fig_2.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_trajectory_S_not_constant_initialpsi_constant/end_psi_{(end_psi/np.pi):.2f}/energy_trajectory_S_not_constant_initialpsi_constant_{initial_pitch_angle_deg:.2f}_Earth_L_9_psi_vs_energy.pdf', dpi=100)

fig_3.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_trajectory_S_not_constant_initialpsi_constant/end_psi_{(end_psi/np.pi):.2f}/energy_trajectory_S_not_constant_initialpsi_constant_{initial_pitch_angle_deg:.2f}_Earth_L_9_psi_vs_S_omegat.png', dpi=100)
fig_3.savefig(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_trajectory_S_not_constant_initialpsi_constant/end_psi_{(end_psi/np.pi):.2f}/energy_trajectory_S_not_constant_initialpsi_constant_{initial_pitch_angle_deg:.2f}_Earth_L_9_psi_vs_S_omegat.pdf', dpi=100)

#dataをcsvファイルとして出力
np.savetxt(f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_trajectory_S_not_constant_initialpsi_constant/end_psi_{(end_psi/np.pi):.2f}/energy_trajectory_S_not_constant_initialpsi_constant_{initial_pitch_angle_deg:.2f}_Earth_L_9.csv', data, delimiter=',')

#plt.show()