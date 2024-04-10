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
plt.rcParams["font.size"] = 55

initial_Kperp_eq_min_eV = 1E0
initial_Kperp_eq_max_eV = 1.1E1
initial_Kperp_eq_mesh_number = 11

initial_S_value_min = 1E-2
initial_S_value_max = 1E0
initial_S_value_mesh_number = 50

separate_number_psi = 30

grid_scale = 'linear'

select_upper_energy_ionospheric_end_eV = 11E3
select_lower_energy_ionospheric_end_eV = 10E3

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/test_particle_simulation_detrapped_phase_edgedefinition/auroral_precipitation'
file_path = f'{dir_name}/Kperp_eq_{initial_Kperp_eq_min_eV:.4f}_{initial_Kperp_eq_max_eV:.4f}_eV_{initial_Kperp_eq_mesh_number}_S_{initial_S_value_min:.4f}_{initial_S_value_max:.4f}_{initial_S_value_mesh_number}_{separate_number_psi}_{grid_scale}.csv'
figure_name = os.path.basename(file_path).replace('.csv', f'_select_{select_lower_energy_ionospheric_end_eV:.1f}_{select_upper_energy_ionospheric_end_eV:.1f}_eV.png')

data = np.loadtxt(file_path, delimiter=',', skiprows=1)

# Kperp_eq, S_value, psi, capital_theta, mlat_rad, energy_perp_eV, energy_para_eV, energy_eV, energy_perp_ionospheric_end_eV, energy_para_ionospheric_end_eV, energy_ionospheric_end_eV
# Kperp_eq = 1のみを抽出する。
#data = data[data[:, 0] == 1.0, :]

data_initial_Kperp_eq = data[:, 0]
data_initial_S_value = data[:, 1]
data_initial_psi = data[:, 2]
data_initial_capital_theta = data[:, 3]
data_initial_mlat_rad = data[:, 4]
data_initial_energy_perp_eV = data[:, 5]
data_initial_energy_para_eV = data[:, 6]
data_initial_energy_eV = data[:, 7]
data_energy_perp_ionospheric_end_eV = data[:, 8]
data_energy_para_ionospheric_end_eV = data[:, 9]
data_energy_ionospheric_end_eV = data[:, 10]

B_ratio = data_energy_perp_ionospheric_end_eV / data_initial_Kperp_eq
B_ratio_constant = np.mean(B_ratio)


# select data
# select_lower_energy_ionospheric_end_eV <= data_energy_ionospheric_end_eV <= select_upper_energy_ionospheric_end_eV
data_initial_Kperp_eq_select = data_initial_Kperp_eq[(select_lower_energy_ionospheric_end_eV <= data_energy_ionospheric_end_eV) & (data_energy_ionospheric_end_eV <= select_upper_energy_ionospheric_end_eV)]
data_initial_S_value_select = data_initial_S_value[(select_lower_energy_ionospheric_end_eV <= data_energy_ionospheric_end_eV) & (data_energy_ionospheric_end_eV <= select_upper_energy_ionospheric_end_eV)]
data_initial_psi_select = data_initial_psi[(select_lower_energy_ionospheric_end_eV <= data_energy_ionospheric_end_eV) & (data_energy_ionospheric_end_eV <= select_upper_energy_ionospheric_end_eV)]
data_initial_capital_theta_select = data_initial_capital_theta[(select_lower_energy_ionospheric_end_eV <= data_energy_ionospheric_end_eV) & (data_energy_ionospheric_end_eV <= select_upper_energy_ionospheric_end_eV)]
data_initial_mlat_rad_select = data_initial_mlat_rad[(select_lower_energy_ionospheric_end_eV <= data_energy_ionospheric_end_eV) & (data_energy_ionospheric_end_eV <= select_upper_energy_ionospheric_end_eV)]

print(f'number of data: {len(data_initial_Kperp_eq_select)}')

##### test particle simulation #####

number_parallel = os.cpu_count()

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
    plasma_beta_ion_now = plasma_beta_ion(mlat_rad)
    d_S_d_t_now = d_S_d_t(mu, theta, psi, mlat_rad)
    delta_now = delta(mlat_rad)
    if S_value_now <= 1E0 and S_value_now >= 0E0:
        function_phase_trapping = (theta / 2E0 / trapping_frequency_now)**2E0 - 5E-1 * (np.cos(psi) + np.sqrt(1E0 - S_value_now**2E0) - S_value_now * (psi + np.pi - np.arcsin(S_value_now)))
        function_phase_trapping_time_derivative = (theta / trapping_frequency_now)**2E0 * (plasma_beta_ion_now * (1E0 + tau(mlat_rad)) + tau(mlat_rad)) / (plasma_beta_ion_now * (1E0 + tau(mlat_rad)) + 2E0 * tau(mlat_rad)) * (theta + wave_frequency) * delta_now + 5E-1 * (psi + np.pi - np.arcsin(S_value_now)) * d_S_d_t_now
        function_saddle_point = psi + np.pi - np.arcsin(S_value_now)
    else:
        function_phase_trapping = np.nan
        function_phase_trapping_time_derivative = np.nan
        function_saddle_point = np.nan
    function_resonant_scattering = (theta / 2E0 / trapping_frequency_now)**2E0 - 5E-1 * (np.cos(psi) + S_value_now * (np.pi - psi) + 1E0)

    #characterize the region
    if function_resonant_scattering > 0E0:
        region = 2.4 + time_now / time_end
    else:
        if S_value_now <= 1E0 and S_value_now >= 0E0:
            if function_phase_trapping <= 0E0 and function_saddle_point >= 0E0:
                region = 0 + time_now / time_end
            else:
                region = 1.2 + time_now / time_end
        else:
            region = 1.2 + time_now / time_end
    
    return function_phase_trapping, function_phase_trapping_time_derivative, function_saddle_point, function_resonant_scattering, region



# runge-kutta method
def S_value_for_TPS(mu, theta, mlat_rad):
    return energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad) * (1E0 + Gamma(mlat_rad)) * delta(mlat_rad) * (1E0 + theta / wave_frequency)**2E0 + magnetic_flux_density(mlat_rad) * mu / energy_wave_potential(mlat_rad) * delta(mlat_rad)    #[]

def d_psi_d_t(theta):
    return theta    #[rad/s]

def d_theta_d_t(mu, theta, psi, mlat_rad):
    return - trapping_frequency(mlat_rad)**2E0 * (np.sin(psi) + S_value_for_TPS(mu, theta, mlat_rad))    #[rad/s^2]

def d_mlat_rad_d_t(theta, mlat_rad):
    return (theta + wave_frequency) / kpara(mlat_rad) / r_eq / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)    #[rad/s]

dt = 1E-3    #[s]
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


# simulation
def main(args):
    count_i = args[0]

    initial_Kperp_eq = data_initial_Kperp_eq_select[count_i]
    initial_S_value = data_initial_S_value_select[count_i]
    initial_psi = data_initial_psi_select[count_i]
    initial_capital_theta = data_initial_capital_theta_select[count_i]
    initial_mlat_rad = data_initial_mlat_rad_select[count_i]

    # initial condition
    mu_value = initial_Kperp_eq * elementary_charge / magnetic_flux_density(initial_mlat_rad)
    initial_theta = 2E0 * trapping_frequency(initial_mlat_rad) * initial_capital_theta

    time = 0E0

    # data_array
    mlat_rad_array = np.array([initial_mlat_rad])
    theta_array = np.array([initial_theta])
    psi_array = np.array([initial_psi])
    time_array = np.array([time])
    vpara_array = np.array([vpara(initial_theta, initial_mlat_rad)])
    Kperp_energy_array = np.array([Kperp_energy(mu_value, initial_mlat_rad)])
    Kpara_energy_array = np.array([Kpara_energy(initial_theta, initial_mlat_rad)])
    Ktotal_energy_array = np.array([Ktotal_energy(mu_value, initial_theta, initial_mlat_rad)])
    d_Ktotal_d_t_array = np.array([d_Ktotal_d_t(initial_theta, initial_psi, initial_mlat_rad)])
    trapping_frequency_array = np.array([trapping_frequency(initial_mlat_rad)])
    S_value_array = np.array([S_value(mu_value, initial_theta, initial_mlat_rad)])

    # iteration
    mlat_rad_0 = initial_mlat_rad
    theta_0 = initial_theta
    psi_0 = initial_psi

    while True:
        mlat_rad_1, theta_1, psi_1 = RK4(mlat_rad_0, theta_0, psi_0, mu_value)
        time += dt

        mlat_rad_array = np.append(mlat_rad_array, mlat_rad_1)
        theta_array = np.append(theta_array, theta_1)
        psi_array = np.append(psi_array, psi_1)
        time_array = np.append(time_array, time)
        vpara_array = np.append(vpara_array, vpara(theta_1, mlat_rad_1))
        Kperp_energy_array = np.append(Kperp_energy_array, Kperp_energy(mu_value, mlat_rad_1))
        Kpara_energy_array = np.append(Kpara_energy_array, Kpara_energy(theta_1, mlat_rad_1))
        Ktotal_energy_array = np.append(Ktotal_energy_array, Ktotal_energy(mu_value, theta_1, mlat_rad_1))
        d_Ktotal_d_t_array = np.append(d_Ktotal_d_t_array, d_Ktotal_d_t(theta_1, psi_1, mlat_rad_1))
        trapping_frequency_array = np.append(trapping_frequency_array, trapping_frequency(mlat_rad_1))
        S_value_array = np.append(S_value_array, S_value(mu_value, theta_1, mlat_rad_1))

        mlat_rad_0 = mlat_rad_1
        theta_0 = theta_1
        psi_0 = psi_1

        if mlat_rad_1 >= mlat_upper_limit_rad:
            break

        if psi_1 != psi_1:
            print(r'Error!: psi is nan')
            quit()
        else:
            mlat_rad_0 = mlat_rad_1
            theta_0 = theta_1
            psi_0 = psi_1
    
    return mlat_rad_array, theta_array, psi_array, time_array, vpara_array, Kperp_energy_array, Kpara_energy_array, Ktotal_energy_array, d_Ktotal_d_t_array, trapping_frequency_array, S_value_array


# plot
fig = plt.figure(figsize=(50, 40), dpi=100)
gs = fig.add_gridspec(7, 3, height_ratios=[1, 1, 1, 1, 1, 1, 0.05])

ax_1 = fig.add_subplot(gs[0:3, 0], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$v_{\parallel}$ [$c$]')
ax_2 = fig.add_subplot(gs[3:6, 0], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'Kinetic energy $K$ [eV]', yscale='log')
ax_3 = fig.add_subplot(gs[0:2, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$', xlim=[-1, 1])
ax_4 = fig.add_subplot(gs[2:4, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$S$', yscale='log', xlim=[-1, 1])
#ax_5 = fig.add_subplot(gs[4:6, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'MLAT $\lambda$ [deg]', xlim=[-1, 1])
ax_5 = fig.add_subplot(gs[4:6, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$t$ [$\mathrm{s}$]', xlim=[-1, 1])
ax_6 = fig.add_subplot(gs[0:2, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$K$ [$\mathrm{eV}$]', xlim=[-1, 1])
#ax_7 = fig.add_subplot(gs[2:4, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$1 + \omega / \theta$', xlim=[-1, 1])
ax_7 = fig.add_subplot(gs[2:4, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\mathrm{d} K / \mathrm{d} t$ [$\mathrm{eV/s}$]', xlim=[-1, 1])
ax_8 = fig.add_subplot(gs[4:6, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'MLAT $\lambda$ [deg]', xlim=[-1, 1])

cmap_color = cm.get_cmap('turbo')
color_target = data_initial_psi_select / np.pi
vmin = np.min(color_target)
vmax = np.max(color_target)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=norm)
sm.set_array([])

cbarax = fig.add_subplot(gs[6, :])
cbar = fig.colorbar(sm, cax=cbarax, orientation='horizontal')
cbar.set_label(r'$\psi_{\mathrm{i}}$ [$\pi$ rad]')

if __name__ == '__main__':
    num_process = os.cpu_count()

    args = []
    for count_i in range(len(data_initial_Kperp_eq_select)):
        args.append([count_i])
    
    results = []
    with Pool(num_process) as p:
        results = p.map(main, args)
    
    for count_parallel in range(len(results)):
        mlat_rad_array = results[count_parallel][0]
        theta_array = results[count_parallel][1]
        psi_array = results[count_parallel][2]
        time_array = results[count_parallel][3]
        vpara_array = results[count_parallel][4]
        Kperp_energy_array = results[count_parallel][5]
        Kpara_energy_array = results[count_parallel][6]
        Ktotal_energy_array = results[count_parallel][7]
        d_Ktotal_d_t_array = results[count_parallel][8]
        trapping_frequency_array = results[count_parallel][9]
        S_value_array = results[count_parallel][10]

        # plot
        ax_1.scatter(mlat_rad_array * 180E0 / np.pi, vpara_array / speed_of_light, c=color_target[count_parallel]*np.ones_like(mlat_rad_array), cmap=cmap_color, vmin=vmin, vmax=vmax, s=1E0)
        ax_1.scatter(mlat_rad_array[0] * 180E0 / np.pi, vpara_array[0] / speed_of_light, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
        ax_1.scatter(mlat_rad_array[-1] * 180E0 / np.pi, vpara_array[-1] / speed_of_light, c='orange', s=200, marker='D', edgecolors='k', zorder=1)

        ax_2.scatter(mlat_rad_array * 180E0 / np.pi, Ktotal_energy_array / elementary_charge, c=color_target[count_parallel]*np.ones_like(mlat_rad_array), cmap=cmap_color, vmin=vmin, vmax=vmax, s=1E0)
        ax_2.scatter(mlat_rad_array[0] * 180E0 / np.pi, Ktotal_energy_array[0] / elementary_charge, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
        ax_2.scatter(mlat_rad_array[-1] * 180E0 / np.pi, Ktotal_energy_array[-1] / elementary_charge, c='orange', s=200, marker='D', edgecolors='k', zorder=1)

        ax_3.scatter(psi_array / np.pi, theta_array / 2E0 / trapping_frequency_array, c=color_target[count_parallel]*np.ones_like(mlat_rad_array), cmap=cmap_color, vmin=vmin, vmax=vmax, s=1E0)
        ax_3.scatter(psi_array[0] / np.pi, theta_array[0] / 2E0 / trapping_frequency_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
        ax_3.scatter(psi_array[-1] / np.pi, theta_array[-1] / 2E0 / trapping_frequency_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=1)

        ax_4.scatter(psi_array / np.pi, S_value_array, c=color_target[count_parallel]*np.ones_like(mlat_rad_array), cmap=cmap_color, vmin=vmin, vmax=vmax, s=1E0)
        ax_4.scatter(psi_array[0] / np.pi, S_value_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
        ax_4.scatter(psi_array[-1] / np.pi, S_value_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=1)

        #ax_5.scatter(psi_array / np.pi, mlat_rad_array * 180E0 / np.pi, c=color_target[count_parallel]*np.ones_like(mlat_rad_array), cmap=cmap_color, vmin=vmin, vmax=vmax, s=1E0)
        #ax_5.scatter(psi_array[0] / np.pi, mlat_rad_array[0] * 180E0 / np.pi, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
        #ax_5.scatter(psi_array[-1] / np.pi, mlat_rad_array[-1] * 180E0 / np.pi, c='orange', s=200, marker='D', edgecolors='k', zorder=1)
        ax_5.scatter(psi_array / np.pi, time_array, c=color_target[count_parallel]*np.ones_like(mlat_rad_array), cmap=cmap_color, vmin=vmin, vmax=vmax, s=1E0)
        ax_5.scatter(psi_array[0] / np.pi, time_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
        ax_5.scatter(psi_array[-1] / np.pi, time_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=1)

        ax_6.scatter(psi_array / np.pi, Ktotal_energy_array / elementary_charge, c=color_target[count_parallel]*np.ones_like(mlat_rad_array), cmap=cmap_color, vmin=vmin, vmax=vmax, s=1E0)
        ax_6.scatter(psi_array[0] / np.pi, Ktotal_energy_array[0] / elementary_charge, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
        ax_6.scatter(psi_array[-1] / np.pi, Ktotal_energy_array[-1] / elementary_charge, c='orange', s=200, marker='D', edgecolors='k', zorder=1)

        ax_7.scatter(psi_array / np.pi, d_Ktotal_d_t_array / elementary_charge, c=color_target[count_parallel]*np.ones_like(mlat_rad_array), cmap=cmap_color, vmin=vmin, vmax=vmax, s=1E0)
        ax_7.scatter(psi_array[0] / np.pi, d_Ktotal_d_t_array[0] / elementary_charge, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
        ax_7.scatter(psi_array[-1] / np.pi, d_Ktotal_d_t_array[-1] / elementary_charge, c='orange', s=200, marker='D', edgecolors='k', zorder=1)
        #ax_7.scatter(psi_array / np.pi, 1E0 + wave_frequency / theta_array, c=color_target[count_parallel]*np.ones_like(mlat_rad_array), cmap=cmap_color, vmin=vmin, vmax=vmax, s=1E0)
        #ax_7.scatter(psi_array[0] / np.pi, 1E0 + wave_frequency / theta_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
        #ax_7.scatter(psi_array[-1] / np.pi, 1E0 + wave_frequency / theta_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=1)

        ax_8.scatter(psi_array / np.pi, mlat_rad_array * 180E0 / np.pi, c=color_target[count_parallel]*np.ones_like(mlat_rad_array), cmap=cmap_color, vmin=vmin, vmax=vmax, s=1E0)
        ax_8.scatter(psi_array[0] / np.pi, mlat_rad_array[0] * 180E0 / np.pi, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=1)
        ax_8.scatter(psi_array[-1] / np.pi, mlat_rad_array[-1] * 180E0 / np.pi, c='orange', s=200, marker='D', edgecolors='k', zorder=1)


axes = [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6, ax_7, ax_8]

mlat_deg_for_background = np.linspace(0.1E0, mlat_upper_limit_deg, 1000)
mlat_rad_for_background = mlat_deg_for_background * np.pi / 180E0
energy_wave_phase_speed_for_background = energy_wave_phase_speed(mlat_rad_for_background)
energy_wave_potential_for_background = energy_wave_potential(mlat_rad_for_background)
Vph_para_for_background = wave_phase_speed(mlat_rad_for_background)

xlim_enlarged_1 = ax_1.get_xlim()
ylim_enlarged_1 = ax_1.get_ylim()
xlim_enlarged_2 = ax_2.get_xlim()
ylim_enlarged_2 = ax_2.get_ylim()
ylim_enlarged_3 = ax_3.get_ylim()

ax_1.plot(mlat_deg_for_background, Vph_para_for_background / speed_of_light, c='r', linewidth=4E0, label=r'$V_{\mathrm{ph \parallel}}$', alpha=0.6)

ax_2.plot(mlat_deg_for_background, energy_wave_phase_speed_for_background / elementary_charge, c='r', linewidth=4E0, label=r'$K_{\mathrm{ph \parallel}}$', alpha=0.6)
ax_2.plot(mlat_deg_for_background, energy_wave_potential_for_background / elementary_charge * np.ones(len(mlat_deg_for_background)), c='g', linewidth=4E0, label=r'$K_{\mathrm{E}}$', alpha=0.6)

ax_1.set_xlim(xlim_enlarged_1)
ax_1.set_ylim(ylim_enlarged_1)
ax_2.set_xlim(xlim_enlarged_2)
ax_2.set_ylim(ylim_enlarged_2)

ylim_enlarged_3 = list(ylim_enlarged_3)
if ylim_enlarged_3[0] < -3E0:
    ylim_enlarged_3[0] = -3E0
if ylim_enlarged_3[1] > 1E0:
    ylim_enlarged_3[1] = 1E0
ax_3.set_ylim(tuple(ylim_enlarged_3))

ax_7.set_yscale('symlog', linthresh=1E0)
#ax_7.set_ylim([1E-1, 1E1])

for ax in axes:
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.text(-0.15, 1.0, '(' + chr(97 + axes.index(ax)) + ')', transform=ax.transAxes)

ax_1.legend()
ax_2.legend()

fig.suptitle(r'$K(\lambda = \lambda_{\mathrm{iono}}) \in$ [' + f'{select_lower_energy_ionospheric_end_eV:.1f}, {select_upper_energy_ionospheric_end_eV:.1f}' + r'] $\mathrm{eV}$')

fig.tight_layout(w_pad=0.3, h_pad=0.0)

fig_path = f'{dir_name}/{figure_name}'
fig.savefig(fig_path)
fig.savefig(fig_path.replace('.png', '.pdf'))
plt.close('all')