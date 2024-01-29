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
plt.rcParams["font.size"] = 40


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


# iteration

def h_function_plus(S_value, mlat_rad):
    return S_value - (np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)) + np.sqrt(2E0) * np.sqrt(np.sqrt(1E0 - S_value**2E0) - S_value * (np.pi / 2E0 - np.arcsin(S_value))))**2E0 * (1E0 + Gamma(mlat_rad)) * delta(mlat_rad)   #[]

def h_function_minus(S_value, mlat_rad):
    return S_value - (np.sqrt(energy_wave_phase_speed(mlat_rad) / energy_wave_potential(mlat_rad)) - np.sqrt(2E0) * np.sqrt(np.sqrt(1E0 - S_value**2E0) - S_value * (np.pi / 2E0 - np.arcsin(S_value))))**2E0 * (1E0 + Gamma(mlat_rad)) * delta(mlat_rad)   #[]

def vpara_s_plus_max_function(S_value, mlat_rad):
    return wave_phase_speed(mlat_rad) + 2E0 * np.sqrt(energy_wave_potential(mlat_rad) / electron_mass) * np.sqrt(np.sqrt(1E0 - S_value**2E0) - S_value * (np.pi / 2E0 - np.arcsin(S_value))) #[m/s]

def vpara_s_minus_min_function(S_value, mlat_rad):
    return wave_phase_speed(mlat_rad) - 2E0 * np.sqrt(energy_wave_potential(mlat_rad) / electron_mass) * np.sqrt(np.sqrt(1E0 - S_value**2E0) - S_value * (np.pi / 2E0 - np.arcsin(S_value))) #[m/s]

def vpara_to_Kpara(vpara):
    return electron_mass / 2E0 * vpara**2E0 / elementary_charge #[eV]


# initial value
mlat_deg_array = np.linspace(0.01E0, 50E0, 5000)
mlat_rad_array = mlat_deg_array * np.pi / 180E0

diff_S = 1E-6

# iteration
def iteration(mlat_rad, sign):
    if sign == 1:
        h_function = h_function_plus
        vpara_function = vpara_s_plus_max_function
        S_initial_value = diff_S
    elif sign == -1:
        h_function = h_function_minus
        vpara_function = vpara_s_minus_min_function
        S_initial_value = diff_S
    elif sign == -2:
        if h_function_minus(1E0, mlat_rad) >= 0E0:
            return 1E0, wave_phase_speed(mlat_rad), vpara_to_Kpara(wave_phase_speed(mlat_rad))
        h_function = h_function_minus
        vpara_function = vpara_s_minus_min_function
        S_initial_value = 1E0 - diff_S
    else:
        print('Error: sign is not 1, -1 or -2')
        quit()
    
    S_value_old = S_initial_value
    iteration_count = 0
    while True:
        func_0 = h_function(S_value_old, mlat_rad)
        func_1 = (h_function(S_value_old + diff_rad, mlat_rad) - h_function(S_value_old - diff_rad, mlat_rad)) / 2E0 / diff_S
        diff = func_0 / func_1
        if np.abs(diff) > 1E-3:
            diff = np.sign(diff) * 1E-3
        S_value_new = S_value_old - diff
        if np.abs(S_value_new - S_value_old) < 1E-6:
            vpara = vpara_function(S_value_new, mlat_rad)
            Kpara = vpara_to_Kpara(vpara)
            return S_value_new, vpara, Kpara
        else:
            if S_value_new != S_value_new:
                #print('Error: S_value_new is nan')
                return np.nan, np.nan, np.nan
            if iteration_count > 10000:
                #print('Error: iteration_count > 10000')
                return np.nan, np.nan, np.nan
            S_value_old = S_value_new
            iteration_count += 1

def main(args):
    mlat_rad = args[0]
    sign_array = np.array([1, -1, -2])

    for count_sign in range(3):
        sign = sign_array[count_sign]
        if sign == 1:
            S_value_plus_min, vpara_s_plus_max, Kpara_s_plus = iteration(mlat_rad, sign)
        elif sign == -1:
            S_value_minus_min, vpara_s_minus_min, Kpara_s_minus_min = iteration(mlat_rad, sign)
        elif sign == -2:
            S_value_minus_max, vpara_s_minus_max, Kpara_s_minus_max = iteration(mlat_rad, sign)
        else:
            print('Error in main: sign is not 1, -1 or -2')
            quit()

    return S_value_plus_min, vpara_s_plus_max, Kpara_s_plus, S_value_minus_min, vpara_s_minus_min, Kpara_s_minus_min, S_value_minus_max, vpara_s_minus_max, Kpara_s_minus_max


S_value_plus_min_array = np.zeros_like(mlat_rad_array)
vpara_s_plus_max_array = np.zeros_like(mlat_rad_array)
S_value_minus_min_array = np.zeros_like(mlat_rad_array)
vpara_s_minus_min_array = np.zeros_like(mlat_rad_array)
S_value_minus_max_array = np.zeros_like(mlat_rad_array)
vpara_s_minus_max_array = np.zeros_like(mlat_rad_array)

Kpara_s_plus_max_array = np.zeros_like(mlat_rad_array)
Kpara_s_plus_min_array = np.zeros_like(mlat_rad_array)
Kpara_s_minus_max_array = np.zeros_like(mlat_rad_array)
Kpara_s_minus_min_array = np.zeros_like(mlat_rad_array)

if __name__ == '__main__':
    
    num_process = os.cpu_count()

    args = []
    for count_mlat in range(len(mlat_rad_array)):
        args.append([mlat_rad_array[count_mlat]])
    
    results = []
    with Pool(num_process) as p:
        results = p.map(main, args)
    
    for count_mlat in range(len(mlat_rad_array)):
        S_value_plus_min_array[count_mlat] = results[count_mlat][0]
        vpara_s_plus_max_array[count_mlat] = results[count_mlat][1]
        S_value_minus_min_array[count_mlat] = results[count_mlat][3]
        vpara_s_minus_min_array[count_mlat] = results[count_mlat][4]
        S_value_minus_max_array[count_mlat] = results[count_mlat][6]
        vpara_s_minus_max_array[count_mlat] = results[count_mlat][7]

        if vpara_s_plus_max_array[count_mlat] == vpara_s_plus_max_array[count_mlat]:
            Kpara_s_plus_max_array[count_mlat] = results[count_mlat][2]
            if vpara_s_minus_min_array[count_mlat] <= 0E0:
                Kpara_s_plus_min_array[count_mlat] = 0E0
                Kpara_s_minus_max_array[count_mlat] = results[count_mlat][5]
                Kpara_s_minus_min_array[count_mlat] = 0E0
            elif vpara_s_minus_min_array[count_mlat] > 0E0:
                Kpara_s_plus_min_array[count_mlat] = results[count_mlat][5]
                Kpara_s_minus_max_array[count_mlat] = np.nan
                Kpara_s_minus_min_array[count_mlat] = np.nan
            else:
                Kpara_s_minus_max_array[count_mlat] = np.nan
                Kpara_s_minus_min_array[count_mlat] = np.nan
                Kpara_s_plus_min_array[count_mlat] = np.nan
        else:
            if vpara_s_minus_max_array[count_mlat] == vpara_s_minus_max_array[count_mlat]:
                Kpara_s_plus_max_array[count_mlat] = results[count_mlat][8]
                if vpara_s_minus_min_array[count_mlat] <= 0E0:
                    Kpara_s_plus_min_array[count_mlat] = 0E0
                    Kpara_s_minus_max_array[count_mlat] = results[count_mlat][5]
                    Kpara_s_minus_min_array[count_mlat] = 0E0
                elif vpara_s_minus_min_array[count_mlat] > 0E0:
                    Kpara_s_plus_min_array[count_mlat] = results[count_mlat][5]
                    Kpara_s_minus_max_array[count_mlat] = np.nan
                    Kpara_s_minus_min_array[count_mlat] = np.nan
                else:
                    Kpara_s_minus_max_array[count_mlat] = np.nan
                    Kpara_s_minus_min_array[count_mlat] = np.nan
                    Kpara_s_plus_min_array[count_mlat] = np.nan
            else:
                Kpara_s_plus_max_array[count_mlat] = np.nan
                Kpara_s_plus_min_array[count_mlat] = np.nan
                Kpara_s_minus_max_array[count_mlat] = np.nan
                Kpara_s_minus_min_array[count_mlat] = np.nan

# plot

wave_phase_speed_array = np.zeros_like(mlat_rad_array)
energy_wave_phase_speed_eV_array = np.zeros_like(mlat_rad_array)
energy_wave_potential_eV_array = np.zeros_like(mlat_rad_array)
vpara_s_minus_array = np.zeros_like(mlat_rad_array)
for count_mlat in range(len(mlat_rad_array)):
    wave_phase_speed_array[count_mlat] = wave_phase_speed(mlat_rad_array[count_mlat])
    energy_wave_phase_speed_eV_array[count_mlat] = energy_wave_phase_speed(mlat_rad_array[count_mlat]) / elementary_charge
    energy_wave_potential_eV_array[count_mlat] = energy_wave_potential(mlat_rad_array[count_mlat]) / elementary_charge
    if vpara_s_minus_max_array[count_mlat] == vpara_s_minus_max_array[count_mlat]:
        vpara_s_minus_array[count_mlat] = np.min([vpara_s_minus_max_array[count_mlat], wave_phase_speed_array[count_mlat]])
    else:
        vpara_s_minus_array[count_mlat] = np.nan

fig = plt.figure(figsize=(20, 30), dpi=100)
fig.suptitle(r'$\mu = 0, \psi = \psi_{\mathrm{s}}$')

ax_1 = fig.add_subplot(211, xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$v_{\parallel}$ [$c$]', xlim=(0E0, 50))

ax_1.fill_between(mlat_deg_array, wave_phase_speed_array / speed_of_light, vpara_s_plus_max_array / speed_of_light, facecolor='yellow', alpha=0.2)
ax_1.fill_between(mlat_deg_array, vpara_s_minus_min_array / speed_of_light, vpara_s_minus_array / speed_of_light, facecolor='aqua', alpha=0.2)

ax_1.plot(mlat_deg_array, vpara_s_plus_max_array / speed_of_light, c='orange', linewidth=4, label=r'$v_{\parallel \mathrm{s} + \mathrm{max}}$', alpha=0.6)
ax_1.plot(mlat_deg_array, vpara_s_minus_max_array / speed_of_light, c='green', linewidth=4, label=r'$v_{\parallel \mathrm{s} - \mathrm{max}}$', alpha=0.6)
ax_1.plot(mlat_deg_array, vpara_s_minus_min_array / speed_of_light, c='dodgerblue', linewidth=4, label=r'$v_{\parallel \mathrm{s} - \mathrm{min}}$', alpha=0.6)

ax_1_ylim = ax_1.get_ylim()
ax_1.plot(mlat_deg_array, wave_phase_speed_array / speed_of_light, c='r', linewidth=4, label=r'$V_{\mathrm{ph} \parallel}$', alpha=0.6)
ax_1.hlines(0E0, 0E0, mlat_upper_limit_deg, color='k', linewidth=4, alpha=0.3)
ax_1.set_ylim(ax_1_ylim)

ax_1.legend()
ax_1.minorticks_on()
ax_1.grid(which='both', alpha=0.3)

ax_2 = fig.add_subplot(212, xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$K_{\parallel}$ [eV]', xlim=(0E0, 50), ylim=(1E1, 1E5), yscale='log')

ax_2.fill_between(mlat_deg_array, Kpara_s_plus_max_array, Kpara_s_plus_min_array, facecolor='plum', alpha=0.2)
ax_2.fill_between(mlat_deg_array, Kpara_s_minus_max_array, Kpara_s_minus_min_array, facecolor='cornflowerblue', alpha=0.2)

ax_2.plot(mlat_deg_array, Kpara_s_plus_max_array, c='deeppink', linewidth=4, label=r'$K_{\parallel \mathrm{lim} (v_{\parallel}>0)}$', alpha=0.6)
ax_2.plot(mlat_deg_array, Kpara_s_plus_min_array, c='deeppink', linewidth=4, alpha=0.6)
ax_2.plot(mlat_deg_array, Kpara_s_minus_max_array, c='blue', linewidth=4, label=r'$K_{\parallel \mathrm{lim} (v_{\parallel}<0)}$', alpha=0.6)
ax_2.plot(mlat_deg_array, Kpara_s_minus_min_array, c='blue', linewidth=4, alpha=0.6)

ax_2.plot(mlat_deg_array, energy_wave_phase_speed_eV_array, c='r', linewidth=4, label=r'$K_{\mathrm{ph} \parallel}$', alpha=0.6)
ax_2.plot(mlat_deg_array, energy_wave_potential_eV_array, c='green', linewidth=4, label=r'$K_{\mathrm{E}}$', alpha=0.6)

ax_2.legend()
ax_2.minorticks_on()
ax_2.grid(which='both', alpha=0.3)



#ax_3 = fig.add_subplot(313, xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$S$', xlim=(0E0, 50), yscale='log', ylim=(1E-3, 1.1E0))
#
#ax_3.plot(mlat_deg_array, S_value_plus_min_array, c='orange', linewidth=4, label=r'$S_{+ \mathrm{min}}$', alpha=0.6)
#ax_3.plot(mlat_deg_array, S_value_minus_max_array, c='green', linewidth=4, label=r'$S_{- \mathrm{max}}$', alpha=0.6)
#ax_3.plot(mlat_deg_array, S_value_minus_min_array, c='dodgerblue', linewidth=4, label=r'$S_{- \mathrm{min}}$', alpha=0.6)
#
#ax_3.legend()
#ax_3.minorticks_on()
#ax_3.grid(which='both', alpha=0.3)

plt.tight_layout()

dir_path = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/energy_limit_trapping_S_not_constant'
os.makedirs(dir_path, exist_ok=True)
fig_path = f'{dir_path}/Earth_L_{lshell_number:.1f}'
plt.savefig(f'{fig_path}.png')
plt.savefig(f'{fig_path}.pdf')
plt.close()