import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# magnetic latitude
mlat_max_deg = 50.0   # [deg]
mlat_min_deg = 0.0    # [deg]
mlat_deg = np.linspace(mlat_min_deg, mlat_max_deg, 10000)   # [deg]
mlat_rad = mlat_deg * np.pi / 180.0   # [rad]


# constants
speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]
electric_constant = 8.8541878128E-12  #[F m-1]
magnetic_constant = 1.25663706212E-6  #[N A-2]

# magnetic field
planet_radius   = 7.1492E7  #[m]
lshell_number   = 5.91E0
r_eq            = planet_radius * lshell_number #[m]
dipole_moment   = 4.2E-4 * (4E0 * np.pi * planet_radius**3E0) / magnetic_constant #[Am]
B0_eq           = (1E-7 * dipole_moment) / r_eq**3E0

b0 = B0_eq / np.cos(mlat_rad)**6E0 * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0)     #[T]
db0_dz = 3E0 * np.sin(mlat_rad) * (5E0 * np.sin(mlat_rad)**2E0 + 3E0) / np.cos(mlat_rad)**8E0 / (3E0 * np.sin(mlat_rad)**2E0 + 1E0) / r_eq * B0_eq   #[T/m]

coordinate_FA = r_eq * (np.arcsinh(np.sqrt(3)*np.sin(mlat_rad)) / 2E0 / np.sqrt(3) + np.sin(mlat_rad) * np.sqrt(5E0 - 3E0 * np.cos(2E0 * mlat_rad)) / 2E0 / np.sqrt(2)) #[m]

# background plasma parameters
oxygen_number_density_eq = 1.163E9    # [m^-3]
oxygen_number_density_scale_height = 1.2842    # [R_J]
oxygen_number_density = oxygen_number_density_eq * np.exp(- (coordinate_FA / oxygen_number_density_scale_height / planet_radius)**2E0)    #[m^-3]

oxygen_mass = 2.677950266103E-26   # [kg] # O+
oxygen_temperature = 5E1   # [eV]

proton_number_density_eq = 5.8E7    # [m^-3]
proton_number_density = proton_number_density_eq * np.ones(mlat_rad.shape)    #[m^-3]

proton_mass = 1.672621898E-27    # [kg]
proton_temperature = 8.6E0    # [eV]

ion_number_density = oxygen_number_density + proton_number_density    #[m^-3]
ion_mass_density = oxygen_number_density * oxygen_mass + proton_number_density * proton_mass    #[kg m^-3]
ion_mass = ion_mass_density / ion_number_density    #[kg]

ion_pressure = ion_number_density * proton_temperature * elementary_charge    #[N m^-2]
ion_temperature = ion_pressure / ion_number_density / elementary_charge    #[eV]

electron_mass = 9.10938356E-31    # [kg]
electron_temperature = 5E0  # [eV]

tau = ion_temperature / electron_temperature    #[]

Alfven_speed = b0 / np.sqrt(magnetic_constant * ion_mass_density)    #[m/s]
#relativistic_Alfven_speed = Alfven_speed / np.sqrt(speed_of_light**2E0 + Alfven_speed**2E0) * speed_of_light   #[m/s]

plasma_beta_ion = 2E0 * magnetic_constant * ion_pressure / b0**2E0  #[]

ion_gyrofrequency = elementary_charge * b0 / ion_mass    #[rad/s]
ion_thermal_speed = np.sqrt(2E0 * ion_temperature * elementary_charge / ion_mass)    #[m/s]
ion_gyroradius = ion_thermal_speed / ion_gyrofrequency    #[m]

electron_gyrofrequency = elementary_charge * b0 / electron_mass    #[rad/s]
electron_thermal_speed = np.sqrt(2E0 * electron_temperature * elementary_charge / electron_mass)    #[m/s]
electron_gyroradius = electron_thermal_speed / electron_gyrofrequency    #[m]


# wave parameters

uperp_eq = 5.4E4 #[m/s]
kperp_rhoi = ion_thermal_speed[0] / uperp_eq    #[rad]

kperp = kperp_rhoi / ion_gyroradius    #[rad m^-1]
wavelength_perp = 2E0 * np.pi / kperp    #[m]

kpara_eq = ion_gyrofrequency[0] / electron_thermal_speed    #[rad m^-1]

wave_frequency = np.sqrt((1E0 + tau[0]) / (plasma_beta_ion[0] * (1E0 + tau[0]) + 2E0 * tau[0])) * kperp_rhoi * kpara_eq * Alfven_speed[0]    #[rad s^-1]

kpara = wave_frequency / Alfven_speed / kperp_rhoi * np.sqrt(plasma_beta_ion + 2E0 * tau / (1E0 + tau))    #[rad m^-1]
wavelength_para = 2E0 * np.pi / kpara    #[m]

wave_phase_speed_parallel = wave_frequency / kpara    #[m/s]

electric_field_eq = 1E-3 #[V/m]
wave_modified_potential = electric_field_eq / kpara[0]     #[V]
#wave_modified_potential = 3E3    #[V]

energy_wave_phase_speed = 5E-1 * electron_mass * wave_phase_speed_parallel**2E0 / elementary_charge    #[eV]
energy_wave_potential = 5E-1 * elementary_charge * wave_modified_potential / elementary_charge    #[eV]


# energy prediction
Kf_epsilon = plasma_beta_ion * (1E0 + tau) / (plasma_beta_ion * (1E0 + tau) + 2E0 * tau)    #[]
Kf_x = db0_dz / kpara / b0    #[]

# 微分
def diff(f):
    diff_f = np.zeros(f.shape)
    for count_i in range(len(coordinate_FA)):
        if count_i == 0:
            diff_f[count_i] = 0E0
        elif count_i == len(coordinate_FA) - 1:
            diff_f[count_i] = (f[count_i] - f[count_i - 1]) / (coordinate_FA[count_i] - coordinate_FA[count_i - 1])
        else:
            diff_f[count_i] = ((f[count_i + 1] - f[count_i]) / (coordinate_FA[count_i + 1] - coordinate_FA[count_i]) + (f[count_i] - f[count_i - 1]) / (coordinate_FA[count_i] - coordinate_FA[count_i - 1])) / 2E0
    return diff_f

diff_b0 = diff(b0) / kpara / b0
diff_ion_number_density = diff(ion_number_density) / kpara / ion_number_density
diff_ion_temperature = diff(ion_temperature) / kpara / ion_temperature
diff_1_plus_tau = diff(1E0 + tau) / kpara / (1E0 + tau)

Kf_y = (1E0 + 2E0 * Kf_epsilon) * Kf_x - (1E0 + Kf_epsilon) * diff_ion_number_density - Kf_epsilon * diff_ion_temperature - Kf_epsilon / plasma_beta_ion / (1E0 + tau) * diff_1_plus_tau    #[]

Kf_eV = 1E0 / Kf_x * ((1E0 - Kf_y * (np.pi / 2E0 - 1E0)) * energy_wave_potential + 2E0 * Kf_y * np.sqrt(energy_wave_phase_speed * energy_wave_potential) * np.sqrt(np.pi / 2E0 - 1E0) - Kf_y * energy_wave_phase_speed)     #[eV]

cos_pitch_angle_limit = np.zeros(len(mlat_deg))    #[]
for count_i in range (len(mlat_deg)):
    cos_pitch_angle_limit[count_i] = (energy_wave_potential / Kf_eV[count_i] - Kf_x[count_i]) / Kf_y[count_i]    #[]
    if cos_pitch_angle_limit[count_i] < 0E0:
        cos_pitch_angle_limit[count_i] = np.nan
        Kf_eV[count_i] = np.nan
    elif cos_pitch_angle_limit[count_i] > 1E0:
        cos_pitch_angle_limit[count_i] = np.nan
        Kf_eV[count_i] = np.nan
    else:
        cos_pitch_angle_limit[count_i] = np.sqrt(cos_pitch_angle_limit[count_i])

pitch_angle_limit_rad = np.arccos(cos_pitch_angle_limit)    #[rad]
pitch_angle_limit_deg = pitch_angle_limit_rad * 180E0 / np.pi    #[deg]

equatorial_pitch_angle_limit_rad = np.arcsin(np.sqrt(B0_eq / b0) * np.sin(pitch_angle_limit_rad))   #[rad]
equatorial_pitch_angle_limit_deg = equatorial_pitch_angle_limit_rad * 180E0 / np.pi    #[deg]

#trapped phase energy range
trapping_frequency = kpara * np.sqrt(energy_wave_potential * elementary_charge / electron_mass) #[rad s^-1]

lower_energy_trapped = np.zeros(len(mlat_deg))
upper_energy_trapped = (np.sqrt(2E0*energy_wave_potential) + np.sqrt(energy_wave_phase_speed))**2E0     #[eV]
upper_energy_trapped_eV = upper_energy_trapped     #[eV]

for count_i in range(len(mlat_deg)):
    if trapping_frequency[count_i] > 0.5 * wave_frequency:
        lower_energy_trapped[count_i] = 0E0   #[J]
    else:
        lower_energy_trapped[count_i] = (np.sqrt(2E0*energy_wave_potential) - np.sqrt(energy_wave_phase_speed[count_i]))**2E0    #[eV]

lower_energy_trapped_eV = lower_energy_trapped    #[eV]

kpara_Alfven_speed = kpara * Alfven_speed    #[rad/s]

model_S_10000 = 10000E0 / energy_wave_potential * (Kf_x + 0.5 * Kf_y)
model_S_1000 = 1000E0 / energy_wave_potential * (Kf_x + 0.5 * Kf_y)
model_S_100 = 100E0 / energy_wave_potential * (Kf_x + 0.5 * Kf_y)
model_S_10 = 10E0 / energy_wave_potential * (Kf_x + 0.5 * Kf_y)
model_S_1 = 1E0 / energy_wave_potential * (Kf_x + 0.5 * Kf_y)

# plot

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35


#print(ion_gyrofrequency[0])
#print(wave_frequency)
#
##wavelengthのplot
#fig = plt.figure(figsize=(10, 10), dpi=100)
#ax1 = fig.add_subplot(121)
#ax1.plot(mlat_deg, wavelength_perp, label=r'perp', color='red')
#ax1.plot(mlat_deg, wavelength_para, label=r'para', color='blue')
#ax1.set_xlabel('mlat [deg]')
#ax1.set_ylabel('wavelength [m]')
#ax1.set_yscale('log')
#ax1.minorticks_on()
#ax1.grid(which='both', alpha=0.5)
#ax1.legend()
#
##plasma betaのplot
#ax2 = fig.add_subplot(122)
#ax2.plot(mlat_deg, plasma_beta_ion, label=r'ion beta', color='red')
#ax2.plot(mlat_deg, electron_mass/ion_mass, label=r'mass ratio', color='blue')
#ax2.set_xlabel('mlat [deg]')
#ax2.set_ylabel('plasma beta')
#ax2.set_yscale('log')
#ax2.minorticks_on()
#ax2.grid(which='both', alpha=0.5)
#ax2.legend()
#
#plt.show()


#fig = plt.figure(figsize=(14, 14))
#ax = fig.add_subplot(111, xlabel=r'MLAT [deg]', ylabel=r'$1 / k_{\parallel} v_{\mathrm{A}}$ [s/rad]', xlim=(0, 50))
#ax.plot(mlat_deg, 1E0 / kpara_Alfven_speed, color='blue', linewidth=4)
#ax.minorticks_on()
#ax.grid(which='both', alpha=0.3)
#
#plt.tight_layout()
#
#plt.show()
#
#quit()

fig = plt.figure(figsize=(14, 21))
ax1 = fig.add_subplot(311, xlabel=r'MLAT [deg]', ylabel=r'Pitch angle [deg]', xlim=(0, 50), ylim=(0, 90))
#ax1.plot(mlat_deg, (energy_wave_potential / energy_limit - delta) / epsilon)
#ax1.plot(mlat_deg, ( - delta) / epsilon)
#ax1.plot(mlat_deg, (energy_wave_potential / energy_limit) / epsilon)
#ax1.plot(mlat_deg, epsilon)
ax1.plot(mlat_deg, pitch_angle_limit_deg, color='blue', linewidth=4, label=r'typical PA')
ax1.plot(mlat_deg, equatorial_pitch_angle_limit_deg, color='red', linewidth=4, label=r'typical equatorial PA')
ax1.minorticks_on()
ax1.grid(which='both', alpha=0.3)
ax1.legend(loc='lower left', fontsize=30)

ax2 = fig.add_subplot(312, xlabel=r'MLAT [deg]', ylabel=r'Energy [eV]', xlim=(0, 50))
ax2.plot(mlat_deg, energy_wave_phase_speed, color='red', linewidth=4, label=r'$K_{\mathrm{ph} \parallel}$', alpha=0.5)
ax2.plot(mlat_deg, energy_wave_potential*np.ones(len(mlat_deg)), color='green', linewidth=4, label=r'$K_{\mathrm{E}}$', alpha=0.5)
ax2.plot(mlat_deg, Kf_eV * cos_pitch_angle_limit**2E0, color='orange', linewidth=4, label=r'$K_{\mathrm{f} \parallel}$', alpha=0.5)
ax2_ylim = ax2.get_ylim()
ax2.plot(mlat_deg, Kf_eV, color='blue', linewidth=4, label=r'typical energy $K_{\mathrm{f}}$', alpha=0.5)
ax2.set_ylim(ax2_ylim)
ax2.minorticks_on()
ax2.grid(which='both', alpha=0.3)
ax2.legend(loc='upper left', fontsize=30)

ax3 = fig.add_subplot(313, xlabel=r'MLAT [deg]', ylabel=r'Energy [eV]', xlim=(0, 50))
ax3.plot(mlat_deg, energy_wave_phase_speed, color='red', linewidth=4, label=r'$K_{\mathrm{ph} \parallel}$')
ax3.plot(mlat_deg, energy_wave_potential*np.ones(len(mlat_deg)), color='green', linewidth=4, label=r'$K_{\mathrm{E}}$')
ax3.plot(mlat_deg, lower_energy_trapped_eV, color='orange', linewidth=4, label=r'$K \mathrm{cos}^{2} \alpha$ in trapped phase')
ax3.plot(mlat_deg, upper_energy_trapped_eV, color='orange', linewidth=4)
ax3.minorticks_on()
ax3.grid(which='both', alpha=0.3)
ax3.legend(loc='upper left', fontsize=30)

plt.tight_layout()

plt.show()