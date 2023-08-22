import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 45

# parameter
number_density_ion = 1E6    #[m-3]
temperature_ion = 1E3   #[eV]
temperature_electron = 1E2  #[eV]

planet_radius = 6371E3 #[m]
lshell_number = 9E0
r_equator = planet_radius * lshell_number  #[m]
dipole_moment = 7.75E22 #[Am2]
magnetic_field_equator = dipole_moment / (r_equator**3) * 1E-7 #[T]

channel = 4
# 1:plasma beta ion, 2:δBpara/δEpara, 3:F_δBpara/F_δEpara, 4:Gamma trapping

# constant
speed_of_light = 2.99792458E8 #[m/s]
elementary_charge = 1.60217662E-19 #[C]
mass_proton = 1.672621898E-27 #[kg]
mass_electron = 9.10938356E-31 #[kg]
mu0 = 4E-7 * np.pi #[N/A^2]

pressure_ion = number_density_ion * temperature_ion * elementary_charge #[Pa]
pressure_electron = number_density_ion * temperature_electron * elementary_charge #[Pa]

deg2rad = np.pi / 180.0

# mlat
mlat_deg = np.linspace(-60, 60, 1000)
mlat_rad = mlat_deg * deg2rad

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave/0.15Hz'

if channel == 1:
    # dipole magnetic field
    magnetic_flux_density = magnetic_field_equator / np.cos(mlat_rad)**6E0 * np.sqrt(1E0 + 3E0*np.sin(mlat_rad)**2) #[T]
    # plasma beta
    beta_ion = pressure_ion / (magnetic_flux_density**2 / (2E0*mu0))
    # plasma beta profile
    fig = plt.subplots(figsize=(14, 14), dpi=100, tight_layout=True)
    ax = plt.subplot(111, yscale='log')
    ax.plot(mlat_deg, beta_ion, color='blue', linewidth=4, label=r'$\beta_{\mathrm{i}}$')
    ax.plot(mlat_deg, np.ones(mlat_deg.shape)*mass_electron/mass_proton, color='dimgrey', linewidth=2, linestyle='--', label=r'$m_{\mathrm{e}}/m_{\mathrm{i}}$')
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.set_xlabel(r'MLAT [deg]')
    ax.set_ylabel(r'$\beta_{\mathrm{i}}$')
    ax.legend()

if channel == 2:
    # dipole magnetic field
    magnetic_flux_density = magnetic_field_equator / np.cos(mlat_rad)**6E0 * np.sqrt(1E0 + 3E0*np.sin(mlat_rad)**2) #[T]
    # plasma beta
    beta_ion = pressure_ion / (magnetic_flux_density**2 / (2E0*mu0))
    #δBpara/δEpara
    number_density_ion_cgs = number_density_ion / 1E6 #[cm-3]
    pressure_ion_cgs = pressure_ion * 1E1 #[Ba]
    pressure_electron_cgs = pressure_electron * 1E1 #[Ba]

    kperp_rhoi = 1E0 #[]
    wave_frequency = 2*np.pi / 2 #[Hz]
    elementary_charge_cgs = elementary_charge/1E1*speed_of_light*1E2 #[statC]
    mass_proton_cgs = mass_proton * 1E3 #[g]

    #δBpara/δEpara
    delta_b_para_over_delta_e_para = np.sqrt(4E0*np.pi*number_density_ion_cgs/mass_proton_cgs) * kperp_rhoi / wave_frequency * elementary_charge_cgs \
        * (pressure_ion_cgs + pressure_electron_cgs) / (2*pressure_ion_cgs + pressure_electron_cgs) * np.sqrt((pressure_ion_cgs + pressure_electron_cgs) / (2*pressure_ion_cgs + beta_ion * (pressure_ion_cgs + pressure_electron_cgs)))

    #plot
    fig = plt.subplots(figsize=(14, 14), dpi=100, tight_layout=True)
    ax = plt.subplot(111, yscale='log')
    ax.plot(mlat_deg, delta_b_para_over_delta_e_para, color='blue', linewidth=4, label=r'$\delta B_{\parallel}/\delta E_{\parallel}$')
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.set_xlabel(r'MLAT [deg]')
    ax.set_ylabel(r'$\delta B_{\parallel}/\delta E_{\parallel}$')
    ax.legend()

if channel == 3:
    electron_energy = np.linspace(100, 1000, 1000) #[eV]
    MLAT_DEG, ELECTRON_ENERGY = np.meshgrid(mlat_deg, electron_energy)
    MLAT_RAD = MLAT_DEG * deg2rad

    speed_of_light_cgs = speed_of_light * 1E2 #[cm/s]
    elementary_charge_cgs = elementary_charge/1E1*speed_of_light_cgs #[statC]
    mass_proton_cgs = mass_proton * 1E3 #[g]

    ELECTRON_ENERGY_CGS = ELECTRON_ENERGY * elementary_charge * 1E7 #[erg]

    #δBpara/δEpara
    number_density_ion_cgs = number_density_ion / 1E6 #[cm-3]
    pressure_ion_cgs = pressure_ion * 1E1 #[Ba]
    pressure_electron_cgs = pressure_electron * 1E1 #[Ba]
    magnetic_flux_density = magnetic_field_equator / np.cos(MLAT_RAD)**6E0 * np.sqrt(1E0 + 3E0*np.sin(MLAT_RAD)**2) #[T]
    magnetic_flux_density_cgs = magnetic_flux_density * 1E4 #[G]
    beta_ion = pressure_ion / (magnetic_flux_density**2 / (2E0*mu0))

    dB0_dz = 3E0 * np.sin(MLAT_RAD) * (5E0 * np.sin(MLAT_RAD)**2E0 + 3E0) / np.cos(MLAT_RAD)**8E0 / (3E0 * np.sin(MLAT_RAD)**2E0 + 1E0) / r_equator * magnetic_field_equator   #[T/m]
    dB0_dz_cgs = dB0_dz * 1E4 / 1E2 #[G/cm]

    kperp_rhoi = 2*np.pi #[rad]
    wave_frequency = 2*np.pi * 0.15 #[Hz]

    wave_scalar_potential   = 2000E0     #[V]
    wave_scalar_potential_cgs = wave_scalar_potential * 1E8 / speed_of_light_cgs #[statV]

    delta_B_para_over_B0 = beta_ion / 2 * (1E0 + pressure_electron_cgs / pressure_ion_cgs) * number_density_ion_cgs * elementary_charge_cgs / pressure_ion_cgs * wave_scalar_potential_cgs #[]

    #δBpara/δEpara
    delta_b_para_over_delta_e_para = np.sqrt(4E0*np.pi*number_density_ion_cgs/mass_proton_cgs) * kperp_rhoi / wave_frequency * elementary_charge_cgs \
        * (pressure_ion_cgs + pressure_electron_cgs) / (2*pressure_ion_cgs + pressure_electron_cgs) * np.sqrt((pressure_ion_cgs + pressure_electron_cgs) / (2*pressure_ion_cgs + beta_ion * (pressure_ion_cgs + pressure_electron_cgs)))

    kpara_cgs = wave_frequency / kperp_rhoi * np.sqrt(4E0 * np.pi * number_density_ion_cgs * mass_proton_cgs) / magnetic_flux_density_cgs * np.sqrt(beta_ion + 2E0 / (1E0 + pressure_electron_cgs/pressure_ion_cgs)) * np.sign(MLAT_DEG) #[cm-1]
    #δBpara/δEpara
    delta_b_para_over_delta_e_para = np.sqrt(4E0*np.pi*number_density_ion_cgs/mass_proton_cgs) * kperp_rhoi / wave_frequency * elementary_charge_cgs \
        * (pressure_ion_cgs + pressure_electron_cgs) / (2*pressure_ion_cgs + pressure_electron_cgs) * np.sqrt((pressure_ion_cgs + pressure_electron_cgs) / (2*pressure_ion_cgs + beta_ion * (pressure_ion_cgs + pressure_electron_cgs)))
    
    #F_ratio
    F_delta_B_para_over_F_delta_E_para = np.abs(1 / elementary_charge_cgs * delta_b_para_over_delta_e_para / magnetic_flux_density_cgs / (1E0 + delta_B_para_over_B0) * (1 / magnetic_flux_density_cgs * dB0_dz_cgs + kpara_cgs) * ELECTRON_ENERGY_CGS) #[]

    #カラーマップでplot
    fig = plt.subplots(figsize=(14, 14), dpi=100, tight_layout=True)
    ax = plt.subplot(111)
    color = ax.pcolormesh(MLAT_DEG, ELECTRON_ENERGY, np.log10(F_delta_B_para_over_F_delta_E_para), cmap='turbo', shading='auto')
    ct = ax.contour(MLAT_DEG, ELECTRON_ENERGY, np.log10(F_delta_B_para_over_F_delta_E_para), colors='black', linewidths=1, levels=[-5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5])
    ax.clabel(ct)
    plt.colorbar(color, ax=ax, label=r'$\mathrm{log}_{10}(F_{\delta B_{\parallel}}/F_{\delta E_{\parallel}})$')
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.set_xlabel(r'MLAT [deg]')
    ax.set_ylabel(r'Electron Energy [eV]')
    fig_name = f'{dir_name}/deltaBpara_over_deltaEpara_ratio.png'
    plt.savefig(fig_name)
    plt.close()

if channel == 4:
    electron_energy = np.linspace(100, 1000, 1000) #[eV]
    MLAT_DEG, ELECTRON_ENERGY = np.meshgrid(mlat_deg, electron_energy)
    MLAT_RAD = MLAT_DEG * deg2rad

    speed_of_light_cgs = speed_of_light * 1E2 #[cm/s]
    elementary_charge_cgs = elementary_charge/1E1*speed_of_light_cgs #[statC]
    mass_proton_cgs = mass_proton * 1E3 #[g]
    mass_electron_cgs = mass_electron * 1E3 #[g]

    ELECTRON_ENERGY_CGS = ELECTRON_ENERGY * elementary_charge * 1E7 #[erg]

    number_density_ion_cgs = number_density_ion / 1E6 #[cm-3]
    pressure_ion_cgs = pressure_ion * 1E1 #[Ba]
    pressure_electron_cgs = pressure_electron * 1E1 #[Ba]
    magnetic_flux_density = magnetic_field_equator / np.cos(MLAT_RAD)**6E0 * np.sqrt(1E0 + 3E0*np.sin(MLAT_RAD)**2) #[T]
    magnetic_flux_density_cgs = magnetic_flux_density * 1E4 #[G]
    beta_ion = pressure_ion / (magnetic_flux_density**2 / (2E0*mu0))
    
    dB0_dz = 3E0 * np.sin(MLAT_RAD) * (5E0 * np.sin(MLAT_RAD)**2E0 + 3E0) / np.cos(MLAT_RAD)**8E0 / (3E0 * np.sin(MLAT_RAD)**2E0 + 1E0) / r_equator * magnetic_field_equator   #[T/m]
    dB0_dz_cgs = dB0_dz * 1E4 / 1E2 #[G/cm]

    kperp_rhoi = 2*np.pi #[rad]
    wave_frequency = 2*np.pi * 0.15 #[rad/s]

    wave_scalar_potential   = 2000E0     #[V]
    wave_scalar_potential_cgs = wave_scalar_potential * 1E8 / speed_of_light_cgs #[statV]
    wave_modified_scalar_potential_cgs = (2E0 + pressure_electron_cgs / pressure_ion_cgs) * wave_scalar_potential_cgs #[statV]

    Phi_B = 4E0 * np.pi * number_density_ion_cgs * elementary_charge_cgs * (1E0 + pressure_electron_cgs / pressure_ion_cgs) * wave_scalar_potential_cgs #[G^2]

    kpara_cgs = wave_frequency / kperp_rhoi * np.sqrt(4E0 * np.pi * number_density_ion_cgs * mass_proton_cgs) / magnetic_flux_density_cgs * np.sqrt(beta_ion + 2E0 / (1E0 + pressure_electron_cgs/pressure_ion_cgs)) * np.sign(MLAT_DEG) #[cm-1]

    d_Phi_B_B0_dz = - Phi_B / magnetic_flux_density_cgs**2E0 * dB0_dz_cgs #[G/cm]

    Gamma = np.sqrt((1E0 - 1E0/2E0 * Phi_B / (magnetic_flux_density_cgs**2E0 + Phi_B) * ELECTRON_ENERGY_CGS / elementary_charge_cgs / wave_modified_scalar_potential_cgs * (ELECTRON_ENERGY_CGS + 2E0 * mass_electron_cgs * speed_of_light_cgs**2E0) / (ELECTRON_ENERGY_CGS + mass_electron_cgs * speed_of_light_cgs**2E0))**2E0 \
                    + (1E0/2E0 * magnetic_flux_density_cgs**2E0 / (magnetic_flux_density_cgs**2E0 + Phi_B) * ELECTRON_ENERGY_CGS / elementary_charge_cgs / wave_modified_scalar_potential_cgs * (ELECTRON_ENERGY_CGS + 2E0 * mass_electron_cgs * speed_of_light_cgs**2E0) / (ELECTRON_ENERGY_CGS + mass_electron_cgs * speed_of_light_cgs**2E0) / kpara_cgs / magnetic_flux_density_cgs * d_Phi_B_B0_dz)**2E0) #[]
    

    #カラーマップでplot
    fig = plt.subplots(figsize=(14, 14), dpi=100, tight_layout=True)
    ax = plt.subplot(111)
    color = ax.pcolormesh(MLAT_DEG, ELECTRON_ENERGY, Gamma, cmap='turbo', shading='auto')
    ct = ax.contour(MLAT_DEG, ELECTRON_ENERGY, Gamma, colors='black', linewidths=1, levels=[0.96, 0.97, 0.98, 0.99])
    ax.clabel(ct)
    plt.colorbar(color, ax=ax, label=r'$\Gamma$ index')
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    ax.set_xlabel(r'MLAT [deg]')
    ax.set_ylabel(r'Electron Energy [eV]')
    fig_name = f'{dir_name}/Gamma_trapping.png'
    plt.savefig(fig_name)
    plt.close()



#plt.show()