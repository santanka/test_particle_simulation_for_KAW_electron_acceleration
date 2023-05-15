import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 25

# parameter
number_density_ion = 1E6    #[m-3]
temperature_ion = 1E3   #[eV]
temperature_electron = 1E2  #[eV]

planet_radius = 6371E3 #[m]
lshell_number = 9E0
r_equator = planet_radius * lshell_number  #[m]
dipole_moment = 7.75E22 #[Am2]
magnetic_field_equator = dipole_moment / (r_equator**3) * 1E-7 #[T]

print(magnetic_field_equator)

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
mlat_deg = np.linspace(-73.8, 73.8, 10000)
mlat_rad = mlat_deg * deg2rad

# dipole magnetic field
magnetic_flux_density = magnetic_field_equator / np.cos(mlat_rad)**6E0 * np.sqrt(1E0 + 3E0*np.sin(mlat_rad)**2) #[T]

# plasma beta
beta_ion = pressure_ion / (magnetic_flux_density**2 / (2E0*mu0))

print(pressure_ion)
print(magnetic_flux_density)
print(np.max(magnetic_flux_density))
print(np.min(magnetic_flux_density))
print(beta_ion)
print(np.max(magnetic_flux_density)/np.min(magnetic_flux_density))

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

plt.show()