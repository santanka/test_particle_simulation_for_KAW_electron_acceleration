import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 25

xmin = -1.5
#xmax = 6.5  #Io
#ymax = 2.5  #Io
xmax = 10.5 #Europa
ymax = 4    #Europa

fig = plt.figure(figsize=(12, 7.5), dpi=500, tight_layout=True)
ax = fig.add_subplot(111, xlabel=r'[$\mathrm{R_{J}}$]', ylabel=r'[$\mathrm{R_{J}}$]')

ax.axvline(0, color='black', lw=2, alpha=0.6, linestyle=':')
ax.axhline(0, color='black', lw=2, alpha=0.6, linestyle=':')


# Jupiter
Radius_eq = 71492E3
Radius_polar = 66854E3
L_poler = Radius_polar / Radius_eq
theta_array = np.linspace(0, 2*np.pi, 1000)
ax.plot(np.cos(theta_array), L_poler * np.sin(theta_array), color='black', lw=2, alpha=0.6)

# dipole field
#L_value = 5.91 # Io
L_value = 9.38  # Europa

mlat_max = np.arccos(np.sqrt((1E0 - L_poler**2E0 + np.sqrt((1E0 - L_poler**2E0)**2E0 + 4E0 * L_poler**2E0 * L_value**2E0)) / 2E0 / L_value**2E0))
mlat_array = np.linspace(-mlat_max, mlat_max, 1000)
ax.plot(L_value * np.cos(mlat_array)**3E0, L_value * np.sin(mlat_array) * np.cos(mlat_array)**2E0, color='purple', lw=2, alpha=0.6)

# alpha_rot
alpha_rot = 9.6 * np.pi / 180

x_array = np.linspace(xmin, xmax, 1000)
y_array = np.linspace(-ymax, ymax, 1000)

ax.plot(-np.tan(alpha_rot) * y_array, y_array, color='blue', lw=2, alpha=0.6, linestyle=':')
ax.plot(x_array, np.tan(alpha_rot) * x_array, color='blue', lw=2, alpha=0.6, linestyle=':')

def centrifugal_equator_mlat_rad(alpha_rot_rad):
    tan_lambda_0 = 2E0 / 3E0 * np.tan(alpha_rot_rad) / (1E0 + np.sqrt(1E0 + 8E0 / 9E0 * np.tan(alpha_rot_rad) ** 2))
    lambda_0_rad = np.arctan(tan_lambda_0)
    return lambda_0_rad
lambda_0 = centrifugal_equator_mlat_rad(alpha_rot)

ax.plot(x_array, np.tan(lambda_0) * x_array, color='red', lw=2, alpha=0.6)


# typical mlat
mlat_typical = 8 * np.pi / 180

ax.scatter(L_value * np.cos(mlat_typical)**3E0, L_value * np.sin(mlat_typical) * np.cos(mlat_typical)**2E0, color='green', s=100, edgecolor='black', lw=1)

x_array_typical = np.linspace(0, L_value * np.cos(mlat_typical)**3E0, 1000)
ax.plot(x_array_typical, np.tan(mlat_typical) * x_array_typical, color='green', lw=2, alpha=0.6)

x_array_Rceq_max = L_value * np.cos(lambda_0) * np.cos(mlat_typical)**2E0 * np.cos(mlat_typical - lambda_0)
x_array_Rceq = np.linspace(0, x_array_Rceq_max, 1000)
#ax.plot(x_array_Rceq, np.tan(lambda_0) * x_array_Rceq, color='green', lw=2, alpha=0.6)

mlat_typical_array = np.linspace(lambda_0, mlat_typical, 1000)
x_array_height = L_value * np.cos(mlat_typical_array)**3E0
y_array_height = L_value * np.sin(mlat_typical_array) * np.cos(mlat_typical_array)**2E0
ax.plot(x_array_height, y_array_height, color='green', lw=2, alpha=0.6)



ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

fig.tight_layout()
plt.axis('scaled')
plt.xlim(xmin, xmax)
plt.ylim(-ymax, ymax)

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan3'
os.makedirs(dir_name, exist_ok=True)
fig.savefig(f'{dir_name}/schematic_diagram_Europa.pdf')
fig.savefig(f'{dir_name}/schematic_diagram_Europa.png')
plt.close()