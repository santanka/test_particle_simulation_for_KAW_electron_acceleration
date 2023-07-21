from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 定数の設定
speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]

lshell_number   = 9E0
planet_radius = 6371E3

r_eq            = planet_radius * lshell_number #[m]
dipole_moment   = 7.75E22 #[Am]
B0_eq           = (1E-7 * dipole_moment) / r_eq**3E0 * 1E4  #[G]

mass_ion        = 1.672621898E-24   #[g]
number_density_ion = 1E0    #[cm-3]
temperature_ion = 1E3   #[eV]
temperature_electron = 1E2  #[eV]
pressure_ion        = number_density_ion * temperature_ion * elementary_charge * 1E7    #cgs
pressure_electron   = number_density_ion * temperature_electron * elementary_charge * 1E7   #cgs

a = 2E0 / (1E0 + pressure_electron/pressure_ion)
b = 8E0 * np.pi * pressure_ion / B0_eq**2E0

# 積分範囲の設定
lambda_value = 1.0
lower_limit = 0
upper_limit = np.linspace(0, 40, 1000) / 180 * np.pi

# 積分する関数の定義
def integrand(lambda_prime):
    return np.sqrt(a + b * np.cos(lambda_prime)**12 / (1 + 3*np.sin(lambda_prime)**2)) * np.cos(lambda_prime)**7

# 積分の実行
result = np.zeros(len(upper_limit))
for i in range(len(upper_limit)):
    result[i], error = integrate.quad(integrand, lower_limit, upper_limit[i])

wave_frequency = 2*np.pi * 0.15    #[rad/s]

kperp_rhoi = 2E0 * np.pi    #[rad]

result = result * r_eq*1E2 * np.sqrt(4*np.pi*mass_ion*number_density_ion) / B0_eq * wave_frequency / kperp_rhoi

initial_wavephase = 0E0

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

fig = plt.figure(figsize=(14, 14), dpi=100, tight_layout=True)
ax = fig.add_subplot(111, xlabel=r'MLAT [deg]', ylabel=r'time [s]', xlim=(-40, 40), ylim=(0, 40)) #

ax.vlines(0, 0, 40, color='k', linestyle='--', linewidth=2)

color_list = ['b', 'm', 'g', 'r', 'c']

for count_i in range(5):
    time = 1 / wave_frequency * (result + initial_wavephase + 2*np.pi*count_i)
    ax.plot(upper_limit/np.pi*180, time, label=r"$\psi_{\parallel}$ = $\psi_{0}-$"+str(2*count_i)+r"$\pi$", linewidth=4, color=color_list[count_i])
    ax.plot(-upper_limit/np.pi*180, time, linewidth=4, color=color_list[count_i])

ax.minorticks_on()
ax.grid(which="both", alpha=0.3)
ax.set_axisbelow(True)

ax.legend()
plt.show()