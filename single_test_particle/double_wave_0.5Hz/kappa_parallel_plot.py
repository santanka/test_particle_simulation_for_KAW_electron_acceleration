import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 30
plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})

fig = plt. figure()
ax1 = fig.add_subplot(231, xlabel = r'South ← MLAT [degree] → North', ylabel = r'$\kappa_{\parallel}$: imaginary wave number')
ax2 = fig.add_subplot(232, xlabel = r'South ← MLAT [degree] → North', ylabel = r'$\psi_{i}$: imaginary wave phase')
ax3 = fig.add_subplot(233, xlabel = r'South ← MLAT [degree] → North', ylabel = r'$g(MLAT)$: wave growth rate')
ax4 = fig.add_subplot(234, xlabel = r'South ← MLAT [degree] → North', ylabel = r'$\gamma$: imaginary frequency')
ax5 = fig.add_subplot(235, xlabel = r'time [s]', ylabel = r'$\gamma$: imaginary frequency')
ax6 = fig.add_subplot(236, xlabel = r'time [s]', ylabel = r'$g(t)$: wave growth rate')

size = 100000

sigmoid_axis = 7.5E0

gradient_parameter_1 = 0.4E0
gradient_parameter_2 = 0.5E0
gradient_parameter_3 = 1E0

R_Earth = 6371E3    #[m]
L_number = 9E0
mlat_deg = np.linspace(0E0, 15E0, size)
mlat_rad = mlat_deg * np.pi / 180

wave_frequency = 2E0 * np.pi

number_density_eq = 1E6 #[/m^3]
temperature_ion_eq = 1E3    #[eV]
temperature_electron_eq = 1E2   #[eV]

elementary_charge = 1.6021766208E-19    #[C]
moment = 7.75E22    #[A m^2]
mass_ion = 1.672621898E-27  #[kg]

def kappa_function(gradient_parameter):
    return - 1E0 / R_Earth / L_number / np.cos(mlat_rad) / np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0) * 180E0 * gradient_parameter \
        / np.pi / (np.tanh(gradient_parameter * (np.abs(mlat_deg) - sigmoid_axis)) + 1E0) \
            / np.cosh(gradient_parameter * (np.abs(mlat_deg) - sigmoid_axis))**2E0 * np.sign(mlat_rad)

def wave_phase_imaginary(gradient_parameter):
    return - np.log(2E0 / (np.tanh(-sigmoid_axis * gradient_parameter) + 1) / 2E0 * (np.tanh(gradient_parameter * (np.abs(mlat_deg) - sigmoid_axis)) + 1E0))

def g_function_exp(gradient_parameter):
    return (np.tanh(-sigmoid_axis * gradient_parameter) + 1) / 2E0 * np.exp(- wave_phase_imaginary(gradient_parameter))

def number_density():
    return number_density_eq

def pressure_ion():
    return temperature_ion_eq * elementary_charge * number_density()

def pressure_electron():
    return temperature_electron_eq * elementary_charge * number_density()

def magnetic_flux_density():
    return 1E-7 * moment / (L_number * R_Earth)**3E0 * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad)**2E0) / np.cos(mlat_rad)**6E0

def Alfven_speed():
    return magnetic_flux_density() / np.sqrt(4E0 * np.pi * 1E-7 * number_density() * mass_ion)

def beta_ion():
    return 2E0 * 4E0*np.pi*1E-7 * pressure_ion() / magnetic_flux_density()**2E0

def k_para():
    return np.sign(mlat_deg) / 2E0 / np.pi * wave_frequency / Alfven_speed() * np.sqrt(beta_ion() + 2E0 / (1E0 + pressure_electron()/pressure_ion()))

def gamma_function(gradient_parameter):
    return - wave_frequency * kappa_function(gradient_parameter) / k_para()

def time_middle():
    v_phase_inverse = k_para() / wave_frequency
    time_ = np.zeros(size)
    for count_i in range(size):
        if (count_i == 0):
            time_[count_i] = 0E0
        else:
            time_[count_i] = 5E-1 * R_Earth * L_number * (v_phase_inverse[count_i-1] * np.cos(mlat_rad[count_i-1]) * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad[count_i-1])) \
                + v_phase_inverse[count_i] * np.cos(mlat_rad[count_i]) * np.sqrt(1E0 + 3E0 * np.sin(mlat_rad[count_i]))) \
                * abs(mlat_rad[count_i] - mlat_rad[count_i-1]) + time_[count_i-1]
    return time_

def mlat_deg_middle():
    mlat_deg_ = np.zeros(size)
    for count_i in range(size):
        if (count_i == 0):
            mlat_deg_[count_i] = 0E0
        else:
            mlat_deg_[count_i] = 5E-1 * (mlat_deg[count_i-1] + mlat_deg[count_i])
    return mlat_deg_

def gamma_function_time_middle(gradient_parameter):
    gamma_ = gamma_function(gradient_parameter)
    gamma_time_ = np.zeros(size)
    for count_i in range(size):
        if (count_i == 0):
            gamma_time_[count_i] = 0E0
        else:
            gamma_time_[count_i] = 5E-1 * (gamma_[count_i-1] + gamma_[count_i])
    return gamma_time_

def wave_phase_imaginary_middle(gradient_parameter):
    mlat_deg_ = mlat_deg_middle()
    return - np.log(2E0 / (np.tanh(-sigmoid_axis * gradient_parameter) + 1) / 2E0 * (np.tanh(gradient_parameter * (np.abs(mlat_deg_) - sigmoid_axis)) + 1E0))

def g_function_exp_middle(gradient_parameter):
    wave_phase_imaginary_ = wave_phase_imaginary_middle(gradient_parameter)
    return (np.tanh(-sigmoid_axis * gradient_parameter) + 1E0) / 2E0 * np.exp( - wave_phase_imaginary_)




kappa_1_5 = kappa_function(gradient_parameter_1)
kappa_2   = kappa_function(gradient_parameter_2)
kappa_4   = kappa_function(gradient_parameter_3)

ax1.plot(mlat_deg, kappa_1_5, c=r'blue', label=r'$a = $'+str(gradient_parameter_1), linestyle=r'solid', linewidth=r'4')
ax1.plot(mlat_deg, kappa_2, c=r'red', label=r'$a = $'+str(gradient_parameter_2), linestyle=r'solid', linewidth=r'4')
ax1.plot(mlat_deg, kappa_4, c=r'green', label=r'$a = $'+str(gradient_parameter_3), linestyle=r'solid', linewidth=r'4')

ax1.minorticks_on()
ax1.grid(which=r'both', alpha=0.5)
ax1.legend()

wave_phase_imaginary_1_5 = wave_phase_imaginary(gradient_parameter_1)
wave_phase_imaginary_2   = wave_phase_imaginary(gradient_parameter_2)
wave_phase_imaginary_4   = wave_phase_imaginary(gradient_parameter_3)

ax2.plot(mlat_deg, wave_phase_imaginary_1_5, c=r'blue', label=r'$a = $'+str(gradient_parameter_1), linestyle=r'solid', linewidth=r'4')
ax2.plot(mlat_deg, wave_phase_imaginary_2, c=r'red', label=r'$a = $'+str(gradient_parameter_2), linestyle=r'solid', linewidth=r'4')
ax2.plot(mlat_deg, wave_phase_imaginary_4, c=r'green', label=r'$a = $'+str(gradient_parameter_3), linestyle=r'solid', linewidth=r'4')

ax2.minorticks_on()
ax2.grid(which=r'both', alpha=0.5)
ax2.legend()

g_function_1_5 = g_function_exp(gradient_parameter_1)
g_function_2   = g_function_exp(gradient_parameter_2)
g_function_4   = g_function_exp(gradient_parameter_3)

ax3.plot(mlat_deg, g_function_1_5, c=r'blue', label=r'$a = $'+str(gradient_parameter_1), linestyle=r'solid', linewidth=r'4')
ax3.plot(mlat_deg, g_function_2, c=r'red', label=r'$a = $'+str(gradient_parameter_2), linestyle=r'solid', linewidth=r'4')
ax3.plot(mlat_deg, g_function_4, c=r'green', label=r'$a = $'+str(gradient_parameter_3), linestyle=r'solid', linewidth=r'4')

ax3.minorticks_on()
ax3.grid(which=r'both', alpha=0.5)
ax3.legend()

gamma_1_5 = gamma_function(gradient_parameter_1)
gamma_2   = gamma_function(gradient_parameter_2)
gamma_4   = gamma_function(gradient_parameter_3)

ax4.plot(mlat_deg, gamma_1_5, c=r'blue', label=r'$a = $'+str(gradient_parameter_1), linestyle=r'solid', linewidth=r'4')
ax4.plot(mlat_deg, gamma_2, c=r'red', label=r'$a = $'+str(gradient_parameter_2), linestyle=r'solid', linewidth=r'4')
ax4.plot(mlat_deg, gamma_4, c=r'green', label=r'$a = $'+str(gradient_parameter_3), linestyle=r'solid', linewidth=r'4')

ax4.minorticks_on()
ax4.grid(which=r'both', alpha=0.5)
ax4.legend()

gamma_time_1_5 = gamma_function_time_middle(gradient_parameter_1)
gamma_time_2   = gamma_function_time_middle(gradient_parameter_2)
gamma_time_4   = gamma_function_time_middle(gradient_parameter_3)

time_middle_ = time_middle()

ax5.plot(time_middle_, gamma_time_1_5, c=r'blue', label=r'$a = $'+str(gradient_parameter_1), linestyle=r'solid', linewidth=r'4')
ax5.plot(time_middle_, gamma_time_2, c=r'red', label=r'$a = $'+str(gradient_parameter_2), linestyle=r'solid', linewidth=r'4')
ax5.plot(time_middle_, gamma_time_4, c=r'green', label=r'$a = $'+str(gradient_parameter_3), linestyle=r'solid', linewidth=r'4')

ax5.minorticks_on()
ax5.grid(which=r'both', alpha=0.5)
ax5.legend()

g_function_time_1_5 = g_function_exp_middle(gradient_parameter_1)
g_function_time_2   = g_function_exp_middle(gradient_parameter_2)
g_function_time_4   = g_function_exp_middle(gradient_parameter_3)

ax6.plot(time_middle_, g_function_time_1_5, c=r'blue', label=r'$a = $'+str(gradient_parameter_1), linestyle=r'solid', linewidth=r'4')
ax6.plot(time_middle_, g_function_time_2, c=r'red', label=r'$a = $'+str(gradient_parameter_2), linestyle=r'solid', linewidth=r'4')
ax6.plot(time_middle_, g_function_time_4, c=r'green', label=r'$a = $'+str(gradient_parameter_3), linestyle=r'solid', linewidth=r'4')

ax6.minorticks_on()
ax6.grid(which=r'both', alpha=0.5)
ax6.legend()

plt.tight_layout()
plt.show()