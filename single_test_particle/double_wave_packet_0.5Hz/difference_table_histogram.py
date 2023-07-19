import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

wave_scalar_potential = 600E0 # [V]
initial_wave_phase = 0E0 # [deg]
gradient_parameter = 2E0 # []
wave_threshold = 5E0 # [deg]

wavekind_list = [r'EparaBpara', r'Epara']

filename_base_1 = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave_packet/results_particle_{str(int(wave_scalar_potential))}V' \
        + f'_gradient_{int(gradient_parameter)}_threshold_{int(wave_threshold)}_wavephase_{int(initial_wave_phase)}_{wavekind_list[0]}'

filename_base_2 = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave_packet/results_particle_{str(int(wave_scalar_potential))}V' \
        + f'_gradient_{int(gradient_parameter)}_threshold_{int(wave_threshold)}_wavephase_{int(initial_wave_phase)}_{wavekind_list[1]}'

filename_energy_1 = f'{filename_base_1}/energy_{wavekind_list[0]}.csv'
filename_energy_2 = f'{filename_base_2}/energy_{wavekind_list[1]}.csv'

# read data
data_energy_1 = np.genfromtxt(filename_energy_1, delimiter=',', skip_header=1)
data_energy_2 = np.genfromtxt(filename_energy_2, delimiter=',', skip_header=1)

data_energy_1 = data_energy_1[:, 1:]
data_energy_2 = data_energy_2[:, 1:]

# make histogram
data_energy_1_flat = data_energy_1.flatten()
data_energy_2_flat = data_energy_2.flatten()

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

fig = plt.figure(figsize=(28, 14), dpi=100, tight_layout=True)
fig.suptitle(r'histogram of energy increase')
ax1 = fig.add_subplot(121)
ax1.hist(data_energy_1_flat, bins=55, alpha=0.5, range=(0, 5500))
ax1.yaxis.set_major_locator(mpl.ticker.MultipleLocator())
ax1.grid(which='major', alpha=0.3)
ax1.set_xlim(0, 5500)
ax1.set_ylim(0, 20)
ax1.set_xlabel(r'energy [eV]')
ax1.set_ylabel(r'count')
ax1.set_title(r'$\delta E_{\parallel}$' ' \& ' r'$\delta B_{\parallel}$')

ax2 = fig.add_subplot(122)
ax2.hist(data_energy_2_flat, bins=55, alpha=0.5, range=(0, 5500))
ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator())
ax2.grid(which='major', alpha=0.3)
ax2.set_xlim(0, 5500)
ax2.set_ylim(0, 20)
ax2.set_xlabel(r'energy [eV]')
ax2.set_ylabel(r'count')
ax2.set_title(r'only $\delta E_{\parallel}$')

plt.show()