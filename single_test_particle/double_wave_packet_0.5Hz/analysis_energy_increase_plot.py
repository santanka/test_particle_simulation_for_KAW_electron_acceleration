import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

wave_scalar_potential   = 600E0     #[V]
initial_wavephase       = 0E0       #[deg]
gradient_parameter      = 2E0       #[]
wave_threshold          = 5E0       #[deg]

wavekind                = r'Epara'

dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave_packet/results_particle_{str(int(wave_scalar_potential))}V' \
    + f'_gradient_{int(gradient_parameter)}_threshold_{int(wave_threshold)}_wavephase_{int(initial_wavephase)}_{wavekind}'
file_name = f'{dir_name}/energy_{wavekind}.csv'
fig_name = f'{dir_name}/energy_increase_{wavekind}.png'

# read data
data = np.genfromtxt(file_name, delimiter=',', skip_header=1)

initial_pitch_angle = np.linspace(5E0, 85E0, data.shape[0])
initial_energy = np.linspace(100E0, 1000E0, data.shape[1]-1)

energy_increase = np.zeros((data.shape[0], data.shape[1]-1))
v_para = np.zeros((data.shape[0], data.shape[1]-1))

speed_of_light = 2.99792458E8
electron_mass = 9.10938356E-31
elementary_charge = 1.60217662E-19

initial_energy = initial_energy * elementary_charge
initial_velocity = np.sqrt(initial_energy**2E0 + 2E0*electron_mass*initial_energy*speed_of_light) / (initial_energy + electron_mass*speed_of_light**2E0) * speed_of_light

for i in range(data.shape[0]):
    for j in range(data.shape[1]-1):
        energy_increase[i, j] = (data[i, j+1] - 100E0*(j+1)) / 100E0
        v_para[i, j] = initial_velocity[j] * np.cos(np.deg2rad(initial_pitch_angle[i]))



mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 25

# plot
fig = plt.figure(figsize=(14, 14), dpi=100, tight_layout=True)
fig.suptitle(str(wavekind) + r', wavephase @ 0 deg = ' + str(int(initial_wavephase)) + r' [deg]')

ax = fig.add_subplot(111)
for count_i in range(6, 10):
    ax.plot(initial_pitch_angle, energy_increase[:, count_i], linewidth=4, label=f'{(count_i+1)*100}V', alpha=0.5)
ax.minorticks_on()
ax.grid(which='both', alpha=0.3)
ax.set_xlabel(r'initial pitch angle [deg]')
ax.set_ylabel(r'Energy increase [$\times 100$ eV]')
ax.legend()
fig.savefig(fig_name)