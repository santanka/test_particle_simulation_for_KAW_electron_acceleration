import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

#フォントの設定
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 35

#定数
S_number_array = [0E0, 5E-1, 1E0, 1.5E0]

# wave_phaseの範囲
wave_phase_min = -np.pi
wave_phase_max = np.pi
wave_phase_num = 10000
wave_phase = np.linspace(wave_phase_min, wave_phase_max, wave_phase_num)

# theta_0_psiの範囲
theta_0_psi_min = -np.pi
theta_0_psi_max = np.pi * 12E0
theta_0_psi_num = 80

# psi_0_thetaの範囲
psi_0_theta_min = - 5E0
psi_0_theta_max = 5E0
psi_0_theta_num = 30


# plot
fig = plt.figure(figsize=(20, 20), dpi=100)
ax_1 = fig.add_subplot(221, xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$', xlim=(-1, 1), ylim=(-3, 3))
ax_2 = fig.add_subplot(222, xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$', xlim=(-1, 1), ylim=(-3, 3))
ax_3 = fig.add_subplot(223, xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$', xlim=(-1, 1), ylim=(-3, 3))
ax_4 = fig.add_subplot(224, xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$', xlim=(-1, 1), ylim=(-3, 3))
axes = [ax_1, ax_2, ax_3, ax_4]

for count_i in range(len(S_number_array)):
    S_number = S_number_array[count_i]
    ax = axes[count_i]
    ax.set_title(r'$S$ $=$ ' + f'{S_number:.1f}')
    #(a)のような図番号を付ける
    ax.text(-0.15, 1.0, '(' + chr(97 + count_i) + ')', transform=ax.transAxes, fontsize=40)

    #各theta_0_psiを通る軌道のプロット
    for theta_0_psi in np.linspace(theta_0_psi_min, theta_0_psi_max, theta_0_psi_num):
        theta_plus = np.sqrt(0.5*((np.cos(wave_phase) - np.cos(theta_0_psi)) - S_number * (wave_phase - theta_0_psi)))
        theta_minus = - np.sqrt(0.5*((np.cos(wave_phase) - np.cos(theta_0_psi)) - S_number * (wave_phase - theta_0_psi)))
        ax.plot(wave_phase/np.pi, theta_plus, color='k', lw=1, alpha=0.3)
        ax.plot(wave_phase/np.pi, theta_minus, color='k', lw=1, alpha=0.3)
    
    #各psi_0_thetaを通る軌道のプロット
    if S_number == 0E0:
        for psi_0_theta in np.linspace(psi_0_theta_min, psi_0_theta_max, psi_0_theta_num):
            theta_plus = np.sqrt(0.5*((np.cos(wave_phase) - np.cos(0E0)) - S_number * (wave_phase - 0E0)) + psi_0_theta**2E0)
            theta_minus = - np.sqrt(0.5*((np.cos(wave_phase) - np.cos(0E0)) - S_number * (wave_phase - 0E0)) + psi_0_theta**2E0)
            ax.plot(wave_phase/np.pi, theta_plus, color='k', lw=1, alpha=0.3)
            ax.plot(wave_phase/np.pi, theta_minus, color='k', lw=1, alpha=0.3)

    #鞍点、安定点の計算
    stable_point = - np.arcsin(S_number)
    saddle_point = - np.pi - stable_point
    print(f'stable_point = {stable_point/np.pi:.2f} pi')
    print(f'saddle_point = {saddle_point/np.pi:.2f} pi')

    #鞍点を通過する軌道のプロット
    if S_number <= 1E0:
        theta_plus_saddle = np.sqrt(0.5*((np.cos(wave_phase) - np.cos(saddle_point)) - S_number * (wave_phase - saddle_point)))
        theta_minus_saddle = - np.sqrt(0.5*((np.cos(wave_phase) - np.cos(saddle_point)) - S_number * (wave_phase - saddle_point)))
        ax.plot(wave_phase/np.pi, theta_plus_saddle, color='orange', lw=4, alpha=1)
        ax.plot(wave_phase/np.pi, theta_minus_saddle, color='orange', lw=4, alpha=1)

        #鞍点と安定点のプロット
        ax.scatter(saddle_point/np.pi, 0, marker='o', color='b', s=100, zorder=100)
        ax.scatter(stable_point/np.pi, 0, marker='o', color='r', s=100, zorder=100)
    
    #psi = piを通過する軌道のプロット
    if S_number > 0E0:
        theta_plus_pi = np.sqrt(0.5*((np.cos(wave_phase) - np.cos(np.pi)) - S_number * (wave_phase - np.pi)))
        theta_minus_pi = - np.sqrt(0.5*((np.cos(wave_phase) - np.cos(np.pi)) - S_number * (wave_phase - np.pi)))
        ax.plot(wave_phase/np.pi, theta_plus_pi, color='purple', lw=4, alpha=1)
        ax.plot(wave_phase/np.pi, theta_minus_pi, color='purple', lw=4, alpha=1)
    
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)

fig.tight_layout()
dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/psi_theta_trajectory'
os.makedirs(dir_name, exist_ok=True)
fig.savefig(f'{dir_name}/psi_theta_trajectory.png')
fig.savefig(f'{dir_name}/psi_theta_trajectory.pdf')