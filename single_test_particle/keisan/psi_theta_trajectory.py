import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

#フォントの設定
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 40

#定数
#S_number_array = [0E0, 2.5E-1, 5E-1, 7.5E-1, 1E0, 1.25E0]
S_number_array = [0E0, 2.5E-1, 5E-1, 1E0]

# wave_phaseの範囲
wave_phase_min = -np.pi
wave_phase_max = np.pi
wave_phase_num = 10000
wave_phase = np.linspace(wave_phase_min, wave_phase_max, wave_phase_num)

# theta_0_psiの範囲
theta_0_psi_min = - np.pi
theta_0_psi_max = np.pi
theta_0_psi_num = 21

# psi_pi_thetaの範囲
psi_pi_theta_min = 0E0
psi_pi_theta_max = 3E0
psi_pi_theta_num = 16


# plot
square_num = np.ceil(np.sqrt(len(S_number_array)))
square_num = int(square_num)
fig = plt.figure(figsize=(square_num*5, square_num*5), dpi=100)

plt.rcParams["font.size"] = square_num * 10
#ax_1 = fig.add_subplot(231, xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\Theta$', xlim=(-1, 1), ylim=(-3, 3))
#ax_2 = fig.add_subplot(232, xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\Theta$', xlim=(-1, 1), ylim=(-3, 3))
#ax_3 = fig.add_subplot(233, xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\Theta$', xlim=(-1, 1), ylim=(-3, 3))
#ax_4 = fig.add_subplot(234, xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\Theta$', xlim=(-1, 1), ylim=(-3, 3))
#ax_5 = fig.add_subplot(235, xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\Theta$', xlim=(-1, 1), ylim=(-3, 3))
#ax_6 = fig.add_subplot(236, xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\Theta$', xlim=(-1, 1), ylim=(-3, 3))
#axes = [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6]

#S_number_arrayの数だけsubplotを作成、長方形または正方形になるように調整
axes = []
for count_i in range(len(S_number_array)):
    ax = fig.add_subplot(square_num, square_num, count_i + 1, xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$', xlim=(-1, 1), ylim=(-3, 3))
    axes.append(ax)


def non_saddle_point_function(psi, S_number):
    return np.cos(psi) + np.sqrt(1E0 - S_number**2E0) - S_number * (psi + np.pi - np.arcsin(S_number))

def non_saddle_point_iteration(S_number):
    diff_psi = 1E-5
    psi_old = - np.arcsin(S_number) + diff_psi
    while True:
        func_1 = non_saddle_point_function(psi_old, S_number)
        func_2 = (non_saddle_point_function(psi_old + diff_psi, S_number) - non_saddle_point_function(psi_old - diff_psi, S_number)) / 2E0 / diff_psi
        diff = func_1 / func_2
        if np.abs(diff) > 1E-3:
            diff = 1E-3 * np.sign(diff)
        psi_new = psi_old - diff
        if np.abs(psi_new - psi_old) < 1E-10:
            break
        psi_old = psi_new
    return psi_new


for count_i in range(len(S_number_array)):
    S_number = S_number_array[count_i]
    ax = axes[count_i]
    ax.set_title(r'$S$ $=$ ' + f'{S_number:.2f}')
    #(a)のような図番号を付ける
    ax.text(-0.15, 1.0, '(' + chr(97 + count_i) + ')', transform=ax.transAxes)

    #各theta_0_psiを通る軌道のプロット
    for theta_0_psi in np.linspace(theta_0_psi_min, theta_0_psi_max, theta_0_psi_num):
        theta_plus = np.sqrt(0.5*((np.cos(wave_phase) - np.cos(theta_0_psi)) - S_number * (wave_phase - theta_0_psi)))
        theta_minus = - np.sqrt(0.5*((np.cos(wave_phase) - np.cos(theta_0_psi)) - S_number * (wave_phase - theta_0_psi)))
        ax.plot(wave_phase/np.pi, theta_plus, color='k', lw=1, alpha=0.3)
        ax.plot(wave_phase/np.pi, theta_minus, color='k', lw=1, alpha=0.3)
    
    #各psi_pi_thetaを通る軌道のプロット
    for psi_pi_theta in np.linspace(psi_pi_theta_min, psi_pi_theta_max, psi_pi_theta_num):
        theta_plus = np.sqrt(0.5*((np.cos(wave_phase) + 1E0) + S_number * (np.pi - wave_phase)) + psi_pi_theta**2E0)
        theta_minus = - np.sqrt(0.5*((np.cos(wave_phase) + 1E0) + S_number * (np.pi - wave_phase)) + psi_pi_theta**2E0)
        ax.plot(wave_phase/np.pi, theta_plus, color='k', lw=1, alpha=0.3)
        ax.plot(wave_phase/np.pi, theta_minus, color='k', lw=1, alpha=0.3)

    #鞍点、安定点の計算
    #非鞍点(theta = 0)の点の計算
    if S_number <= 1E0:
        stable_point = - np.arcsin(S_number)
        saddle_point = - np.pi - stable_point
        print(f'stable_point = {stable_point/np.pi:.2f} pi')
        print(f'saddle_point = {saddle_point/np.pi:.2f} pi')
        non_saddle_point = non_saddle_point_iteration(S_number)
        print(f'non_saddle_point = {non_saddle_point/np.pi:.2f} pi')
    

    #鞍点を通過する軌道のプロット
    if S_number <= 1E0:
        theta_plus_saddle_1 = np.sqrt(0.5*((np.cos(wave_phase) - np.cos(saddle_point)) - S_number * (wave_phase - saddle_point)))
        theta_minus_saddle_1 = - np.sqrt(0.5*((np.cos(wave_phase) - np.cos(saddle_point)) - S_number * (wave_phase - saddle_point)))
        wave_phase_saddle = np.linspace(saddle_point, non_saddle_point, 1000)
        theta_plus_saddle = np.sqrt(0.5*((np.cos(wave_phase_saddle) - np.cos(saddle_point)) - S_number * (wave_phase_saddle - saddle_point)))
        theta_minus_saddle = - np.sqrt(0.5*((np.cos(wave_phase_saddle) - np.cos(saddle_point)) - S_number * (wave_phase_saddle - saddle_point)))
        ax.plot(wave_phase/np.pi, theta_plus_saddle_1, color='k', lw=1, alpha=0.6)
        ax.plot(wave_phase/np.pi, theta_minus_saddle_1, color='k', lw=1, alpha=0.6)
        ax.plot(wave_phase_saddle/np.pi, theta_plus_saddle, color='orange', lw=1.3*square_num, alpha=1)
        ax.plot(wave_phase_saddle/np.pi, theta_minus_saddle, color='orange', lw=1.3*square_num, alpha=1)

        #鞍点と安定点のプロット
        ax.scatter(saddle_point/np.pi, 0, marker='o', color='b', s=33*square_num, zorder=100)
        ax.scatter(non_saddle_point/np.pi, 0, marker='o', color='g', s=33*square_num, zorder=100)
        ax.scatter(stable_point/np.pi, 0, marker='o', color='r', s=33*square_num, zorder=100)
    
    #psi = piを通過する軌道のプロット
    #if S_number > 0E0:
    #    theta_plus_pi = np.sqrt(0.5*((np.cos(wave_phase) - np.cos(np.pi)) - S_number * (wave_phase - np.pi)))
    #    theta_minus_pi = - np.sqrt(0.5*((np.cos(wave_phase) - np.cos(np.pi)) - S_number * (wave_phase - np.pi)))
    #    ax.plot(wave_phase/np.pi, theta_plus_pi, color='purple', lw=4, alpha=1)
    #    ax.plot(wave_phase/np.pi, theta_minus_pi, color='purple', lw=4, alpha=1)
    
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)

#余白を削除


fig.tight_layout()
dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/psi_theta_trajectory'
os.makedirs(dir_name, exist_ok=True)
fig.savefig(f'{dir_name}/psi_theta_trajectory_except_nonresonant.png')
fig.savefig(f'{dir_name}/psi_theta_trajectory_except_nonresonant.pdf')