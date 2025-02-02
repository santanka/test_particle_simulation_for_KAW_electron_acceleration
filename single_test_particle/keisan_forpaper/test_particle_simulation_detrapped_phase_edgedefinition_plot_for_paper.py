import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import datetime
import os
import netCDF4 as nc

# Font setting
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
font_size = 55
plt.rcParams["font.size"] = font_size

# Load data
def load_data_path(initial_Kperp_eq_min_eV, initial_Kperp_eq_max_eV, initial_S_value_min, initial_S_value_max, separate_number_mesh, separate_number_psi, psi_exit, initial_Kperp_eq_main, initial_S_value_main):
    dir_name = f'/test_particle_simulation_detrapped_phase_edgedefinition_forpaper/Kperp_eq_{initial_Kperp_eq_min_eV:.4f}_{initial_Kperp_eq_max_eV:.4f}_eV_S_{initial_S_value_min:.4f}_{initial_S_value_max:.4f}_{separate_number_mesh}_{separate_number_psi}_linear_psi_exit_{(psi_exit/np.pi):.2f}_forpaper'
    data_name = dir_name + f'/Kperp_eq_{initial_Kperp_eq_main:.4f}_eV_S_{initial_S_value_main:.4f}_psi_exit_{(psi_exit/np.pi):.2f}.nc'
    return data_name

def load_data(data_name):
    data = nc.Dataset(data_name, 'r', format='NETCDF4')
    return data

def figure_plot(data_name, data):

    # Load data
    initial_condition_group = data['initial_condition']
    initial_mu_main = initial_condition_group['mu'][:]
    initial_S_value_main = initial_condition_group['init_S_value'][:]
    initial_psi_main = initial_condition_group['init_psi'][:]

    # plot setting
    fig_name = data_name.replace('.nc', '.png')
    fig = plt.figure(figsize=(40, 40))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.05])

    cmap_color = cm.turbo
    color_vmin = np.nanmin(initial_psi_main) / np.pi
    color_vmax = np.nanmax(initial_psi_main) / np.pi
    if color_vmin == color_vmax:
        color_vmin = color_vmin - 1E-3
        color_vmax = color_vmax + 1E-3
    norm = mpl.colors.Normalize(vmin=color_vmin, vmax=color_vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_color)
    sm.set_array([])
    ax_cbar = fig.add_subplot(gs[3, :])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(r'$\psi_{\mathrm{i}}$ [$\pi$ $\mathrm{rad}$]')

    ax_1_1 = fig.add_subplot(gs[0, 0], xlabel=r'$\lambda$ [deg]', ylabel=r'$v_{\parallel}$ [$c$]')
    ax_2_1 = fig.add_subplot(gs[1, 0], xlabel=r'$\lambda$ [deg]', ylabel=r'$K$ [eV]', yscale='log')
    ax_3_1 = fig.add_subplot(gs[2, 0], xlabel=r'$\lambda$ [deg]', ylabel=r'$\alpha$ [$\mathrm{deg}$]', yticks=[0, 30, 60, 90, 120, 150, 180])
    ax_1_2 = fig.add_subplot(gs[0, 1], xlabel=r'$\lambda$ [deg]', ylabel=r'$S$', yscale='log')
    ax_2_2 = fig.add_subplot(gs[1, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$')
    ax_3_2 = fig.add_subplot(gs[2, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\lambda$ [deg]')
    ax_1_3 = fig.add_subplot(gs[0, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$S$', yscale='log')
    ax_2_3 = fig.add_subplot(gs[1, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$K$ [eV]', yscale='log')
    ax_3_3 = fig.add_subplot(gs[2, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\mathrm{d} K / \mathrm{d} t$ [eV/s]')

    ax_2_2.set_yscale('symlog', linthresh=1E0)
    ax_3_3.set_yscale('symlog', linthresh=1E2)

    fig.suptitle(r'$\mu = %.3f \, \mathrm{eV/nT}, \, S_{\mathrm{i}} = %.4f$' % (initial_mu_main, initial_S_value_main))

    for group_name in data.groups.keys():
        if group_name == 'background' or group_name == 'initial_condition':
            continue
        group = data[group_name]

        # Load data
        mlat_rad_particle = group['mlat_rad'][:]    # [rad]
        theta_particle = group['theta'][:]        # [rad/s]
        psi_particle = group['psi'][:]          # [rad]
        S_value_particle = group['S_value'][:]    # []
        vpara_particle = group['vpara'][:]      # [c]
        energy_particle = group['energy'][:]    # [eV]
        pitch_angle_particle = group['alpha'][:]    # [deg]
        trapping_freqeuncy_particle = group['trapping_frequency'][:]    # [rad/s]
        dKdt_particle = group['d_K_d_t_eV_s'][:]        # [eV/s]

        # Plot
        initial_psi_color = psi_particle[0] / np.pi * np.ones_like(psi_particle)

        mlat_deg_particle = mlat_rad_particle * 180. / np.pi    # [deg]

        ax_1_1.scatter(mlat_deg_particle, vpara_particle, s=0.5, c=initial_psi_color, cmap=cmap_color, norm=norm)
        ax_1_1.scatter(mlat_deg_particle[0], vpara_particle[0], s=200, marker='o', edgecolors='k', zorder=1, c='lightgrey')
        ax_1_1.scatter(mlat_deg_particle[-1], vpara_particle[-1], s=200, marker='D', edgecolors='k', zorder=1, c='orange')

        ax_2_1.scatter(mlat_deg_particle, energy_particle, s=0.5, c=initial_psi_color, cmap=cmap_color, norm=norm)
        ax_2_1.scatter(mlat_deg_particle[0], energy_particle[0], s=200, marker='o', edgecolors='k', zorder=1, c='lightgrey')
        ax_2_1.scatter(mlat_deg_particle[-1], energy_particle[-1], s=200, marker='D', edgecolors='k', zorder=1, c='orange')

        ax_3_1.scatter(mlat_deg_particle, pitch_angle_particle, s=0.5, c=initial_psi_color, cmap=cmap_color, norm=norm)
        ax_3_1.scatter(mlat_deg_particle[0], pitch_angle_particle[0], s=200, marker='o', edgecolors='k', zorder=1, c='lightgrey')
        ax_3_1.scatter(mlat_deg_particle[-1], pitch_angle_particle[-1], s=200, marker='D', edgecolors='k', zorder=1, c='orange')

        ax_1_2.scatter(mlat_deg_particle, S_value_particle, s=0.5, c=initial_psi_color, cmap=cmap_color, norm=norm)
        ax_1_2.scatter(mlat_deg_particle[0], S_value_particle[0], s=200, marker='o', edgecolors='k', zorder=1, c='lightgrey')
        ax_1_2.scatter(mlat_deg_particle[-1], S_value_particle[-1], s=200, marker='D', edgecolors='k', zorder=1, c='orange')

        ax_2_2.scatter(psi_particle / np.pi, theta_particle / 2. / trapping_freqeuncy_particle, s=0.5, c=initial_psi_color, cmap=cmap_color, norm=norm)
        ax_2_2.scatter(psi_particle[0] / np.pi, theta_particle[0] / 2. / trapping_freqeuncy_particle[0], s=200, marker='o', edgecolors='k', zorder=1, c='lightgrey')
        ax_2_2.scatter(psi_particle[-1] / np.pi, theta_particle[-1] / 2. / trapping_freqeuncy_particle[-1], s=200, marker='D', edgecolors='k', zorder=1, c='orange')

        ax_3_2.scatter(psi_particle / np.pi, mlat_deg_particle, s=0.5, c=initial_psi_color, cmap=cmap_color, norm=norm)
        ax_3_2.scatter(psi_particle[0] / np.pi, mlat_deg_particle[0], s=200, marker='o', edgecolors='k', zorder=1, c='lightgrey')
        ax_3_2.scatter(psi_particle[-1] / np.pi, mlat_deg_particle[-1], s=200, marker='D', edgecolors='k', zorder=1, c='orange')

        ax_1_3.scatter(psi_particle / np.pi, S_value_particle, s=0.5, c=initial_psi_color, cmap=cmap_color, norm=norm)
        ax_1_3.scatter(psi_particle[0] / np.pi, S_value_particle[0], s=200, marker='o', edgecolors='k', zorder=1, c='lightgrey')
        ax_1_3.scatter(psi_particle[-1] / np.pi, S_value_particle[-1], s=200, marker='D', edgecolors='k', zorder=1, c='orange')

        ax_2_3.scatter(psi_particle / np.pi, energy_particle, s=0.5, c=initial_psi_color, cmap=cmap_color, norm=norm)
        ax_2_3.scatter(psi_particle[0] / np.pi, energy_particle[0], s=200, marker='o', edgecolors='k', zorder=1, c='lightgrey')
        ax_2_3.scatter(psi_particle[-1] / np.pi, energy_particle[-1], s=200, marker='D', edgecolors='k', zorder=1, c='orange')

        ax_3_3.scatter(psi_particle / np.pi, dKdt_particle, s=0.5, c=initial_psi_color, cmap=cmap_color, norm=norm)
        ax_3_3.scatter(psi_particle[0] / np.pi, dKdt_particle[0], s=200, marker='o', edgecolors='k', zorder=1, c='lightgrey')
        ax_3_3.scatter(psi_particle[-1] / np.pi, dKdt_particle[-1], s=200, marker='D', edgecolors='k', zorder=1, c='orange')

    xlim_ax_1_1 = ax_1_1.get_xlim()
    ylim_ax_1_1 = ax_1_1.get_ylim()
    xlim_ax_2_1 = ax_2_1.get_xlim()
    ylim_ax_2_1 = ax_2_1.get_ylim()
    xlim_ax_3_1 = ax_3_1.get_xlim()
    ylim_ax_3_1 = ax_3_1.get_ylim()

    # load background data
    background_group = data['background']
    mlat_deg_background = background_group['mlat_deg'][:]   # [deg]
    Vphpara_background = background_group['Vphpara_for_background'][:]  # [c]
    Kphpara_background = background_group['Kphpara_for_background'][:]  # [eV]
    K_E_background = background_group['K_E_for_background'][:]  # [eV]
    Kperp_background = background_group['Kperp_for_background'][:]  # [eV]
    loss_cone_angle_background = background_group['loss_cone_for_background'][:]    # [deg]

    # plot background data
    ax_1_1.plot(mlat_deg_background, Vphpara_background, c='r', linewidth=4, zorder=-1, alpha=0.6, linestyle='-.')
    ax_1_1.axhline(y=0, c='k', linewidth=4, zorder=-1, alpha=0.3, linestyle='--')

    ax_2_1.plot(mlat_deg_background, Kphpara_background + Kperp_background, c='r', linewidth=4, zorder=-1, alpha=0.6, linestyle='-.')
    ax_2_1.plot(mlat_deg_background, K_E_background, c='g', linewidth=4, zorder=-1, alpha=0.6, linestyle='-.')
    ax_2_1.plot(mlat_deg_background, Kperp_background, c='orange', linewidth=4, zorder=-1, alpha=0.6, linestyle='-.')

    ax_3_1.plot(mlat_deg_background, loss_cone_angle_background, c='k', linewidth=4, zorder=-1, alpha=0.6, linestyle='-.')
    ax_3_1.plot(mlat_deg_background, 180. - loss_cone_angle_background, c='k', linewidth=4, zorder=-1, alpha=0.6, linestyle='-.')
    ax_3_1.axhline(y=90., c='k', linewidth=4, zorder=-1, alpha=0.3, linestyle='--')
    ylim_ax_3_1 = [np.nanmax([ylim_ax_3_1[0], 0]), np.nanmin([ylim_ax_3_1[1], 180])]

    ax_1_1.set_xlim(xlim_ax_1_1)
    ax_1_1.set_ylim(ylim_ax_1_1)
    ax_2_1.set_xlim(xlim_ax_2_1)
    ax_2_1.set_ylim(ylim_ax_2_1)
    ax_3_1.set_xlim(xlim_ax_3_1)
    ax_3_1.set_ylim(ylim_ax_3_1)

    ax_1_2.axhline(y=1., c='k', linewidth=4, zorder=-1, alpha=0.3, linestyle='--')

    xlim_ax_2_2 = ax_2_2.get_xlim()
    ylim_ax_2_2 = ax_2_2.get_ylim()

    psi_array_background = np.linspace(-np.pi, np.pi, 10000)
    Theta_array_background = np.sqrt(5E-1 * (np.cos(psi_array_background) + np.sqrt(1E0 - initial_S_value_main**2E0) - initial_S_value_main * (psi_array_background + np.pi - np.arcsin(initial_S_value_main))))
    ax_2_2.plot(psi_array_background / np.pi, Theta_array_background, c='k', linewidth=4, zorder=-1, alpha=0.6, linestyle='-.')
    ax_2_2.plot(psi_array_background / np.pi, -Theta_array_background, c='k', linewidth=4, zorder=-1, alpha=0.6, linestyle='-.')
    ax_2_2.axhline(y=0., c='k', linewidth=4, zorder=-1, alpha=0.3, linestyle='--')

    ax_1_3.axhline(y=1., c='k', linewidth=4, zorder=-1, alpha=0.3, linestyle='--')
    ax_3_3.axhline(y=0., c='k', linewidth=4, zorder=-1, alpha=0.3, linestyle='--')

    for count_j in range(int(xlim_ax_2_2[0]), int(xlim_ax_2_2[1]) + 1):
        ax_2_2.axvline(x=count_j, color='k', linewidth=4, zorder=-1, alpha=0.3, linestyle='--')
        ax_3_2.axvline(x=count_j, color='k', linewidth=4, zorder=-1, alpha=0.3, linestyle='--')
        ax_1_3.axvline(x=count_j, color='k', linewidth=4, zorder=-1, alpha=0.3, linestyle='--')
        ax_2_3.axvline(x=count_j, color='k', linewidth=4, zorder=-1, alpha=0.3, linestyle='--')
        ax_3_3.axvline(x=count_j, color='k', linewidth=4, zorder=-1, alpha=0.3, linestyle='--')

    ax_2_2.set_xlim(xlim_ax_2_2)
    ax_2_2.set_ylim(ylim_ax_2_2)
    ax_3_2.set_xlim(xlim_ax_2_2)
    ax_1_3.set_xlim(xlim_ax_2_2)
    ax_2_3.set_xlim(xlim_ax_2_2)
    ax_3_3.set_xlim(xlim_ax_2_2)

    axes = [ax_1_1, ax_2_1, ax_3_1, ax_1_2, ax_2_2, ax_3_2, ax_1_3, ax_2_3, ax_3_3]
    for ax in axes:
        ax.minorticks_on()
        ax.grid(which='both', alpha=0.3)
        ax.text(-0.20, 1, '(' + chr(97 + axes.index(ax)) + ')', transform=ax.transAxes)
    
    fig.tight_layout(w_pad=0.3, h_pad=0.0)
    fig.savefig(fig_name)
    fig.savefig(fig_name.replace('.png', '.pdf'))
    plt.close()

def main(initial_Kperp_eq_min_eV, initial_Kperp_eq_max_eV, initial_S_value_min, initial_S_value_max, separate_number_mesh, separate_number_psi, psi_exit, initial_Kperp_eq_main, initial_S_value_main):
    data_name = load_data_path(initial_Kperp_eq_min_eV, initial_Kperp_eq_max_eV, initial_S_value_min, initial_S_value_max, separate_number_mesh, separate_number_psi, psi_exit, initial_Kperp_eq_main, initial_S_value_main)
    print(data_name)
    data = load_data(data_name)
    print(data)
    figure_plot(data_name, data)

    return

# Run
# initial_Kperp_eq_min_eV, initial_Kperp_eq_max_eV, initial_S_value_min, initial_S_value_max, separate_number_mesh, separate_number_psi, grid_scale, psi_exit, initial_Kperp_eq_main, initial_S_value_main
main(1E0, 1E3, 1E-1, 1E0, 20, 10, -1E1 * np.pi, 1E0, 1E-1)