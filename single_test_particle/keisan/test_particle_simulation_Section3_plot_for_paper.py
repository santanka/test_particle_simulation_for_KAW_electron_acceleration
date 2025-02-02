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
def load_data_path(initial_K_eV, initial_pitch_angle_deg, initial_mlat_deg, initial_psi):
    dir_name = f'/mnt/j/KAW_simulation_data/single_test_particle/keisan/test_particle_simulation_Section3/{initial_psi/np.pi:.2f}_pi'
    data_name = f'{dir_name}/{initial_K_eV:.2f}eV_{initial_pitch_angle_deg:.2f}deg_{initial_mlat_deg:.2f}deg.nc'
    return data_name

def load_data(data_name):
    data = nc.Dataset(data_name, 'r')
    return data

def figure_plot(data_name, data):

    # Load data
    time_array = data['time'][:]    # [s]
    mlat_rad_array = data['mlat_rad'][:]    # [rad]
    vpara_array = data['vpara'][:]    # [c]
    Ktotal_array = data['energy'][:]    # [eV]
    pitch_angle_array = data['alpha'][:]    # [deg]
    S_value_array = data['S_value'][:]  # []
    psi_array = data['psi'][:]  # [rad]
    theta_array = data['theta'][:]  # [rad/s]
    trapping_frequency_array = data['trapping_frequency'][:]  # [rad/s]
    detrapped_point_array = data['detrapped_point'][:]

    detrapped_point = np.where(detrapped_point_array == 1)[0]
    trapped_point = np.where(detrapped_point_array == 2)[0]

    # Plot
    fig = plt.figure(figsize=(25*1.5, 20*1.5), dpi=100)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.05])

    cmap_color = cm.turbo
    color_target = time_array
    vmin_color = np.min(color_target)
    vmax_color = np.max(color_target)
    norm_color = mpl.colors.Normalize(vmin=vmin_color, vmax=vmax_color)
    scalarMap_color = plt.cm.ScalarMappable(norm=norm_color, cmap=cmap_color)
    scalarMap_color.set_array([])

    ax_cbar = fig.add_subplot(gs[2, :])
    cbar = fig.colorbar(scalarMap_color, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(r'Time $t$ [s]')

    ax_1_1 = fig.add_subplot(gs[0, 0], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$v_{\parallel}$ [$c$]')
    ax_1_2 = fig.add_subplot(gs[0, 1], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$K$ [eV]', yscale='log')
    ax_1_3 = fig.add_subplot(gs[0, 2], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$\alpha$ [deg]', yticks=[0, 30, 60, 90, 120, 150, 180])
    ax_2_1 = fig.add_subplot(gs[1, 0], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'$S$', yscale='log')
    ax_2_2 = fig.add_subplot(gs[1, 1], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$\theta / 2 \omega_{\mathrm{t}}$', xlim=(-1E0, 1E0))
    ax_2_3 = fig.add_subplot(gs[1, 2], xlabel=r'$\psi$ [$\pi$ rad]', ylabel=r'$K$ [eV]', yscale='log', xlim=(-1E0, 1E0))

    ax_1_1.scatter(mlat_rad_array * 180E0 / np.pi, vpara_array, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_1_1.scatter(mlat_rad_array[0] * 180E0 / np.pi, vpara_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_1_1.scatter(mlat_rad_array[-1] * 180E0 / np.pi, vpara_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_1_1.scatter(mlat_rad_array[detrapped_point] * 180E0 / np.pi, vpara_array[detrapped_point], c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_1_1.scatter(mlat_rad_array[trapped_point] * 180E0 / np.pi, vpara_array[trapped_point], c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)

    ax_1_2.scatter(mlat_rad_array * 180E0 / np.pi, Ktotal_array, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_1_2.scatter(mlat_rad_array[0] * 180E0 / np.pi, Ktotal_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_1_2.scatter(mlat_rad_array[-1] * 180E0 / np.pi, Ktotal_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_1_2.scatter(mlat_rad_array[detrapped_point] * 180E0 / np.pi, Ktotal_array[detrapped_point], c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_1_2.scatter(mlat_rad_array[trapped_point] * 180E0 / np.pi, Ktotal_array[trapped_point], c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)

    ax_1_3.scatter(mlat_rad_array * 180E0 / np.pi, pitch_angle_array, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_1_3.scatter(mlat_rad_array[0] * 180E0 / np.pi, pitch_angle_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_1_3.scatter(mlat_rad_array[-1] * 180E0 / np.pi, pitch_angle_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_1_3.scatter(mlat_rad_array[detrapped_point] * 180E0 / np.pi, pitch_angle_array[detrapped_point], c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_1_3.scatter(mlat_rad_array[trapped_point] * 180E0 / np.pi, pitch_angle_array[trapped_point], c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)

    ax_2_1.scatter(mlat_rad_array * 180E0 / np.pi, S_value_array, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_2_1.scatter(mlat_rad_array[0] * 180E0 / np.pi, S_value_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_2_1.scatter(mlat_rad_array[-1] * 180E0 / np.pi, S_value_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_2_1.scatter(mlat_rad_array[detrapped_point] * 180E0 / np.pi, S_value_array[detrapped_point], c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_2_1.scatter(mlat_rad_array[trapped_point] * 180E0 / np.pi, S_value_array[trapped_point], c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_2_1.axhline(y=1E0, color='k', linewidth=4E0, zorder=0, alpha=0.3, linestyle='--')

    ax_2_2.scatter(psi_array / np.pi, theta_array / trapping_frequency_array / 2E0, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_2_2.scatter(psi_array[0] / np.pi, theta_array[0] / trapping_frequency_array[0] / 2E0, c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_2_2.scatter(psi_array[-1] / np.pi, theta_array[-1] / trapping_frequency_array[-1] / 2E0, c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_2_2.scatter(psi_array[detrapped_point] / np.pi, theta_array[detrapped_point] / trapping_frequency_array[detrapped_point] / 2E0, c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_2_2.scatter(psi_array[trapped_point] / np.pi, theta_array[trapped_point] / trapping_frequency_array[trapped_point] / 2E0, c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_2_2.axhline(y=0E0, color='k', linewidth=4E0, zorder=0, alpha=0.3, linestyle='--')

    ax_2_3.scatter(psi_array / np.pi, Ktotal_array, c=color_target, cmap=cmap_color, vmin=vmin_color, vmax=vmax_color, s=1, zorder=1)
    ax_2_3.scatter(psi_array[0] / np.pi, Ktotal_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax_2_3.scatter(psi_array[-1] / np.pi, Ktotal_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax_2_3.scatter(psi_array[detrapped_point] / np.pi, Ktotal_array[detrapped_point], c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax_2_3.scatter(psi_array[trapped_point] / np.pi, Ktotal_array[trapped_point], c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)

    mlat_deg_for_background = data['mlat_deg_for_background'][:]   # [deg]
    parallel_phase_speed_for_background = data['Vphpara_for_background'][:]   # [c]
    energy_wave_phase_speed_for_background = data['Kphpara_for_background'][:]  # [eV]
    energy_wave_potential_for_background = data['K_E_for_background'][:]    # [eV]
    energy_perp_for_background = data['Kperp_for_background'][:]    # [eV]
    loss_cone_for_background = data['loss_cone_for_background'][:]  # [deg]

    xlim_enlarged_1_1 = ax_1_1.get_xlim()
    ylim_enlarged_1_1 = ax_1_1.get_ylim()
    xlim_enlarged_1_2 = ax_1_2.get_xlim()
    ylim_enlarged_1_2 = ax_1_2.get_ylim()
    xlim_enlarged_1_3 = ax_1_3.get_xlim()
    ylim_enlarged_1_3 = ax_1_3.get_ylim()

    ylim_enlarged_1_3 = ax_1_3.get_ylim()
    ylim_enlarged_1_3 = [np.nanmax([ylim_enlarged_1_3[0], 0E0]), np.nanmin([ylim_enlarged_1_3[1], 180E0])]

    ax_1_1.plot(mlat_deg_for_background, parallel_phase_speed_for_background, c='r', linewidth=4E0, zorder=0, alpha=0.6, label=r'$V_{\mathrm{ph} \parallel}$')
    ax_1_1.axhline(y=0E0, color='k', linewidth=4E0, zorder=0, alpha=0.3, linestyle='--')

    ax_1_2.plot(mlat_deg_for_background, (energy_wave_phase_speed_for_background + energy_perp_for_background), c='r', linewidth=4E0, zorder=0, alpha=0.6, label=r'$K_{\perp} + K_{\mathrm{ph \parallel}}$')
    ax_1_2.plot(mlat_deg_for_background, energy_wave_potential_for_background, c='g', linewidth=4E0, label=r'$K_{\mathrm{E}}$', alpha=0.6)
    ax_1_2.plot(mlat_deg_for_background, energy_perp_for_background, c='orange', linewidth=4E0, label=r'$K_{\perp}$', alpha=0.6)

    ax_1_3.plot(mlat_deg_for_background, loss_cone_for_background, c='k', linewidth=4E0, zorder=0, alpha=0.6, label=r'Loss cone')
    ax_1_3.plot(mlat_deg_for_background, 180E0 - loss_cone_for_background, c='k', linewidth=4E0, zorder=0, alpha=0.6)
    ax_1_3.axhline(y=90E0, color='k', linewidth=4E0, zorder=0, alpha=0.3, linestyle='--')

    ax_1_1.set_xlim(xlim_enlarged_1_1)
    ax_1_1.set_ylim(ylim_enlarged_1_1)
    ax_1_2.set_xlim(xlim_enlarged_1_2)
    ax_1_2.set_ylim(ylim_enlarged_1_2)
    ax_1_3.set_xlim(xlim_enlarged_1_3)
    ax_1_3.set_ylim(ylim_enlarged_1_3)

    ylim_enlarged_2_2 = [np.nanmin(theta_array / 2E0 / trapping_frequency_array)-0.1, np.nanmax(theta_array / 2E0 / trapping_frequency_array)+0.1]
    if ylim_enlarged_2_2[0] < -3E0:
        ylim_enlarged_2_2[0] = -3E0
    if ylim_enlarged_2_2[1] > 3E0:
        ylim_enlarged_2_2[1] = 3E0
    ax_2_2.set_ylim(ylim_enlarged_2_2)
    
    axes = [ax_1_1, ax_1_2, ax_1_3, ax_2_1, ax_2_2, ax_2_3]

    for ax in axes:
        ax.minorticks_on()
        ax.grid(which='both', alpha=0.3)
        ax.text(-0.20, 1.0, '(' + chr(97 + axes.index(ax)) + ')', transform=ax.transAxes)

    fig.suptitle(r'$K_{\mathrm{i}} = %.1f$ eV, $\alpha_{\mathrm{i}} = %.1f$ deg, $\lambda_{\mathrm{i}} = %.1f$ deg, $\psi_{\mathrm{i}} = %.1f \pi$ rad' % (Ktotal_array[0], pitch_angle_array[0], mlat_rad_array[0] * 180E0 / np.pi, psi_array[0] / np.pi))

    fig.tight_layout(w_pad=0.3, h_pad=0.0)

    # Save figure
    fig_name = f'{data_name[:-3]}_test'
    fig.savefig(fig_name + '.png')
    fig.savefig(fig_name + '.pdf')
    plt.close()

    return

def figure_contour_plot(data_name, data):

    # Load data
    time_array = data['time'][:]    # [s]
    mlat_rad_array = data['mlat_rad'][:]    # [rad]
    Ktotal_energy_array = data['energy'][:]    # [eV]
    alpha_array = data['alpha'][:]    # [deg]
    psi_array = data['psi'][:]    # [rad]

    mlat_deg_contour = data['mlat_deg_for_background'][:]    # [deg]
    time_array_contour = data['time_background'][:]    # [s]
    mesh_time_array_contour, mesh_mlat_deg_contour = np.meshgrid(time_array_contour, mlat_deg_contour)
    force_electric_field_array = data['F_Epara'][:][:]    # [mV/m]
    
    detrapped_point_array = data['detrapped_point'][:]
    detrapped_point = np.where(detrapped_point_array == 1)[0]
    trapped_point = np.where(detrapped_point_array == 2)[0]

    # Plot
    fig = plt.figure(figsize=(20, 20), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])

    cmap_color = cm.bwr
    color_target = force_electric_field_array
    vmax_color = np.max(np.abs(color_target))
    vmin_color = - vmax_color
    norm_color = mpl.colors.Normalize(vmin=vmin_color, vmax=vmax_color)
    scalarMap_color = plt.cm.ScalarMappable(norm=norm_color, cmap=cmap_color)
    scalarMap_color.set_array([])

    ax_cbar = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(scalarMap_color, cax=ax_cbar, orientation='vertical')
    cbar.set_label(r'Force of $\delta E_{\parallel}$ [$10^{-3}$ eV/m]')

    ax = fig.add_subplot(gs[0, 0], xlabel=r'MLAT $\lambda$ [deg]', ylabel=r'Time $t$ [s]')
    ax.contourf(mesh_mlat_deg_contour, mesh_time_array_contour, force_electric_field_array, cmap=cmap_color, norm=norm_color, levels=1000)
    ax.scatter(mlat_rad_array * 180E0 / np.pi, time_array, c='k', s=1, zorder=1)
    ax.scatter(mlat_rad_array[0] * 180E0 / np.pi, time_array[0], c='lightgrey', s=200, marker='o', edgecolors='k', zorder=100)
    ax.scatter(mlat_rad_array[-1] * 180E0 / np.pi, time_array[-1], c='orange', s=200, marker='D', edgecolors='k', zorder=100)
    ax.scatter(mlat_rad_array[detrapped_point] * 180E0 / np.pi, time_array[detrapped_point], c='magenta', s=1000, marker='*', edgecolors='k', zorder=100)
    ax.scatter(mlat_rad_array[trapped_point] * 180E0 / np.pi, time_array[trapped_point], c='cyan', s=1000, marker='*', edgecolors='k', zorder=100)

    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    xlim_max = np.max(mlat_rad_array * 180E0 / np.pi)
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(time_array[0], time_array[-1])

    fig.suptitle(r'$K_{\mathrm{i}} = %.1f$ eV, $\alpha_{\mathrm{i}} = %.1f$ deg, $\lambda_{\mathrm{i}} = %.1f$ deg, $\psi_{\mathrm{i}} = %.1f \pi$ rad' % (Ktotal_energy_array[0], alpha_array[0], mlat_rad_array[0] * 180E0 / np.pi, psi_array[0] / np.pi), fontsize=font_size*0.9)

    fig.tight_layout()

    # Save figure
    fig_name = f'{data_name[:-3]}_test_contour'
    fig.savefig(fig_name + '.png')
    fig.savefig(fig_name + '.pdf')
    plt.close()


def main(initial_K_eV, initial_pitch_angle_deg, initial_mlat_deg, initial_psi):
    data_name = load_data_path(initial_K_eV, initial_pitch_angle_deg, initial_mlat_deg, initial_psi)
    data = load_data(data_name)

    figure_plot(data_name, data)
    figure_contour_plot(data_name, data)

    return

# Run
# initial_K_eV, initial_pitch_angle_deg, initial_mlat_deg, initial_psi
main(100, 5, 1, -5E-1 * np.pi)