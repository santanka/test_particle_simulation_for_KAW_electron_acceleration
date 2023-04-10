import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mathtext
from matplotlib import cm
from numpy.lib.twodim_base import tri
mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants

df  = np.genfromtxt(r"/home/satanka/Documents/fort/TPS-DAW/TPS-KAW/results_particle_nowave/myrank000/particle_trajectory05-086.dat")
df2 = np.genfromtxt(r"/home/satanka/Documents/fort/TPS-DAW/TPS-KAW/results_particle_nowave/myrank000/potential_prof.dat")

channel = 1
#channel概要
#1:電子軌道, 2:エネルギー時間変化, 3:赤道ピッチ角時間変化, 4:ピッチ角時間変化, 5:エネルギー時間変化&赤道ピッチ角時間変化,
#6:磁気緯度vs波の周波数, 7:磁気緯度vs波長, 8:磁気緯度 vs Alfven速度, 9:磁気緯度 vs delta_E_para (t=0)
#10:磁気緯度 vs delta_B_para (t=0), 11:磁力線座標 vs delta_B_perp (t=0), 12:磁力線座標 vs delta_E_perp_perp (t=0)
#13:磁力線座標 vs delta_E_perp_phi (t=0), 14:磁力線座標 vs 静電ポテンシャル, 15:磁気緯度vs垂直波長&Larmor半径
#16:時間vs第一断熱不変量, 17:磁力線座標vs波の位相 (t=0), 18:電子ジャイロ位相時間変化, 19:電子が見る波の位相時間変化
#20:電子が見る波の位相vs共鳴速度との差, 21:電子が見る波の位相の垂直成分の時間変化, 22:電子が見る波の位相の平行成分の時間変化
#23:力の平行成分の時間変化, 24:力の平行成分の時間変化(delta_B_paraのミラー力), 25:u_particleの時間変化, 26:wave growth phaseの時間変化

trigger = 1 #(1: wave_check)

switch_delta_B_para = 0E0
switch_delta_E_para = 0E0

#規格化定数
c  = 299792458E0
q  = 1.6021766208E-19
m  = 9.10938356E-31

R_E = 6371E3
L = 9E0
T_ion = 1E3 #[eV]
T_electron = 1E2 #[eV]
moment  = 7.75E22 #the Earth's dipole moment model

n_i = 1E0 #[cm^-3]
m_i = 1.672621898E-24 #[g]
p_i = n_i * T_ion*q*1E7 #cgs
p_e = n_i * T_electron*q*1E7 #cgs

ep0 = 600 * 1E8 / (c*1E2)
MLAT_position_threshold = 2E0 * np.pi / 180E0
wave_initial_to_threshold = 1E3
ep = ep0 / wave_initial_to_threshold
z_position_threshold = (R_E*L*1E2) * (np.arcsinh(np.sqrt(3E0)*np.sin(MLAT_position_threshold))/2/np.sqrt(3) + np.sin(MLAT_position_threshold)*np.sqrt(5-3*np.cos(2*MLAT_position_threshold)) /2/np.sqrt(2))


mu_0    = 4E0 * np.pi * 1E-7
B0_eq     = (1E-7 * moment) / (L * R_E)**3
Omega0_eq = q * B0_eq / m
z_unit = c / Omega0_eq
t_unit = 1E0 / Omega0_eq
J_unit = m * c**2E0
V_unit = m * c**2E0 / q
#print(t_unit, z_unit)


limit_under = 0
limit = 300000

time = df[limit_under : limit, 1]   #df[:, 1]
z_particle = df[limit_under : limit, 2]/R_E   #df[:, 2]/R_E
u_z_particle = df[limit_under : limit, 3]   #df[:, 3]
u_perp_particle = df[limit_under : limit, 4]    #df[:, 4]
u_phase_particle = df[limit_under : limit, 5]   #df[:, 5]
energy_particle = df[limit_under : limit, 6]    #df[:, 6]
pitch_angle = np.mod(np.arctan(u_perp_particle/u_z_particle), np.pi)
pitch_angle_eq = df[limit_under : limit, 7] #df[:, 7] #deg
wave_phase = df[limit_under : limit, 8]     #df[:, 8] #rad
wave_growth_phase = df[limit_under : limit, 9]     #df[:, 9]
v_z_particle = u_z_particle / np.sqrt(1 + (u_z_particle**2 + u_perp_particle**2)/c**2)
v_perp_particle = u_perp_particle / np.sqrt(1 + (u_z_particle**2 + u_perp_particle**2)/c**2)
gamma = np.sqrt(1 + (u_z_particle**2 + u_perp_particle**2)/c**2)

if (trigger == 1):
    z_position = df2[:, 0]
    wave_number_para = df2[:, 1]
    wave_number_perp = df2[:, 2]
    wave_frequency = df2[:, 3]
    V_resonant = df2[:, 4]
    electrostatic_potential = df2[:, 5]
    EE_wave_para = df2[:, 6]
    EE_wave_para_nonphase =  df2[:, 1] * df2[:, 5] * (2. + T_electron/T_ion)
    EE_wave_perp_perp = df2[:, 7]
    EE_wave_perp_phi = df2[:, 8]
    BB_wave_para = df2[:, 9]
    BB_wave_perp = df2[:, 10]
    Alfven_velocity = df2[:, 11]
    V_resonant_wide_plus = V_resonant + np.sqrt(np.abs(q * EE_wave_para_nonphase / m / wave_number_para))
    V_resonant_wide_minus = V_resonant - np.sqrt(np.abs(q * EE_wave_para_nonphase / m / wave_number_para))
    ion_Larmor_radius = df2[:, 12]

if (channel == 1):
    length = len(z_particle)
    MLAT = np.zeros(length)
    for jj in range(length):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_particle(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
                + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
                - z_particle[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-5):
                break

            MLAT0 = MLAT1
        
        MLAT[jj] = MLAT1
    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='MLAT [degree]', ylabel=r'$v_{\parallel}/c$')    
    #ax.plot(z_particle, v_z_particle/c, zorder=1, color='b')
    plt.scatter(MLAT/np.pi*180, v_z_particle/c, c=time, cmap=cm.turbo, marker='.', lw=0)
    pp = plt.colorbar()
    pp.set_label('time [s]')
    ax.scatter(MLAT[0]/np.pi*180, v_z_particle[0]/c, marker='o', color='r', label='start', zorder=3, s=200)
    ax.scatter(MLAT[length-1]/np.pi*180, v_z_particle[length-1]/c, marker='D', color='r', label='end', zorder=3, s=200)  #[len(z_particle)-1]
    if (trigger == 1):
        length2 = len(z_position)
        MLAT_position = np.zeros(length2)
        for jj in range(length2):
            MLAT0 = 1.
            for ii in range(1000000):
                if (ii == 1000000):
                    print("Error!: solution is not found. z_position = " + str(z_position(jj)))
                
                ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
                + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
                - z_position[jj]*R_E
                gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

                MLAT1 = float(MLAT0 - ff / gg)
            
                if (abs(MLAT1 - MLAT0) <= 1E-5):
                    break

                MLAT0 = MLAT1
        
            MLAT_position[jj] = MLAT1
        
        ax.plot(MLAT_position/np.pi*180, V_resonant/c, linestyle='-.', color='red', linewidth='4')
        #ax.plot(z_position, Alfven_velocity/c, linestyle='-.', color='orange', linewidth='4')
        #ax.plot(z_position, V_resonant_wide_plus/c, linestyle='-.', color='green', linewidth='4')
        #ax.plot(z_position, V_resonant_wide_minus/c, linestyle='-.', color='green', linewidth='4')
    #fig.suptitle('particle trajectory')
    ax.minorticks_on()
    ax.grid(which="both")
    ax.set_axisbelow(True)
    ax.legend()
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

if (channel == 2):
    fig = plt.figure()
    plt.rcParams["font.size"] = 50
    ax = fig.add_subplot(111, xlabel='time [s]', ylabel='energy [eV]')    
    plt.scatter(time, energy_particle, c=time, cmap=cm.turbo, marker='.', lw=0)
    pp = plt.colorbar()
    pp.set_label('time [s]')
    #fig.suptitle('Evolution of particle energy')
    ax.minorticks_on()
    ax.grid(which="both")

if (channel == 3):
    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    ax = fig.add_subplot(111, xlabel='time [s]', ylabel='equatorial pitch angle [degree]')    
    ax.plot(time, pitch_angle_eq)
    #fig.suptitle('Evolution of equatorial pitch angle')
    ax.minorticks_on()
    ax.grid(which="both")

if(channel == 4):
    fig = plt.figure()
    plt.rcParams["font.size"] = 50
    ax = fig.add_subplot(111, xlabel='time [s]', ylabel='pitch angle [degree]')    
    #ax.plot(time, pitch_angle/np.pi*180)
    plt.scatter(time, pitch_angle/np.pi*180, c=time, cmap=cm.turbo, marker='.', lw=0)
    pp = plt.colorbar()
    pp.set_label('time [s]')
    #fig.suptitle('Evolution of equatorial pitch angle')
    ax.minorticks_on()
    ax.grid(which="both")

if (channel == 5):
    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    ax1 = fig.add_subplot(121)
    ax1.plot(time[0:900000], energy_particle[0:900000])
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('energy [eV]')
    ax1.minorticks_on()
    ax1.grid(which="both")
    ax2 = fig.add_subplot(122)
    ax2.plot(time[0:900000], pitch_angle_eq[0:900000])
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('equatorial pitch angle [degree]')
    ax2.minorticks_on()
    ax2.grid(which="both")

    

if (channel == 6 and trigger == 1):
    length2 = len(z_position)
    MLAT_position = np.zeros(length2)
    for jj in range(length2):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_position(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
            + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
            - z_position[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-5):
                break

            MLAT0 = MLAT1
        
        MLAT_position[jj] = MLAT1
    
    fig = plt.figure()
    plt.rcParams["font.size"] = 50
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='MLAT [degree]', ylabel='wave frequency [Hz]', yscale='log')    
    ax.plot(MLAT_position/np.pi*180, wave_frequency/2/np.pi, linewidth='4')
    #fig.suptitle('wave frequency [Hz]')
    ax.minorticks_on()
    ax.grid(which="both")

if (channel == 7 and trigger == 1):
    lambda_E = np.arccos(L**(-0.5))
    lam = np.linspace(-lambda_E, lambda_E, 10000)
    ss = L * ((1. / 2.) * np.sin(lam) * np.sqrt(3. * np.sin(lam)**2. + 1.) \
        + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(lam) + np.sqrt(3. * np.sin(lam)**2. + 1.)))
    dlog_dz = 3 * np.sin(lam) / R_E / L / np.log(10) / np.sqrt(1 + 3 * np.sin(lam)**2) * (1/(1 + 3 * np.sin(lam)**2) + 2/(np.cos(lam)**2))

    length2 = len(z_position)
    MLAT_position = np.zeros(length2)
    for jj in range(length2):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_position(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
            + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
            - z_position[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-5):
                break

            MLAT0 = MLAT1
        
        MLAT_position[jj] = MLAT1
    
    plt.rcParams["font.size"] = 50
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    plt.rcParams['text.usetex'] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel=r'z [$\rm{R_E}$]', ylabel='length [km]', yscale='log')    
    ax.plot(MLAT_position/np.pi*180, np.abs(2*np.pi/wave_number_para/10**3), label=r'$\lambda_{\parallel} = \frac{2\pi}{k_{\parallel}}$', linewidth='4')
    ax.plot(MLAT_position/np.pi*180, np.abs(2*np.pi/wave_number_perp/10**3), label=r'$\lambda_{\perp} = \frac{2\pi}{k_{\perp}}$', linewidth='4')
    ax.plot(lam/np.pi*180, 1/abs(dlog_dz) / 1E3, label=r'$\left( \frac{d \left( \rm{log}_{10} \frac{B_0}{B_E} \right)}{dz} \right)^{-1}$', linewidth='4')
    #ax.plot(z_position, np.abs(ion_Larmor_radius/ 10**3), label=r'$\rho_i$', linewidth='4')
    #ax.plot(z_position, ion_Larmor_radius*wave_number_perp)
    #fig.suptitle('wavelength [km]')
    ax.minorticks_on()
    ax.grid(which="both")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

if (channel == 8 and trigger == 1):
    fig = plt.figure()
    plt.rcParams["font.size"] = 50
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='z [$R_E$]', ylabel='$v_A / c$', yscale='log')    
    ax.plot(z_position, Alfven_velocity/c, linewidth='4')
    #fig.suptitle('Alfven velocity [km/s]')
    ax.minorticks_on()
    ax.grid(which="both")
    

if (channel == 9 and trigger == 1):
    length2 = len(z_position)
    MLAT_position = np.zeros(length2)
    for jj in range(length2):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_position(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
            + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
            - z_position[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-5):
                break

            MLAT0 = MLAT1
        
        MLAT_position[jj] = MLAT1
    fig = plt.figure()
    plt.rcParams["font.size"] = 50
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='MLAT [degree]', ylabel='$\delta E_{\parallel} [mV/m]$')    
    ax.plot(MLAT_position/np.pi*180, EE_wave_para * 1E3, linewidth='4')
    #fig.suptitle('Ewpara [mV/m]')
    ax.minorticks_on()
    ax.grid(which="both")

if (channel == 10 and trigger == 1):
    length2 = len(z_position)
    MLAT_position = np.zeros(length2)
    for jj in range(length2):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_position(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
            + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
            - z_position[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-5):
                break

            MLAT0 = MLAT1
        
        MLAT_position[jj] = MLAT1
    fig = plt.figure()
    plt.rcParams["font.size"] = 50
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='MLAT [degree]', ylabel='$\delta B_{\parallel} [nT]$')    
    ax.plot(MLAT_position/np.pi*180, BB_wave_para * 1E9, linewidth='4')
    #fig.suptitle('Ewpara [mV/m]')
    ax.minorticks_on()
    ax.grid(which="both")

if (channel == 11 and trigger == 1):
    fig = plt.figure()
    plt.rcParams["font.size"] = 50
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    plt.rcParams['text.usetex'] = True
    ax = fig.add_subplot(111, xlabel='z [$R_E$]', ylabel='$\delta B_{\perp} [nT]$')    
    ax.plot(z_position, BB_wave_perp * 1E9, linewidth='4')
    #fig.suptitle('Ewpara [mV/m]')
    ax.minorticks_on()
    ax.grid(which="both")

if (channel == 12 and trigger == 1):
    fig = plt.figure()
    plt.rcParams["font.size"] = 50
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    plt.rcParams['text.usetex'] = True
    ax = fig.add_subplot(111, xlabel='z [$R_E$]', ylabel='$\delta E_{\perp \perp} [mV/m]$')    
    ax.plot(z_position, EE_wave_perp_perp * 1E3, linewidth='4')
    #fig.suptitle('Ewpara [mV/m]')
    ax.minorticks_on()
    ax.grid(which="both")

if (channel == 13 and trigger == 1):
    fig = plt.figure()
    plt.rcParams["font.size"] = 50
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    plt.rcParams['text.usetex'] = True
    ax = fig.add_subplot(111, xlabel='z [$R_E$]', ylabel='$\delta E_{\perp \phi} [mV/m]$')    
    ax.plot(z_position, EE_wave_perp_phi * 1E3, linewidth='4')
    #fig.suptitle('Ewpara [mV/m]')
    ax.minorticks_on()
    ax.grid(which="both")
    

if (channel == 14 and trigger == 1):
    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    ax = fig.add_subplot(111, xlabel='z [RE]', ylabel='electrostatic potential [V]')    
    ax.plot(z_position, electrostatic_potential, linewidth='4')
    #fig.suptitle('electrostatic potential [V]')
    ax.minorticks_on()
    ax.grid(which="both")
    



if (channel == 15 and trigger == 1):
    size = len(z_position)
    MLAT = np.zeros(size)
    
    for jj in range(size):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_position(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
                + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
                - z_position[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-5):
                break

            MLAT0 = MLAT1
        
        MLAT[jj] = MLAT1
        print(z_position[jj], MLAT[jj])
    
    Ke_100 = 100 * q
    Ke_1000 = 1000 * q
    cyclotron_radius_100 = 1/q/c/B0_eq * np.cos(MLAT)**6 / np.sqrt(1+3*np.sin(MLAT)**2) * np.sqrt(Ke_100*(Ke_100+m*c*c))
    cyclotron_radius_1000 = 1/q/c/B0_eq * np.cos(MLAT)**6 / np.sqrt(1+3*np.sin(MLAT)**2) * np.sqrt(Ke_1000*(Ke_1000+m*c*c))

    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    ax = fig.add_subplot(111, xlabel='z [RE]', ylabel='length [km]', yscale='log')
    #ax.plot(z_position, np.abs(2*np.pi/wave_number_para/10**3), label='para')
    ax.plot(z_position, np.abs(2*np.pi/wave_number_perp/10**3), label='wave_perp', linewidth='4')
    ax.plot(z_position, cyclotron_radius_100/10**3, label='100 eV', linewidth='4')
    ax.plot(z_position, cyclotron_radius_1000/10**3, label='1 keV', linewidth='4')
    ax.plot(z_position, np.abs(ion_Larmor_radius/ 10**3), label='ion_acoustic_gyroradius', linewidth='4')
    #ax.plot(z_position, ion_acoustic_gyroradius*wave_number_perp)
    fig.suptitle('length [km]')
    ax.grid()
    ax.legend()


if (channel == 16):
    size = len(z_particle)
    MLAT = np.zeros(size)
    
    for jj in range(size):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_particle(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
                + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
                - z_particle[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-7):
                break

            MLAT0 = MLAT1
        
        MLAT[jj] = MLAT1
    
    B0 = B0_eq / np.cos(MLAT)**6 * np.sqrt(1+3*np.sin(MLAT)**2) * 1E4 #[G]
    Alpha = 4E0 * np.pi * (1E0 + p_e / p_i) * (q/1E1*c*1E2) * n_i / B0 * ep
    delta_B_para = Alpha * np.cos(wave_phase) * np.exp(-wave_growth_phase) * switch_delta_B_para #[G]

    mu = np.zeros(size-63)
    time_ave = np.zeros(size-63)
    for ii in range(size-63):
        for jj in range(63):
            mu[ii] = m * (v_perp_particle[ii+jj])**2. / 2. / ((B0[ii+jj]+delta_B_para[ii+jj])*1E-4)/63. + mu[ii]
            time_ave[ii] = time_ave[ii] + time[ii + jj]/63.
    #mu = m * (u_perp_particle)**2. / 2. / ((B0+delta_B_para)*1E-4)

    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='time [$s$]', ylabel=r'$\mu / \mu (t=0)$')
    ax.plot(time_ave, mu/mu[0])
    #fig.suptitle('1st adiabatic invariant [Am^2]')
    ax.minorticks_on()
    ax.grid(which="both")


if (channel == 17 and trigger == 1):
    wave_phase = np.zeros(len(z_position))
    for jj in range(len(z_position)-1):
        wave_phase[jj+1] = wave_phase[jj] + (wave_number_para[jj]+wave_number_para[jj+1])/2E0 * (z_position[jj+1]-z_position[jj])
    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='z [$R_E$]', ylabel='wave phase (t=0) [rad]')
    ax.plot(z_position, wave_phase)
    ax.minorticks_on()
    ax.grid(which="both")

if (channel == 18):
    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    ax = fig.add_subplot(111, xlabel='time [s]', ylabel='gyro phase [rad]')    
    ax.plot(time, u_phase_particle)
    #fig.suptitle('Evolution of particle energy')
    ax.minorticks_on()
    ax.grid(which="both")

if (channel == 19):
    wave_phase = wave_phase + np.pi
    wave_phase = np.mod(wave_phase, 2*np.pi) - np.pi
    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    ax = fig.add_subplot(111, xlabel='time [s]', ylabel='wave phase [rad]')    
    ax.plot(time, wave_phase)
    #fig.suptitle('Evolution of particle energy')
    ax.minorticks_on()
    ax.grid(which="both")

if (channel == 20):
    size = len(z_particle)
    MLAT = np.zeros(size)
    
    for jj in range(size):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_particle(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
                + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
                - z_particle[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-7):
                break

            MLAT0 = MLAT1
        
        MLAT[jj] = MLAT1
    
    B0 = B0_eq / np.cos(MLAT)**6 * np.sqrt(1+3*np.sin(MLAT)**2) * 1E4 #[G]
    n_i = 1E0 #[cm^-3]
    m_i = 1.672621898E-24 #[g]
    p_i = n_i * 1000*q*1E7
    p_e = n_i * 100*q*1E7
    kpara = np.sqrt(2E0*np.pi*n_i*m_i*p_i) / B0**2E0 * np.sqrt(4E0 * np.pi + B0**2E0 / (p_i+p_e)) * MLAT/abs(MLAT)
    omega = 2*np.pi / 2
    V_res = omega/kpara /1E2

    theta = (kpara*1E2)*(v_z_particle - V_res)

    wave_phase = wave_phase + np.pi
    wave_phase = np.mod(wave_phase, 2*np.pi) - np.pi

    fig = plt.figure()
    plt.rcParams["font.size"] = 50
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='wave phase $\psi$ [rad]', ylabel=r'$\frac{v_{\parallel}}{V_{R \parallel}}-1$')
    ax.plot(wave_phase, theta/omega)
    ax.scatter(wave_phase[0], theta[0]/omega, marker='o', color='r', label='start', zorder=3, s=200)
    ax.scatter(wave_phase[size-1], theta[size-1]/omega, marker='D', color='r', label='goal', zorder=3, s=200)  #[len(z_particle)-1]
    #fig.suptitle('1st adiabatic invariant [Am^2]')
    ax.minorticks_on()
    ax.grid(which="both")
    ax.legend()

if (channel == 21):
    size = len(z_particle)
    MLAT = np.zeros(size)
    
    for jj in range(size):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_particle(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
                + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
                - z_particle[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-7):
                break

            MLAT0 = MLAT1
        
        MLAT[jj] = MLAT1
    
    B0 = B0_eq / np.cos(MLAT)**6 * np.sqrt(1+3*np.sin(MLAT)**2) * 1E4 #[G]
    n_i = 1E0 #[cm^-3]
    m_i = 1.672621898E-24 #[g]
    p_i = n_i * 1000*q*1E7 #cgs
    p_e = n_i * 100*q*1E7 #cgs
    kpara = np.sqrt(2E0*np.pi*n_i*m_i*p_i) / B0**2E0 * np.sqrt(4E0 * np.pi + B0**2E0 / (p_i+p_e)) * MLAT/abs(MLAT) #[rad/cm]
    rho_i = c*1E2 * np.sqrt(2*m_i*p_i/n_i) / (q/1E1*c*1E2) / B0 #[cm]
    kperp = 2*np.pi/rho_i #[rad/cm]
    delta_B_para = 4*np.pi*p_i/B0**2 * (1 + p_e/p_i) * n_i * (q/1E1*c*1E2) / p_i * B0 * (0 * 1E8 / (c*1E2)) * np.cos(wave_phase) #[G]
    Omega_e = (q/1E1*c*1E2) * (B0+delta_B_para) / (m*1E3) / (c*1E2) #[rad/s]
    
    #SI
    r_c = gamma*v_perp_particle/Omega_e #[m]
    omega = 2*np.pi / 2 #[rad/s]
    V_res = omega/kpara /1E2 #[m/s]

    psi_perp = kperp*1E2 *r_c*np.sin(u_phase_particle) #[rad]

    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='time [s]', ylabel=r'$\psi_\perp$')
    ax.plot(time, psi_perp)
    #fig.suptitle('1st adiabatic invariant [Am^2]')
    ax.minorticks_on()
    ax.grid(which="both")

if (channel == 22):
    size = len(z_particle)
    MLAT = np.zeros(size)
    
    for jj in range(size):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_particle(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
                + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
                - z_particle[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-7):
                break

            MLAT0 = MLAT1
        
        MLAT[jj] = MLAT1
    
    B0 = B0_eq / np.cos(MLAT)**6 * np.sqrt(1+3*np.sin(MLAT)**2) * 1E4 #[G]
    n_i = 1E0 #[cm^-3]
    m_i = 1.672621898E-24 #[g]
    p_i = n_i * 1000*q*1E7
    p_e = n_i * 100*q*1E7
    kpara = np.sqrt(2E0*np.pi*n_i*m_i*p_i) / B0**2E0 * np.sqrt(4E0 * np.pi + B0**2E0 / (p_i+p_e)) * MLAT/abs(MLAT) * 1E2 #[rad/m]
    omega = 2*np.pi / 2 #[rad/s]
    
    #SI
    delta_psi_para = np.zeros(size-1)
    for ii in range(size-1):
        delta_psi_para[ii] = ((kpara[ii] + kpara[ii+1])/2 * (v_z_particle[ii] + v_z_particle[ii+1])/2 - omega)* (time[ii+1]-time[ii])

    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='time [s]', ylabel=r'$\psi_\parallel$')
    ax.plot(time[:size-1], delta_psi_para)
    #fig.suptitle('1st adiabatic invariant [Am^2]')
    ax.minorticks_on()
    ax.grid(which="both")

if(channel == 23):
    size = len(z_particle)
    MLAT = np.zeros(size)
    
    for jj in range(size):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_particle(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
                + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
                - z_particle[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-7):
                break

            MLAT0 = MLAT1
        
        MLAT[jj] = np.mod(MLAT1+np.pi, 2*np.pi) - np.pi
    
    B0 = B0_eq / np.cos(MLAT)**6E0 * np.sqrt(1E0+3E0*np.sin(MLAT)**2E0) * 1E4 #[G]
    
    kpara = np.sqrt(2E0*np.pi*n_i*m_i*p_i) / B0**2E0 * np.sqrt(4E0 * np.pi + B0**2E0 / (p_i+p_e)) * np.sign(MLAT) #[rad/cm]
    
    dB0_dz = 3E0 * np.sin(MLAT) * (5E0 * np.sin(MLAT)**2E0 + 3E0) / np.cos(MLAT)**8E0 / (3E0 * np.sin(MLAT)**2E0 + 1E0) / (R_E*L*1E2) * (B0_eq*1E4)
    Alpha = 4E0 * np.pi * (1E0 + p_e / p_i) * (q/1E1*c*1E2) * n_i / B0 * ep
    d_Alpha_dz = - Alpha / B0 * dB0_dz
    delta_B_para = Alpha * np.cos(wave_phase) * np.exp(-wave_growth_phase) * switch_delta_B_para #[G]
    delta_E_para = (2E0 + p_e / p_i) * kpara * ep * np.sin(wave_phase) * np.exp(-wave_growth_phase) * switch_delta_E_para #[G]
    
    particle_Larmor_radius = (m*1E3) * (u_perp_particle*1E2) * (c*1E2) / (q/1E1*c*1E2) / (B0 + delta_B_para) #[cm]
    rho_i = (c*1E2) * np.sqrt(2*m_i*p_i/n_i) / (q/1E1*c*1E2) / B0 #[cm]
    kperp = 2*np.pi/rho_i * np.ones(size) #[rad/cm]
    wave_growth_number_perp = np.zeros(size)
    wave_growth_number_para = np.zeros(size)
    for ii in range(size):
        if(abs(MLAT[ii]) <= MLAT_position_threshold):
            wave_growth_number_para[ii] = - 1 / z_position_threshold * np.log(wave_initial_to_threshold) * np.sign(MLAT[ii])
        elif(abs(MLAT[ii]) > MLAT_position_threshold):
            wave_growth_number_para[ii] = 0E0
    
    Delta_real = np.zeros(size)
    Delta_imag = np.zeros(size)
    for ii in range(size):
        if(particle_Larmor_radius[ii]*np.sin(u_phase_particle[ii]) != 0E0):
            Delta_real[ii] = (1E0 + wave_growth_number_perp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]) \
                - np.exp(wave_growth_number_perp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii])) * np.cos(kperp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]))) \
                    / (particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]))**2E0
            Delta_imag[ii] = (- kperp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]) \
                + np.exp(wave_growth_number_perp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii])) * np.sin(kperp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]))) \
                    / (particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]))**2E0
            
        elif(particle_Larmor_radius[ii]*np.sin(u_phase_particle[ii]) == 0E0):
            Delta_real[ii] = 5E-1 * (kperp[ii]**2E0 - wave_growth_number_perp[ii]**2E0)
            Delta_imag[ii] = kperp[ii] * wave_growth_number_perp[ii]
    
    cos_delta = ((kperp**2E0 - wave_growth_number_perp**2E0) * np.cos(wave_phase) + 2E0 * kperp * wave_growth_number_perp * np.sin(wave_phase)) / (kperp**2E0 + wave_growth_number_perp**2E0)
    sin_delta = ((kperp**2E0 - wave_growth_number_perp**2E0) * np.sin(wave_phase) - 2E0 * kperp * wave_growth_number_perp * np.cos(wave_phase)) / (kperp**2E0 + wave_growth_number_perp**2E0)

    Xi_function = np.zeros(size)
    for ii in range(size):
        if(MLAT[ii] != 0E0):
            Xi_function[ii] = 2E0 / (kperp[ii]**2E0 + wave_growth_number_perp[ii]**2E0) * np.exp(- wave_growth_phase[ii]) \
                * ((d_Alpha_dz[ii] + kpara[ii] * Alpha[ii]) * (Delta_real[ii] * cos_delta[ii] - Delta_imag[ii] * sin_delta[ii]) - wave_growth_number_para[ii] * Alpha[ii] * (Delta_real[ii] * sin_delta[ii] + Delta_imag[ii] * cos_delta[ii]))

        elif(MLAT[ii] == 0E0):
            Xi_function[ii] = 0E0
    
    dB_dz = dB0_dz + Xi_function #[G/cm]

    F_mirror_background = - (m*1E3) * (u_perp_particle*1E2)**2E0 / 2E0 / (B0 + delta_B_para) / gamma * dB0_dz
    F_mirror_wave = - (m*1E3) * (u_perp_particle*1E2)**2E0 / 2E0 / (B0 + delta_B_para) / gamma * Xi_function
    F_electric = - (q/1E1*c*1E2) * delta_E_para

    F_mirror_background_B0only = - (m*1E3) * (u_perp_particle*1E2)**2E0 / 2E0 / B0 / gamma * dB0_dz

    #cgs-gauss -> SI
    F_mirror_background = F_mirror_background * 1E-5
    F_mirror_wave = F_mirror_wave * 1E-5
    F_electric = F_electric * 1E-5
    F_parallel = F_mirror_background + F_mirror_wave + F_electric

    F_mirror_background_B0only = F_mirror_background_B0only * 1E-5

    fig = plt.figure()
    plt.rcParams["font.size"] = 50
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='time [s]', ylabel='Force [N]')
    ax.plot(time, F_mirror_background, color="purple", alpha=0.5, label=r'$F_{mirror B_0}$', lw=4)
    if(switch_delta_B_para == 1E0):
        ax.plot(time, F_mirror_wave, color='green', alpha=0.5, label=r'$F_{mirror \delta B_{\parallel}}$', lw=4)
    if(switch_delta_E_para == 1E0):
        ax.plot(time, F_electric, color='b', alpha=0.5, label=r'$F_{\delta E_{\parallel}}$', lw=4)
    #ax.plot(time, F_mirror_background_B0only, color="red", alpha=0.5, label=r'$F_{mirror B_0 only}$', lw=4)
    plt.scatter(time, F_parallel, c=time, cmap=cm.turbo, marker='.', lw=0)
    pp = plt.colorbar()
    pp.set_label('time [s]')
    ax.minorticks_on()
    ax.grid(which="both")
    ax.legend()

if(channel == 24):
    size = len(z_particle)
    MLAT = np.zeros(size)
    
    for jj in range(size):
        MLAT0 = 1.
        for ii in range(1000000):
            if (ii == 1000000):
                print("Error!: solution is not found. z_position = " + str(z_particle(jj)))
                
            ff = R_E*L * ((1. / 2.) * np.sin(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.) \
                + (1. / (2. * np.sqrt(3.))) * np.log(np.sqrt(3.) * np.sin(MLAT0) + np.sqrt(3. * np.sin(MLAT0)**2. + 1.))) \
                - z_particle[jj]*R_E
            gg = R_E*L * np.cos(MLAT0) * np.sqrt(3. * np.sin(MLAT0)**2. + 1.)

            MLAT1 = float(MLAT0 - ff / gg)
            
            if (abs(MLAT1 - MLAT0) <= 1E-7):
                break

            MLAT0 = MLAT1
        
        MLAT[jj] = np.mod(MLAT1+np.pi, 2*np.pi) - np.pi
    
    B0 = B0_eq / np.cos(MLAT)**6E0 * np.sqrt(1E0+3E0*np.sin(MLAT)**2E0) * 1E4 #[G]
    
    kpara = np.sqrt(2E0*np.pi*n_i*m_i*p_i) / B0**2E0 * np.sqrt(4E0 * np.pi + B0**2E0 / (p_i+p_e)) * np.sign(MLAT) #[rad/cm]
    
    dB0_dz = 3E0 * np.sin(MLAT) * (5E0 * np.sin(MLAT)**2E0 + 3E0) / np.cos(MLAT)**8E0 / (3E0 * np.sin(MLAT)**2E0 + 1E0) / (R_E*L*1E2) * (B0_eq*1E4)
    Alpha = 4E0 * np.pi * (1E0 + p_e / p_i) * (q/1E1*c*1E2) * n_i / B0 * ep
    d_Alpha_dz = - Alpha / B0 * dB0_dz
    delta_B_para = Alpha * np.cos(wave_phase) * np.exp(-wave_growth_phase) * switch_delta_B_para #[G]
    delta_E_para = (2E0 + p_e / p_i) * kpara * ep * np.sin(wave_phase) * np.exp(-wave_growth_phase) * switch_delta_E_para #[G]
    
    particle_Larmor_radius = (m*1E3) * (u_perp_particle*1E2) * (c*1E2) / (q/1E1*c*1E2) / (B0 + delta_B_para) #[cm]
    rho_i = (c*1E2) * np.sqrt(2*m_i*p_i/n_i) / (q/1E1*c*1E2) / B0 #[cm]
    kperp = 2*np.pi/rho_i * np.ones(size) #[rad/cm]
    wave_growth_number_perp = np.zeros(size)
    wave_growth_number_para = np.zeros(size)
    for ii in range(size):
        if(abs(MLAT[ii]) <= MLAT_position_threshold):
            wave_growth_number_para[ii] = - 1 / z_position_threshold * np.log(wave_initial_to_threshold) * np.sign(MLAT[ii])
        elif(abs(MLAT[ii]) > MLAT_position_threshold):
            wave_growth_number_para[ii] = 0E0
    
    Delta_real = np.zeros(size)
    Delta_imag = np.zeros(size)
    for ii in range(size):
        if(particle_Larmor_radius[ii]*np.sin(u_phase_particle[ii]) != 0E0):
            Delta_real[ii] = (1E0 + wave_growth_number_perp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]) \
                - np.exp(wave_growth_number_perp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii])) * np.cos(kperp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]))) \
                    / (particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]))**2E0
            Delta_imag[ii] = (- kperp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]) \
                + np.exp(wave_growth_number_perp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii])) * np.sin(kperp[ii] * particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]))) \
                    / (particle_Larmor_radius[ii] * np.sin(u_phase_particle[ii]))**2E0
            
        elif(particle_Larmor_radius[ii]*np.sin(u_phase_particle[ii]) == 0E0):
            Delta_real[ii] = 5E-1 * (kperp[ii]**2E0 - wave_growth_number_perp[ii]**2E0)
            Delta_imag[ii] = kperp[ii] * wave_growth_number_perp[ii]
    
    cos_delta = ((kperp**2E0 - wave_growth_number_perp**2E0) * np.cos(wave_phase) + 2E0 * kperp * wave_growth_number_perp * np.sin(wave_phase)) / (kperp**2E0 + wave_growth_number_perp**2E0)
    sin_delta = ((kperp**2E0 - wave_growth_number_perp**2E0) * np.sin(wave_phase) - 2E0 * kperp * wave_growth_number_perp * np.cos(wave_phase)) / (kperp**2E0 + wave_growth_number_perp**2E0)

    B0_function = np.zeros(size)
    kpara_function = np.zeros(size)
    kappa_para_funciton = np.zeros(size)
    Xi_function = np.zeros(size)

    for ii in range(size):
        if(MLAT[ii] != 0E0):
            B0_function[ii] = 2E0 / (kperp[ii]**2E0 + wave_growth_number_perp[ii]**2E0) * np.exp(- wave_growth_phase[ii]) \
                * d_Alpha_dz[ii] * (Delta_real[ii] * cos_delta[ii] - Delta_imag[ii] * sin_delta[ii])
            kpara_function[ii] = 2E0 / (kperp[ii]**2E0 + wave_growth_number_perp[ii]**2E0) * np.exp(- wave_growth_phase[ii]) \
                * kpara[ii] * Alpha[ii] * (Delta_real[ii] * cos_delta[ii] - Delta_imag[ii] * sin_delta[ii])
            kappa_para_funciton[ii] = 2E0 / (kperp[ii]**2E0 + wave_growth_number_perp[ii]**2E0) * np.exp(- wave_growth_phase[ii]) \
                * (- wave_growth_number_para[ii] * Alpha[ii] * (Delta_real[ii] * sin_delta[ii] + Delta_imag[ii] * cos_delta[ii]))
            Xi_function[ii] = B0_function[ii] + kpara_function[ii] + kappa_para_funciton[ii]

        elif(MLAT[ii] == 0E0):
            B0_function[ii] = 0E0
            kpara_function[ii] = 0E0
            kappa_para_funciton[ii] = 0E0
            Xi_function[ii] = 0E0
    
    dB_dz = dB0_dz + Xi_function #[G]

    F_mirror_background = - (m*1E3) * (u_perp_particle*1E2)**2E0 / 2E0 / (B0 + delta_B_para) / gamma * dB0_dz
    F_mirror_wave_B0 = - (m*1E3) * (u_perp_particle*1E2)**2E0 / 2E0 / (B0 + delta_B_para) / gamma * B0_function
    F_mirror_wave_kpara = - (m*1E3) * (u_perp_particle*1E2)**2E0 / 2E0 / (B0 + delta_B_para) / gamma * kpara_function
    F_mirror_wave_kappa_para = - (m*1E3) * (u_perp_particle*1E2)**2E0 / 2E0 / (B0 + delta_B_para) / gamma * kappa_para_funciton
    F_mirror_wave = F_mirror_wave_B0 + F_mirror_wave_B0 + F_mirror_wave_kpara + F_mirror_wave_kappa_para
    F_electric = - q / 1E1 * (c*1E2) * delta_E_para

    #cgs-gauss -> SI
    F_mirror_background = F_mirror_background * 1E-5
    F_mirror_wave_B0 = F_mirror_wave_B0 * 1E-5
    F_mirror_wave_kpara = F_mirror_wave_kpara * 1E-5
    F_mirror_wave_kappa_para = F_mirror_wave_kappa_para * 1E-5
    F_mirror_wave = F_mirror_wave * 1E-5
    F_electric = F_electric * 1E-5
    F_parallel = F_mirror_background + F_mirror_wave + F_electric

    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    plt.rcParams.update({'mathtext.default': 'default', 'mathtext.fontset': 'stix'})
    ax = fig.add_subplot(111, xlabel='time [s]', ylabel='Force [N]')
    ax.plot(time, F_mirror_background, color="purple", alpha=0.5, label=r'$F_{B_0}$', lw=4)
    if(switch_delta_B_para == 1E0):
        ax.plot(time, F_mirror_wave_B0, color='red', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (B_0)$', lw=4)
        ax.plot(time, F_mirror_wave_kpara, color='orange', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (k_{\parallel})$', lw=4)
        ax.plot(time, F_mirror_wave_kappa_para, color='magenta', alpha=0.5, label=r'$F_{\delta B_{\parallel}} (\kappa_{\parallel})$', lw=4)
    if(switch_delta_E_para == 1E0):
        ax.plot(time, F_electric, color='b', alpha=0.5, label=r'$F_{\delta E_{\parallel}}$', lw=4)
    #plt.scatter(time, F_parallel, c=time, cmap=cm.turbo, marker='.', lw=0)
    #pp = plt.colorbar()
    #pp.set_label('time [s]')
    ax.minorticks_on()
    ax.grid(which="both")
    ax.legend()#bbox_to_anchor=(1.30, 1), loc='upper left', borderaxespad=0

if (channel == 25):
    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    ax1 = fig.add_subplot(121)
    ax1.plot(time, u_z_particle/c)
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('parallel velocity [/c]')
    ax1.minorticks_on()
    ax1.grid(which="both")
    ax2 = fig.add_subplot(122)
    ax2.plot(time, u_perp_particle/c)
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('perpendicular speed [/c]')
    ax2.minorticks_on()
    ax2.grid(which="both")

if (channel == 26):
    fig = plt.figure()
    plt.rcParams["font.size"] = 40
    ax = fig.add_subplot(111, xlabel='time [s]', ylabel='wave phase [rad]')    
    ax.plot(time, wave_growth_phase)
    #fig.suptitle('Evolution of particle energy')
    ax.minorticks_on()
    ax.grid(which="both")




plt.tight_layout()
plt.show()


