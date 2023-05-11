subroutine z_position_to_radius_MLAT(z_position, radius, MLAT)
    use lshell_setting

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: z_position
    DOUBLE PRECISION, INTENT(OUT) :: radius, MLAT
    DOUBLE PRECISION :: ff, gg, MLAT0, MLAT1
    INTEGER :: ii

    MLAT0 = 1d0
    do ii = 1, 1000000
        if (ii == 1000000) then
            print *, "Error!: solution is not found. z_position = ", z_position
            print *, ff, gg, MLAT0, MLAT1
        endif

        ff = r_eq * ((1d0 / 2d0) * DSIN(MLAT0) * DSQRT(3d0 * DSIN(MLAT0)**2 + 1d0) &
            & + (1d0 / (2d0 * DSQRT(3d0))) * ASINH(DSQRT(3d0) * DSIN(MLAT0))) &
            & - z_position
        gg = r_eq * DCOS(MLAT0) * DSQRT(3d0 * DSIN(MLAT0)**2 + 1d0)

        MLAT1 = MLAT0 - ff / gg
        if (DABS(MLAT1 - MLAT0) <= 1d-4) exit
        MLAT0 = MLAT1
    end do !ii

    MLAT = MLAT1
    radius = r_eq * DCOS(MLAT)**2d0

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_position_to_BB(z_position, BB)

    implicit none
      
    DOUBLE PRECISION, INTENT(IN)  :: z_position
    DOUBLE PRECISION, INTENT(OUT) :: BB
    DOUBLE PRECISION :: radius, MLAT
    
    call z_position_to_radius_MLAT(z_position, radius, MLAT)
  
    BB = DSQRT(1d0 + 3d0 * DSIN(MLAT)**2) / DCOS(MLAT)**6

    if (isnan(BB)) then
        print *, 'z_position_to_BB: BB = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_position_to_number_density(number_density)
    use constants_in_the_simulations

    implicit none

    DOUBLE PRECISION, INTENT(OUT) :: number_density

    number_density = number_density_eq

    if (number_density /= number_density) then
        print *, number_density
        print *, number_density_eq
        print *, 'z_position_to_number_density: number_density = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_position_to_alfven_velocity(BB, alfven_velocity)
    use constants_in_the_simulations
    use lshell_setting

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: BB
    DOUBLE PRECISION, INTENT(OUT) :: alfven_velocity
    DOUBLE PRECISION :: number_density

    CALL z_position_to_number_density(number_density)

    alfven_velocity = BB / DSQRT(4d0 * pi * number_density * ion_mass)

    if (isnan(alfven_velocity)) then
        print *, 'z_position_to_alfven_velocity: alfven_velocity = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_position_to_ion_Larmor_radius(BB, ion_Larmor_radius)
    use constant_parameter
    use lshell_setting

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: BB
    DOUBLE PRECISION, INTENT(OUT) :: ion_Larmor_radius
    DOUBLE PRECISION :: number_density
    
    CALL z_position_to_number_density(number_density)

    ion_Larmor_radius = c_normal * SQRT(2d0 * ion_mass * Temperature_ion) / charge / BB
    
    if (isnan(ion_Larmor_radius)) then
        print *, 'z_position_to_ion_acoustic_gyroradius: ion_Larmor_radius = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_position_to_wave_frequency(wave_frequency)
    use constant_parameter
    use lshell_setting

    implicit none

    DOUBLE PRECISION, INTENT(OUT) :: wave_frequency

    wave_frequency = 2d0 * pi / 2d0 * t_unit

    if (wave_frequency /= wave_frequency) then
        print *, wave_frequency
        print *, 2d0 * pi / 2d0 * t_unit
        print *, 'z_position_to_wave_frequency: wave_frequency = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_position_to_wave_number_perp(BB, wave_number_perp)
    use constant_parameter
    use constants_in_the_simulations

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: BB
    DOUBLE PRECISION, INTENT(OUT) :: wave_number_perp
    DOUBLE PRECISION :: ion_Larmor_radius

    CALL z_position_to_ion_Larmor_radius(BB, ion_Larmor_radius)

    wave_number_perp = 2d0 * pi / ion_Larmor_radius

    if (isnan(wave_number_perp)) then
        print *, 'z_position_to_wave_number_perp: wave_number_perp = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_position_to_beta_ion(BB, beta_ion)
    use lshell_setting

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: BB
    DOUBLE PRECISION, INTENT(OUT) :: beta_ion
    DOUBLE PRECISION :: number_density

    CALL z_position_to_number_density(number_density)

    beta_ion = 8d0 * pi * number_density * Temperature_ion / BB**2d0

    if (isnan(beta_ion)) then
        print *, 'z_position_to_beta_ion: beta_ion = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_position_to_wave_number_para(z_position, BB, wave_number_perp, wave_number_para, channel)
    use lshell_setting

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: z_position, BB, wave_number_perp
    integer, intent(in) :: channel  !channel = 1 or 2
    DOUBLE PRECISION, INTENT(OUT) :: wave_number_para
    DOUBLE PRECISION :: wave_frequency, alfven_velocity, ion_Larmor_radius, beta_ion, channel_pm

    CALL z_position_to_wave_frequency(wave_frequency)
    CALL z_position_to_alfven_velocity(BB, alfven_velocity)
    CALL z_position_to_ion_Larmor_radius(BB, ion_Larmor_radius)
    CALL z_position_to_beta_ion(BB, beta_ion)

    if ( channel == 1 ) then
        channel_pm = +1d0
    else if ( channel == 2 ) then
        channel_pm = -1d0
    else
        print *, "ERROR!: Unexpected wave channel was entered."
    end if

    wave_number_para = 1 / (wave_number_perp * ion_Larmor_radius) * wave_frequency / alfven_velocity &
        & * sqrt(beta_ion + 2d0 / (1d0 + Temperature_electron/Temperature_ion)) * channel_pm

    if (isnan(wave_number_para)) then
        print *, 'z_position_to_wave_number_para: wave_number_para = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine wave_number_para_to_wave_phase_initial(wave_number_para_pre, wave_number_para, wave_phase_pre, wave_phase, pm_flag)
    use constants_in_the_simulations

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: wave_number_para_pre, wave_number_para, wave_phase_pre
    double precision, intent(in) :: pm_flag !pm_flag > 0 -> 1d0, pm_flag < 0 -> -1d0
    DOUBLE PRECISION, INTENT(OUT) :: wave_phase

    wave_phase = wave_phase_pre + sign(1d0, pm_flag) * (wave_number_para_pre + wave_number_para) / 2d0 * d_z

    if (isnan(wave_phase)) then
        print *, 'wave_number_para_to_wave_phase_initial: wave_phase = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_position_to_electrostatic_potential(z_position, wave_phase, electrostatic_potential, channel)
    use constant_parameter
    use lshell_setting
    use constants_in_the_simulations

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: z_position, wave_phase
    integer, intent(in) :: channel ! channel = 1 or 2
    DOUBLE PRECISION, INTENT(OUT) :: electrostatic_potential
    DOUBLE PRECISION :: radius, MLAT, g_function, channel_pm, BB, wave_number_perp, wave_number_para, h_function

    if ( channel == 1 ) then
        channel_pm = +1d0
    else if ( channel == 2 ) then
        channel_pm = -1d0
    else
        print *, "ERROR!: Unexpected wave channel was entered."
    end if

    CALL z_position_to_radius_MLAT(z_position, radius, MLAT)

    g_function = 5d-1 &
        & * (tanh(channel_pm * gradient_parameter * (rad2deg * MLAT - channel_pm * mlat_deg_wave_threshold/2d0)) + 1d0)

    if (isnan(g_function)) then
        print *, 'z_position_to_electrostatic_potential: g_function = NaN'
    end if


    if (switch_wave_packet == 0) then
        electrostatic_potential = electrostatic_potential_0 * g_function

    else if (switch_wave_packet == 1 .and. wave_phase >= initial_wave_phase - 8d0*pi .and. wave_phase <= initial_wave_phase) then
        call z_position_to_BB(z_position, BB)
        call z_position_to_wave_number_perp(BB, wave_number_perp)
        call z_position_to_wave_number_para(z_position, BB, wave_number_perp, wave_number_para, channel)
        h_function = 5d-1 * (1d0 - cos(0.25d0 * (wave_phase - initial_wave_phase)))
        electrostatic_potential = electrostatic_potential_0 * g_function * h_function
    
    else
        electrostatic_potential = 0d0
    
    end if


    if (isnan(electrostatic_potential)) then
        print *, 'z_position_to_electrostatic_potential: electrostatic_potential = NaN'
    end if
    
end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine time_to_wave_phase_update(wave_phase, wave_frequency)
    use constants_in_the_simulations

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: wave_frequency
    DOUBLE PRECISION, INTENT(INOUT) :: wave_phase

    wave_phase = wave_phase - wave_frequency * d_t

    if (isnan(wave_phase)) then
        print *, 'time_to_wave_phase_update: wave_phase = NaN'
    end if

end subroutine

!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_particle_to_position(z_particle, z_position, i_z_left, i_z_right, ratio)
    use constants_in_the_simulations, only: n_z

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: z_position(-n_z:n_z), z_particle
    INTEGER, INTENT(OUT) :: i_z_left, i_z_right
    DOUBLE PRECISION, INTENT(OUT) :: ratio
    DOUBLE PRECISION :: difference(-n_z:n_z), difference_min
    INTEGER :: i_min(1)

    difference = ABS(z_position - z_particle)
    i_min = MINLOC(difference) - (n_z + 1) !array_difference : 0~2*n_z+1

    if (i_min(1) >= n_z) then
        i_z_left = n_z - 1
        i_z_right = n_z
        ratio = 1d0

    else if (i_min(1) <= -n_z) then
        i_z_left = - n_z
        i_z_right = - n_z + 1
        ratio = 1d0

    else
        difference_min = z_position(i_min(1)) - z_particle
        if (difference_min > 0) then
            i_z_left = i_min(1) - 1
            i_z_right = i_min(1)
        else if (difference_min <= 0) then
            i_z_left = i_min(1)
            i_z_right = i_min(1) + 1
        end if

        ratio = (z_particle - z_position(i_z_left)) / (z_position(i_z_right) - z_position(i_z_left))

    end if

    if (isnan(ratio)) then
        print *, 'z_particle_to_position: ratio = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine u_particle_to_gamma(u_particle, gamma)

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: u_particle(0:2)
    DOUBLE PRECISION, INTENT(OUT) :: gamma

    gamma = DSQRT(1d0 + u_particle(0)**2d0 + u_particle(1)**2d0)

    if (isnan(gamma)) then
        print *, 'u_particle_to_gamma: gamma = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine u_particle_to_v_particle_para(u_particle, v_particle_para)

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: u_particle(0:2)
    DOUBLE PRECISION, INTENT(OUT) :: v_particle_para
    DOUBLE PRECISION :: gamma

    CALL u_particle_to_gamma(u_particle, gamma)

    v_particle_para = u_particle(0) / gamma

    if (isnan(v_particle_para)) then
        print *, 'u_particle_to_v_particle_para: v_particle_para = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_particle_to_dB_dz(z_particle, u_particle, particle_Larmor_radius, B0, wave_number_para_1, wave_number_para_2, &
    & wave_number_perp, wave_phase_1, wave_phase_2, dB_dz)

    use constant_parameter
    use lshell_setting
    use constants_in_the_simulations

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: z_particle, B0, wave_number_para_1, wave_number_para_2, wave_phase_1, wave_phase_2
    double precision, intent(in) :: particle_Larmor_radius, wave_number_perp
    double precision, dimension(0:2), intent(in) :: u_particle
    DOUBLE PRECISION, INTENT(OUT) :: dB_dz
    DOUBLE PRECISION :: radius, MLAT, dB0_dz, Xi_1, Xi_2, dg_dz_1, dg_dz_2, deg_mlat, beta_ion, number_density, Delta_r, Delta_i, Alpha
    double precision :: g_function_1, g_function_2
    double precision :: h_function_1, h_function_2, dh_dz_1, dh_dz_2


    CALL z_position_to_radius_MLAT(z_particle, radius, MLAT)
    CALL z_position_to_beta_ion(B0, beta_ion)

    dB0_dz = 3d0 * DSIN(MLAT) * (5d0 * DSIN(MLAT)**2d0 + 3d0) / DCOS(MLAT)**8d0 / (3d0 * DSIN(MLAT)**2d0 + 1d0) / r_eq

    if( switch_BB_wave_para == 1d0) then
        deg_mlat = rad2deg * MLAT

        dg_dz_1 = 90d0 / pi / r_eq / cos(MLAT) / sqrt(1d0 + 3d0 * sin(MLAT)**2d0) * gradient_parameter &
            & / cosh(gradient_parameter * (deg_mlat - mlat_deg_wave_threshold / 2d0))**2d0
        dg_dz_2 = - 90d0 / pi / r_eq / cos(MLAT) / sqrt(1d0 + 3d0 * sin(MLAT)**2d0) * gradient_parameter &
            & / cosh(- gradient_parameter * (deg_mlat + mlat_deg_wave_threshold / 2d0))**2d0

        call z_position_to_number_density(number_density)

        Alpha = 4d0 * pi * (1d0 + Temperature_electron / Temperature_ion) * number_density * charge * electrostatic_potential_0

        if ( wave_number_perp * particle_Larmor_radius * sin(u_particle(2)) /= 0d0 ) then
            Delta_r = (1d0 - cos(wave_number_perp * particle_Larmor_radius * sin(u_particle(2)))) &
                & / (wave_number_perp * particle_Larmor_radius * sin(u_particle(2)))**2d0
            Delta_i = (sin(wave_number_perp * particle_Larmor_radius * sin(u_particle(2))) &
                & - wave_number_perp * particle_Larmor_radius * sin(u_particle(2))) &
                & / (wave_number_perp * particle_Larmor_radius * sin(u_particle(2)))**2d0

        else if (wave_number_perp * particle_Larmor_radius * sin(u_particle(2)) == 0d0) then
            Delta_r = 5d-1
            Delta_i = 0d0

        end if

        CALL z_position_to_radius_MLAT(z_particle, radius, MLAT)

        g_function_1 = 5d-1 * (tanh(gradient_parameter * (rad2deg * MLAT - mlat_deg_wave_threshold / 2d0)) + 1d0)
        g_function_2 = 5d-1 * (tanh(- gradient_parameter * (rad2deg * MLAT + mlat_deg_wave_threshold / 2d0)) + 1d0)

        if (switch_wave_packet == 0d0) then
            h_function_1 = 1d0
            h_function_2 = 1d0
            dh_dz_1 = 0d0
            dh_dz_2 = 0d0

        else if (switch_wave_packet == 1d0) then
            if (wave_phase_1 >= initial_wave_phase - 8d0*pi .and. wave_phase_1 <= initial_wave_phase) then
                h_function_1 = 5d-1 * (1d0 - cos(0.25d0 * (wave_phase_1 - initial_wave_phase)))
                dh_dz_1 = 1d0 / 8d0 * wave_number_para_1 * sin(0.25d0 * (wave_phase_1 - initial_wave_phase))
            else
                h_function_1 = 0d0
                dh_dz_1 = 0d0
            end if

            if (wave_phase_2 >= initial_wave_phase - 8d0*pi .and. wave_phase_2 <= initial_wave_phase) then
                h_function_2 = 5d-1 * (1d0 - cos(0.25d0 * (wave_phase_2 - initial_wave_phase)))
                dh_dz_2 = 1d0 / 8d0 * wave_number_para_2 * sin(0.25d0 * (wave_phase_2 - initial_wave_phase))
            else
                h_function_2 = 0d0
                dh_dz_2 = 0d0
            end if
        end if

        Xi_1 = 2d0 * Alpha * ((- B0**(-2d0) * dB0_dz * g_function_1 * h_function_1 + B0**(-1d0) * dg_dz_1 * h_function_1 &
            & + B0**(-1d0) * g_function_1 * dh_dz_1) * (Delta_r * cos(wave_phase_1) - Delta_i * sin(wave_phase_1)) &
            & - wave_number_para_1 * g_function_1 * h_function_1 / B0 * (Delta_r * sin(wave_phase_1) + Delta_i * cos(wave_phase_1)))
            
        Xi_2 = 2d0 * Alpha * ((- B0**(-2d0) * dB0_dz * g_function_2 * h_function_2 + B0**(-1d0) * dg_dz_2 * h_function_2 &
            & + B0**(-1d0) * g_function_2 * dh_dz_2) * (Delta_r * cos(wave_phase_2) - Delta_i * sin(wave_phase_2)) &
            & - wave_number_para_2 * g_function_2 * h_function_2 / B0 * (Delta_r * sin(wave_phase_2) + Delta_i * cos(wave_phase_2)))
            
    else
        Xi_1 = 0d0
        Xi_2 = 0d0
    
    end if

    dB_dz = dB0_dz + Xi_1 + Xi_2

    if (isnan(dB_dz)) then
        print *, 'z_particle_to_dB_dz: dB_dz = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine electrostatic_potential_to_EE_wave_para(electrostatic_potential, wave_number_para, wave_phase, EE_wave_para)
    
    use lshell_setting
    use constants_in_the_simulations

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: electrostatic_potential, wave_number_para, wave_phase
    DOUBLE PRECISION, INTENT(OUT) :: EE_wave_para

    EE_wave_para = (2d0 + Temperature_electron / Temperature_ion) * wave_number_para * electrostatic_potential * SIN(wave_phase)
    EE_wave_para = EE_wave_para * switch_EE_wave_para

    if (isnan(EE_wave_para)) then
        print *, 'electrostatic_potential_to_EE_wave_para: EE_wave_para = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine electrostatic_potential_to_EE_wave_perp_perp(electrostatic_potential, wave_frequency, wave_number_perp, wave_phase, &
    & z_position, u_particle_phase, EE_wave_perp_perp)
    
    use lshell_setting
    use constants_in_the_simulations

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: electrostatic_potential, wave_frequency, wave_number_perp, wave_phase, z_position
    DOUBLE PRECISION, INTENT(IN) :: u_particle_phase
    DOUBLE PRECISION, INTENT(OUT) :: EE_wave_perp_perp
    DOUBLE PRECISION :: BB, beta_ion, ion_Larmor_radius, ion_gyrofrequency

    CALL z_position_to_BB(z_position, BB)
    CALL z_position_to_beta_ion(BB, beta_ion)
    CALL z_position_to_ion_Larmor_radius(BB, ion_Larmor_radius)

    ion_gyrofrequency = charge * BB / ion_mass / c_normal

    EE_wave_perp_perp = (COS(u_particle_phase) * SIN(wave_phase) + &
                        & wave_frequency / ion_gyrofrequency * beta_ion / (wave_number_perp * ion_Larmor_radius)**2d0 * &
                        & (1d0 + Temperature_electron / Temperature_ion) * SIN(u_particle_phase) * COS(wave_phase)) &
                        & * wave_number_perp * electrostatic_potential

    EE_wave_perp_perp = EE_wave_perp_perp * switch_EE_wave_perp_perp

    if (isnan(EE_wave_perp_perp)) then
        print *, 'electrostatic_potential_to_EE_wave_perp_perp: EE_wave_perp_perp = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine electrostatic_potential_to_EE_wave_perp_phi(electrostatic_potential, wave_frequency, wave_number_perp, wave_phase, &
    & z_position, u_particle_phase, EE_wave_perp_phi)
    
    use lshell_setting
    use constants_in_the_simulations

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: electrostatic_potential, wave_frequency, wave_number_perp, wave_phase, z_position
    DOUBLE PRECISION, INTENT(IN) :: u_particle_phase
    DOUBLE PRECISION, INTENT(OUT) :: EE_wave_perp_phi
    DOUBLE PRECISION :: BB, beta_ion, ion_Larmor_radius, ion_gyrofrequency

    CALL z_position_to_BB(z_position, BB)
    CALL z_position_to_beta_ion(BB, beta_ion)
    CALL z_position_to_ion_Larmor_radius(BB, ion_Larmor_radius)

    ion_gyrofrequency = charge * BB / ion_mass / c_normal

    EE_wave_perp_phi = (SIN(u_particle_phase) * SIN(wave_phase) - &
                        & wave_frequency / ion_gyrofrequency * beta_ion / (wave_number_perp * ion_Larmor_radius)**2d0 * &
                        & (1d0 + Temperature_electron / Temperature_ion) * COS(u_particle_phase) * COS(wave_phase)) &
                        & * wave_number_perp * electrostatic_potential

    EE_wave_perp_phi = EE_wave_perp_phi * switch_EE_wave_perp_phi

    if (isnan(EE_wave_perp_phi)) then
        print *, 'electrostatic_potential_to_EE_wave_perp_phi: EE_wave_perp_phi = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine electrostatic_potential_to_BB_wave_para(electrostatic_potential, wave_phase, z_position, BB_wave_para)
    
    use lshell_setting
    use constants_in_the_simulations

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: electrostatic_potential, wave_phase, z_position
    DOUBLE PRECISION, INTENT(OUT) :: BB_wave_para
    DOUBLE PRECISION :: BB, beta_ion

    CALL z_position_to_BB(z_position, BB)
    CALL z_position_to_beta_ion(BB, beta_ion)

    BB_wave_para = beta_ion / 2d0 * (1 + Temperature_electron / Temperature_ion) * charge / Temperature_ion * BB &
                    & * electrostatic_potential * COS(wave_phase)

    BB_wave_para = BB_wave_para * switch_BB_wave_para

    if (isnan(BB_wave_para)) then
        print *, 'electrostatic_potential_to_BB_wave_para: BB_wave_para = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine electrostatic_potential_to_BB_wave_perp(electrostatic_potential, wave_phase, wave_frequency, wave_number_para, &
    & wave_number_perp, BB_wave_perp)
    
    use lshell_setting
    use constants_in_the_simulations

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: electrostatic_potential, wave_phase, wave_frequency, wave_number_para, wave_number_perp
    DOUBLE PRECISION, INTENT(OUT) :: BB_wave_perp
    
    BB_wave_perp = (1d0 + Temperature_electron / Temperature_ion) * wave_number_para * c_normal / wave_frequency &
                    & * wave_number_perp * electrostatic_potential * SIN(wave_phase)

    BB_wave_perp = BB_wave_perp * switch_BB_wave_perp

    if (isnan(BB_wave_perp)) then
        print *, 'electrostatic_potential_to_BB_wave_perp: BB_wave_perp = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine calculation_particle_Larmor_radius(u_particle, BB, BB_wave_para, particle_Larmor_radius)
    use lshell_setting

    implicit none
    
    double precision, dimension(0:2), intent(in) :: u_particle
    double precision, intent(in) :: BB, BB_wave_para
    double precision, intent(out) :: particle_Larmor_radius

    particle_Larmor_radius = electron_mass * u_particle(1) * c_normal / charge / (BB + BB_wave_para)
    
end subroutine calculation_particle_Larmor_radius
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine wave_phase_plus_perp_contribution(wave_phase, wave_number_perp, electrostatic_potential_1, electrostatic_potential_2, &
    & z_position, BB, u_particle, wave_phase_update)

    use constant_parameter
    use lshell_setting

    implicit none

    double precision, dimension(2), intent(in) :: wave_phase
    DOUBLE PRECISION, INTENT(IN) :: wave_number_perp, electrostatic_potential_1, electrostatic_potential_2, z_position, BB
    DOUBLE PRECISION, DIMENSION(0:2), INTENT(IN) :: u_particle
    DOUBLE PRECISION, dimension(2), INTENT(OUT) :: wave_phase_update
    DOUBLE PRECISION :: phase_1_old, phase_1_new, phase_2_old, phase_2_new, BB_wave_para_1, BB_wave_para_2, BB_wave_para_sum
    double precision :: ff_1, gg_1, ff_2, gg_2
    INTEGER :: ii

    phase_1_old = wave_phase(1)
    phase_2_old = wave_phase(2)

    do ii = 1, 1000000
        if (ii == 1000000) then
            print *, "Error!: solution is not found. wave_phase = ", wave_phase
            print *, "                               wave_number_perp = ", wave_number_perp
            print *, "                               z_position = ", z_position
            print *, "                               u_perp = ", u_particle(1)
            print *, "                               u_phase = ", MOD(u_particle(2), 2d0*pi)
        endif

        CALL electrostatic_potential_to_BB_wave_para(electrostatic_potential_1, phase_1_old, z_position, BB_wave_para_1)
        CALL electrostatic_potential_to_BB_wave_para(electrostatic_potential_2, phase_2_old, z_position, BB_wave_para_2)

        BB_wave_para_sum = BB_wave_para_1 + BB_wave_para_2

        ff_1 = phase_1_old - wave_phase(1) &
            & - wave_number_perp * u_particle(1) * electron_mass * c_normal / charge / (BB + BB_wave_para_sum) * SIN(u_particle(2))
        gg_1 = 1 - wave_number_perp * u_particle(1) * electron_mass * c_normal / charge / (BB + BB_wave_para_sum)**2d0 &
            & * SIN(u_particle(2)) * BB_wave_para_1 * TAN(phase_1_old)

        phase_1_new = phase_1_old - ff_1 / gg_1


        ff_2 = phase_2_old - wave_phase(2) &
            & - wave_number_perp * u_particle(1) * electron_mass * c_normal / charge / (BB + BB_wave_para_sum) * SIN(u_particle(2))
        gg_2 = 1 - wave_number_perp * u_particle(1) * electron_mass * c_normal / charge / (BB + BB_wave_para_sum)**2d0 &
            & * SIN(u_particle(2)) * BB_wave_para_2 * TAN(phase_2_old)

        phase_2_new = phase_2_old - ff_2 / gg_2

        if (sqrt((phase_1_new - phase_1_old)**2d0 + (phase_2_new - phase_2_old)**2d0) <= 1d-5) exit
        phase_1_old = phase_1_new
        phase_2_old = phase_2_new

    end do !ii

    wave_phase_update(1) = phase_1_new
    wave_phase_update(2) = phase_2_new

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine Motion_of_Equation(z_position, wave_phase_1, wave_phase_2, z_p, u_p, force, wave_phase_p)
    !p -> particle

    use constant_parameter, only: pi
    use lshell_setting
    use constants_in_the_simulations, only: n_z

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: z_position(-n_z:n_z), wave_phase_1(-n_z:n_z), wave_phase_2(-n_z:n_z)
    DOUBLE PRECISION, INTENT(IN) :: z_p, u_p(0:2)
    DOUBLE PRECISION, INTENT(OUT) :: force(0:2)
    double precision, dimension(2), intent(out) :: wave_phase_p
    double precision :: wave_phase_1_p, wave_phase_2_p
    DOUBLE PRECISION :: gamma, ratio, BB_p, dB_dz_p, wave_number_perp_p, wave_number_para_1_p, wave_number_para_2_p
    double precision :: wave_frequency_p, force_wave(0:2), electrostatic_potential_1_p, electrostatic_potential_2_p
    double precision, dimension(2) :: wave_phase_p_update
    double precision :: EE_wave_para_1_p, EE_wave_perp_perp_1_p, EE_wave_perp_phi_1_p, BB_wave_para_1_p, BB_wave_perp_1_p
    double precision :: EE_wave_para_2_p, EE_wave_perp_perp_2_p, EE_wave_perp_phi_2_p, BB_wave_para_2_p, BB_wave_perp_2_p
    double precision :: EE_wave_para_sum_p, EE_wave_perp_perp_sum_p, EE_wave_perp_phi_sum_p, BB_wave_para_sum_p, BB_wave_perp_sum_p
    double precision :: particle_Larmor_radius_p
    INTEGER :: i_z_left, i_z_right

    CALL u_particle_to_gamma(u_p, gamma)
    CALL z_particle_to_position(z_p, z_position, i_z_left, i_z_right, ratio)
    CALL z_position_to_BB(z_p, BB_p)

    wave_phase_1_p = (1d0 - ratio) * wave_phase_1(i_z_left) + ratio * wave_phase_1(i_z_right)
    wave_phase_2_p = (1d0 - ratio) * wave_phase_2(i_z_left) + ratio * wave_phase_2(i_z_right)

    CALL z_position_to_electrostatic_potential(z_p, wave_phase_1_p, electrostatic_potential_1_p, 1)
    CALL z_position_to_electrostatic_potential(z_p, wave_phase_2_p, electrostatic_potential_2_p, 2)
    
    CALL z_position_to_wave_frequency(wave_frequency_p)
    CALL z_position_to_wave_number_perp(BB_p, wave_number_perp_p)
    
    CALL z_position_to_wave_number_para(z_p, BB_p, wave_number_perp_p, wave_number_para_1_p, 1)
    CALL z_position_to_wave_number_para(z_p, BB_p, wave_number_perp_p, wave_number_para_2_p, 2)

    wave_phase_p(1) = wave_phase_1_p
    wave_phase_p(2) = wave_phase_2_p

    CALL wave_phase_plus_perp_contribution(wave_phase_p, wave_number_perp_p, electrostatic_potential_1_p, &
                                        & electrostatic_potential_2_p, z_p, BB_p, u_p, wave_phase_p_update)
    wave_phase_p = wave_phase_p_update

    wave_phase_1_p = wave_phase_p(1)
    wave_phase_2_p = wave_phase_p(2)


    CALL electrostatic_potential_to_EE_wave_para(electrostatic_potential_1_p, wave_number_para_1_p, wave_phase_1_p,EE_wave_para_1_p)
    CALL electrostatic_potential_to_EE_wave_perp_perp(electrostatic_potential_1_p, wave_frequency_p, wave_number_perp_p, &
                                                    & wave_phase_1_p, z_p, u_p(2), EE_wave_perp_perp_1_p)
    CALL electrostatic_potential_to_EE_wave_perp_phi(electrostatic_potential_1_p, wave_frequency_p, wave_number_perp_p, &
                                                    & wave_phase_1_p, z_p, u_p(2), EE_wave_perp_phi_1_p)
    CALL electrostatic_potential_to_BB_wave_para(electrostatic_potential_1_p, wave_phase_1_p, z_p, BB_wave_para_1_p)
    CALL electrostatic_potential_to_BB_wave_perp(electrostatic_potential_1_p, wave_phase_1_p, wave_frequency_p, &
                                                & wave_number_para_1_p, wave_number_perp_p, BB_wave_perp_1_p)
    
    
    CALL electrostatic_potential_to_EE_wave_para(electrostatic_potential_2_p, wave_number_para_2_p, wave_phase_2_p,EE_wave_para_2_p)
    CALL electrostatic_potential_to_EE_wave_perp_perp(electrostatic_potential_2_p, wave_frequency_p, wave_number_perp_p, &
                                                    & wave_phase_2_p, z_p, u_p(2), EE_wave_perp_perp_2_p)
    CALL electrostatic_potential_to_EE_wave_perp_phi(electrostatic_potential_2_p, wave_frequency_p, wave_number_perp_p, &
                                                    & wave_phase_2_p, z_p, u_p(2), EE_wave_perp_phi_2_p)
    CALL electrostatic_potential_to_BB_wave_para(electrostatic_potential_2_p, wave_phase_2_p, z_p, BB_wave_para_2_p)
    CALL electrostatic_potential_to_BB_wave_perp(electrostatic_potential_2_p, wave_phase_2_p, wave_frequency_p, &
                                                & wave_number_para_2_p, wave_number_perp_p, BB_wave_perp_2_p)

    EE_wave_para_sum_p = EE_wave_para_1_p + EE_wave_para_2_p
    EE_wave_perp_perp_sum_p = EE_wave_perp_perp_1_p + EE_wave_perp_perp_2_p
    EE_wave_perp_phi_sum_p = EE_wave_perp_phi_1_p + EE_wave_perp_phi_2_p
    BB_wave_para_sum_p = BB_wave_para_1_p + BB_wave_para_2_p
    BB_wave_perp_sum_p = BB_wave_perp_1_p + BB_wave_perp_2_p

    call calculation_particle_Larmor_radius(u_p, BB_p, BB_wave_para_sum_p, particle_Larmor_radius_p)

    call z_particle_to_dB_dz(z_p, u_p, particle_Larmor_radius_p, BB_p, wave_number_para_1_p, wave_number_para_2_p, &
        & wave_number_perp_p, wave_phase_1_p, wave_phase_2_p, dB_dz_p)
    

    !force(EE_wave & BB_wave_perp)
    force_wave(0) = - charge * EE_wave_para_sum_p / electron_mass - u_p(1)**2d0 / 2d0 / (BB_p + BB_wave_para_sum_p) / gamma * 2d0 &
        & / particle_Larmor_radius_p * BB_wave_perp_sum_p * cos(u_p(2))
    force_wave(1) = - charge * EE_wave_perp_perp_sum_p / electron_mass + u_p(1)*u_p(0) / 2d0 / (BB_p + BB_wave_para_sum_p) / gamma &
        & * 2d0 / particle_Larmor_radius_p * BB_wave_perp_sum_p *cos(u_p(2))
    force_wave(2) = + charge * EE_wave_perp_phi_sum_p / electron_mass / gamma / c_normal * gamma * c_normal / u_p(1) &
            & - charge * BB_wave_perp_sum_p / electron_mass / gamma / c_normal * u_p(0) / u_p(1) * SIN(u_p(2))

    !force
    force(0) = - u_p(1)**2d0 / 2d0 / (BB_p + BB_wave_para_sum_p) / gamma * dB_dz_p + force_wave(0)
    force(1) = u_p(0) * u_p(1) / 2d0 / (BB_p + BB_wave_para_sum_p) / gamma * dB_dz_p + force_wave(1)
    force(2) = charge * (BB_p + BB_wave_para_sum_p) / electron_mass / gamma / c_normal + force_wave(2)

    if (isnan(force(0))) then
        print *, 'Motion_of_Equation: force(0) = NaN'
    end if
    if (isnan(force(1))) then
        print *, 'Motion_of_Equation: force(1) = NaN'
    end if
    if (isnan(force(2))) then
        print *, 'Motion_of_Equation: force(2) = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine particle_update_by_runge_kutta(z_in, wave_phase_in_1, wave_phase_in_2, z_particle, u_particle, equator_flag, edge_flag, &
    & wave_phase_1)

    use constants_in_the_simulations, only: d_t, n_z, L_z, d_z

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: z_in(-n_z:n_z), wave_phase_in_1(-n_z:n_z), wave_phase_in_2(-n_z:n_z)
    DOUBLE PRECISION, INTENT(INOUT) :: z_particle, u_particle(0:2)
    DOUBLE PRECISION, dimension(2), INTENT(OUT) :: wave_phase_1
    INTEGER, INTENT(OUT) :: edge_flag, equator_flag
    DOUBLE PRECISION :: ff_RK_1(0:2), ff_RK_2(0:2), ff_RK_3(0:2), ff_RK_4(0:2), u_particle_s(0:2)
    DOUBLE PRECISION :: kk_RK_1, kk_RK_2, kk_RK_3, kk_RK_4
    DOUBLE PRECISION, dimension(2) :: wave_phase_2, wave_phase_3, wave_phase_4
    

    u_particle_s(:) = u_particle(:)

    !RK4
    CALL u_particle_to_v_particle_para(u_particle_s, kk_RK_1)
    CALL Motion_of_Equation(z_in, wave_phase_in_1, wave_phase_in_2, &
        & z_particle, u_particle, ff_RK_1, wave_phase_1)

    CALL u_particle_to_v_particle_para(u_particle_s + ff_RK_1 / 2d0 * d_t, kk_RK_2)
    CALL Motion_of_Equation(z_in, wave_phase_in_1, wave_phase_in_2, &
        & z_particle + kk_RK_1 / 2d0 * d_t, u_particle_s + ff_RK_1 / 2d0 * d_t, ff_RK_2, wave_phase_2)

    CALL u_particle_to_v_particle_para(u_particle_s + ff_RK_2 / 2d0 * d_t, kk_RK_3)
    CALL Motion_of_Equation(z_in, wave_phase_in_1, wave_phase_in_2, &
        & z_particle + kk_RK_2 / 2d0 * d_t, u_particle_s + ff_RK_2 / 2d0 * d_t, ff_RK_3, wave_phase_3)

    CALL u_particle_to_v_particle_para(u_particle_s + ff_RK_3 * d_t, kk_RK_4)
    CALL Motion_of_Equation(z_in, wave_phase_in_1, wave_phase_in_2, &
        & z_particle + kk_RK_3 * d_t, u_particle_s + ff_RK_3 * d_t, ff_RK_4, wave_phase_4)

    !particle update
    u_particle(:) = u_particle(:) + (ff_RK_1(:) + 2d0 * ff_RK_2(:) + 2d0 * ff_RK_3(:) + ff_RK_4(:)) * d_t / 6d0
    z_particle = z_particle + (kk_RK_1 + 2d0 * kk_RK_2 + 2d0 * kk_RK_3 + kk_RK_4) * d_t / 6d0
    
    if (z_particle >= L_z) then !mirror
        z_particle = L_z - (z_particle - L_z)
        u_particle(0) = - DABS(u_particle(0))
        edge_flag = 1

    else if (z_particle <= -L_z) then
        z_particle = -L_z - (z_particle + L_z)
        u_particle(0) = - DABS(u_particle(0))
        edge_flag = 1
    end if

    equator_flag = 0
    if ( z_particle <= z_in(1) .and. z_particle >= z_in(-1) ) then
        equator_flag = 1
    end if

    if (isnan(u_particle(0))) then
        print *, 'particle_update_by_runge_kutta: u_particle(0) = NaN'
    end if
    if (isnan(u_particle(1))) then
        print *, 'particle_update_by_runge_kutta: u_particle(1) = NaN'
    end if
    if (isnan(u_particle(2))) then
        print *, 'particle_update_by_runge_kutta: u_particle(2) = NaN'
    end if
    if (isnan(z_particle)) then
        print *, 'particle_update_by_runge_kutta: z_particle = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_particle_to_wave_phase(z_position, z_particle, u_p, wave_phase_1, wave_phase_2, wave_phase_p)
    use constants_in_the_simulations, only: n_z

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: z_position(-n_z:n_z), wave_phase_1(-n_z:n_z), wave_phase_2(-n_z:n_z)
    DOUBLE PRECISION, INTENT(IN) :: z_particle, u_p(0:2)
    DOUBLE PRECISION, dimension(2), INTENT(OUT) :: wave_phase_p
    INTEGER :: i_z_left, i_z_right
    DOUBLE PRECISION :: ratio, BB_p, electrostatic_potential_1_p, electrostatic_potential_2_p, wave_number_perp_p
    double precision, dimension(2) :: wave_phase_p_update

    
    CALL z_particle_to_position(z_particle, z_position, i_z_left, i_z_right, ratio)
    CALL z_position_to_BB(z_particle, BB_p)

    wave_phase_p(1) = (1d0 - ratio) * wave_phase_1(i_z_left) + ratio * wave_phase_1(i_z_right)
    wave_phase_p(2) = (1d0 - ratio) * wave_phase_2(i_z_left) + ratio * wave_phase_2(i_z_right)

    CALL z_position_to_electrostatic_potential(z_particle, wave_phase_p(1), electrostatic_potential_1_p, 1)
    CALL z_position_to_electrostatic_potential(z_particle, wave_phase_p(2), electrostatic_potential_2_p, 2)
    CALL z_position_to_wave_number_perp(BB_p, wave_number_perp_p)

    CALL wave_phase_plus_perp_contribution(wave_phase_p, wave_number_perp_p, electrostatic_potential_1_p, &
        & electrostatic_potential_2_p, z_particle, BB_p, u_p, wave_phase_p_update)
    
    wave_phase_p = wave_phase_p_update

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine u_particle_to_energy(u_particle, energy)
    use lshell_setting
    
    implicit none

    DOUBLE PRECISION, INTENT(IN) :: u_particle(0:2)
    DOUBLE PRECISION, INTENT(OUT) :: energy
    DOUBLE PRECISION :: gamma

    gamma = DSQRT(1 + (u_particle(0)**2d0 + u_particle(1)**2d0) / c_normal**2d0)
    energy = gamma - 1d0

    if (isnan(energy)) then
        print *, 'u_particle_to_energy: energy = NaN'
    end if
    
end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine u_particle_to_alpha_eq(z_particle, u_particle, alpha_particle_eq)

    use constant_parameter, only: rad2deg

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: z_particle, u_particle(0:2)
    DOUBLE PRECISION, INTENT(OUT) :: alpha_particle_eq
    DOUBLE PRECISION :: BB_particle

    CALL z_position_to_BB(z_particle, BB_particle)
    alpha_particle_eq = ASIN(SIN(ATAN2(u_particle(1), u_particle(0))) / SQRT(BB_particle)) * rad2deg

    if (isnan(alpha_particle_eq)) then
        print *, 'u_particle_to_alpha_eq: alpha_particle_eq = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine u_particle_to_pitch_angle(u_particle, alpha_particle)

    use constant_parameter, only: rad2deg

    implicit none

    DOUBLE PRECISION, INTENT(IN) :: u_particle(0:2)
    DOUBLE PRECISION, INTENT(OUT) :: alpha_particle
    
    alpha_particle = atan2(u_particle(1), u_particle(0)) * rad2deg

    if (isnan(alpha_particle)) then
        print *, 'u_particle_to_alpha_eq: alpha_particle_eq = NaN'
    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine write_time(string_in)
    implicit none
    CHARACTER(10) :: date1, date2, date3
    CHARACTER(20) :: string_out
    CHARACTER(20), OPTIONAL, INTENT(IN) :: string_in

    INTEGER       :: date_time(10)

    if (present(string_in)) then
        string_out = string_in
    else
        write(string_out, *) '     ' 
    end if

    call date_and_time(date1, date2, date3, date_time)
    write(*, '(I4, A1, I2.2, A1, I2.2, A1, I2.2, A1, I2.2, A1, I2.2, A1, A20)') &
        & date_time(1), '/', date_time(2), '/', date_time(3), ' ',         &
        & date_time(5), ':', date_time(6), ':', date_time(7), ' ',         &
        & string_out
end subroutine