program main
    !$ use omp_lib

    use mt19937
    use constant_parameter
    use lshell_setting
    use variables

    implicit none

!--MPI----------------------------------------------------------------------
    include 'mpif.h'
    INTEGER(KIND=4) ierr, nprocs, myrank
    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
!--MPI----------------------------------------------------------------------

    call write_time(string)

    write(command, '(A22)') 'mkdir results_particle'
    call system(command)
    write(command, '(A29, I3.3)') 'mkdir results_particle/myrank', myrank
    call system(command)
    
    !------------------------
    !initial setting of field
    !------------------------
    do i_z = -n_z, n_z
        z_position(i_z) = DBLE(i_z) * d_z
        call z_position_to_BB(z_position(i_z), BB(i_z))
        call z_position_to_wave_frequency(wave_frequency(i_z))
        call z_position_to_wave_number_perp(BB(i_z), wave_number_perp(i_z))
        call z_position_to_wave_number_para(z_position(i_z), BB(i_z), wave_number_perp(i_z), wave_number_para_1(i_z), 1)
        call z_position_to_wave_number_para(z_position(i_z), BB(i_z), wave_number_perp(i_z), wave_number_para_2(i_z), 2)
    end do !i_z

    !initial wave_phase profile
    wave_phase_1(0) = initial_wave_phase
    wave_phase_2(0) = initial_wave_phase

    do i_z = 1, n_z
        call wave_number_para_to_wave_phase_initial(wave_number_para_1(i_z-1), wave_number_para_1(i_z), wave_phase_1(i_z-1), &
        & wave_phase_1(i_z), 1d0)
        call wave_number_para_to_wave_phase_initial(wave_number_para_1(1-i_z), wave_number_para_1(-i_z), wave_phase_1(1-i_z), &
        & wave_phase_1(-i_z), -1d0)

        call wave_number_para_to_wave_phase_initial(wave_number_para_2(i_z-1), wave_number_para_2(i_z), wave_phase_2(i_z-1), &
        & wave_phase_2(i_z), 1d0)
        call wave_number_para_to_wave_phase_initial(wave_number_para_2(1-i_z), wave_number_para_2(-i_z), wave_phase_2(1-i_z), &
        & wave_phase_2(-i_z), -1d0)
    end do !i_z


    !----------------------------
    !initial setting of particles
    !----------------------------
    CALL system_clock(count=clock)
    clock = 4267529
    CALL sgrnd(clock)
    print *, clock

    !count the quantity of the data
    WRITE(file_data, '(A37)') 'initial_condition_test_number_180.csv'
    !WRITE(file_data, '(A17, I3.3, A4)') 'initial_condition', myrank, '.dat'
    OPEN (500, file = file_data)
    N_particle = 0
    
    i = 0
    do !s
        i = i + 1
        read(500, *, end = 99) 
        N_particle = N_particle + 1
    end do !s

    99 close(500)

    print *, 'N_particle =', N_particle

    !allocate the capacity
    allocate(alpha0(1:N_particle))
    allocate(gamma0(1:N_particle))
    allocate(energy0(1:N_particle))
    allocate(alpha_eq(1:N_particle))
    allocate(z_particle(1:N_particle))
    allocate(u_particle(0:2, 1:N_particle))
    allocate(equator_time(1:N_particle))
    allocate(equator_flag(1:N_particle))
    allocate(wave_flag(1:N_particle))
    allocate(edge_flag(1:N_particle))

    
    !get the values
    !WRITE(file_data, '(A26)') 'initial_condition_test.csv'
    open (500, file = file_data)

    do i = 1, N_particle
        read(500,*,iostat=ios) energy0(i), alpha0(i), z_particle(i), alpha_eq(i)
        !print *, energy0(i), alpha0(i), z_particle(i), alpha_eq(i)

        energy0(i) = energy0(i) * 1d3 * (q/c*1d1) * 1d7 / J_unit ![keV]→[eV]→[erg]→[]

        gamma0(i) = energy0(i) / electron_mass / c_normal**2d0 + 1d0
        !print *, i, myrank, energy0(i), gamma0(i)

        v_particle = SQRT(1d0 - 1d0/gamma0(i)**2) * c_normal
        v_particle_para = v_particle * COS(alpha0(i)*deg2rad)
        v_particle_perp = v_particle * SIN(alpha0(i)*deg2rad)
        !print *, i, myrank, v_particle, v_particle_para, v_particle_perp

        u_particle(0, i) = gamma0(i) * v_particle_para
        u_particle(1, i) = gamma0(i) * v_particle_perp
        u_particle(2, i) = 2d0 * pi * grnd()
        !print *, i, myrank, u_particle(:, i)
    end do !i

    98 CLOSE(500)

    !flag & sign reset
    equator_flag = 0
    equator_time = 0
    wave_flag = 0
    edge_flag = 0




    !-----------------------
    !initial setting of wave
    !-----------------------
    if (wave_existance .eqv. .True.) then
        WRITE(file_check, '(A23, I3.3, A19)') 'results_particle/myrank', myrank, '/potential_prof.dat'
        OPEN(unit = 10, file = file_check)
        do i_z = -n_z, n_z
            CALL z_position_to_ion_Larmor_radius(BB(i_z), ion_Larmor_radius(i_z))

            CALL z_position_to_electrostatic_potential(z_position(i_z), electrostatic_potential_1(i_z), 1)
            CALL z_position_to_electrostatic_potential(z_position(i_z), electrostatic_potential_2(i_z), 2)

            call electrostatic_potential_to_EE_wave_para(electrostatic_potential_1(i_z), wave_number_para_1(i_z), &
                & wave_phase_1(i_z), EE_wave_para_1(i_z))
            CALL electrostatic_potential_to_EE_wave_perp_perp(electrostatic_potential_1(i_z), wave_frequency(i_z), &
                & wave_number_perp(i_z), wave_phase_1(i_z), z_position(i_z), 0d0, EE_wave_perp_perp_1(i_z))
            CALL electrostatic_potential_to_EE_wave_perp_phi(electrostatic_potential_1(i_z), wave_frequency(i_z), &
                & wave_number_perp(i_z), wave_phase_1(i_z), z_position(i_z), 0d0, EE_wave_perp_phi_1(i_z))
            CALL electrostatic_potential_to_BB_wave_para(electrostatic_potential_1(i_z), wave_phase_1(i_z), z_position(i_z), &
                & BB_wave_para_1(i_z))
            CAll electrostatic_potential_to_BB_wave_perp(electrostatic_potential_1(i_z), wave_phase_1(i_z), wave_frequency(i_z), &
                & wave_number_para_1(i_z), wave_number_perp(i_z), BB_wave_perp_1(i_z))

            call electrostatic_potential_to_EE_wave_para(electrostatic_potential_2(i_z), wave_number_para_2(i_z), &
                & wave_phase_2(i_z), EE_wave_para_2(i_z))
            CALL electrostatic_potential_to_EE_wave_perp_perp(electrostatic_potential_2(i_z), wave_frequency(i_z), &
                & wave_number_perp(i_z), wave_phase_2(i_z), z_position(i_z), 0d0, EE_wave_perp_perp_2(i_z))
            CALL electrostatic_potential_to_EE_wave_perp_phi(electrostatic_potential_2(i_z), wave_frequency(i_z), &
                & wave_number_perp(i_z), wave_phase_2(i_z), z_position(i_z), 0d0, EE_wave_perp_phi_2(i_z))
            CALL electrostatic_potential_to_BB_wave_para(electrostatic_potential_2(i_z), wave_phase_2(i_z), z_position(i_z), &
                & BB_wave_para_2(i_z))
            CAll electrostatic_potential_to_BB_wave_perp(electrostatic_potential_2(i_z), wave_phase_2(i_z), wave_frequency(i_z), &
                & wave_number_para_2(i_z), wave_number_perp(i_z), BB_wave_perp_2(i_z))
            
            call z_position_to_alfven_velocity(BB(i_z), alfven_velocity(i_z))
            CALL z_position_to_beta_ion(BB(i_z), beta_ion(i_z))
            CALL z_position_to_number_density(number_density(i_z))
            !print *, z_position(i_z), 'ep = ', electrostatic_potential_1(i_z), 'ion_Larmor_radius =', ion_Larmor_radius(i_z)
            !print *, z_position(i_z), 'wnperp = ', wave_number_perp(i_z), 'Te = ', Temperature_electron
            !print *, z_position(i_z), 'Ti = ', Temperature_ion, 'ep0 = ', electrostatic_potential_0
            !print *, z_position(i_z), 'V_unit = ', V_unit
            !print *, z_position(i_z), 'wnpara = ', wave_number_para_1(i_z)
            !print *, z_position(i_z), 'Ewpara = ', EE_wave_para_1(i_z)
            !print *
            WRITE(10, '(30E15.7)') z_position(i_z) * z_unit * 1d-2 / R_E, & ![]→[cm]→[m]→[/R_E]
                                & wave_number_para_1(i_z) / z_unit * 1d2 , & ![rad]→[rad/cm]→[rad/m]
                                & wave_number_perp(i_z) / z_unit * 1d2, & ![rad]→[rad/cm]→[rad/m]
                                & wave_frequency(i_z) / t_unit, & ![rad]→[rad/s]
                                & wave_frequency(i_z)/wave_number_para_1(i_z) * c / 1d2, & ![]→[cm/s]→[m/s]
                                & electrostatic_potential_1(i_z) * V_unit * c / 1d8, & ![]→[statV]→[V]
                                & EE_wave_para_1(i_z) * B0_eq * c / 1d6, & ![]→[statV/cm]→[V/m]
                                & EE_wave_perp_perp_1(i_z) * B0_eq * c / 1d6, & ![]→[statV/cm]→[V/m]
                                & EE_wave_perp_phi_1(i_z) * B0_eq * c / 1d6, & ![]→[statV/cm]→[V/m]
                                & BB_wave_para_1(i_z) * B0_eq / 1d4, & ![]→[G]→[T]
                                & BB_wave_perp_1(i_z) * B0_eq / 1d4, & ![]→[G]→[T]
                                & MOD(wave_phase_1(i_z), 2*pi), &
                                & wave_number_para_2(i_z) / z_unit * 1d2 , & ![rad]→[rad/cm]→[rad/m]
                                & wave_number_perp(i_z) / z_unit * 1d2, & ![rad]→[rad/cm]→[rad/m]
                                & wave_frequency(i_z) / t_unit, & ![rad]→[rad/s]
                                & wave_frequency(i_z)/wave_number_para_2(i_z) * c / 1d2, & ![]→[cm/s]→[m/s]
                                & electrostatic_potential_2(i_z) * V_unit * c / 1d8, & ![]→[statV]→[V]
                                & EE_wave_para_2(i_z) * B0_eq * c / 1d6, & ![]→[statV/cm]→[V/m]
                                & EE_wave_perp_perp_2(i_z) * B0_eq * c / 1d6, & ![]→[statV/cm]→[V/m]
                                & EE_wave_perp_phi_2(i_z) * B0_eq * c / 1d6, & ![]→[statV/cm]→[V/m]
                                & BB_wave_para_2(i_z) * B0_eq / 1d4, & ![]→[G]→[T]
                                & BB_wave_perp_2(i_z) * B0_eq / 1d4, & ![]→[G]→[T]
                                & MOD(wave_phase_2(i_z), 2*pi), &
                                & alfven_velocity(i_z) * c / 1d2, & ![]→[cm/s]→[m/s]
                                & ion_Larmor_radius(i_z) * z_unit * 1d-2, & ![]→[cm]→[m]
                                & beta_ion(i_z), &
                                & BB(i_z) * B0_eq / 1d4, & ![]→[G]→[T]
                                & Temperature_ion * J_unit * 1d-7 / (q/c*1d1), &![]→[erg]→[J]→[eV]
                                & Temperature_electron * J_unit * 1d-7 / (q/c*1d1), &![]→[erg]→[J]→[eV]
                                & number_density(i_z) / z_unit**3d0 * 1d6 ![]→[cm^-3]→[m^-3]
        end do !i_z
        wave_exist_parameter = 1d0
        CLOSE(10)
    else
        wave_exist_parameter = 0d0
    end if
    
    

    !---------
    !file open
    !---------
    !do i_thr = 0, N_thr
    !    N_file = 20 + i_thr
    !    WRITE(file_equator, '(A7, I3.3, A17, I2.2, A4)') 'results', myrank, '/count_at_equator', i_thr, '.dat'
    !    OPEN(unit = N_file, file = file_equator)
    !end do !i_thr

    
    do i_thr = 0, n_thread-1
        N_file = 20 + i_thr
        WRITE(file_particle, '(A23, I3.3, A20, I2.2, A4)') 'results_particle/myrank', myrank, '/particle_trajectory', &
            & i_thr, '.dat'
        !print *, N_file, file_particle
        OPEN(unit = N_file, file = file_particle)
    end do !i_particle



    !----------------
    !simulation start
    !----------------
    do i_time = 1, n_time
        time = DBLE(i_time) * d_t


        !-----------------
        !update wave_phase
        !-----------------
        do i_z = -n_z, n_z
            CALL time_to_wave_phase_update(wave_phase_1(i_z), wave_frequency(i_z))
            CALL time_to_wave_phase_update(wave_phase_2(i_z), wave_frequency(i_z))
        end do !i_z

        if(mod(i_time, 100) == 1) print *, 'pass time_to_wave_phase_update: i_time =', i_time


        !$omp parallel num_threads(n_thread) &
        !$omp & private(i_particle, z_particle_sim, u_particle_sim, equator_flag_sim, wave_flag_sim, edge_flag_sim, &
        !$omp & N_file, alpha_particle_eq, energy_particle, BB_particle, alpha, v_particle, v_particle_para, v_particle_perp, &
        !$omp & wave_phase_sim, wave_phase_update, wave_phase_update2)
        !$omp do
        do i_particle = 1, N_particle
            !if(mod(i_time, 100) == 1) print *, 'pass time_to_wave_phase_update: i_time =', i_time, ', i_particle =', i_particle, &
            !    & ', edge_flag =', edge_flag(i_particle)

            if(edge_flag(i_particle) /= 1) then
            
                !if(mod(i_time, 100) == 1 .and. mod(i_particle, 100) == 1) then
                !    print *, myrank, i_time
                !end if

                z_particle_sim = z_particle(i_particle)
                u_particle_sim(:) = u_particle(:, i_particle)
                equator_flag_sim = equator_flag(i_particle)
                wave_flag_sim = wave_flag(i_particle)
                edge_flag_sim = edge_flag(i_particle)


                N_file = 20 + omp_get_thread_num()
                if(mod(i_time, 100) == 1 .and. mod(i_particle, 100) == 1) then
                    print *, myrank, i_time, N_file, i_particle, z_particle(i_particle), z_particle_sim, u_particle_sim(:)
                end if

                call particle_update_by_runge_kutta(z_position, wave_phase_1, wave_phase_2, z_particle_sim, &
                    & u_particle_sim, equator_flag_sim, edge_flag_sim, wave_exist_parameter, wave_phase_update)
                
                wave_phase_sim = wave_phase_update

                if(mod(i_time, 100) == 1 .and. mod(i_particle, 100) == 1) then
                    print *, 'z_particle_sim = ', z_particle_sim, ', u_particle_sim(:) = ', u_particle_sim
                end if

                CALL z_particle_to_wave_phase(z_position, z_particle_sim, u_particle_sim, wave_phase_1, wave_phase_2, &
                    & wave_phase_update2)

                wave_phase_sim = wave_phase_update2

                CALL u_particle_to_energy(u_particle_sim, energy_particle)
                CALL u_particle_to_alpha_eq(z_particle_sim, u_particle_sim, alpha_particle_eq)
                
                WRITE(unit = N_file, fmt = '(I7.7, 9E15.7)') i_particle, &
                                                        & time * t_unit, & ![]→[s]
                                                        & z_particle_sim * z_unit * 1d-2, & ![]→[cm]→[m]
                                                        & u_particle_sim(0) * c * 1d-2, & ![]→[cm/s]→[m/s]
                                                        & u_particle_sim(1) * c * 1d-2, & ![]→[cm/s]→[m/s]
                                                        & MODULO(u_particle_sim(2), 2d0 * pi) , & ![rad]
                                                        & energy_particle * J_unit * 1d-7 / (q/c*1d1) , & ![]→[erg]→[J]→[eV]
                                                        & alpha_particle_eq, & ![degree]
                                                        & MODULO(wave_phase_sim(1), 2d0*pi), & ![rad]
                                                        & MODULO(wave_phase_sim(2), 2d0*pi) ![rad]
                if(mod(i_time, 100) == 1 .and. mod(i_particle, 100) == 1) then
                    print *, 'pass write: i_time =', i_time, ', i_particle =', i_particle
                end if

                !if (edge_flag_sim == 1) then
                !    z_particle_sim = grnd() * ABS(z_particle_sim - L_z) + z_particle_sim
                !    CALL z_position_to_BB(z_particle_sim, BB_particle)
                !    alpha = ASIN(SQRT(BB_particle * SIN(alpha_eq(i_particle) * deg2rad)**2d0))
                !    v_particle = SQRT(1d0 - 1d0 / gamma0(i_particle)**2d0)
                !    v_particle_para = - v_particle * COS(alpha)
                !    v_particle_perp = v_particle * SIN(alpha)
                !    u_particle_sim(0) = gamma0(i_particle) * v_particle_para
                !    u_particle_sim(1) = gamma0(i_particle) * v_particle_perp
                !    u_particle_sim(2) = 2d0 * pi * grnd()
                !    edge_flag_sim = 0
                !end if

                z_particle(i_particle) = z_particle_sim
                u_particle(:, i_particle) = u_particle_sim(:)
                wave_flag(i_particle) = wave_flag_sim
                edge_flag(i_particle) = edge_flag_sim
                
            end if
        end do !i_particle
        !$omp end do nowait
        !$omp end parallel

        !-------
        !out put
        !-------
        if (mod(i_time, 1000) == 0) then
            WRITE(string, '(F6.2, A2, A6, I2.2)') DBLE(i_time) / DBLE(n_time) * 100d0, ' %'
            CALL write_time(string)
        end if
            
    end do !i_time

    print *, "end"


    !-MPI------------------
    CALL MPI_FINALIZE(ierr)
    !-MPI------------------

    
end program main