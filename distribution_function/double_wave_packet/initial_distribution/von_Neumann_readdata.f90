program distribution_read_and_calculate

    !$use omp_lib
    use mt19937
    use constant_parameter
    use lshell_setting

    implicit none

    !--------------------------------------
    !  constants_in_the_simulation
    !--------------------------------------

    double precision, parameter :: temperature_perp          = 1d2    ![eV]
    double precision, parameter :: temperature_para          = 1d2    ![eV]
    double precision, parameter :: energy_max                = 1d4    ![eV]
    double precision, parameter :: energy_min                = 0d0    ![eV]
    double precision, parameter :: pitch_angle_max           = 90d0    ![degree]
    double precision, parameter :: pitch_angle_min           = 0d0      ![degree]
    double precision, parameter :: distribution_function_max = 1d1    ![c^2]
    double precision, parameter :: distribution_function_min = 0d0    ![c^2]
    integer, parameter          :: particle_total_number     = 20000000
    double precision, parameter :: d_pitch_angle             = 1d0 ![degree]
    double precision, parameter :: d_energy                  = 5d1 ![eV]
    integer, parameter          :: phase_space_number        = 1000

    integer, parameter          :: z_number   = 3500
    double precision, parameter :: d_z        = 0.5d0
    double precision, parameter :: z_max      = z_number * d_z
    double precision, parameter :: d_time     = 1.0d0

    double precision, parameter :: losscone_deg = asin((4d0 * L**6d0 - 3d0 * L**5d0)**(-2.5d-1)) * rad2deg
    
    !-------------------------------------
    ! variables
    !-------------------------------------
    integer :: clock
    double precision :: z_position(0 : z_number)
    double precision :: magnetic_flux_density(0 : z_number)
    character(128) :: command
    character(74)    :: file_alpha_energy
    character(128)    :: file_phase_space
    character(51)   :: file_initial_condition
    character(55)   ::file_initial_condition_old
    character(128)    :: file_bounce_period

    integer :: count_a, count_number, count_E, count_vpara, count_vperp, count_z, count_tau, count_particle, iostat
    double precision :: variable_a, variable_E, variable_v

    !------------------------------
    ! for distribution function
    !------------------------------

    double precision :: equatorial_pitch_angle, gamma_velocity_abs, energy
    double precision :: gamma
    double precision :: distribution_function_random, distribution_function
    double precision :: gamma_velocity_para, gamma_velocity_perp, velocity_para, velocity_perp
    double precision :: pitch_angle_floor, energy_floor, pitch_angle_top, energy_top
    double precision :: v_para_floor, v_perp_floor, v_para_top, v_perp_top


    !-------------------------------------
    ! for count
    !-------------------------------------

    integer :: count_alpha_energy(int(pitch_angle_min / d_pitch_angle):int(pitch_angle_max / d_pitch_angle) - 1, &
        & int(energy_min / d_energy):int(energy_max / d_energy) - 1)
    integer :: weight_count_alpha_energy(int(pitch_angle_min / d_pitch_angle):int(pitch_angle_max / d_pitch_angle) - 1, &
        & int(energy_min / d_energy):int(energy_max / d_energy) - 1)
    integer :: count_phase_space(0 : phase_space_number - 1, 0 : phase_space_number - 1)
    INTEGER :: total_alpha_energy
    integer :: particle_num_for_plot_a_E
    integer :: phase_space_for_plot
    integer :: particle_num(int(pitch_angle_min / d_pitch_angle):int(pitch_angle_max / d_pitch_angle) - 1, &
        & int(energy_min / d_energy):int(energy_max / d_energy) - 1)
    integer :: particle_num_old(int(pitch_angle_min / d_pitch_angle):int(pitch_angle_max / d_pitch_angle) - 1, &
        & int(energy_min / d_energy):int(energy_max / d_energy) - 1)
    double precision :: norm_tau_b
    integer :: particle_num_quotient, particle_num_modulo

    !------------------------------
    ! for initial position
    !------------------------------

    double precision :: time
    double precision :: tau_b
    !double precision, parameter :: delta_t = 200  ![/t_unit] <- これなに？
    double precision, parameter :: delta_t = 5000000    ![/t_unit]
    !double precision, parameter :: delta_t = 2    ![/t_unit]
    double precision :: gamma_velocity(0:2), gamma_velocity_1(0:2), gamma_velocity_2(0:2)
    double precision :: magnetic_flux_density_at_mirror_point, z_position_at_mirror_point
    double precision :: z_particle, z_particle_1, z_particle_2, z_small, z_large, z_particle_1_to_mirror, z_particle_2_to_mirror
    double precision :: magnetic_flux_density_small, magnetic_flux_density_large, magnetic_flux_density_output
    double precision :: variable_time, pitch_angle_1, pitch_angle_2, pitch_angle_output, z_particle_output
    double precision :: equatorial_pitch_angle_output, energy_output
    double precision :: grnd_save

    !------------------------------
    ! for parallelization
    !------------------------------

    !INTEGER, PARAMETER :: n_parallelization = 100
    character(8)  :: date ! yyyymmdd
    character(10) :: time_check ! hhmmss.fff
    character(5)  :: zone ! shhmm
    integer :: value(8)   ! yyyy mm dd diff hh mm ss fff

    !--MPI-------------------------------------------------------------------
    include 'mpif.h'
    INTEGER(KIND=4)  ierr,nprocs,myrank
    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD,nprocs,ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD,myrank,ierr)
    !--MPI-------------------------------------------------------------------


    !-------------------------------------------
    !   initial setting
    !-------------------------------------------

    do count_z = 0, z_number
        z_position(count_z) = dble(count_z) * d_z
        call z_to_B(z_position(count_z), magnetic_flux_density(count_z))
    end do   !count_z

    call system_clock(count=clock)
    clock = 4267529
    call sgrnd(clock)
    if (myrank == 1) write(*,*) clock
    
    
    !------------------------------
    ! read data
    !------------------------------

    myrank = myrank + 1
    file_alpha_energy = ''
    write(file_alpha_energy, '(A67, I3.3, A4)') 'results/distribution_alpha_energy_str/distribution_alpha_energy_str', myrank,'.dat'
    if (myrank == 1) print *, trim(file_alpha_energy)
    open(12, file = file_alpha_energy, action='read', status='old', iostat=iostat)

    variable_a = 0.0d0
    variable_E = 0.0d0
    particle_num_for_plot_a_E = 0
    count_alpha_energy = 0
    particle_num = 0

    do count_a = int(pitch_angle_min / d_pitch_angle), int(pitch_angle_max / d_pitch_angle) - 1
        do count_E = int(energy_min / d_energy), int(energy_max / d_energy) - 1
            read(12, '(2E15.7, 3I10)') variable_a, variable_E, particle_num_for_plot_a_E, count_alpha_energy(count_a, count_E), &
                & particle_num(count_a, count_E)
                !if(myrank == 1 .and. mod(count_a, 5) == 0) then
                !    print *, count_a, count_E, particle_num(count_a, count_E)
                !end if
        end do  ! count_E
        read(12, '()')
    end do  ! count_a

    close(12)

    do count_a = int(pitch_angle_min / d_pitch_angle), int(pitch_angle_max / d_pitch_angle) - 1
        variable_a = dble(count_a) * d_pitch_angle

        do count_E = int(energy_min / d_energy), int(energy_max / d_energy) - 1
            variable_E = dble(count_E) * d_energy

            call energy_to_v(variable_E + d_energy/2d0, gamma, variable_v)
            !tau_b [s]: particle trajectory time from one mirror point to another mirror point
            tau_b      = 2d0 * r_eq / sqrt(1d0 - 1d0/gamma**2) *(1.30d0 - 0.56d0 * sin((variable_a + d_pitch_angle / 2d0) *deg2rad))
            norm_tau_b = 2d0 * r_eq / sqrt(1d0 - 1d0/gamma**2) * (1.30d0 - 0.56d0 * sin( 89.5d0 * deg2rad))
            weight_count_alpha_energy(count_a, count_E) = count_alpha_energy(count_a, count_E)* tau_b /norm_tau_b
        end do  ! count_E
    end do  ! count_a


    !----------------------------------------------------------
    !   read old data
    !----------------------------------------------------------

    file_initial_condition = ''
    file_initial_condition_old = ''
    write(file_initial_condition, '(A44, I3.3, A4)') 'results/initial_conditions/initial_condition', myrank, '.dat'
    write(file_initial_condition_old, '(A48, I3.3, A4)') 'results/initial_conditions/initial_condition_old', myrank, '.dat'
    if (myrank == 1) print *, trim(file_initial_condition)
    if (myrank == 1) print *, trim(file_initial_condition_old)

    write(command, '(A3, A51, A1, A55)') 'cp ', trim(file_initial_condition), ' ', trim(file_initial_condition_old)
    if (myrank == 1) print *, trim(command)
    call system(trim(command))
    open(14, file = file_initial_condition_old, action='read', status='old', iostat=iostat)

    energy_output = 0d0
    pitch_angle_output = 0d0
    z_particle_output = 0d0
    equatorial_pitch_angle_output = 0d0
    particle_num_old = 0d0
    do
        read (14, '(E15.7, 3(",", E15.7))', iostat = iostat) energy_output, pitch_angle_output, z_particle_output, &
            & equatorial_pitch_angle_output
        if (iostat == 0) then
            if (equatorial_pitch_angle_output > 90d0) then
                equatorial_pitch_angle_output = 180d0 - equatorial_pitch_angle_output
            end if
            count_a = int(equatorial_pitch_angle_output / d_pitch_angle)
            count_E = int(energy_output / d_energy)
            particle_num_old(count_a, count_E) = particle_num_old(count_a, count_E) + 1
        else if (iostat /= 0) then
            exit
        end if
    end do

    close(14)

    count_a = myrank
    variable_a = dble(count_a) * d_pitch_angle
    equatorial_pitch_angle  = (variable_a + d_pitch_angle / 2d0) * deg2rad
    do count_E = 0, 2
        variable_E = dble(count_E) * d_energy
        energy = variable_E + d_energy / 2d0
        gamma      = energy / m_e + 1d0
        variable_v = sqrt(1d0 - 1d0 / gamma**2d0)
        tau_b = 2d0 * r_eq / variable_v * (1.30d0 - 0.56d0 * sin(equatorial_pitch_angle))
        if (particle_num(count_a, count_E) /= 0) then
            particle_num_quotient = particle_num_old(count_a, count_E) / particle_num(count_a, count_E)
            particle_num_modulo = mod(particle_num_old(count_a, count_E), particle_num(count_a, count_E))
        else
            particle_num_quotient = 0
            particle_num_modulo = 0
        end if
        print *, myrank, count_E, particle_num_old(myrank, count_E), particle_num(myrank, count_E), int(tau_b / delta_t), &
            & particle_num_quotient, particle_num_modulo
    end do ! count_E

    !stop


    !----------------------------------------------------------
    !   decide_z_distribution
    !----------------------------------------------------------

    open(15, file = file_initial_condition, status='old', action='write', position='append')

    tau_b = 0d0

    do count_E = int(energy_min / d_energy), int(energy_max / d_energy) - 1

        count_a = myrank  !loss cone ~ 1.5 degrees in L=9

        variable_a = dble(count_a) * d_pitch_angle
        variable_E = dble(count_E) * d_energy
        
        if (weight_count_alpha_energy(count_a, count_E) >= 1) then
            equatorial_pitch_angle  = (variable_a + d_pitch_angle / 2d0) * deg2rad
            energy                  = variable_E + d_energy / 2d0
            gamma                   = energy / m_e + 1d0
            variable_v              = sqrt(1d0 - 1d0 / gamma**2d0)
            velocity_para           = variable_v * cos(equatorial_pitch_angle)
            velocity_perp           = variable_v * sin(equatorial_pitch_angle)
            gamma_velocity(0)       = gamma * velocity_para
            gamma_velocity(1)       = gamma * velocity_perp
            gamma_velocity(2)       = 2d0 * pi * grnd()
            tau_b                   = 2d0 * r_eq / variable_v * (1.30d0 - 0.56d0 * sin(equatorial_pitch_angle))

            magnetic_flux_density_at_mirror_point = 1d0 / sin(equatorial_pitch_angle)**2d0   ![/B0_eq]
            do count_z = 1, z_number
                z_position_at_mirror_point = z_position(count_z - 1)
                if(magnetic_flux_density(count_z) > magnetic_flux_density_at_mirror_point) exit
            end do   !count_z

            time = 0d0
            z_particle = 0d0

            if (particle_num(count_a, count_E) /= 0) then
                particle_num_quotient = particle_num_old(count_a, count_E) / particle_num(count_a, count_E)
                particle_num_modulo = mod(particle_num_old(count_a, count_E), particle_num(count_a, count_E))
            else
                particle_num_quotient = 0
                particle_num_modulo = 0
            end if

            print *, myrank, count_E, particle_num_old(myrank, count_E), particle_num(myrank, count_E), int(tau_b / delta_t), &
            & particle_num_quotient, particle_num_modulo

            if (particle_num(count_a, count_E) >= 1 .and. particle_num_quotient + 1 <= int(tau_b / delta_t)) then

                do count_tau = particle_num_quotient + 1, int(tau_b / delta_t)
                
                    variable_time = delta_t * dble(count_tau)

                    if ( variable_time <= tau_b ) then
                        z_particle_1 = z_particle
                        gamma_velocity_1 = gamma_velocity

                        do while(time <= variable_time)
                            call runge_kutta(z_position, z_particle, gamma_velocity(0:2))
                            z_particle_2 = z_particle
                            gamma_velocity_2 = gamma_velocity
                            time = time + 1d0 * d_time
                        end do

                        if ( z_particle_1 <= z_particle_2 ) then
                            z_small = z_particle_1
                            z_large = z_particle_2
                        else
                            z_small = z_particle_2
                            z_large = z_particle_1
                        end if

                        call z_to_B(z_small, magnetic_flux_density_small)
                        call z_to_B(z_large, magnetic_flux_density_large)

                        z_particle_1_to_mirror = z_position_at_mirror_point - z_particle_1
                        z_particle_2_to_mirror = z_position_at_mirror_point - z_particle_2

                        do count_particle = particle_num_modulo + 1, particle_num(count_a, count_E)
                            grnd_save = 0d0
                            do while(grnd_save == 0d0)
                                grnd_save = grnd() - 0.5
                            end do

                            if ( gamma_velocity_1(0) >= 0d0 .and. gamma_velocity_2(0) >= 0d0 ) then
                                pitch_angle_1 = asin(sqrt(magnetic_flux_density_small * sin(equatorial_pitch_angle)**2d0))
                                pitch_angle_2 = asin(sqrt(magnetic_flux_density_large * sin(equatorial_pitch_angle)**2d0))
                                if ( grnd_save >= 0 ) then
                                    pitch_angle_output = grnd() * (pitch_angle_2 - pitch_angle_1) + pitch_angle_1
                                else
                                    pitch_angle_output = pi - (grnd() * (pitch_angle_2 - pitch_angle_1) + pitch_angle_1)
                                end if
                                z_particle_output = (grnd() * (z_particle_2 - z_particle_1) + z_particle_1) * sign(1d0, grnd_save)
                            end if

                            if ( gamma_velocity_1(0) >= 0d0 .and. gamma_velocity_2(0) < 0d0 ) then

                                if ( grnd() < z_particle_1_to_mirror / (z_particle_1_to_mirror + z_particle_2_to_mirror)) then
                                    pitch_angle_1 = asin(sqrt(magnetic_flux_density_small * sin(equatorial_pitch_angle)**2d0))
                                    pitch_angle_2 = pi / 2d0
                                    if ( grnd_save > 0 ) then
                                        pitch_angle_output = grnd() * (pitch_angle_2 - pitch_angle_1) + pitch_angle_1
                                    else
                                        pitch_angle_output = pi - (grnd() * (pitch_angle_2 - pitch_angle_1) + pitch_angle_1)
                                    end if
                                    z_particle_output = (grnd() * (z_position_at_mirror_point - z_particle_1) + z_particle_1) &
                                        & * sign(1d0, grnd_save)

                                else
                                    pitch_angle_1 = pi / 2d0
                                    pitch_angle_2 = pi - asin(sqrt(magnetic_flux_density_large * sin(equatorial_pitch_angle)**2d0))
                                    if ( grnd_save > 0 ) then
                                        pitch_angle_output = grnd() * (pitch_angle_2 - pitch_angle_1) + pitch_angle_1
                                    else
                                        pitch_angle_output = pi - (grnd() * (pitch_angle_2 - pitch_angle_1) + pitch_angle_1)
                                    end if
                                    z_particle_output = (grnd() * (z_position_at_mirror_point - z_particle_2) + z_particle_2) &
                                        & * sign(1d0, grnd_save)

                                end if
                            end if

                            if ( gamma_velocity_1(0) < 0d0 .and. gamma_velocity_2(0) < 0d0 ) then
                                pitch_angle_1 = pi - asin(sqrt(magnetic_flux_density_small * sin(equatorial_pitch_angle)**2d0))
                                pitch_angle_2 = pi - asin(sqrt(magnetic_flux_density_large * sin(equatorial_pitch_angle)**2d0))
                                if ( grnd_save > 0 ) then
                                    pitch_angle_output = grnd() * (pitch_angle_2 - pitch_angle_1) + pitch_angle_1
                                else
                                    pitch_angle_output = pi - (grnd() * (pitch_angle_2 - pitch_angle_1) + pitch_angle_1)
                                end if
                                z_particle_output = (grnd() * (z_particle_1 - z_particle_2) + z_particle_2) * sign(1d0, grnd_save)
                            end if

                            call z_to_B(z_particle_output, magnetic_flux_density_output)
                            equatorial_pitch_angle_output = asin(sin(pitch_angle_output) / sqrt(magnetic_flux_density_output))

                            if (pitch_angle_output > pi / 2d0) then
                                equatorial_pitch_angle_output = pi - equatorial_pitch_angle_output
                            end if
                            
                            if ( abs(z_particle_output) <= z_max .and. losscone_deg < equatorial_pitch_angle_output * rad2deg &
                                & .and. equatorial_pitch_angle_output * rad2deg < 180d0 - losscone_deg ) then

                                energy_output = grnd() * d_energy + variable_E

                                if ( equatorial_pitch_angle_output == equatorial_pitch_angle_output ) then
                                    write(15,'(E15.7, 3(",", E15.7))') energy_output, pitch_angle_output * rad2deg, &
                                        & z_particle_output, equatorial_pitch_angle_output * rad2deg
                                end if
                            end if
                        end do   !count_particle
                    end if
                end do   !count_tau
            end if
        end if

        if (mod(count_E, 1) == 0 .and. count_number > 0) then
            write(*,*) 'pitch_angle', count_a, variable_a, 'energy', myrank, variable_E, count_number, &
                & particle_num(count_a, count_E), dble(weight_count_alpha_energy(count_a, count_E)), tau_b, delta_t, tau_b/delta_t
        end if

    end do

    write(*,*) "end", myrank

    close(15)

    !-MPI-------------------------------------------------------------------
    call MPI_FINALIZE(ierr)
    !-MPI-------------------------------------------------------------------

!---------------------------------------------------------------------------------
contains

subroutine runge_kutta(z_f, z_p, u_p)
    
    implicit none

    DOUBLE PRECISION, INTENT(IN)    :: z_f(0:z_number)
    DOUBLE PRECISION, INTENT(INOUT) :: z_p, u_p(0:2)
    DOUBLE PRECISION :: l1(0:2), l2(0:2), l3(0:2), l4(0:2), u_p_s(0:2)
    DOUBLE PRECISION :: k1, k2, k3, k4

    u_p_s(:) = u_p(:)

    call dz_dt(u_p_s, k1)
    call force(z_f, z_p, u_p_s, l1)

    call dz_dt(u_p_s + l1 / 2d0, k2)
    call force(z_f, z_p + k1 / 2d0, u_p_s + l1 / 2d0, l2)

    call dz_dt(u_p_s + l2 / 2d0, k3)
    call force(z_f, z_p + k2 / 2d0, u_p_s + l2 / 2d0, l3)

    call dz_dt(u_p_s + l3, k4)
    call force(z_f, z_p + k3, u_p_s + l3, l4)

    u_p(:) = u_p(:) + (l1 + 2d0 * l2 + 2d0 * l3 + l4) * d_time / 6d0
    z_p    = z_p    + (k1 + 2d0 * k2 + 2d0 * k3 + k4) * d_time / 6d0

    if (z_p <= 0.d0 ) then
        z_p    = - z_p
        u_p(0) =   DABS(u_p(0))

    else if (z_p >= z_max) then
        z_p    = z_max - (z_p - z_max)
        u_p(0) = - DABS(u_p(0))

    end if
    
end subroutine runge_kutta
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine force(z, z_p, p, f)
    !z_position(z) [/z_unit], z_particle(z_p) [/z_unit], p(=gamma*v)[/c] -> force [t_unit^2 / z_unit]

    implicit none

    DOUBLE PRECISION, PARAMETER   :: pi = 4d0*DATAN(1d0)
    DOUBLE PRECISION, INTENT(IN)  :: z(0 : z_number), z_p
    DOUBLE PRECISION, INTENT(IN)  :: p(0:2)
    DOUBLE PRECISION, INTENT(OUT) :: f(0:2)

    DOUBLE PRECISION :: gamma, B_p, dB_dz
    DOUBLE PRECISION :: ratio
    INTEGER :: i_z_left, i_z_right

    call p_to_gamma(p(0:2), gamma)
    call z_p_to_position(z_p, z, i_z_left, i_z_right, ratio)

    call z_to_B(z_p, B_p)
    call z_to_dB_dz(z_p, dB_dz)


    f(0) = - p(1)**2     * dB_dz / (2d0 * gamma * B_p)    ![t_unit^2 / z_unit]
    f(1) = + p(0) * p(1) * dB_dz / (2d0 * gamma * B_p)    ![t_unit^2 / z_unit]
    f(2) = + B_p / gamma                                  ![t_unit]

end subroutine force
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_to_dB_dz(z, dB_dz)
    !z_position [/z_unit] -> dB/dz [z_unit / B0_eq]
    use lshell_setting, only: r_eq
    implicit none

    DOUBLE PRECISION, INTENT(IN)  :: z
    DOUBLE PRECISION, INTENT(OUT) :: dB_dz
    DOUBLE PRECISION :: r, lambda

    call z_to_r_lambda(z, r, lambda)

    dB_dz = 3d0 * DSIN(lambda) / DCOS(lambda)**8 * (3d0 + 5d0 * DSIN(lambda)**2) / (1d0 + 3d0 * DSIN(lambda)**2) / r_eq

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine dz_dt(p, v)
    !p(=gamma*v)[/c] -> v [/c]

    implicit none

    DOUBLE PRECISION, INTENT(IN)  :: p(0:2)
    DOUBLE PRECISION, INTENT(OUT) :: v
    DOUBLE PRECISION :: gamma

    call p_to_gamma(p(0:2), gamma)
    v = p(0) / gamma

end subroutine dz_dt
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine p_to_gamma(p, gamma)
    !p(=gamma*v) [/c] -> gamma []

    implicit none

    DOUBLE PRECISION, INTENT(IN)  :: p(0:2)
    DOUBLE PRECISION, INTENT(OUT) :: gamma

    gamma = SQRT(1 + p(0)**2 + p(1)**2)

end subroutine p_to_gamma
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine energy_to_v(energy, gamma, v)
    !energy[eV] -> gamma [], v[/c]

    implicit none

    DOUBLE PRECISION, INTENT(IN)  :: energy
    DOUBLE PRECISION, INTENT(OUT) :: gamma, v

    gamma = energy / m_e + 1d0
    v = DSQRT(1d0 - 1d0/gamma**2)

end subroutine energy_to_v
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_p_to_position(z_p, z, i_z_left, i_z_right, ratio)
    ! z_p = (1d0 - ratio) * z(i_z_left) + ratio * z(i_z_right)
    ! discretize z_p
    implicit none

    DOUBLE PRECISION, INTENT(IN)  :: z(0 : z_number), z_p
    INTEGER, INTENT(OUT)          :: i_z_left, i_z_right
    DOUBLE PRECISION, INTENT(OUT) :: ratio
    DOUBLE PRECISION :: diff(0 : z_number), d
    INTEGER :: i_min(1)

    diff  = ABS(z - z_p)
    i_min = MINLOC(diff)

    if (i_min(1) >= z_number) then
        i_z_left  = z_number - 1
        i_z_right = z_number
        ratio = 1d0

    else if( i_min(1) <= -z_number) then
        i_z_left  = z_number - 1
        i_z_right = z_number
        ratio = 1d0

    else
        d = z(i_min(1)) - z_p
        if (d > 0) then
            i_z_left  = i_min(1) - 1
            i_z_right = i_min(1)
        else if (d <= 0) then
            i_z_left  = i_min(1)
            i_z_right = i_min(1) + 1
        end if

        ratio     = (z_p - z(i_z_left)) / (z(i_z_right) - z(i_z_left))

    end if

end subroutine
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_to_r_lambda(z, r, lambda)
    !z_position [/z_unit] -> r_position [/z_unit], lambda(MLAT [rad])

    implicit none

    DOUBLE PRECISION, INTENT(IN)  :: z
    DOUBLE PRECISION, INTENT(OUT) :: r, lambda
    DOUBLE PRECISION :: f, g, lambda0, lambda1
    INTEGER :: i

    lambda0 = 1d0
    do i = 1, 1000000
        if (i == 1000000) then
            write(*, *) "Error: solution is not found. z = ", z
        end if
        f = r_eq * (5d-1 * sin(lambda0) * sqrt(3d0 * sin(lambda0)**2d0 + 1d0) &
            & + 1d0 / 2d0 / sqrt(3d0) * asinh(sqrt(3d0) * sin(lambda0))) &
            & - z
        g = r_eq * DCOS(lambda0) * DSQRT(3d0 * DSIN(lambda0)**2d0 + 1d0)

        lambda1 = lambda0 - f / g
        if (DABS(lambda1 - lambda0) <=  1d-4) exit
        lambda0 = lambda1
    end do

    lambda = lambda1  ![rad]
    r      = r_eq * DCOS(lambda)**2  ![/z_unit]

end subroutine z_to_r_lambda
!
!!----------------------------------------------------------------------------------------------------------------------------------
!
subroutine z_to_B(z, B)
    !z_position [/z_unit] -> magnetic flux density [/B0_eq]

    implicit none

    DOUBLE PRECISION, INTENT(IN)  :: z
    DOUBLE PRECISION, INTENT(OUT) :: B
    DOUBLE PRECISION :: r, lambda

    call z_to_r_lambda(z, r, lambda)

    B = DSQRT(1d0 + 3d0 * DSIN(lambda)**2d0) / DCOS(lambda)**6d0   ![/B0_eq]

end subroutine z_to_B
!
!!----------------------------------------------------------------------------------------------------------------------------------
!

end program distribution_read_and_calculate