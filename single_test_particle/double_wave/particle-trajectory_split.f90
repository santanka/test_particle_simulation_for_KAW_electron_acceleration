program main
    !$use omp_lib

    implicit none

    integer, parameter :: thread_number = 36
    integer, parameter :: particle_in_thread = 5
    CHARACTER(112), dimension(thread_number) :: file_input
    character(116), dimension(thread_number, particle_in_thread) :: file_output
    INTEGER :: particle_number
    DOUBLE PRECISION :: time_simulation, z_position, u_para, u_perp, u_phi, energy, pitch_angle_eq, wave_phase, wave_growth_phase

    character(8)  :: date ! yyyymmdd
    character(10) :: time ! hhmmss.fff
    character(5)  :: zone ! shhmm
    integer :: value(8)   ! yyyy mm dd diff hh mm ss fff
    
    integer, dimension(thread_number) :: thread_file_number
    integer, dimension(thread_number, particle_in_thread) :: output_file_number
    integer :: thread_i, particle_i
    character(2) :: thread_i_character
    character(3) :: particle_i_character


    !$omp parallel private(thread_i, thread_i_character, particle_i, particle_i_character, particle_number, time_simulation, &
    !$omp & z_position, u_para, u_perp, u_phi, energy, pitch_angle_eq, wave_phase, wave_growth_phase)
    !$omp do
    do thread_i = 0, thread_number-1
    
        write(thread_i_character, '(I2.2)') thread_i
        file_input(thread_i + 1) = &
            & '/mnt/j/KAW_simulation_data/single_test_particle/double_wave/results_particle/myrank000/particle_trajectory' // &
            & thread_i_character // '.dat'

        thread_file_number(thread_i + 1) = 100 + thread_i

        open(unit = thread_file_number(thread_i + 1), file = file_input(thread_i + 1))

        do particle_i = 1, particle_in_thread

            call date_and_time(date, time, zone, value)
            print *, thread_i, particle_i, value
        
            rewind(thread_file_number(thread_i + 1))
            write(particle_i_character, '(I3.3)') particle_i + particle_in_thread * thread_i
            
            file_output(thread_i +1, particle_i) &
                & = '/mnt/j/KAW_simulation_data/single_test_particle/double_wave/results_particle/myrank000/particle_trajectory' //&
                & thread_i_character // '-' // particle_i_character // '.dat'

            output_file_number(thread_i +1, particle_i) = 200 + particle_i + particle_in_thread * thread_i

            open(unit = output_file_number(thread_i+1, particle_i), file = file_output(thread_i +1, particle_i))

            do
                
                read(unit = thread_file_number(thread_i + 1), fmt = *, end = 99) particle_number, time_simulation, z_position, &
                    & u_para, u_perp, u_phi, energy, pitch_angle_eq, wave_phase, wave_growth_phase

                if ( particle_number == particle_i + particle_in_thread * thread_i ) then
                    
                    write(unit = output_file_number(thread_i+1, particle_i), fmt = '(I3.3, 9E15.7)') particle_number, &
                        & time_simulation, z_position, u_para, u_perp, u_phi, energy, pitch_angle_eq, wave_phase, wave_growth_phase

                end if

            end do

            99 close(unit = output_file_number(thread_i+1, particle_i))
            
        end do !particle_i

        close(unit = thread_file_number(thread_i + 1))

    end do !thread_i
    !$omp end do
    !$omp end parallel
    
end program main