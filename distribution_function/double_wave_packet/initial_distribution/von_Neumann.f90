program distribution_calculater

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
   character(128)    :: file_alpha_energy
   character(128)    :: file_phase_space
   character(128)    :: file_initial_condition
   character(128)    :: file_bounce_period

   integer :: count_a, count_number, count_E, count_vpara, count_vperp, count_z, count_tau, count_particle
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

   integer :: count_alpha_energy(nint(pitch_angle_min / d_pitch_angle):nint(pitch_angle_max / d_pitch_angle) - 1, &
      & nint(energy_min / d_energy):nint(energy_max / d_energy) - 1)
   integer :: weight_count_alpha_energy(nint(pitch_angle_min / d_pitch_angle):nint(pitch_angle_max / d_pitch_angle) - 1, &
      & nint(energy_min / d_energy):nint(energy_max / d_energy) - 1)
   integer :: count_phase_space(0 : phase_space_number - 1, 0 : phase_space_number - 1)
   INTEGER :: total_alpha_energy
   integer :: particle_num_for_plot_a_E
   integer :: phase_space_for_plot
   integer :: particle_num
   double precision :: norm_tau_b

   !------------------------------
   ! for initial position
   !------------------------------

   double precision :: time
   double precision :: tau_b
   !double precision, parameter :: delta_t = 200  ![/t_unit] <- これなに？
   double precision, parameter :: delta_t = 2000    ![/t_unit]
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
   write(*,*) clock

   write(command, '(A13)') 'mkdir results'
   call system(command)
   write(command, '(A32)') 'mkdir results/initial_conditions'
   call system(command)
   write(command, '(A43)') 'mkdir results/distribution_alpha_energy_str'
   call system(command)
   write(command, '(A42)') 'mkdir results/distribution_phase_space_str'
   call system(command)
   write(command, '(A27)') 'mkdir results/bounce_period'
   call system(command)

   !-------------------------------------------
   ! decide distribution function
   !------------------------------------------
   myrank = myrank + 1
   write(file_alpha_energy, '(A67, I3.3, A4)') &
      & 'results/distribution_alpha_energy_str/distribution_alpha_energy_str' , myrank, '.dat'
   open (11, file = file_alpha_energy, status='replace')
   write(file_phase_space, '(A65, I3.3, A4)') &
      & 'results/distribution_phase_space_str/distribution_phase_space_str' , myrank, '.dat'
   open (12, file = file_phase_space, status='replace')
   write(file_bounce_period, '(A35, I3.3, A4)') &
      & 'results/bounce_period/bounce_period' , myrank, '.dat'
   open (13, file = file_bounce_period, status='replace')

   count_number = 0
   count_alpha_energy = 0
   count_phase_space = 0

   do while(count_number <= particle_total_number)

      if (mod(count_number, 1000000) == 0 .and. myrank == 1) then
         call date_and_time(date, time_check, zone, value)
         print *, count_number, myrank, value
      end if

      energy = grnd() * (energy_max - energy_min) + energy_min    ![eV]
      equatorial_pitch_angle = grnd() * (pitch_angle_max - pitch_angle_min) + pitch_angle_min      ![degree]
      gamma = energy / m_e + 1d0    ![]
      gamma_velocity_abs = sqrt(gamma**2d0 - 1d0)     ![/c]
      gamma_velocity_para = gamma_velocity_abs * cos(equatorial_pitch_angle * deg2rad)      ![/c]
      gamma_velocity_perp = gamma_velocity_abs * sin(equatorial_pitch_angle * deg2rad)      ![/c]
      velocity_para = gamma_velocity_para / gamma    ![/c]
      velocity_perp = gamma_velocity_perp / gamma    ![/c]
      distribution_function_random = grnd() * (distribution_function_max - distribution_function_min) + distribution_function_min
      ![c^2]

      distribution_function = 2d0 * velocity_perp / (2d0 * pi * temperature_perp / m_e) / sqrt(2d0 * pi * temperature_para / m_e) &
         & * exp(- velocity_perp**2d0 * m_e / temperature_perp / 2d0) * exp(- velocity_para**2d0 * m_e / temperature_para / 2d0)
      
      if ( distribution_function_random <= distribution_function ) then
         
         count_number = count_number + 1

         do count_a = nint(pitch_angle_min / d_pitch_angle), nint(pitch_angle_max / d_pitch_angle) - 1
            do count_E = nint(energy_min / d_energy), nint(energy_max / d_energy) - 1
               pitch_angle_floor = dble(count_a) * d_pitch_angle
               pitch_angle_top = dble(count_a + 1) * d_pitch_angle
               energy_floor = dble(count_E) * d_energy
               energy_top = dble(count_E + 1) * d_energy
               if ( pitch_angle_floor <= equatorial_pitch_angle .and. equatorial_pitch_angle < pitch_angle_top ) then
                  if ( energy_floor <= energy .and. energy < energy_top ) then
                     count_alpha_energy(count_a, count_E) &
                        & = count_alpha_energy(count_a, count_E) + 1
                  end if
               end if
            end do   !count_E
         end do   !count_a

         do count_vpara = 0, phase_space_number - 1
            do count_vperp = 0, phase_space_number - 1
               v_para_floor = dble(count_vpara) / dble(phase_space_number)
               v_para_top = dble(count_vpara + 1) / dble(phase_space_number)
               v_perp_floor = dble(count_vperp) / dble(phase_space_number)
               v_perp_top = dble(count_vperp + 1) / dble(phase_space_number)
               if ( v_para_floor <= velocity_para .and. velocity_para < v_para_top ) then
                  if ( v_perp_floor <= velocity_perp .and. velocity_perp < v_perp_top ) then
                     if (count_vpara < 0 .or. count_vpara > phase_space_number - 1 .or. count_vperp < 0 &
                        & .or. count_vperp > phase_space_number - 1 ) then
                        print *, count_vpara, count_vperp
                     end if
                     count_phase_space(count_vpara, count_vperp) &
                        & = count_phase_space(count_vpara, count_vperp) + 1
                  end if
               end if
            end do   !count_vperp
         end do   !count_vpara

      end if
   end do

   do count_a = nint(pitch_angle_min / d_pitch_angle), nint(pitch_angle_max / d_pitch_angle) - 1
      variable_a = dble(count_a) * d_pitch_angle

      do count_E = nint(energy_min / d_energy), nint(energy_max / d_energy) - 1
         variable_E = dble(count_E) * d_energy

         call energy_to_v(variable_E + d_energy/2d0, gamma, variable_v)
         !tau_b [s]: particle trajectory time from one mirror point to another mirror point
         tau_b      = 2d0 * r_eq / sqrt(1d0 - 1d0/gamma**2) * (1.30d0 - 0.56d0 * sin((variable_a + d_pitch_angle / 2d0) *deg2rad))
         norm_tau_b = 2d0 * r_eq / sqrt(1d0 - 1d0/gamma**2) * (1.30d0 - 0.56d0 * sin( 89.5d0 * deg2rad))
         weight_count_alpha_energy(count_a, count_E) = count_alpha_energy(count_a, count_E)* tau_b /norm_tau_b
         !weighting_at_pitch_angle
         particle_num = dble(weight_count_alpha_energy(count_a, count_E)) / (tau_b / delta_t)
         particle_num_for_plot_a_E = dble(count_alpha_energy(count_a, count_E)) / (2d0 * pi * sqrt(1d0 - 1d0 / gamma**2d0) &
            & * sin((variable_a + d_pitch_angle / 2d0) * deg2rad))
         write(11,'(2E15.7, 3I10)') variable_a, variable_E, particle_num_for_plot_a_E, count_alpha_energy(count_a, count_E), &
            & particle_num    !delta_tの意味がわからない
         !if (particle_num > 0) then
         !   print *, variable_a, variable_E, particle_num_for_plot_a_E, count_alpha_energy(count_a, count_E), particle_num
         !end if
      end do   !count_E
      write(11,*)''
   end do   !count_a
   close(11)

   do count_vpara = 0, phase_space_number-1
      do count_vperp = 0, phase_space_number-1
         v_para_floor = dble(count_vpara) / dble(phase_space_number)
         v_perp_floor = dble(count_vperp) / dble(phase_space_number)
         phase_space_for_plot = count_phase_space(count_vpara, count_vperp) / (2d0*pi*(v_perp_floor+5d-1/dble(phase_space_number)))
         write(12,'(2E15.7, 2I10)') v_para_floor, v_perp_floor, phase_space_for_plot, count_phase_space(count_vpara, count_vperp)
      end do   !count_vperp
      write(12,*)''
   end do   !count_vpara
   close(12)

   do count_a = nint(pitch_angle_min / d_pitch_angle), nint(pitch_angle_max / d_pitch_angle) - 1
      variable_a = dble(count_a) * d_pitch_angle
      do count_E = nint(energy_min / d_energy), nint(energy_max / d_energy) - 1
         variable_E = dble(count_E) * d_energy
         energy = variable_E + d_energy / 2d0
         gamma = energy / m_e + 1d0
         tau_b = 2d0 * r_eq / sqrt(1d0 - 1d0 / gamma**2d0) * (1.30d0 - 0.56d0 * sin((variable_a + d_pitch_angle / 2d0) * deg2rad))
         write(13,'(3E15.7)') variable_a, variable_E, tau_b
      end do   !count_E
      write(13,*)''
   end do   !count_a
   close(13)



   !----------------------------------------------------------
   !   decide_z_distribution
   !----------------------------------------------------------

   write(file_initial_condition, '(A44, I3.3, A4)') 'results/initial_conditions/initial_condition', myrank, '.dat'
   print *, file_initial_condition
   open (14, file = file_initial_condition)
   
   particle_num = 0
   tau_b        = 0
   
   do count_a = 1, 89  !loss cone ~ 1.5 degrees in L=9

      count_number = 0
      variable_a = dble(count_a) * d_pitch_angle

      count_E = myrank

      variable_E = dble(count_E) * d_energy
      !print *, 'check point 0', count_a, count_E, weight_count_alpha_energy(count_a, count_E), myrank
      
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
         !print *, "check point 1", count_a, count_E, myrank

         magnetic_flux_density_at_mirror_point = 1d0 / sin(equatorial_pitch_angle)**2d0   ![/B0_eq]
         do count_z = 1, z_number
            z_position_at_mirror_point = z_position(count_z - 1)
            if(magnetic_flux_density(count_z) > magnetic_flux_density_at_mirror_point) exit
         end do   !count_z

         !print *, "check point 2", count_a, count_E, myrank


         particle_num = dble(weight_count_alpha_energy(count_a, count_E)) / (tau_b / delta_t)   !delta_tの意味がわからない

         !if (weight_count_alpha_energy(count_a, count_E) > 0) then
         !   print *, "check point 3", weight_count_alpha_energy(count_a, count_E), tau_b/delta_t, particle_num, myrank
         !end if

         if (particle_num >= 1) then

            time = 0d0
            z_particle = 0d0

            !print *, "check point 4", count_a, count_E, particle_num, myrank
            
            do count_tau = 1, 1000000
               
               variable_time = delta_t * dble(count_tau)

               if ( variable_time <= tau_b ) then

                  !print *, "check point 5", count_a, count_E, particle_num, count_tau, myrank
                  
                  z_particle_1 = z_particle
                  gamma_velocity_1 = gamma_velocity

                  do while(time <= variable_time)
                     call runge_kutta(z_position, z_particle, gamma_velocity(0:2))
                     z_particle_2 = z_particle
                     gamma_velocity_2 = gamma_velocity
                     time = time + 1d0 * d_time
                  end do

                  !print *, "check point 6", count_a, count_E, particle_num, count_tau, myrank

                  if ( z_particle_1 <= z_particle_2 ) then
                     z_small = z_particle_1
                     z_large = z_particle_2
                  else
                     z_small = z_particle_2
                     z_large = z_particle_1
                  end if

                  !print *, "check point 7", count_a, count_E, particle_num, count_tau, myrank

                  call z_to_B(z_small, magnetic_flux_density_small)
                  call z_to_B(z_large, magnetic_flux_density_large)

                  z_particle_1_to_mirror = z_position_at_mirror_point - z_particle_1
                  z_particle_2_to_mirror = z_position_at_mirror_point - z_particle_2

                  !print *, "check point 8: count_tau", count_tau, particle_num, count_number, myrank

                  do count_particle = 1, particle_num
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
                           write(14,'(4E15.7)') energy_output, pitch_angle_output * rad2deg, z_particle_output, &
                              & equatorial_pitch_angle_output * rad2deg
                           count_number = count_number + 1
                        end if
                        
                     end if

                  end do   !count_particle
               end if
            end do   !count_tau
         end if
      end if
            
         
      if (mod(count_a, 1) == 0 .and. count_number > 0) then
         write(*,*) 'pitch_angle', count_a, variable_a, 'energy', myrank, variable_E, count_number, particle_num, &
            & dble(weight_count_alpha_energy(count_a, count_E)), tau_b, delta_t, tau_b/delta_t
      end if

   end do
   
   write(*,*) "end"

   close(14)

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
   i_min = MINLOC(diff) !- ( z_number + 1 ) !array_z_0~2*z_number+1

!    if (i_min(1) > z_number) then
!      write(*,*) '!-------------------------'
!      write(*,*) 'i_min = ', i_min(1)
!      write(*,*) '!-------------------------'
!      i_min(1) = z_number
!    end if

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
end program distribution_calculater

