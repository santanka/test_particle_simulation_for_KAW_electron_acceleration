module constants_in_the_simulations
  use constant_parameter
  use lshell_setting

  implicit none

  !-------------------------------------
  ! simulation variation
  !-------------------------------------
  LOGICAL, PARAMETER :: frequency_sweep  = .false. ! .true.
  LOGICAL, PARAMETER :: wave_existance   = .true. ! .true.
  LOGICAL, PARAMETER :: categorization   = .false. ! .false.

  ! wave component: on = 1d0, off = 0d0
  double precision, parameter :: switch_EE_wave_para = 0d0
  double precision, parameter :: switch_EE_wave_perp_perp = 0d0
  double precision, parameter :: switch_EE_wave_perp_phi = 0d0
  double precision, parameter :: switch_BB_wave_para = 0d0
  double precision, parameter :: switch_BB_wave_perp = 0d0

  double precision, parameter :: switch_wave_packet = 0E0
  
  !-------------------------------------
  ! initial setting of simulation system
  !-------------------------------------
  INTEGER, PARAMETER          :: n_time = 200000  ! !80000 (10.9932 [s])
  INTEGER, PARAMETER          :: n_z = 3500 ! (n + 1) for dB_dz
  DOUBLE PRECISION, PARAMETER :: d_t = 1.0d0 / Omega0_eq
  DOUBLE PRECISION, PARAMETER :: d_z = 0.5d0 / Omega0_eq * c_normal
  DOUBLE PRECISION, PARAMETER :: L_t = DBLE(n_time) * d_t
  DOUBLE PRECISION, PARAMETER :: L_z = DBLE(n_z) * d_z ! (n - 1) for dB_dz

  !-------------------------------------
  ! initial setting of wave
  !-------------------------------------
  DOUBLE PRECISION, PARAMETER :: electrostatic_potential_0 = 600d0 * 1d8 / c / V_unit ![V]->[statV]->[]
  double precision, parameter :: gradient_parameter = 2d0
  double precision, parameter :: mlat_deg_wave_threshold = 5d0  ![deg]
  double precision, parameter :: initial_wave_phase = 0d0 * deg2rad ![rad]

  !-------------------------------------
  ! initial setting of particle
  !-------------------------------------
  INTEGER            :: N_particle, N_select_particle
  INTEGER, PARAMETER :: n_thread = 2
  
  end module constants_in_the_simulations
  