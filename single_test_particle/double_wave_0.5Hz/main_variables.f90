module variables
    use lshell_setting
    use constants_in_the_simulations
  
    implicit none
  
    !-------------------------------------
    ! variables
    !-------------------------------------
  
    INTEGER          :: i_time, i_z, i_particle, i_phase, i_alpha, i_grid, i_thr
    INTEGER          :: i, j, ios, a, E, i_v_para, i_v_perp
    INTEGER          :: N_file, N_file_particle
    INTEGER, PARAMETER :: N_thr = 20
    DOUBLE PRECISION :: time
    DOUBLE PRECISION :: z_position(-n_z : n_z)
    DOUBLE PRECISION :: BB(-n_z : n_z)
    DOUBLE PRECISION :: beta_ion(-n_z : n_z)
    DOUBLE PRECISION :: number_density(-n_z : n_z)
    DOUBLE PRECISION :: alfven_velocity(-n_z : n_z)
    DOUBLE PRECISION :: ion_Larmor_radius(-n_z : n_z)
    DOUBLE PRECISION :: wave_phase_1(-n_z : n_z), wave_phase_2(-n_z : n_z)
    DOUBLE PRECISION :: rnd
    CHARACTER(64)    :: file_output, file_wave, file_data
    CHARACTER(64)    :: file_energy, file_alpha, file_distribution, file_phase_space
    CHARACTER(64)    :: file_equator, file_particle, file_check
    CHARACTER(20)    :: string
    CHARACTER(64) :: command
    
    !-------------------------------------
    ! for wave
    !-------------------------------------
  
    DOUBLE PRECISION :: wave_frequency(-n_z : n_z)
    DOUBLE PRECISION :: wave_number_perp(-n_z : n_z)
    DOUBLE PRECISION :: wave_number_para_1(-n_z : n_z), wave_number_para_2(-n_z : n_z)
    DOUBLE PRECISION :: electrostatic_potential_1(-n_z : n_z), EE_wave_para_1(-n_z : n_z), EE_wave_perp_perp_1(-n_z : n_z)
    DOUBLE PRECISION :: EE_wave_perp_phi_1(-n_z : n_z), BB_wave_para_1(-n_z : n_z), BB_wave_perp_1(-n_z : n_z)
    DOUBLE PRECISION :: electrostatic_potential_2(-n_z : n_z), EE_wave_para_2(-n_z : n_z), EE_wave_perp_perp_2(-n_z : n_z)
    DOUBLE PRECISION :: EE_wave_perp_phi_2(-n_z : n_z), BB_wave_para_2(-n_z : n_z), BB_wave_perp_2(-n_z : n_z)
    DOUBLE PRECISION :: V_g(-n_z : n_z), V_g_0
    DOUBLE PRECISION :: z_front, B_front, V_g_front
    DOUBLE PRECISION :: z_edge,  B_edge,  V_g_edge
    DOUBLE PRECISION :: wave_exist_parameter
  
    !-------------------------------------
    ! for particle
    !-------------------------------------
  
    INTEGER          :: clock  !, N_p_real, i_rdm
    DOUBLE PRECISION, allocatable :: alpha0(:), gamma0(:), energy0(:), alpha_eq(:)
    DOUBLE PRECISION :: alpha, gamma, energy, B0_p
    INTEGER          :: alpha_loop, energy_loop, phi_loop
    DOUBLE PRECISION :: v_particle, zeta
    DOUBLE PRECISION :: v_particle_para, v_particle_perp
    DOUBLE PRECISION :: v_0, v_1
    DOUBLE PRECISION, allocatable :: z_particle(:), u_particle(:,:), u_particle_eq(:,:), v_eq(:,:)
    DOUBLE PRECISION :: z_particle_sim, u_particle_sim(0:2)
    double precision, dimension(2) :: wave_phase_update, wave_phase_update2, wave_phase_sim
    DOUBLE PRECISION,allocatable :: equator_time(:)
    INTEGER,allocatable          :: equator_flag(:), wave_flag(:), edge_flag(:)
    INTEGER :: equator_flag_sim, wave_flag_sim, edge_flag_sim
  
    !------------------------------
    ! decide z
    !------------------------------
    INTEGER          :: n_z_mirror
    DOUBLE PRECISION :: B_mirror, z_mirror, norm, cumulative(0:n_z)
  
    
    !-------------------------------------
    ! categorization
    !-------------------------------------
  
    DOUBLE PRECISION, allocatable :: sign_theta0(:), sign_theta1(:)
    DOUBLE PRECISION :: BB_particle, alpha_particle_eq, energy_particle
    DOUBLE PRECISION :: freq_p, ampl_p, k_p, V_g_p
    DOUBLE PRECISION :: gamma_p, V_R_p, theta_p, Cw_p, w_tr_p, dB_dz_p, dk_dB_p, S_p
    INTEGER, allocatable :: cross_theta_0(:)
    INTEGER, allocatable :: Cw_flag(:), S_flag(:)
    INTEGER :: Cw_flag_sim, S_flag_sim, cross_theta_0_sim
    DOUBLE PRECISION :: sign_theta0_sim, sign_theta1_sim
    
  end module variables
  