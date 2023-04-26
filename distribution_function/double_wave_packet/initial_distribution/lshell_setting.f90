module lshell_setting
  use constant_parameter
  implicit none

  DOUBLE PRECISION, PARAMETER :: mu_0    = 4d0 * pi * 1d-7
  DOUBLE PRECISION, PARAMETER :: moment  = 7.75d22 ! the Earth's dipole moment model
  DOUBLE PRECISION, PARAMETER :: R_E     = 6371d3  ! radius of the Earth

  DOUBLE PRECISION, PARAMETER :: L         = 9d0 ! L-shell
  DOUBLE PRECISION, PARAMETER :: B0_eq     = (1d-7 * moment) / (L * R_E)**3
  DOUBLE PRECISION, PARAMETER :: Omega0_eq = q * B0_eq / m
  DOUBLE PRECISION, PARAMETER :: fce_eq    = Omega0_eq / (2d0 * pi)

  DOUBLE PRECISION, PARAMETER :: z_unit = c / Omega0_eq
  DOUBLE PRECISION, PARAMETER :: t_unit = 1d0 / Omega0_eq
  DOUBLE PRECISION, PARAMETER :: r_eq   = R_E * L / z_unit

end module lshell_setting