module constant_parameter
    implicit none
  
    !-------------------------------------
    ! mathematical and physical constants
    !-------------------------------------
    DOUBLE PRECISION, PARAMETER :: pi = 4d0*DATAN(1d0)
    DOUBLE PRECISION, PARAMETER :: c  = 299792458d0 * 1d2 ![m]→[cm]
    DOUBLE PRECISION, PARAMETER :: q  = 1.6021766208d-19 / 1d1 * c ![C]→[statC]
    DOUBLE PRECISION, PARAMETER :: m  = 9.10938356d-31 * 1d3 ![kg]→[g]
  
    DOUBLE PRECISION, PARAMETER :: rad2deg = 180d0 / pi
    DOUBLE PRECISION, PARAMETER :: deg2rad = pi / 180d0
    
  end module constant_parameter
  