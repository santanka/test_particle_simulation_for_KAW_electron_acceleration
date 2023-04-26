from scipy import integrate
import numpy as np

# 定数の設定
speed_of_light = 299792458E0    #[m s-1]
elementary_charge = 1.6021766208E-19    #[A s]

lshell_number   = 9E0
planet_radius = 6371E3

r_eq            = planet_radius * lshell_number #[m]
dipole_moment   = 7.75E22 #[Am]
B0_eq           = (1E-7 * dipole_moment) / r_eq**3E0 * 1E4  #[G]

mass_ion        = 1.672621898E-24   #[g]
number_density_ion = 1E0    #[cm-3]
temperature_ion = 1E3   #[eV]
temperature_electron = 1E2  #[eV]
pressure_ion        = number_density_ion * temperature_ion * elementary_charge * 1E7    #cgs
pressure_electron   = number_density_ion * temperature_electron * elementary_charge * 1E7   #cgs

a = 2E0 / (1E0 + pressure_electron/pressure_ion)
b = 8E0 * np.pi * pressure_ion / B0_eq**2E0

# 積分範囲の設定
lambda_value = 1.0
lower_limit = 0
upper_limit = 69.17 / 180E0 * np.pi

# 積分する関数の定義
def integrand(lambda_prime):
    return np.sqrt(a + b * np.cos(lambda_prime)**12 / (1 + 3*np.sin(lambda_prime)**2)) * np.cos(lambda_prime)**7

# 積分の実行
result, error = integrate.quad(integrand, lower_limit, upper_limit)

result = result * r_eq*1E2 * np.sqrt(np.pi*mass_ion*number_density_ion) / B0_eq

# 積分結果の出力
print(a, b)
print("Result: ", result)
#print("time: ", 8E0+result/np.pi)