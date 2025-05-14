import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===== Global matplotlib configuration =====
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 25

# ===== Physical constants =====
speed_of_light      = 2.99792458e8           # [m/s]
elementary_charge   = 1.6021766208e-19       # [C]
electric_constant   = 8.8541878128e-12       # [F/m]
magnetic_constant   = 1.25663706212e-6       # [N/A^2]
proton_mass         = 1.672621898e-27        # [kg]
electron_mass       = 9.10938356e-31         # [kg]

# ===== Jupiter system parameters =====
L_number            = 9.65                   # Europa mean L‑shell
Radius_Jupiter      = 7.1492e7               # [m] equatorial radius
Time_rotation       = 9.9258 * 3600.0        # [s]
Omega_Jupiter       = 2.0*np.pi / Time_rotation

# ===== Plasma parameters =====
ion_temperature     = 88.0                   # [eV]
ion_mass_number     = 18.0                   # dominant ion assumed H2O+
n_e_eq              = 158e6                  # [m⁻³] equatorial density

# ---------------------------------------------------------
# Centrifugal scale height (Thomas+2004‑type)
def centrifugal_scale_height():
    return np.sqrt(2.0*ion_temperature*elementary_charge /(3.0*ion_mass_number*proton_mass*Omega_Jupiter**2))
H_c = centrifugal_scale_height()

# ===== Magnetic field: centered dipole =====
g01, g11, h11 = 410993.4e-9, -71305.9e-9, 20958.4e-9  # [T] Gauss coefficients

def dipole_moment():
    return 4*np.pi*Radius_Jupiter**3 / magnetic_constant * np.sqrt(g01**2+g11**2+h11**2)
M_dip = dipole_moment()

def B_dipole(mlat):
    """Magnetic flux density along a dipole field line at MLAT (radians)."""
    return magnetic_constant*M_dip/(4*np.pi*(Radius_Jupiter*L_number)**3) * np.sqrt(1+3*np.sin(mlat)**2)/np.cos(mlat)**6

# Gradient scale length of |B|

def L_B(mlat):
    return Radius_Jupiter*L_number*np.cos(mlat)**2*(1+3*np.sin(mlat)**2)**1.5 /(3*np.sin(mlat)*(3+5*np.sin(mlat)**2))

# ===== Rotation‑defined centrifugal equator =====

def mlat_centrifugal_equator(alpha_rot):
    tan_l0 = 2/3*np.tan(alpha_rot)/(1+np.sqrt(1+8/9*np.tan(alpha_rot)**2))
    return np.arctan(tan_l0)

# Arc length from magnetic equator to MLAT (dipole)

def s_distance(mlat):
    term = (np.arcsin(np.sqrt(3)*np.sin(mlat))/(2*np.sqrt(3)) +
            np.sqrt(5-3*np.cos(2*mlat))*np.sin(mlat)/(2*np.sqrt(2)))
    return term*Radius_Jupiter*L_number

# ===== Density model base class & implementations =====
class DensityModel:
    def __init__(self,label):
        self.label = label
    def ne(self,mlat,alpha):
        raise NotImplementedError
    def L_ne(self,mlat,alpha):
        raise NotImplementedError

class ConstNe(DensityModel):
    """MLAT-independent electron density."""
    def __init__(self,n_eq):
        super().__init__('Const')
        self.n_eq = n_eq
    def ne(self,mlat,alpha):
        return self.n_eq
    def L_ne(self,mlat,alpha):
        return np.inf * np.ones_like(mlat)

class BProportionalNe(DensityModel):
    """Density proportional to |B|."""
    def __init__(self,n_eq):
        super().__init__('∝B')
        self.n_eq = n_eq
        self.B_eq = B_dipole(0.0)
    def ne(self,mlat,alpha):
        return self.n_eq * B_dipole(mlat)/self.B_eq
    def L_ne(self,mlat,alpha):
        return L_B(mlat)

class CentrifugalNe(DensityModel):
    """Gaussian profile with centrifugal scale height."""
    def __init__(self,n_eq,H_c):
        super().__init__('Centrifugal')
        self.n_eq = n_eq
        self.H_c  = H_c
    def _height(self,mlat,alpha):
        lam0 = mlat_centrifugal_equator(alpha)
        return s_distance(mlat) - s_distance(lam0)
    def ne(self,mlat,alpha):
        h = self._height(mlat,alpha)
        return self.n_eq * np.exp(-h**2/self.H_c**2)
    def L_ne(self,mlat,alpha):
        h = self._height(mlat,alpha)
        return -self.H_c**2/(2*h)

# ===== Wave & particle helper functions =====

def omega_ce(mlat):
    return elementary_charge*B_dipole(mlat)/electron_mass

def omega_pe(model,mlat,alpha):
    return elementary_charge/np.sqrt(electron_mass*electric_constant)*np.sqrt(model.ne(mlat,alpha))

def _k(model,mlat,alpha,omega):
    return 2*np.pi*speed_of_light/np.sqrt(omega**2 + omega*omega_pe(model,mlat,alpha)**2 /(omega_ce(mlat)-omega))

def chi(model,mlat,alpha,omega):
    return np.sqrt(1 - (omega/(speed_of_light*_k(model,mlat,alpha,omega)))**2)

def xi(model,mlat,alpha,omega):
    return np.sqrt(omega*(omega_ce(mlat)-omega)/omega_pe(model,mlat,alpha)**2)

def v_phase(model,mlat,alpha,omega):
    return speed_of_light*chi(model,mlat,alpha,omega)*xi(model,mlat,alpha,omega)

def v_group(model,mlat,alpha,omega):
    ch = chi(model,mlat,alpha,omega)
    xi_ = xi(model,mlat,alpha,omega)
    return speed_of_light*xi_/ch /(xi_**2 + omega_ce(mlat)/(2*(omega_ce(mlat)-omega)))

def v_res(model,mlat,alpha,omega,v_perp):
    cycl = omega_ce(mlat)/omega
    vp_c = v_phase(model,mlat,alpha,omega)/speed_of_light
    vp_ratio = v_perp/speed_of_light
    num = 1 - np.sqrt(1 - (1+cycl**2*vp_c**2)*(1 - cycl**2*(1-vp_ratio**2)))
    return v_phase(model,mlat,alpha,omega)*num /(1 + cycl**2*vp_c**2)

def gamma_lor(v_para,v_perp):
    return 1/np.sqrt(1 - (v_para**2+v_perp**2)/speed_of_light**2)

def s0(model,mlat,alpha,omega,v_perp):
    return chi(model,mlat,alpha,omega)*v_perp /(xi(model,mlat,alpha,omega)*speed_of_light)

def s1(model,mlat,alpha,omega,v_perp):
    v_r = v_res(model,mlat,alpha,omega,v_perp)
    g   = gamma_lor(v_r,v_perp)
    return g*(1 - v_r/v_group(model,mlat,alpha,omega))**2

def s2_mod(model, mlat, alpha, omega, v_perp):
    v_r = v_res(model, mlat, alpha, omega, v_perp)
    g   = gamma_lor(v_r, v_perp)
    ch  = chi(model, mlat, alpha, omega)
    xi_ = xi(model, mlat, alpha, omega)

    inv_Lb = 1.0 / L_B(mlat)                 # 配列
    Ln     = model.L_ne(mlat, alpha)         # 配列 or スカラー
    inv_Ln = np.where(np.isfinite(Ln), 1.0 / Ln, 0.0)

    grad_term = inv_Lb + inv_Ln
    return g * omega / (2 * xi_ * ch) * grad_term


def wave_sweep_rate(model,inhomog,alpha,mlat,omega,Bwave,v_perp):
    """Time rate of change of wave frequency (sweep rate)."""
    s_0 = s0(model,mlat,alpha,omega,v_perp)
    s_1 = s1(model,mlat,alpha,omega,v_perp)
    s_2 = s2_mod(model,mlat,alpha,omega,v_perp)
    return -s_0/s_1 * inhomog * omega * elementary_charge * Bwave / electron_mass - s_2/s_1

# ===== Main arrays & parameters =====
mlat_deg_array       = np.linspace(0.2,15.0,10000)
mlat_rad_array       = np.deg2rad(mlat_deg_array)
alpha_rot_deg_list   = [-9.3,0.0,9.3]
alpha_rot_rad_list   = np.deg2rad(alpha_rot_deg_list)

omega_ce_eq          = omega_ce(0.0)
wave_frequency_array = np.array([0.25,0.75])*omega_ce_eq

v_perp_typical       = 0.706*speed_of_light
Bwave_typical        = 4.7e-4 * B_dipole(0.0)
inhomog_factor       = -0.4

# Density models in order: Const, B‑prop, Centrifugal
density_models = [
    ConstNe(n_e_eq),
    BProportionalNe(n_e_eq),
    CentrifugalNe(n_e_eq,H_c)
]

# ===== Plot =====
n_rows = len(alpha_rot_rad_list)
n_cols = len(density_models)
fig, axes = plt.subplots(n_rows,n_cols,figsize=(5*n_cols,5*n_rows),dpi=140,sharex='col',sharey='row')

for j,model in enumerate(density_models):
    axes[0,j].set_title(model.label)

for i,alpha in enumerate(alpha_rot_rad_list):
    for j,model in enumerate(density_models):
        ax = axes[i,j]
        for omega in wave_frequency_array:
            rate = wave_sweep_rate(model,inhomog_factor,alpha,mlat_rad_array,omega,Bwave_typical,v_perp_typical)
            ax.plot(mlat_deg_array,rate/omega_ce_eq,label=fr'$\omega/\Omega_e={omega/omega_ce_eq:.2f}$',linewidth=2,alpha=0.7)
        if j==0:
            ax.set_ylabel(fr'$\alpha_{{rot}}={np.rad2deg(alpha):.1f}^\circ$' + '\n' + fr'$1/\Omega_e\,\partial\omega/\partial t$')
        if i==n_rows-1:
            ax.set_xlabel('MLAT [deg]')
        ax.set_xlim(0,15)
        ax.grid(alpha=0.3)
        if i==0 and j==n_cols-1:
            ax.legend(fontsize=12)

fig.tight_layout()
fig.savefig('wave_sweep_ratio_comparison.png')
plt.close(fig)
