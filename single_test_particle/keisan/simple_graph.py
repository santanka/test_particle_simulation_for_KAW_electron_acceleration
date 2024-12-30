import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 30

# Define the function F
def F(Theta, S):
    return Theta + np.sqrt(Theta**2 + (1 + np.pi / 2 * S))

# Generate x values
Theta = np.linspace(-10, 0, 4000)

S_number = np.linspace(0, 2, 5)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, xlabel=r'$\Theta_{\mathrm{\pi}}$', ylabel=r'$F(\Theta_{\mathrm{\pi}}, S)$')

for S_value in S_number:
    ax.plot(Theta, F(Theta, S_value), label=r'$S = %.1f$' % S_value)

ax.set_xlim(-3, 0)
ax.set_ylim(0, 2.5)

ax.legend()
ax.minorticks_on()
ax.grid(which='both', alpha=0.3)

plt.show()