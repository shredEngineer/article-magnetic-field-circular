import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import EngFormatter

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

I = 1
epsilon_0 = 8.854e-12
c = 3e8

def Az(x, y):
    r = np.sqrt(x**2 + y**2)
    r[r == 0] = 1e-10  # Vermeidung von log(0)
    return I / (2 * np.pi * epsilon_0 * c**2) * np.log(1/r)

x = np.linspace(-1, 1, 500)
y = np.linspace(-1, 1, 500)
X, Y = np.meshgrid(x, y)
Z_xy = Az(X, Y)

fig = plt.figure(figsize=(8, 5), dpi=300, constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z_xy, cmap='cool')

ax.set_xlabel(r"$x$ / m", fontsize=14, labelpad=10)
ax.set_ylabel(r"$y$ / m", fontsize=14, labelpad=10)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r"$A_z$ / Tm", fontsize=14, labelpad=22, rotation=0)
ax.zaxis.set_major_formatter(EngFormatter(unit=""))

ax.view_init(elev=30, azim=45)

# ax.set_title(r"$I$ = 1 A", fontsize=14)

plt.savefig("A_alt.png", dpi=300)
plt.show()
