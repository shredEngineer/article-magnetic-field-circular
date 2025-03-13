import numpy as np
import pyvista as pv

from matplotlib import cm
from matplotlib.colors import ListedColormap

# --- Eigene dunklere Version von 'cool' ---
base_cmap = cm.get_cmap("cool", 256)
dark_cmap_colors = base_cmap(np.linspace(0, 1, 256))
dark_cmap_colors[:, :3] *= 0.6  # <-- RGB abdunkeln (0.0 - 1.0)
dark_cmap = ListedColormap(dark_cmap_colors)

# --- Parameter ---
z_offset = 0.05  # ← Höhe der Pfeile über der Fläche (0.0 = direkt auf Fläche)

# --- Konstanten ---
I = 1
epsilon_0 = 8.854e-12
c = 3e8
mu_0 = 4 * np.pi * 1e-7

# --- Az-Funktion ---
def Az(x, y):
	r = np.sqrt(x**2 + y**2)
	r[r == 0] = 1e-10
	return I / (2 * np.pi * epsilon_0 * c**2) * np.log(1 / r)

# === Fläche (flach!) ===
x_fine = np.linspace(-1, 1, 100)
y_fine = np.linspace(-1, 1, 100)
Xf, Yf = np.meshgrid(x_fine, y_fine)
Zf = np.zeros_like(Xf)
Az_vals = Az(Xf, Yf)

grid = pv.StructuredGrid()
grid.points = np.c_[Xf.ravel(), Yf.ravel(), Zf.ravel()]
grid.dimensions = Xf.shape[0], Xf.shape[1], 1
grid["Az"] = Az_vals.ravel()

# --- Isolinien ---
contours = grid.contour(isosurfaces=25, scalars="Az")

# === Vektoren (in leicht erhobener Ebene, steuerbar mit z_offset) ===
x_coarse = np.linspace(-1, 1, 8)
y_coarse = np.linspace(-1, 1, 8)
Xc, Yc = np.meshgrid(x_coarse, y_coarse)
Zc = np.full_like(Xc, z_offset)

R = np.sqrt(Xc**2 + Yc**2)
R[R == 0] = 1e-10
grad_A_mag = -mu_0 * I / (2 * np.pi * R)
grad_Ax = grad_A_mag * Xc / R
grad_Ay = grad_A_mag * Yc / R
grad_Az = np.zeros_like(grad_Ax)

# --- Gradient-Pfeile (nabla A_z) ---
points = np.c_[Xc.ravel(), Yc.ravel(), Zc.ravel()]
vectors = np.c_[grad_Ax.ravel(), grad_Ay.ravel(), grad_Az.ravel()]
magnitudes = np.linalg.norm(vectors, axis=1)
directions = vectors / magnitudes[:, np.newaxis]

pdata = pv.PolyData(points)
pdata["gradA"] = directions
pdata["magnitude"] = np.log10(magnitudes)

factor = 0.25
pdata.points -= 0.5 * directions * factor  # zentrieren
glyphs = pdata.glyph(orient="gradA", scale=False, factor=factor)

# === Orthogonales Magnetfeld B (azimutal) ===
Bx = -grad_Ay
By = grad_Ax
Bz = np.zeros_like(Bx)

B_vectors = np.c_[Bx.ravel(), By.ravel(), Bz.ravel()]
B_directions = B_vectors / np.linalg.norm(B_vectors, axis=1)[:, np.newaxis]

# ← Punkte mit gleichem z_offset wie oben
B_points = np.c_[Xc.ravel(), Yc.ravel(), Zc.ravel()]
B_pdata = pv.PolyData(B_points)
B_pdata["B"] = B_directions
B_pdata.points -= 0.5 * B_directions * factor
B_glyphs = B_pdata.glyph(orient="B", scale=False, factor=factor)

# --- Draht als Zylinder ---
cylinder = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=0.02, height=2.0)

# === Plotten ===
plotter = pv.Plotter(window_size=(1000, 600))
plotter.set_background("white")
plotter.renderer.SetBackgroundAlpha(0)

plotter.add_mesh(grid, scalars="Az", cmap="cool", show_scalar_bar=False, opacity=1.0)
plotter.add_mesh(contours, color="black", line_width=1)
plotter.add_mesh(glyphs, scalars="magnitude", cmap=dark_cmap, show_scalar_bar=False)
plotter.add_mesh(B_glyphs, color="orange")
plotter.add_mesh(cylinder, color="black")

plotter.view_vector([0, 0, 1], viewup=[0, 1, 0])
plotter.camera.zoom(1.6)
plotter.show_axes()

# === Anzeige + Screenshot ===
plotter.show(auto_close=False, interactive=False)
plotter.screenshot("A_z_surf_B_orth_to_grad_A_z_vec_top.png", transparent_background=True)
plotter.close()
