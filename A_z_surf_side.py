import numpy as np
import pyvista as pv

# --- Konstanten ---
I = 1
epsilon_0 = 8.854e-12
c = 3e8

def Az(x, y):
	r = np.sqrt(x**2 + y**2)
	r[r == 0] = 1e-10
	return I / (2 * np.pi * epsilon_0 * c**2) * np.log(1/r)

# --- Gitter ---
x = np.linspace(-1, 1, 1000)
y = np.linspace(-1, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = Az(X, Y) * 5e5

# --- StructuredGrid mit erhobener Fläche (z = A_z) ---
grid = pv.StructuredGrid()
grid.points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]  # <-- z = A_z!
grid.dimensions = X.shape[0], X.shape[1], 1
grid["Az"] = Z.ravel()  # <-- Skalarfeld

# --- Isolinien auf der Fläche berechnen ---
n_contours = 25
contours = grid.contour(isosurfaces=n_contours, scalars="Az")  # ← jetzt passt's!

# --- Plotter ---
plotter = pv.Plotter(window_size=(1000, 600))
plotter.set_background("white")
plotter.renderer.SetBackgroundAlpha(0)

# --- Erhobene Fläche mit Farbverlauf ---
plotter.add_mesh(grid, scalars="Az", cmap="cool", show_scalar_bar=False, opacity=1.0)

# --- Isolinien auf der Fläche ---
plotter.add_mesh(contours, color="black", line_width=1)

# --- Ansicht ---
plotter.show_axes()
plotter.view_vector([2, 2, 0.9], viewup=[0, 0, 1])
plotter.camera.zoom(1.4)

# --- Screenshot ---
plotter.show(auto_close=False, interactive=False)
plotter.screenshot("A_z_surf_side.png", transparent_background=True)
plotter.close()
