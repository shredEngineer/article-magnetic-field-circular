import numpy as np
import pyvista as pv

# --- Konstanten ---
I = 1
mu_0 = 4 * np.pi * 1e-7

# --- Gitter ---
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
R[R == 0] = 1e-10  # Singularität vermeiden

# --- Magnetfeld-Komponente B_x ---
Bx = -mu_0 * I / (2 * np.pi * R) * Y / R  # Bx(x, y)
Bx_scaled = Bx * 2e5  # Für bessere Sichtbarkeit

# --- StructuredGrid mit "erhobener" Fläche in x-Richtung ---
grid = pv.StructuredGrid()
grid.points = np.c_[Bx_scaled.ravel(), Y.ravel(), X.ravel()]  # ACHTUNG: Bx als x-Koordinate
grid.dimensions = X.shape[0], X.shape[1], 1
grid["Bx"] = Bx_scaled.ravel()  # Skalarfeld bleibt für Farbe

# --- Isolinien auf der Fläche berechnen ---
n_contours = 25
contours = grid.contour(isosurfaces=n_contours, scalars="Bx")

# --- Plotter ---
plotter = pv.Plotter(window_size=(1000, 600))
plotter.set_background("white")
plotter.renderer.SetBackgroundAlpha(0)

# --- Fläche mit Farbverlauf in x-Richtung ---
plotter.add_mesh(grid, scalars="Bx", cmap="cool", show_scalar_bar=False, opacity=1.0)

# --- Isolinien ---
plotter.add_mesh(contours, color="black", line_width=1)

# --- Ansicht ---
plotter.show_axes()
plotter.view_vector([1, 2, 1], viewup=[0, 0, 1])  # leicht schräge Ansicht
plotter.camera.zoom(1.3)

# --- Screenshot ---
plotter.show(auto_close=False, interactive=False)
plotter.screenshot("B_x_surf.png", transparent_background=True)
plotter.close()
