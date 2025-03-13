import numpy as np
import pyvista as pv

# --- Konstanten ---
I = 1  # Strom in A
mu_0 = 4 * np.pi * 1e-7  # Vakuumpermeabilität

# --- Gitter in z=0-Ebene ---
x = np.linspace(-1, 1, 8)
y = np.linspace(-1, 1, 8)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# --- Abstand vom Draht ---
R = np.sqrt(X**2 + Y**2)
R[R == 0] = 1e-10  # Vermeidung von Division durch 0

# --- Gradient von A_z: rein radial ---
grad_A_mag = -mu_0 * I / (2 * np.pi * R)
grad_Ax = grad_A_mag * X / R
grad_Ay = grad_A_mag * Y / R
grad_Az = np.zeros_like(grad_Ax)

# --- Punkte & Vektoren ---
points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
vectors = np.vstack((grad_Ax.ravel(), grad_Ay.ravel(), grad_Az.ravel())).T

# --- Richtung normalisieren für Richtungsglyphs ---
magnitudes = np.linalg.norm(vectors, axis=1)
directions = vectors / magnitudes[:, np.newaxis]

# --- PyVista: PolyData und Glyphs ---
pdata = pv.PolyData(points)
pdata["gradA"] = directions
pdata["magnitude"] = np.log10(magnitudes)

factor = 0.25
pdata.points -= 0.5 * directions * factor  # Pfeile auf Gitter zentrieren
glyphs = pdata.glyph(orient="gradA", scale=False, factor=factor)

# --- Draht als Zylinder ---
cylinder = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=0.02, height=2.0)

# --- Plot ---
plotter = pv.Plotter(window_size=(1000, 600))
plotter.set_background("white")
plotter.renderer.SetBackgroundAlpha(0)

plotter.add_mesh(glyphs, scalars="magnitude", cmap="cool", show_scalar_bar=False)
plotter.add_mesh(cylinder, color="black")

plotter.view_vector([0, 0, 1], viewup=[0, 1, 0])
plotter.camera.zoom(1.6)

plotter.show_axes()

plotter.show(auto_close=False, interactive=False)
plotter.screenshot("grad_A_z_vec_top.png", transparent_background=True)
plotter.close()
