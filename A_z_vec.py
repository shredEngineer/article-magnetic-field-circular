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

# --- Abstand vom Draht (entlang z) ---
R = np.sqrt(X**2 + Y**2)
R[R == 0] = 1e-10  # Vermeidung von log(0)

# --- Vektorpotential: nur A_z-Komponente ---
c = 2  # skalierung damit's nicht negativ wird
Az = mu_0 * I / (2 * np.pi) * np.log(1 / R * c)
Ax = np.zeros_like(Az)
Ay = np.zeros_like(Az)

# --- Punkte & Vektoren ---
points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
vectors = np.vstack((Ax.ravel(), Ay.ravel(), Az.ravel())).T

# --- Richtung normalisieren ---
magnitudes = np.linalg.norm(vectors, axis=1)
directions = vectors / magnitudes[:, np.newaxis]

# --- PolyData ---
pdata = pv.PolyData(points)
pdata["A"] = directions
pdata["magnitude"] = np.log10(magnitudes)

# --- Glyphs (farbige Pfeile in z-Richtung) ---
factor = 0.25
pdata.points -= 0.5 * directions * factor  # Auf Gitterpunkten zentrieren
glyphs = pdata.glyph(orient="A", scale=False, factor=factor)

# --- Plotter ---
plotter = pv.Plotter(window_size=(1000, 600))
plotter.set_background("white")
plotter.renderer.SetBackgroundAlpha(0)

plotter.add_mesh(glyphs, scalars="magnitude", cmap="cool", show_scalar_bar=False)

# --- Draht als Zylinder ---
cylinder = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=0.02, height=2.0)
plotter.add_mesh(cylinder, color="black")

# --- Kamera & Screenshot ---
plotter.view_vector([1.3, 1.3, 0.9], viewup=[0, 0, 1])
plotter.camera.zoom(1.8)
plotter.show_axes()

plotter.show(auto_close=False, interactive=False)
plotter.screenshot("A_z_vec.png", transparent_background=True)
plotter.close()
