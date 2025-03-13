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

R = np.sqrt(X**2 + Y**2)
R[R == 0] = 1e-10

# --- Magnetfeld (nur xy) ---
Bx = -mu_0 * I / (2 * np.pi * R) * Y / R
By = np.zeros_like(Bx)
Bz = np.zeros_like(Bx)

# --- Punkte & Vektoren ---
points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
vectors = np.vstack((Bx.ravel(), By.ravel(), Bz.ravel())).T

# Richtung normalisieren
magnitudes = np.linalg.norm(vectors, axis=1)
directions = vectors / magnitudes[:, np.newaxis]

# --- PolyData ---
pdata = pv.PolyData(points)
pdata["B"] = directions
pdata["magnitude"] = np.log10(magnitudes)

# --- Glyphs (farbige Pfeile) ---
factor = 0.25
pdata.points -= 0.5 * directions * factor  # Auf Gitterpunkten zentrieren
glyphs = pdata.glyph(orient="B", scale=False, factor=factor)

# --- Plotter ---
plotter = pv.Plotter(window_size=(1000, 600))
plotter.set_background("white")  # Muss gesetzt werden (damit Alpha-Wert wirkt)
plotter.renderer.SetBackgroundAlpha(0)  # ← wichtig für Transparenz

plotter.add_mesh(glyphs, scalars="magnitude", cmap="cool", show_scalar_bar=False)

# Draht als Zylinder
cylinder = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=0.02, height=2.0)
plotter.add_mesh(cylinder, color="black")

# Ansicht & Screenshot
plotter.view_vector([1.3, 1.3, 0.9], viewup=[0, 0, 1])
plotter.camera.zoom(1.8)
plotter.show_axes()

# Screenshot mit Transparenz
plotter.show(auto_close=False, interactive=False)
plotter.screenshot("B_x.png", transparent_background=True)
plotter.close()
