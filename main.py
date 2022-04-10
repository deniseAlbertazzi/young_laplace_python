from matplotlib import pyplot as plt
import numpy as np

from YL import Surface, YL
from YL.utils import clean

# Clad width in millimeters
cladW = 1.5

# Clad area in millimeters^2
cladArea = 1.1

# Clad overlap in percentage of cladW
cladOverlap = 0.10

# Clad tracks and layers
# [n-of-tracks, n-of-tracks, n-of-tracks]
cladTracks = np.array([1])

# Clad offset if the offset for subsequent layers in percentage of cladW
cladOffset = 0.1
# must be with a decimal dot

# Material density in g/cm^3
density = 7.81

# Material surface tension in N/m
surface_tension = 1.77

# D is the width of subsequent clads (ie. after the first clad every layer)
D = cladW * (1 - cladOverlap)

print(f"W       = {cladW} mm")
print(f"Area    = {cladArea} mm^2")
print(f"Overlap = {cladOverlap}")
print(f"Offset  = {cladOffset}")
print(f"D       = {D} mm")

# Clad shape can be ellipse, parabola, sine, arc, or YoungLaplace
# shape = Arc(cladW, cladArea)
# shape = Ellipse(cladW, cladArea)
# shape = Sine(cladW, cladArea)
# shape = Parabola(cladW, cladArea)
shape = YL(cladW, cladArea, density, surface_tension)

# Surface holds the data about the current surface and is updated on every
# clad
surface = Surface(np.arange(-5, 30, 0.05))  # -5:0.05:30

# Configure image
fig = plt.figure()
plt.axes().set_aspect("equal", "datalim")
plt.title("Clad profiles using different shapes")
plt.xlabel("X (mm)")
plt.ylabel("Z (mm)")
plt.xlim((-2, 2))
plt.grid()

for j in range(len(cladTracks)):
    A = j * cladW * cladOffset
    for i in range(cladTracks[j]):
        print(f"layer={j}, track={i}")

        # Generate a function for the shape and area of that shape
        X, Z = shape.clad(surface, A)
        X, Z = clean(X, Z)
        plt.plot(X, Z, "-")

        # Add the clad to the surface
        surface.add(X, Z)
        A = A + cladW * (1 - cladOverlap)

        # Optionally rasterize after every layer, to improve performance and
        # apply smoothing. Rasterizing collapses all previous functions into
        # one function, defined only on a range of values for the x-axis.
        # In between those values it is interpolated using a trapezoid.
        # surface.Smooth(0.01);

        # plot(surface.X, surface.Z, 'linewidth', 1)

plt.show()
