import numpy as np
from scipy import integrate

from .surface import Surface


# odefcn is called interatively to calculate df/ds for each new step
def odefcn(t, f, H, C):
    dfds = np.zeros(4)
    dfds[0] = np.cos(f[2])
    dfds[1] = np.sin(f[2])
    if t == 0:
        dfds[2] = H
    else:
        dfds[2] = 2 * H + C * f[1] - np.sin(f[2]) / f[0]
    dfds[3] = f[0] * np.sin(f[2])

    return dfds


# odestop determines when to stop evaluating df/ds, which is when the
# tangential angle reaches 180 degree (ie. a full/round droplet)
def odestop(t, f, H, C):
    # Cut-off at 180 degrees
    value = np.pi - f[2]

    return value


odestop.terminal = True
odestop.direction = 0


def young_laplace(density, surface_tension, H):
    # density in g/cm^3
    # surface_tension in N/m
    # H as the curvature in 1/m
    # X, Z are in mm
    # T is the tangential angle
    # A in mm^2 and is the area of one side of the droplet

    C = 1000 * density * 9.81 / surface_tension  # 1/m^2 in SI base units
    S_span = np.arange(0, 2, 0.0005)
    if H > 10:
        S_span = np.arange(0, 2, 0.00005)  # use a finer grid for large H

    res = integrate.solve_ivp(
        odefcn,
        [0, 2],
        [0, 0, 0, 0],
        t_eval=S_span,
        args=[H, C],
        events=odestop,
        dense_output=True,
    )

    X, Z, theta, area = np.vsplit(res.y, 4)
    X = X * 1000
    Z = Z * 1000
    theta = theta * 180 / np.pi
    area = area * 1000 ** 2

    # Move Z into the first quadrant of the plot
    Z = np.amax(Z) - Z

    # Mirror shape to form a whole droplet
    # Exclude value at zero
    Xflipped = -np.flipud(X[1:])
    Zflipped = np.flipud(Z[1:])

    X = np.vstack((Xflipped, X))
    Z = np.vstack((Zflipped, Z))

    return Surface(X, Z)