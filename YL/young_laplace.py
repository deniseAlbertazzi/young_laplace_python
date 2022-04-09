from typing import Tuple
import numpy as np
from scipy import integrate


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


def young_laplace(
    density: float,
    surface_tension: float,
    H: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # density in g/cm^3
    # surface_tension in N/m
    # H as the curvature in 1/m
    # X, Z are in mm
    # T is the tangential angle
    # A in mm^2 and is the area of one side of the droplet

    C = 1000 * density * 9.81 / surface_tension  # 1/m^2 in SI base units
    step = 0.0005
    end_value = 2
    if H > 10:  # use a finer grid for large H
        step = 0.00005
    end_value += step

    S_span = np.arange(0, end_value, step)
    res = integrate.solve_ivp(
        odefcn,
        [0, end_value],
        [0, 0, 0, 0],
        t_eval=S_span,
        args=[H, C],
        events=odestop,
        dense_output=True,
    )

    X, Z, theta, area = np.vsplit(res.y, 4)
    X = X.reshape((X.size,)) * 1000
    Z = Z.reshape((Z.size,)) * 1000
    # theta = theta.reshape((theta.size,)) * 180 / np.pi
    # area = area.reshape((area.size,)) * 1000 ** 2

    # Move Z into the first quadrant of the plot
    Z = np.amax(Z) - Z

    # Mirror shape to form a whole droplet
    # Exclude value at zero
    Xflipped = -np.flip(X[1:])
    Zflipped = np.flip(Z[1:])

    X = np.hstack((Xflipped, X))
    Z = np.hstack((Zflipped, Z))

    return X, Z
