import numpy as np
from scipy import interpolate


def polyarea(X, Y):
    return 0.5 * np.abs(np.dot(X, np.roll(Y, 1)) - np.dot(Y, np.roll(X, 1)))


def lin_interpolate(X, V, x):
    if len(X) != len(V):
        print("X and V sizes do not match")
        return

    f = interpolate.interp1d(X, V, fill_value="extrapolate")

    return f(x)
