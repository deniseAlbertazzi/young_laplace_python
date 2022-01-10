import numpy as np
from scipy import interpolate


def polyarea(x, y):
    return 0.5 * np.abs(np.dot(x.transpose(), y) - np.dot(y.transpose(), x))


def lin_interpolate(X, V, x):
    if len(X) != len(V):
        print("X and V sizes do not match")
        return

    f = interpolate.interp1d(X, V, fill_value="extrapolate")

    return f(x)
