from typing import Tuple
import numpy as np
from scipy import interpolate


def polyarea(X, Y):
    return 0.5 * np.abs(np.dot(X, np.roll(Y, 1)) - np.dot(Y, np.roll(X, 1)))


def lin_interpolate(X: np.ndarray, V: np.ndarray, x) -> np.ndarray:
    if len(X) != len(V):
        print("X and V sizes do not match")
        return

    z = interpolate.interp1d(X, V, fill_value="extrapolate")(x)
    return z


def clean(X: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    # Get where Z != 0
    indexes = Z != 0

    X = X[indexes]
    Z = Z[indexes]

    return X, Z
