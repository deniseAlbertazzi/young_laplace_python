import numpy as np

from .utils import lin_interpolate


class Surface:
    def __init__(self, X, Z=None):
        self.X = X.reshape((X.size,))
        if Z is not None:
            self.Z = Z.reshape((Z.size,))
        else:
            self.Z = np.zeros(X.size)

    def add(self, X, Z):
        A = np.amin(X)
        imin = np.argmin(X)
        B = np.amax(X)
        imax = np.argmax(X)

        Xa = abs(self.X - A)
        Xb = abs(self.X - B)

        jmin = np.argmin(Xa)
        jmax = np.argmin(Xb)

        clad = lin_interpolate(X[imin:imax], Z[imin:imax], self.X[jmin:jmax])

        self.Z[jmin:jmax] = clad

    def smooth(self, span):
        """Stackoverflow version of matlab's smooth"""
        out0 = np.convolve(self.Z, np.ones(span, dtype=int), "valid") / span
        r = np.arange(1, span - 1, 2)
        start = np.cumsum(self.Z[: span - 1])[::2] / r
        stop = (np.cumsum(self.Z[:-span:-1])[::2] / r)[::-1]
        self.Z = np.concatenate((start, out0, stop))

    def f(self, x):
        """Gets Z value based on X coordinate"""
        return lin_interpolate(self.X, self.Z, x)

    def area(self, Xa, Xb):
        ind = np.logical_and(Xa < self.X, self.X < Xb)
        Xm = self.X[ind]
        Zm = self.Z[ind]

        Za = self.f(Xa).item()
        Zb = self.f(Xb).item()

        return np.trapz(
            np.concatenate((np.array([Za]), Zm, np.array([Zb]))),
            np.concatenate((np.array([Xa]), Xm, np.array([Xb]))),
        )
