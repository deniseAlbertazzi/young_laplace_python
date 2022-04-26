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
        imin = np.where(X == A)[0][0]
        B = np.amax(X)
        imax = np.where(X == B)[0][0]

        X = X[imin:imax]
        Z = Z[imin:imax]

        Xa = self.X - A
        Xb = self.X - B
        Xa_min = abs(Xa).min(0)
        Xb_min = abs(Xb).min(0)

        jmin = np.where(Xa == Xa_min)[0]
        jmax = np.where(Xb == Xb_min)[0]

        if not jmin.size:
            jmin = np.where(Xa == -Xa_min)[0]

        if not jmax.size:
            jmax = np.where(Xb == -Xb_min)[0]

        jmin = jmin[0]
        jmax = jmax[0]

        clad = lin_interpolate(X, Z, self.X[jmin:jmax])

        self.Z[jmin:jmax] = clad

        return self.X, self.Z

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

        return np.trapz([Za] + Zm + [Zb], [Xa] + Xm + [Xb])
