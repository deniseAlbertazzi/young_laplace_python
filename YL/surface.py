import numpy as np
from matplotlib import pyplot as plt

from .utils import lin_interpolate


class Surface:
    def __init__(self, X, Z=None):
        self.X = X.reshape((X.size,))
        if Z is not None:
            self.Z = Z.reshape((Z.size,))
        else:
            self.Z = np.zeros(X.size)

    def clean(self) -> None:
        # If arrays are not long enough to be trimmed
        if not len(self.X) > 3 or not len(self.Z) > 3:
            return

        new_X, new_Z = [self.X[0]], [self.Z[0]]
        # remove border values
        middle_X, middle_Z = self.X[1:-1], self.Z[1:-1]

        for x, z in zip(middle_X, middle_Z):
            if z != 0:
                new_X.append(x)
                new_Z.append(z)

        new_X.append(self.X[-1])
        new_Z.append(self.Z[-1])

        self.X, self.Z = new_X, new_Z

    def __add__(self, surface):
        A = np.amin(surface.X)
        imin = np.where(surface.X == A)[0][0]
        B = np.amax(surface.X)
        imax = np.where(surface.X == B)[0][0]

        X = surface.X[imin:imax]
        Z = surface.Z[imin:imax]

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

        if len(clad) > 1:
            if not clad[0]:
                clad = clad[1:]
                jmin = jmin + 1
            if not clad[-1]:
                clad = clad[0:-1]
                jmax = jmax - 1
        Z[jmin:jmax] = clad

        return Surface(np.asarray(X), np.asarray(Z))

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
        Xm = None
        Zm = None
        for i, x in enumerate(self.X):
            if Xa < x < Xb:
                Xm = x
                Zm = self.Z[i]
                break

        if not Xm:
            Xm = (Xa + Xb) / 2
            Zm = self.f(Xm).item()

        Za = self.f(Xa).item()
        Zb = self.f(Xb).item()

        return np.trapz([Za, Zm, Zb], [Xa, Xm, Xb])

    def plot(self):
        plt.figure()
        plt.axes().set_aspect("equal", "datalim")
        plt.title("Clad profiles using different shapes")
        plt.xlabel("X (mm)")
        plt.ylabel("Z (mm)")
        plt.grid()
        plt.plot(self.X, self.Z, "-")
