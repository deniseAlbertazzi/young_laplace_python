from typing import Tuple
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

from .surface import Surface
from .utils import polyarea
from .young_laplace import young_laplace


class YL:
    def __init__(self, cladW, cladArea, density, surface_tension):
        self.width = float(cladW)
        self.area = float(cladArea)
        self.density = float(density)
        self.surface_tension = float(surface_tension)

    def clad(self, surface: Surface, x):
        x0 = [1000, 0]
        pmin = [0.0001, -self.width / 4]
        pmax = [100000, self.width / 4]
        res = optimize.least_squares(
            self.solve_clad,
            x0=x0,
            args=(surface, x),
            bounds=(pmin, pmax),
            max_nfev=500,
        )

        H, apex = res.x

        # Calculate the final shape using the found parameters
        X, Z = young_laplace(self.density, self.surface_tension, H)
        X, Z, res_area = self.deposit(X, Z, x + apex, surface, self.area)

        plt.plot(X, Z, "-")
        print(f"H={H} apex={apex}")
        print(f"residuals: width={H} apex={apex} area={res_area}")

        return X, Z

    def solve_clad(self, res, surface, x):
        H = res[0]
        apex = res[1]
        X, Z = young_laplace(self.density, self.surface_tension, H)
        X, Z, _ = self.deposit(X, Z, x + apex, surface, self.area)
        return [
            X[-1] - X[0] - self.width,
            (X[-1] + X[0]) / 2 - x,
        ]

    def deposit(self, X, Z, x, surface, area):
        z0 = 0
        zmin = -np.amax(Z) - 0.1
        zmax = 0.1

        X = X + x
        Z = Z + surface.f(x)

        ### HERE
        res = optimize.least_squares(
            self.solve_deposit,
            x0=z0,
            args=(X, Z, surface, area),
            bounds=(zmin, zmax),
        )

        Z += res.x

        X, Z = self.cut_deposit(X, Z, surface)

        return X, Z, res

    def cut_deposit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        surface: Surface,
    ) -> Tuple[np.ndarray, np.ndarray]:
        i = next(i for i, z in enumerate(Z) if z >= surface.f(X[i]))
        j = next(j for j, z in sorted(enumerate(Z), reverse=True) if z >= surface.f(X[j]))
        return X[i : j + 1], Z[i : j + 1]

    def solve_deposit(self, z, X, Z, surface, area):
        new_Z = Z.copy()
        new_X = X.copy()
        new_Z += z
        # edge-case: only cut when it actually penetrates into the surface
        if z.item() < 0:
            new_X, new_Z = self.cut_deposit(new_X, new_Z, surface)
        zMin = np.minimum(new_Z[0], new_Z[-1])

        # bubbleArea contains the area of the bubble itself and adds the
        # triangle under the bubble if the left and right have unequal height,
        # and also adds the rectangular area under the bubble
        bubbleArea = (
            polyarea(new_X, new_Z)
            + zMin * (new_X[-1] - new_X[0])
            + 0.5 * abs(new_Z[-1] - new_Z[0]) * (new_X[-1] - new_X[0])
        )

        # p uses the area of the surface and the target area
        p = bubbleArea - surface.area(new_X[0], new_X[-1]) - area

        # if np.average(X) > 1:
        #     print(surface.area(new_X[0], new_X[-1]))
        #     plt.plot(X, Z, "-")
        return p
