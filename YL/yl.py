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
        new_surface = young_laplace(self.density, self.surface_tension, H)
        new_surface, res_area = self.deposit(new_surface, x + apex, surface, self.area)

        print(f"H={H} apex={apex}")
        print(f"residuals: width={H} apex={apex} area={res_area}")

        return new_surface

    def solve_clad(self, res, surface, x):
        H = res[0]
        apex = res[1]
        new_surface = young_laplace(self.density, self.surface_tension, H)
        new_surface, _ = self.deposit(new_surface, x + apex, surface, self.area)
        X = new_surface.X
        return [X[-1] - X[0] - self.width, (X[-1] + X[0]) / 2 - x]

    def deposit(self, new_surface, x, surface, area):
        z0 = 0
        zmin = -np.amax(new_surface.Z) - 0.1
        zmax = 0.1

        new_surface = Surface(new_surface.X + x, new_surface.Z + surface.f(x))

        res = optimize.least_squares(
            self.solve_deposit, z0, args=(new_surface, surface, area), bounds=(zmin, zmax)
        )

        new_surface.Z += res.x
        new_surface = self.cut_deposit(new_surface, surface)

        return new_surface, res

    def cut_deposit(self, new_surface, surface):
        for i in range(new_surface.X.size):
            if new_surface.Z[i] >= surface.f(new_surface.X[i]):
                new_surface.X = new_surface.X[i:]
                new_surface.Z = new_surface.Z[i:]
                break
        for i in range(new_surface.X.size - 1, -1, -1):
            if new_surface.Z[i] >= surface.f(new_surface.X[i]):
                new_surface.X = new_surface.X[:i]
                new_surface.Z = new_surface.Z[:i]
                break
        return new_surface

    def solve_deposit(self, z, new_surface, surface, area):
        new_surface.Z += z
        if z.item() < 0:  # edge-case: only cut when it actually penetrates into the surface
            new_surface = self.cut_deposit(new_surface, surface)

        X = new_surface.X
        Z = new_surface.Z

        zMin = np.minimum(Z[0], Z[-1])

        # bubbleArea contains the area of the bubble itself and adds the
        # triangle under the bubble if the left and right have unequal height,
        # and also adds the rectangular area under the bubble
        bubbleArea = (
            polyarea(X[0], Z[0]) + zMin * (X[-1] - X[0]) + 0.5 * abs(Z[-1] - Z[0]) * (X[-1] - X[0])
        )

        # p uses the area of the surface and the target area
        return bubbleArea - surface.area(X[0], X[-1]) - area

    def plot(self):
        plt.figure()
        plt.axes().set_aspect("equal", "datalim")
        plt.title("Clad profiles using different shapes")
        plt.xlabel("X (mm)")
        plt.ylabel("Z (mm)")
        plt.grid()
        plt.plot(self.X, self.Z, "-")
