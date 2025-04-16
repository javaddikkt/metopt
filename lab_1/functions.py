from abc import ABC, abstractmethod
import numpy as np

CONST_DELTA = 1e-3
CONST_CENTRE = 0
CONST_SCALE = 10


# ------------------------- Функции-шумы -------------------------------------
def noise(n=0, centre=CONST_CENTRE, scale=CONST_SCALE):
    sum_noise = 0
    for i in range(n):
        sum_noise += (np.random.normal(loc=centre, scale=scale))
    return sum_noise


# ---------------------- Численная аппроксимация градиента, гессиана ------------------
def numeric_grad(fun, x: np.array, delta=1e-5):
    grad_approx = np.zeros_like(x, dtype=float)
    f0 = fun(x)
    for i in range(len(x)):
        x[i] += delta
        f1 = fun(x)
        grad_approx[i] = (f1 - f0) / delta
        x[i] -= delta
    return grad_approx


# TODO: check
def numeric_hess(fun, x: np.array, delta=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    fx = fun(x)
    for i in range(n):
        x[i] += delta
        fxi = fun(x)
        x[i] -= 2 * delta
        fxj = fun(x)
        x[i] += delta
        hess[i, i] = (fxi - 2 * fx + fxj) / delta ** 2
        for j in range(i + 1, n):
            x[i] += delta
            x[j] += delta
            f1 = fun(x)
            x[j] -= 2 * delta
            f2 = fun(x)
            x[i] -= 2 * delta
            f3 = fun(x)
            x[j] += 2 * delta
            f4 = fun(x)
            x[i] += delta
            x[j] -= delta
            hess[i, j] = hess[j, i] = (f1 - f2 - f4 + f3) / (4 * delta ** 2)
    return hess


# --------------------- Интерфейс функции -----------------------
class Function(ABC):
    def __init__(self, with_n_noise=0):
        self.with_n_noise = with_n_noise

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def f(self, x: np.array):
        pass

    def grad(self, x: np.array):
        return numeric_grad(self.f, x, CONST_DELTA)

    @abstractmethod
    def analit_grad(self, x: np.array):
        pass

    def hess(self, x: np.array):
        return numeric_hess(self.f, x, CONST_DELTA)

    @abstractmethod
    def analit_hess(self, x: np.array):
        pass


# --------------------- Классы функций -----------------------
class FSpherical(Function):

    def name(self):
        return "Spherical"

    def f(self, x: np.array):
        return x[0] ** 2 + x[1] ** 2 + noise(self.with_n_noise)

    def analit_grad(self, x: np.array):
        return np.array([2 * x[0], 2 * x[1]], dtype=float)

    def analit_hess(self, x: np.array):
        return np.array([[2, 0], [0, 2]], dtype=float)


class FBadCond(Function):

    def name(self):
        return "Bad Conditionality"

    def f(self, x: np.array):
        return 1000 * x[0] ** 2 + x[1] ** 2 + noise(self.with_n_noise)

    def analit_grad(self, x: np.array):
        return np.array([2000 * x[0], 2 * x[1]], dtype=float)

    def analit_hess(self, x: np.array):
        return np.array([[2000, 0], [0, 2]], dtype=float)


class FBooth(Function):

    def name(self):
        return "Booth"

    def f(self, x: np.array):
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2 + noise(self.with_n_noise)

    def analit_grad(self, x: np.array):
        return np.array([
            10 * x[0] + 8 * x[1] - 34,
            8 * x[0] + 10 * x[1] - 38
        ])

    def analit_hess(self, x: np.array):
        return np.array([[10, 8], [8, 10]], dtype=float)


class FHimmelblau(Function):

    def name(self):
        return "Himmelblau"

    def f(self, x: np.array):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2 + noise(self.with_n_noise)

    def analit_grad(self, x: np.array):
        g1 = x[0] ** 2 + x[1] - 11
        g2 = x[0] + x[1] ** 2 - 7
        df_dx = 2 * g1 * (2 * x[0]) + 2 * g2
        df_dy = 2 * g1 + 2 * g2 * (2 * x[1])
        return np.array([df_dx, df_dy], dtype=float)

    # TODO: count
    def analit_hess(self, x: np.array):
        return np.array([], dtype=float)

