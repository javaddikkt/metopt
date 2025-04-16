import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functions import *

# ----------------------------- Константы ------------------------------------
CONST_LMB = 1
CONST_ALP = 0.5
CONST_BET = 1
CONST_EPS = 1e-5
CONST_EPS_STOP = 1e-2
CONST_STEP = 1.25e-3
CONST_CENTRE = 0
CONST_SCALE = 10
CONST_DELTA = 1e-3


# ---------------------- Методы одномерного поиска ---------------------------
def line_search_dichotomy(f_obj, x, direction, eps=1e-5):
    a, b = 0, 1
    while abs(b - a) > eps:
        c = (a + b) / 2
        x1 = x + (c - eps) * direction
        x2 = x + (c + eps) * direction
        f1 = f_obj.f(x1)
        f2 = f_obj.f(x2)
        if f1 < f2:
            b = c
        else:
            a = c
    return (a + b) / 2


def line_search_golden(f_obj, x, direction, eps=1e-5):
    a, b = 0, 1
    rho = (math.sqrt(5) - 1) / 2
    while abs(b - a) > eps:
        x1 = a + (1 - rho) * (b - a)
        x2 = a + rho * (b - a)
        f1 = f_obj.f(x + x1 * direction)
        f2 = f_obj.f(x + x2 * direction)
        if f1 < f2:
            b = x2
        else:
            a = x1
    return (a + b) / 2


# ------------------- Стратегии выбора длины шага ----------------------------
def fixed_step(step_size=CONST_STEP):
    def func(_f, _x, _i):
        return step_size

    return func


def exponential_step(lmb=CONST_LMB):
    def func(_f, _x, k):
        return (1.0 / math.sqrt(k + 1)) * math.e ** (-lmb * k)

    return func


def polynomial_step(a=CONST_ALP, b=CONST_BET):
    def func(_f, _x, k):
        return (1.0 / math.sqrt(k + 1)) * (b * k + 1) ** (-a)

    return func


def dichotomy_step(eps=CONST_EPS):
    def func(f_obj, x, _k):
        direction = -f_obj.grad(x)
        return line_search_dichotomy(f_obj, x, direction, eps)

    return func


def golden_step(eps=CONST_EPS):
    def func(f_obj, x, _k):
        direction = -f_obj.grad(x)
        return line_search_golden(f_obj, x, direction, eps)

    return func


def get_step_function(step_type='fixed', **kwargs):
    if step_type == 'fixed':
        return fixed_step(kwargs.get('step_size', CONST_STEP))
    elif step_type == 'exponential':
        return exponential_step(kwargs.get('lambda', CONST_LMB))
    elif step_type == 'polynomial':
        return polynomial_step(
            a=kwargs.get('alpha', CONST_ALP),
            b=kwargs.get('beta', CONST_BET)
        )
    elif step_type == 'dichotomy':
        return dichotomy_step(kwargs.get('line_eps', CONST_EPS))
    elif step_type == 'golden':
        return golden_step(kwargs.get('line_eps', CONST_EPS))
    else:
        raise ValueError(f"Unexpected step_type: {step_type}")


# ------------------------ Критерии остановки ---------------------------------
def check_stop(f_obj, x, delta, eps, stop_type='var'):
    if stop_type == "var":
        return np.linalg.norm(delta) < eps
    elif stop_type == "func":
        return abs(f_obj.f(x) - f_obj.f(x + delta)) < eps
    elif stop_type == "grad":
        return np.linalg.norm(f_obj.grad(x)) < eps
    else:
        raise ValueError(f"Unexpected stop_type: {stop_type}")


# ------------------------- Градиентный спуск --------------------------------
def grad_descent(
        f_obj,
        x0,
        step_type='fixed',
        stop_type='var',
        step_params=None,
        iter_bound=1000,
        eps=CONST_EPS_STOP
):
    if step_params is None:
        step_params = {}

    step_func = get_step_function(step_type, **step_params)
    x = np.array(x0, dtype=float)
    points = []
    max_x, max_y = 0.0, 0.0

    for i in range(iter_bound):
        points.append(x.copy())

        direction = -f_obj.grad(x)
        alpha = step_func(f_obj, x, i)
        delta = alpha * direction
        x = x + delta
        max_x = max(abs(x[0] + delta[0]), max_x)
        max_y = max(abs(x[1] + delta[1]), max_y)

        if check_stop(f_obj, x, delta, eps, stop_type):
            points.append(x.copy())
            return points, max_x, max_y, i + 1

    return points, max_x, max_y, iter_bound


# -------------------------- Визуализация ------------------------------------
def draw(
        f_obj,
        x0,
        step_type='fixed',
        stop_type='var',
        step_params=None,
        iter_bound=1000,
        eps=CONST_EPS_STOP
):
    if step_params is None:
        step_params = {}

    points, max_x, max_y, iters = grad_descent(
        f_obj, x0, step_type, stop_type,
        step_params=step_params,
        iter_bound=iter_bound,
        eps=eps
    )

    points = np.array(points)
    grid_points = 500
    x_lim = (-max_x - 1, max_x + 1)
    y_lim = (-max_y - 1, max_y + 1)

    x_vals = np.linspace(*x_lim, grid_points)
    y_vals = np.linspace(*y_lim, grid_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_obj.f([X, Y])
    if f_obj.with_n_noise != 0:
        scale = math.fabs(f_obj.f([max_x, max_y]) - f_obj.f(points[-1])) / 1000
        Z += np.random.normal(loc=0, scale=scale, size=X.shape)
    plt.contour(X, Y, Z, levels=100)
    plt.plot(points[:, 0], points[:, 1], 'o-', markersize=3, color="deeppink")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Iterations: {iters}, Last point: {points[-1]}")
    plt.show()

    print("Функция: ", f_obj.name(), "; Начальная точка: ", x0)
    print("Стратегия: ", step_type, "; Критерий остановки: ", stop_type)
    print("Величина зашумленности: ", f_obj.with_n_noise)
    print("Гиперпараметры: ", step_params)
    print(">>Результаты вычислений<<")
    print("Оптимальное x:", points[-1])
    print("Число итераций:", iters)


# ---------------------------- Пример запуска --------------------------------
if __name__ == "__main__":
    fb = FBooth(with_n_noise=5)
    initial_point = [-5, -5]

    print("=== Запуск градиентного спуска ===")
    draw(fb, initial_point, step_type='golden', stop_type='var', iter_bound=1000)

    print("\n=== Сравнение с scipy.optimize.minimize (CG) ===")
    res = minimize(
        fb.f,
        np.array(initial_point, dtype=float),
        method='CG',
        jac=fb.analit_grad
    )
    print("Оптимальное x:", res.x)
    print("Число итераций:", res.nit)
    print("Статус завершения:", res.message)
