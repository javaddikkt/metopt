import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def draw(f_obj, points, max_x, max_y):
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
    plt.title("Путь")
    plt.show()


def write_params(f_obj, x0, step_type, stop_type, step_params, points, iters):
    print("Функция: ", f_obj.name(), "; Начальная точка: ", x0)
    print("Стратегия: ", step_type, "; Критерий остановки: ", stop_type)
    print("Величина зашумленности: ", f_obj.with_n_noise)
    print("Гиперпараметры: ", step_params)
    print(">>Результаты вычислений<<")
    print("Оптимальное x:", points[-1])
    print("Число итераций:", iters)


