from lab_1.gradient_descent import *
from lab_1.functions import *


def newton(
        f_obj: Function,
        x0,
        step_type='fixed',
        stop_type='var',
        step_params=None,
        iter_bound=1000,
        eps=CONST_EPS_STOP
):
    print("NEWTON")
    if step_params is None:
        step_params = {}

    step_func = get_step_function(step_type, **step_params)
    x = np.array(x0, dtype=float)
    points = []
    max_x, max_y = 0.0, 0.0

    for i in range(iter_bound):
        points.append(x.copy())
        direction = -np.linalg.solve(f_obj.hess(x), f_obj.grad(x))
        alpha = step_func(f_obj, x, i)
        delta = alpha * direction
        x = x + delta
        max_x = max(abs(x[0] + delta[0]), max_x)
        max_y = max(abs(x[1] + delta[1]), max_y)

        if check_stop(f_obj, x, delta, eps, stop_type):
            points.append(x.copy())
            return points, max_x, max_y, i + 1

    return points, max_x, max_y, iter_bound


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

    points, max_x, max_y, iters = newton(
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
    fb = FBooth()
    initial_point = [-5, -5]

    print("=== Запуск градиентного спуска ===")
    draw(fb, initial_point, step_type='dichotomy', stop_type='func', iter_bound=1000)

    print("\n=== Сравнение с scipy.optimize.minimize (Newton-CG) ===")
    res = minimize(
        fb.f,
        np.array(initial_point, dtype=float),
        method='Newton-CG',
        jac=fb.analit_grad
    )
    print("Оптимальное x:", res.x)
    print("Число итераций:", res.nit)
    print("Статус завершения:", res.message)
