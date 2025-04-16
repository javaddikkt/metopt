import numpy as np
import math

from lab_1.gradient_descent import *
from lab_1.functions import *
import optuna


def newton(
        f_obj: Function,
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


def bfgs(
        f_obj: Function,
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
    b = np.eye(2)
    points = []
    max_x, max_y = 0.0, 0.0

    for i in range(iter_bound):
        points.append(x.copy())
        grad_x = f_obj.grad(x)
        direction = - b @ grad_x
        alpha = step_func(f_obj, x, i)
        delta = alpha * direction
        x = x + delta
        max_x = max(abs(x[0] + delta[0]), max_x)
        max_y = max(abs(x[1] + delta[1]), max_y)

        if check_stop(f_obj, x, delta, eps, stop_type):
            points.append(x.copy())
            return points, max_x, max_y, i + 1

        b = next_b(b, delta, f_obj.grad(x) - grad_x)

    return points, max_x, max_y, iter_bound


def next_b(b, s, y):
    y = y.reshape(-1, 1)
    s = s.reshape(-1, 1)
    if y.T @ s == 0:
        return b
    pho = 1 / (y.T @ s)
    return (np.eye(2) - pho * s @ y.T) @ b @ (np.eye(2) - pho * y @ s.T) + (pho * s @ s.T)


def objective(trial, method, f_obj: Function, x0, iter_bound=1000, eps=CONST_EPS_STOP):
    step_type = trial.suggest_categorical('step', ['fixed', 'exponential', 'polynomial', 'dichotomy', 'golden'])
    stop_type = trial.suggest_categorical('stop', ['var', 'func', 'grad'])
    step_params = {}
    if step_type == 'fixed':
        step_size = trial.suggest_float('step_size', 1e-5, 1e-2)
        step_params['step_size'] = step_size
    elif step_type == 'exponential':
        lmb = trial.suggest_float('lambda', 0, 1)
        step_params['lambda'] = lmb
    elif step_type == 'polynomial':
        alpha = trial.suggest_float('alpha', 0, 1)
        beta = trial.suggest_float('beta', 0, 1)
        step_params['alpha'] = alpha
        step_params['beta'] = beta
    else:
        line_eps = trial.suggest_float('line_eps', 1e-7, 1e-2)
        step_params['line_eps'] = line_eps

    points, max_x, max_y, iters = method(f_obj, x0, step_type, stop_type, step_params, iter_bound, eps)
    real_point, text = compare(f_obj, method)
    return np.linalg.norm(real_point - points[-1]), iters
    # draw(f_obj, points, max_x, max_y)
    # write_params(f_obj, x0, step_type, stop_type, step_params, points, iters)


def get_method(method):
    if method == 'bfgs':
        return bfgs
    elif method == 'newton':
        return newton
    print("Unknown method")


def optimize(f_obj, x0, method):
    study = optuna.create_study(directions=['minimize', 'minimize'])
    study.optimize(lambda trial: objective(trial, method=get_method(method), f_obj=f_obj, x0=x0), n_trials=30)
    trial = study.best_trials[-1]
    params = trial.params
    step_params = {}
    for param in params:
        if param != 'step' and param != 'stop':
            step_params[param] = params[param]
    print("=== Подобранные гиперпараметры ===")
    proceed(f_obj, x0, method, params['step'], params['stop'], step_params)

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


def proceed(
        f_obj,
        x0,
        method,
        step_type='fixed',
        stop_type='var',
        step_params=None,
        iter_bound=1000,
        eps=CONST_EPS_STOP
):
    if step_params is None:
        step_params = {}

    print("=== Запуск оптимизации ===")

    points, max_x, max_y, iters = get_method(method)(
        f_obj, x0, step_type, stop_type,
        step_params=step_params,
        iter_bound=iter_bound,
        eps=eps
    )

    draw(f_obj, points, max_x, max_y)
    write_params(f_obj, x0, step_type, stop_type, step_params, points, iters)
    res_x, text = compare(f_obj, method)
    print(text)


def compare(f_obj, method):
    if method == 'bfgs':
        method = 'BFGS'
    else:
        method = 'Newton-CG'

    result = ""
    result += f"\n=== Сравнение с scipy.optimize.minimize ({method}) ===\n"
    res = minimize(
        f_obj.f,
        np.array(initial_point, dtype=float),
        method=method,
        jac=f_obj.analit_grad
    )
    result += f"Оптимальное x: {res.x}\n"
    result += f"Число итераций: {res.nit}\n"
    result += f"Статус завершения: {res.message}\n"
    return res.x, result


# ---------------------------- Пример запуска --------------------------------
if __name__ == "__main__":
    fb = FBadCond()
    initial_point = [-5, -5]

    # proceed(fb, initial_point, method='bfgs', step_type='dichotomy', stop_type='func', iter_bound=1000)
    optimize(fb, initial_point, method='newton')
