from visual import *
from scipy_funcs import *
from optuna_funcs import *
from step import *

ITER_BOUND = 1000


def minimize(
        f_obj,
        x0,
        direction_f,
        step_params
):
    step_func = get_step_function(step_params)
    x = np.array(x0, dtype=float)
    points = []
    max_x, max_y = abs(float(x[0])), abs(float(x[1]))
    b = step_params.get('hess', None)

    for i in range(ITER_BOUND):
        points.append(x.copy())

        alpha = step_func(f_obj, x, i)
        grad = f_obj.grad(x)
        direction = direction_f(f_obj, x, grad, b)
        delta = alpha * direction
        x = x + delta

        if b is not None:
            b = next_b(b, delta, f_obj.grad(x) - grad)

        max_x = max(abs(float(x[0])), max_x)
        max_y = max(abs(float(x[1])), max_y)

        if check_stop(f_obj, x, delta, step_params):
            points.append(x.copy())
            return points, max_x, max_y, i + 1

    return points, max_x, max_y, ITER_BOUND


def grad_descent(
        f_obj: Function,
        x0,
        step_params
):
    return minimize(
        f_obj, x0,
        lambda f, x, grad, b: -grad,
        step_params
    )


def newton(
        f_obj: Function,
        x0,
        step_params
):
    return minimize(
        f_obj, x0,
        lambda f, x, grad, b: -np.linalg.solve(f.hess(x), grad),
        step_params
    )


def bfgs(
        f_obj: Function,
        x0,
        step_params
):
    step_params['hess'] = np.eye(2)
    return minimize(
        f_obj, x0,
        lambda f, x, grad, b: -b @ grad,
        step_params
    )


def next_b(b, s, y):
    y = y.reshape(-1, 1)
    s = s.reshape(-1, 1)
    if y.T @ s == 0:
        return b
    rho = 1 / (y.T @ s)
    return (np.eye(2) - rho * s @ y.T) @ b @ (np.eye(2) - rho * y @ s.T) + (rho * s @ s.T)


def get_method(method):
    if method == 'newton':
        return newton
    if method == 'bfgs':
        return bfgs
    if method == 'grad':
        return grad_descent
    print("Unknown method")


def proceed(
        f_obj,
        x0,
        method,
        step_params
):
    print("=== Результат оптимизации ===")

    points, max_x, max_y, iters = get_method(method)(f_obj, x0, step_params)
    draw(f_obj, points, max_x, max_y)
    write_params(method, f_obj, x0, step_params, points, iters)
    method, res = compare(f_obj, x0, method)
    write_comparison(method, res)


def proceed_optimized(f_obj, x0, method):
    params = optimize(f_obj, x0, get_method(method), compare)

    print("=== Результат работы optuna ===")

    proceed(f_obj, x0, method, params)
