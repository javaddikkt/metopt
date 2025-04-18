from visual import *
from scipy_funcs import *
from optuna_funcs import *
from step import *

CONST_EPS_STOP = 1e-2


def minimize(
        f_obj,
        x0,
        direction_f,
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
    max_x, max_y = x[0], x[1]
    b = step_params.get('hess')

    for i in range(iter_bound):
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

        if check_stop(f_obj, x, delta, eps, stop_type):
            points.append(x.copy())
            return points, max_x, max_y, i + 1

    return points, max_x, max_y, iter_bound


# def grad_descent(
#         f_obj,
#         x0,
#         step_type='fixed',
#         stop_type='var',
#         step_params=None,
#         iter_bound=1000,
#         eps=CONST_EPS_STOP
# ):
#     if step_params is None:
#         step_params = {}
#
#     step_func = get_step_function(step_type, **step_params)
#     x = np.array(x0, dtype=float)
#     points = []
#     max_x, max_y = 0.0, 0.0
#
#     for i in range(iter_bound):
#         points.append(x.copy())
#
#         direction = -f_obj.grad(x)
#         alpha = step_func(f_obj, x, i)
#         delta = alpha * direction
#         x = x + delta
#         max_x = max(abs(x[0] + delta[0]), max_x)
#         max_y = max(abs(x[1] + delta[1]), max_y)
#
#         if check_stop(f_obj, x, delta, eps, stop_type):
#             points.append(x.copy())
#             return points, max_x, max_y, i + 1
#
#     return points, max_x, max_y, iter_bound


# def newton(
#         f_obj: Function,
#         x0,
#         step_type='fixed',
#         stop_type='var',
#         step_params=None,
#         iter_bound=1000,
#         eps=CONST_EPS_STOP
# ):
#     if step_params is None:
#         step_params = {}
#
#     step_func = get_step_function(step_type, **step_params)
#     x = np.array(x0, dtype=float)
#     points = []
#     max_x, max_y = 0.0, 0.0
#
#     for i in range(iter_bound):
#         points.append(x.copy())
#         direction = -np.linalg.solve(f_obj.hess(x), f_obj.grad(x))
#         alpha = step_func(f_obj, x, i)
#         delta = alpha * direction
#         x = x + delta
#         max_x = max(abs(x[0] + delta[0]), max_x)
#         max_y = max(abs(x[1] + delta[1]), max_y)
#
#         if check_stop(f_obj, x, delta, eps, stop_type):
#             points.append(x.copy())
#             return points, max_x, max_y, i + 1
#
#     return points, max_x, max_y, iter_bound


def grad_descent(
        f_obj: Function,
        x0,
        step_type='fixed',
        stop_type='var',
        step_params=None,
        iter_bound=1000,
        eps=CONST_EPS_STOP
):
    return minimize(
        f_obj, x0,
        lambda f, x, grad, b: -grad,
        step_type, stop_type, step_params, iter_bound, eps
    )


def newton(
        f_obj: Function,
        x0,
        step_type='fixed',
        stop_type='var',
        step_params=None,
        iter_bound=1000,
        eps=CONST_EPS_STOP
):
    return minimize(
        f_obj, x0,
        lambda f, x, grad, b: -np.linalg.solve(f.hess(x), grad),
        step_type, stop_type, step_params, iter_bound, eps
    )


def bfgs(
        f_obj: Function,
        x0,
        step_type='fixed',
        stop_type='var',
        step_params=None,
        iter_bound=1000,
        eps=CONST_EPS_STOP
):
    step_params['hess'] = np.eye(2)
    return minimize(
        f_obj, x0,
        lambda f, x, grad, b: -b @ grad,
        step_type, stop_type, step_params, iter_bound, eps
    )


def next_b(b, s, y):
    y = y.reshape(-1, 1)
    s = s.reshape(-1, 1)
    if y.T @ s == 0:
        return b
    rho = 1 / (y.T @ s)
    return (np.eye(2) - rho * s @ y.T) @ b @ (np.eye(2) - rho * y @ s.T) + (rho * s @ s.T)


# def bfgs(
#         f_obj: Function,
#         x0,
#         step_type='fixed',
#         stop_type='var',
#         step_params=None,
#         iter_bound=1000,
#         eps=CONST_EPS_STOP
# ):
#     if step_params is None:
#         step_params = {}
#
#     step_func = get_step_function(step_type, **step_params)
#     x = np.array(x0, dtype=float)
#     b = np.eye(2)
#     points = []
#     max_x, max_y = 0.0, 0.0
#
#     for i in range(iter_bound):
#         points.append(x.copy())
#         grad_x = f_obj.grad(x)
#         direction = - b @ grad_x
#         alpha = step_func(f_obj, x, i)
#         delta = alpha * direction
#         x = x + delta
#         max_x = max(abs(x[0] + delta[0]), max_x)
#         max_y = max(abs(x[1] + delta[1]), max_y)
#
#         if check_stop(f_obj, x, delta, eps, stop_type):
#             points.append(x.copy())
#             return points, max_x, max_y, i + 1
#
#         b = next_b(b, delta, f_obj.grad(x) - grad_x)
#
#     return points, max_x, max_y, iter_bound


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
        step_params, iter_bound, eps
    )

    draw(f_obj, points, max_x, max_y)
    write_params(f_obj, x0, step_type, stop_type, step_params, points, iters)
    res_x, text = compare(f_obj, x0, method)
    print(text)


def proceed_optimized(f_obj, x0, method):
    step, stop, params = optimize(f_obj, x0, get_method(method), compare)
    print("=== Подобранные гиперпараметры ===")
    proceed(f_obj, x0, method, step, stop, params)
