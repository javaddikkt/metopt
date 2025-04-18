import math
import numpy as np

# ----------------------------- Константы ------------------------------------
CONST_LAMBDA = 1
CONST_ALPHA = 0.5
CONST_BETA = 1
CONST_EPS = 1e-5
CONST_STEP = 1.25e-3
CONST_TOL = 1e-4


# ------------------------ Критерии остановки ---------------------------------
def check_stop(f_obj, x, delta, step_params):
    stop_type = step_params.get('stop_type')
    tol = step_params.get('tol')
    if stop_type == "var":
        return np.linalg.norm(delta) < tol
    elif stop_type == "func":
        return abs(f_obj.f(x) - f_obj.f(x + delta)) < tol
    elif stop_type == "grad":
        return np.linalg.norm(f_obj.grad(x)) < tol
    else:
        raise ValueError(f"Unexpected stop_type: {stop_type}")


# ---------------------- Методы одномерного поиска ---------------------------
def line_search_dichotomy(f_obj, x, direction, eps=CONST_EPS):
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


def line_search_golden(f_obj, x, direction, eps=CONST_EPS):
    a, b = 0, 1
    rho = 2 - (math.sqrt(5) + 1) / 2
    x1 = a + rho * (b - a)
    x2 = b - rho * (b - a)
    f1 = f_obj.f(x + x1 * direction)
    f2 = f_obj.f(x + x2 * direction)
    while True:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + rho * (b - a)
            f1 = f_obj.f(x + x1 * direction)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - rho * (b - a)
            f2 = f_obj.f(x + x2 * direction)
        if abs(b - a) <= eps:
            break
    return (x1 + x2) / 2


# ------------------- Стратегии выбора длины шага ----------------------------
def fixed_step(step_size=CONST_STEP):
    return lambda _f, _x, _k: step_size


def exponential_step(lmb=CONST_LAMBDA):
    return lambda _f, _x, k: (1.0 / math.sqrt(k + 1)) * math.e ** (-lmb * k)


def polynomial_step(a=CONST_ALPHA, b=CONST_BETA):
    return lambda _f, _x, k: (1.0 / math.sqrt(k + 1)) * (b * k + 1) ** (-a)


def dichotomy_step(eps=CONST_EPS):
    return lambda f_obj, x, _k: line_search_dichotomy(f_obj, x, -f_obj.grad(x), eps)


def golden_step(eps=CONST_EPS):
    return lambda f_obj, x, _k: line_search_golden(f_obj, x, -f_obj.grad(x), eps)


def get_step_function(params):
    step_type = params.setdefault('step_type', 'fixed')
    params.setdefault('tol', CONST_TOL)

    if step_type == 'fixed':
        step_size = params.setdefault('step_size', CONST_STEP)
        return fixed_step(step_size)

    elif step_type == 'exponential':
        lmb = params.setdefault('lambda', CONST_LAMBDA)
        return exponential_step(lmb)

    elif step_type == 'polynomial':
        alpha = params.setdefault('alpha', CONST_ALPHA)
        beta = params.setdefault('beta', CONST_BETA)
        return polynomial_step(a=alpha, b=beta)

    elif step_type == 'dichotomy':
        line_eps = params.setdefault('line_eps', CONST_EPS)
        return dichotomy_step(line_eps)

    elif step_type == 'golden':
        line_eps = params.setdefault('line_eps', CONST_EPS)
        return golden_step(line_eps)

    else:
        print(f"Unexpected step_type: {step_type}")
