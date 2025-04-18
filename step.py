import math
import numpy as np

# ----------------------------- Константы ------------------------------------
CONST_LMB = 1
CONST_ALP = 0.5
CONST_BET = 1
CONST_EPS = 1e-5
CONST_STEP = 1.25e-3


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
