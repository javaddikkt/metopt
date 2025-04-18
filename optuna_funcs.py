import optuna
import numpy as np
from functions import Function


def objective(trial, method_f, f_obj: Function, x0, compare_f, iter_bound=1000, eps=1e-2):
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

    points, max_x, max_y, iters = method_f(f_obj, x0, step_type, stop_type, step_params, iter_bound, eps)
    real_point, text = compare_f(f_obj, x0, method_f)
    return np.linalg.norm(real_point - points[-1]), iters


def optimize(f_obj, x0, method_f, compare_f):
    study = optuna.create_study(directions=['minimize', 'minimize'])
    study.optimize(lambda trial: objective(trial, method_f=method_f, f_obj=f_obj, x0=x0, compare_f=compare_f), n_trials=30)
    trial = study.best_trials[-1]
    params = trial.params
    step_params = {}
    for param in params:
        if param != 'step' and param != 'stop':
            step_params[param] = params[param]
    return params['step'], params['stop'], step_params
    # print("=== Подобранные гиперпараметры ===")
    # proceed(f_obj, x0, method, params['step'], params['stop'], step_params)