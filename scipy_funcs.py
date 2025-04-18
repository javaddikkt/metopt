from scipy.optimize import minimize
import numpy as np


def compare(f_obj, x0, method):
    if method == 'bfgs':
        method = 'BFGS'
    elif method == 'newton':
        method = 'Newton-CG'
    else:
        method = 'CG'

    result = ""
    result += f"\n=== Сравнение с scipy.optimize.minimize ({method}) ===\n"
    res = minimize(
        f_obj.f,
        np.array(x0, dtype=float),
        method=method,
        jac=f_obj.analit_grad
    )
    result += f"Оптимальное x: {res.x}\n"
    result += f"Число итераций: {res.nit}\n"
    result += f"Статус завершения: {res.message}\n"
    return res.x, result
