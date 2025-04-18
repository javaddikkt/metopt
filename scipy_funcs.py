from scipy.optimize import minimize
import numpy as np


def compare(f_obj, x0, method):
    if method == 'bfgs':
        method = 'BFGS'
    elif method == 'newton':
        method = 'Newton-CG'
    else:
        method = 'CG'

    res = minimize(
        f_obj.f,
        np.array(x0, dtype=float),
        method=method,
        jac=f_obj.analit_grad
    )
    return method, res
