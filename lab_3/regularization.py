import numpy as np

def none_reg():
    return lambda w: (0.0, 0.0)

def l2(alpha=0.0):
    return lambda w: (2 * alpha * w, alpha * np.sum(w * w))

def l1(alpha=0.0,):
    return lambda w: (alpha * np.sign(w), alpha * np.sum(np.abs(w)))

def elastic(alpha=0.0, beta=0.0):
    return lambda w: (
        alpha * np.sign(w) + 2 * beta * w,
        alpha * np.sum(np.abs(w)) + beta * np.sum(w * w),
    )

def make_regularizer(name='none', alpha=0.0, beta=0.0):
    if name == 'none':
        return none_reg()
    if name == 'l2':
        return l2(alpha)
    if name == 'l1':
        return l1(alpha)
    if name == 'elastic':
        return elastic(alpha, beta)
    raise ValueError(f'unknown regularizer {name}')