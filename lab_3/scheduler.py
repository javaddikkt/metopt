import math

def const(lr):
    return lambda step: lr

def time_based(lr, lam):
    return lambda step: lr / (1 + lam * step)

def exponential(lr, lam):
    return lambda step: lr * math.exp(-lam * step)

def make_scheduler(name='const', lr=1e-3, lam=1e-3):
    if name == 'const':
        return const(lr)
    if name == 'time_based':
        return time_based(lr, lam)
    if name == 'exponential':
        return exponential(lr, lam)
    raise ValueError(f'unknown scheduler {name}')