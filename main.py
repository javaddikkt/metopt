from functions import *
from methods import proceed, proceed_optimized


# ---------------------------- Пример запуска --------------------------------
if __name__ == "__main__":
    fb = FHimmelblau()
    initial_point = [-5, -5]

    proceed(fb, initial_point, method='bfgs', step_type='dichotomy', stop_type='func', iter_bound=1000)
    # proceed_optimized(fb, initial_point, method_f='newton')
