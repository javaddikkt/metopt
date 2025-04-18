from functions import *
from methods import proceed, proceed_optimized


# ---------------------------- Пример запуска --------------------------------
if __name__ == "__main__":
    fb = FHimmelblau()
    initial_point = [-5, -5]
    params = {'step_type': 'dichotomy', 'stop_type': 'func'}

    proceed(fb, initial_point, method='bfgs', step_params=params)
    # proceed_optimized(fb, initial_point, method='newton')
