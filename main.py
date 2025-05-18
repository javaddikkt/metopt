from functions import *
from methods import proceed, proceed_optimized


# ---------------------------- Пример запуска --------------------------------
if __name__ == "__main__":
    fb = FBooth()
    initial_point = [0, 0]
    params = {'step_type': 'dichotomy', 'stop_type': 'func'}

    # обычный запуск с заданными параметрами
    # proceed(fb, initial_point, method='bfgs', step_params=params)

    # запуск с подбором параметров библиотекой optuna
    proceed_optimized(fb, initial_point, method='bfgs')
