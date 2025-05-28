import time, tracemalloc, pandas as pd
from stochastic import sgd
import numpy as np
import math

def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))

def benchmark(x, y, cfg=None, test_split=0.2):
    cfg = cfg or {}
    n = len(x)
    idx = int(n * (1 - test_split))
    x_train, x_val = x[:idx], x[idx:]
    y_train, y_val = y[:idx], y[idx:]
    print("Train set size: ", len(x_train))
    tracemalloc.start()
    t0 = time.perf_counter()
    trainer = sgd(**cfg)
    w, b, hist = trainer.fit(x_train, y_train)
    dt = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    preds = x_val @ w + b
    return w, b, pd.DataFrame([
        {
            **cfg,
            'time_sec': dt,
            'peak_mem_mb': peak / 1024 ** 2,
            'rmse_val': rmse(y_val, preds),
            'epochs_run': len(hist) // math.ceil(len(x_train) / trainer.batch_size)
        }
    ])