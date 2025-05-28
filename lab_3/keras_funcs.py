import time
import tracemalloc
import pandas as pd
import tensorflow as tf
from keras import layers, optimizers, losses, callbacks
from metrics import rmse

def build(d_in):
    return tf.keras.Sequential([
        layers.Input(shape=(d_in,)),
        layers.Dense(1)
    ])

opts = {
    'sgd_plain':     optimizers.SGD(1e-2),
    'sgd_momentum':  optimizers.SGD(1e-2, momentum=0.9),
    'sgd_nesterov':  optimizers.SGD(1e-2, momentum=0.9, nesterov=True),
    'adagrad':       optimizers.Adagrad(1e-2),
    'rmsprop':       optimizers.RMSprop(1e-2, rho=0.9),
    'adam':          optimizers.Adam(1e-2),
}

def run(x, y, batch_size=32, max_epochs=1, patience=5):
    rows = []
    for name, opt in opts.items():
        model = build(x.shape[1])
        model.compile(optimizer=opt,
                      loss=losses.MeanSquaredError())

        tracemalloc.start()
        t0 = time.perf_counter()

        history = model.fit(
            x, y,
            batch_size=batch_size,
            epochs=max_epochs,
            verbose=1,
            callbacks=[callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)]
        )

        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        preds = model.predict(x, batch_size=batch_size, verbose=0)
        rmse_val = rmse(y, preds)

        epochs_run = len(history.history['loss'])

        rows.append({
            'optimizer':    name,
            'time_sec':     dt,
            'peak_mem_mb':  peak / (1024 * 1024),
            'rmse_val':     rmse_val,
            'epochs_run':   epochs_run
        })

    df = pd.DataFrame(rows)
    print(df)
