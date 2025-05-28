import time
import tracemalloc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from metrics import rmse

class Linreg(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)
    def forward(self, x):
        return self.lin(x)

opts_torch = {
    'sgd_plain':    lambda p: optim.SGD(p,    lr=1e-2),
    'sgd_momentum': lambda p: optim.SGD(p,    lr=1e-2, momentum=0.9),
    'sgd_nesterov': lambda p: optim.SGD(p,    lr=1e-2, momentum=0.9, nesterov=True),
    'adagrad':      lambda p: optim.Adagrad(p,lr=1e-2),
    'rmsprop':      lambda p: optim.RMSprop(p,lr=1e-2, alpha=0.9),
    'adam':         lambda p: optim.Adam(p,   lr=1e-2),
}

def run(x, y, epochs=1, batch_size=32):
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True)

    rows = []
    for name, make_opt in opts_torch.items():
        model = Linreg(x.shape[1])
        opt   = make_opt(model.parameters())
        loss_fn = nn.MSELoss()

        tracemalloc.start()
        t0 = time.perf_counter()

        for i in range(epochs):
            print(i)
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()

        dt = time.perf_counter() - t0
        curr, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        with torch.no_grad():
            preds = model(x_t).cpu().numpy()
        rmse_val = rmse(y_t.cpu().numpy(), preds)

        rows.append({
            'optimizer':    name,
            'time_sec':     dt,
            'peak_mem_mb':  peak / (1024 * 1024),
            'rmse_val':     rmse_val,
            'epochs_run':   epochs
        })

    df = pd.DataFrame(rows)
    print(df)
