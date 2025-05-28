import math
import numpy as np
from scheduler import make_scheduler
from regularization import make_regularizer

class sgd:
    def __init__(
            self,
            lr=1e-2,
            batch_size=32,
            max_epochs=100,
            scheduler='const',
            lam=1e-3,
            reg='none',
            alpha=0.0,
            beta=0.0,
            stop_tol=1e-5,
            stop_by='args',
            method='plain',
            rho=0.95,
            eps=1e-6
    ):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.stop_tol = stop_tol
        self.stop_by = stop_by
        self.method = method
        self.rho = rho
        self.eps = eps
        self.lr_fn = make_scheduler(scheduler, lr=lr, lam=lam)
        self.reg_fn = make_regularizer(reg, alpha=alpha, beta=beta)

    def batch_iter(self, x, y, rng):
        n = len(x)
        idx = rng.permutation(n)
        for i in range(0, n, self.batch_size):
            ids = idx[i : i + self.batch_size]
            yield x[ids], y[ids]

    def check_stop(self, delta_w, delta_b, delta_j):
        tol = self.stop_tol
        if self.stop_by == "args":
            return np.linalg.norm(delta_w) < tol and math.fabs(delta_b) < tol
        if self.stop_by == "func":
            return math.fabs(delta_j) < tol
        print("Unknown stop type:", self.stop_by)
        return False

    def fit(self, x, y, init_w=None, init_b=0.0):
        x = np.asarray(x, float)
        y = np.asarray(y, float).reshape(-1, 1)
        init_w = init_w if init_w is not None else np.zeros((x.shape[1], 1))
        w = init_w.reshape(-1, 1).astype(float)
        b = float(init_b)
        j = 0.0
        hist = []
        rng = np.random.default_rng(42)

        if self.method == 'adadelta':
            Eg2 = np.zeros_like(w)  # E[g^2]
            Ed2 = np.zeros_like(w)  # E[Δw^2]
            Eg2_b, Ed2_b = 0.0, 0.0  # для смещения

        step = 0
        for i in range(self.max_epochs):
            if i % 10 == 0:
                print("epoch running: ", i)
            for xb, yb in self.batch_iter(x, y, rng):
                m = len(xb)
                w_t = w.reshape(1, -1)[0]
                j_b = 0.0
                grad_w = np.zeros_like(w)
                grad_b = 0
                for i in range(m):
                    x_j = np.array(xb[i]).reshape(-1, 1)
                    y_j = yb[i]
                    r = w_t @ x_j + b - y_j
                    j_b += r ** 2 / m
                    grad_w += (r * x_j) / m
                    grad_b += r / m
                grad_w *= 2
                grad_b *= 2

                delta_grad, delta_j = self.reg_fn(w)
                grad_w += delta_grad
                j_b += delta_j

                if self.method == "adadelta":
                    Eg2 = self.rho * Eg2 + (1 - self.rho) * (grad_w ** 2)
                    Eg2_b = self.rho * Eg2_b + (1 - self.rho) * (grad_b ** 2)

                    RMS_g = np.sqrt(Eg2 + self.eps)
                    RMS_delta = np.sqrt(Ed2 + self.eps)
                    delta_w = - (RMS_delta / RMS_g) * grad_w

                    RMS_g_b = math.sqrt(Eg2_b + self.eps)
                    RMS_delta_b = math.sqrt(Ed2_b + self.eps)
                    delta_b = - (RMS_delta_b / RMS_g_b) * grad_b

                    w_new = w + delta_w
                    b_new = b + delta_b

                    Ed2 = self.rho * Ed2 + (1 - self.rho) * (delta_w ** 2)
                    Ed2_b = self.rho * Ed2_b + (1 - self.rho) * (delta_b ** 2)
                else:
                    eta = self.lr_fn(step)
                    w_new = w - eta * grad_w
                    b_new = b - eta * grad_b

                hist.append(j_b)
                if self.check_stop(w_new - w, b_new - b, j_b - j):
                    print("stopped on: ", j_b - j)
                    return w_new, b_new, hist
                w = w_new
                b = b_new
                j = j_b
                step += 1
                
        return w, b, hist