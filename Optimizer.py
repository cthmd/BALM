import numpy as np
import pyproximal
import pylops
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog

class BALM:
    def __init__(self, obj_f, obj_grad, A, b, x=None, lamb=None, r=1, delta=0.001, max_run=100, stopping_error=1e-6):
        self.obj_f = obj_f
        self.obj_grad = obj_grad
        self.A = A
        self.b = b
        if x == None:
            self.x = np.ones_like(b)
        if lamb == None:
            self.lamb = np.ones_like(b)
        self.max_run = max_run
        self.history = []
        self.stopping_error = stopping_error
        H0 = 1/self.r * self.A @ self.A.T
        H0 = H0 + delta * np.identity(H0.shape[0])
        self.H0_inv = np.linalg.inv(H0)

    def primal_update(self, x, lamb):
        q0k = x + (1/self.r) * self.A.T @ lamb
        quad = pyproximal.Quadratic(Op=pylops.MatrixMult(np.identity(H0.shape[0])))
        x_new = pyproximal.optimization.primal.ProximalPoint(quad, q0k, self.r, show=False, niter=1)
        return x_new

    def dual_update(self, x, x_new, lamb):
        s0k = self.A @ (2*x_new - x) - self.b
        lamb_new = lamb - self.H0_inv @ s0k
        return lamb_new

    def calculate_improvement(self, old, new):
        return np.linalg.norm(old - new)

    def stop(self, old, new):
        improvement = self.calculate_improvement(old, new)
        return (improvement < stopping_error)

    def optimize(self):
        x = self.x
        lamb = self.lamb
        for i in range(self.max_run):
            # x step
            x_new = self.primal_update(x, lamb)
            
            # lambda step
            lamb_new = self.dual_update(x, x_new, lamb)
            
            # update
            self.history.append(self.calculate_improvement(x, x_new))
            if self.stop:
                break
            x = x_new
            lamb = lamb_new
        
        self.x = x
        self.lamb = lamb
        return self.x, self.lamb


