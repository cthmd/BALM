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
        self.r = r
        self.max_run = max_run
        self.history = []
        self.stopping_error = stopping_error
        H0 = 1/self.r * self.A @ self.A.T
        H0 = H0 + delta * np.identity(H0.shape[0])
        self.H0_inv = np.linalg.inv(H0)

    def primal_update(self, x, lamb):
        q0k = x + (1/self.r) * self.A.T @ lamb
        x_new = self.obj_f.prox(q0k, self.r)
        return x_new

    def dual_update(self, x, x_new, lamb):
        s0k = self.A @ (2*x_new - x) - self.b
        lamb_new = lamb - self.H0_inv @ s0k
        return lamb_new

    def calculate_improvement(self, old, new):
        return np.linalg.norm(old - new)

    def stop(self, old, new):
        improvement = self.calculate_improvement(old, new)
        return (improvement < self.stopping_error)

    def optimize(self):
        print("starting")
        x = self.x
        lamb = self.lamb
        for i in range(self.max_run):
            # x step
            x_new = self.primal_update(x, lamb)
            
            # lambda step
            lamb_new = self.dual_update(x, x_new, lamb)
            
            # update
            self.history.append(self.calculate_improvement(x, x_new))
            print(f"iteration {i}: {x}")
            if self.stop(x, x_new):
                break
            x = x_new
            lamb = lamb_new
        
        self.x = x
        self.lamb = lamb
        return self.x, self.lamb

A = np.array([[1, 2, -3, 4, 5, 6, 7, 8],
            [-11, 12, 13, 14, 15, 16, 17, 18],
            [21, -22, 23, 24, 25, 26, -27, 28],
            [31, 32, 33, -34, 35, 36, 37, 38],
            [41, 42, 43, 44, 45, 46, 47, 48],
            [51, 52, -53, 54, 55, 56, 57, 58],
            [61, 62, 63, 64, -65, 66, 67, 68], 
            [71, 72, 73, 74, 75, 76, 77, 78]])
b = np.array([ 186,  542,  458 ,1012, 1644, 1686, 1714, 2724])

obj_f = pyproximal.Quadratic()
opt = BALM(obj_f, None, A, b)
opt.optimize()
