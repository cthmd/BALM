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
            self.x = np.ones_like(A[0])
        if lamb == None:
            self.lamb = np.ones_like(b)
        self.r = r
        self.max_run = max_run
        self.history = []
        self.stopping_error = stopping_error
        H0 = 1/self.r * self.A @ self.A.T
        H0 = H0 + delta * np.identity(H0.shape[0])
        self.H0_inv = np.linalg.inv(H0)

    def calculate_q0k(self, x, lamb):
        q0k = x + (1/self.r) * self.A.T @ lamb
        return q0k

    def primal_update(self, x, lamb):
        q0k = self.calculate_q0k(x, lamb)
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
            
            self.history.append(self.calculate_improvement(x, x_new))
            print(f"iteration {i}: {x}")
            if self.stop(x, x_new):
                x = x_new
                lamb = lamb_new
                break
            x = x_new
            lamb = lamb_new
        
        self.x = x
        self.lamb = lamb
        return self.x, self.lamb

class DP_BALM(BALM):
    def __init__(self, obj_f, obj_grad, A, b, x=None, lamb=None, alpha=1, r=1, delta=0.001, max_run=100, stopping_error=1e-6):
        super().__init__(obj_f, obj_grad, A, b, x=None, lamb=None, r=1, delta=0.001, max_run=100, stopping_error=1e-6)
        self.alpha = alpha
    
    def calculate_q0k(self, x, lamb, lamb_bar):
        q0k = x + (1/self.r) * A.T @ (2 * lamb_bar - lamb)
        return q0k
    
    def primal_update(self, x, lamb):
        lamb_bar = lamb - self.H0_inv @ (self.A @ x - self.b)
        q0k = self.calculate_q0k(x, lamb, lamb_bar)
        x_bar = self.obj_f.prox(q0k, self.r)
        x_new = x + self.alpha * (x_bar - x)
        return x_new

    def dual_update(self, x, x_new, lamb):
        lamb_bar = lamb - self.H0_inv @ (self.A @ x - self.b)
        lamb_new = lamb + self.alpha*(lamb_bar - lamb)
        return lamb_new

class FW_BALM(BALM):
    def __init__(self, obj_f, obj_grad, A, b, x=None, lamb=None, r=1, delta=0.001, max_run=100, stopping_error=1e-6):
        super().__init__(obj_f, obj_grad, A, b, x=None, lamb=None, r=1, delta=0.001, max_run=100, stopping_error=1e-6)

    def lagrangian(self, x, q0k):
        return self.obj_f(x) + self.r/2 * np.linalg.norm(x - q0k, ord=2)**2

    def lagrangian_grad(self, x, q0k):
        return self.obj_grad(x) + self.r * np.linalg.norm(x - q0k, ord=2)

    def primal_update(self, x, lamb):
        q0k = self.calculate_q0k(x, lamb)
        grad_L_eval = self.lagrangian_grad(x, q0k)
        lmo = linprog(c=grad_L_eval, A_eq=self.A, b_eq=self.b)
        s = lmo.x
        print(lmo.nit)
        gamma_opt = minimize(lambda g: self.lagrangian(x + g * (s - x), q0k), 0, bounds=[(0, 1)])
        gamma = gamma_opt.x
        x_new = (1 - gamma) * x + gamma * s
        return x_new

class FW_ALM(FW_BALM):
    def __init__(self, obj_f, obj_grad, A, b, x=None, lamb=None, r=1, delta=0.001, max_run=100, stopping_error=1e-6):
        super().__init__(obj_f, obj_grad, A, b, x=None, lamb=None, r=1, delta=0.001, max_run=100, stopping_error=1e-6)
        
    def lagrangian(self, x, lamb):
        return self.obj_f(x) + np.dot(lamb, self.A @ x - self.b) + self.r/2 * np.linalg.norm(self.A @ x - self.b)**2

    def lagrangian_grad(self, x, lamb):
        return self.obj_grad(x) + np.dot(self.A.T, lamb) + self.r * np.dot(self.A.T, self.A @ x - self.b)

    def primal_update(self, x, lamb):
        grad_L_eval = self.lagrangian_grad(x, lamb)
        lmo = linprog(c=grad_L_eval, A_eq=self.A, b_eq=self.b)
        s = lmo.x
        gamma_opt = minimize(lambda g: self.lagrangian(x + g * (s - x), lamb), 0, bounds=[(0, 1)])
        gamma = gamma_opt.x
        x_new = (1 - gamma) * x + gamma * s
        return x_new

class FW_DP_BALM(DP_BALM, FW_BALM):
    def __init__(self, obj_f, obj_grad, A, b, x=None, lamb=None, alpha=1, r=1, delta=0.001, max_run=100, stopping_error=1e-6):
        super().__init__(obj_f, obj_grad, A, b, x=None, lamb=None, alpha=1, r=1, delta=0.001, max_run=100, stopping_error=1e-6)

    def primal_update(self, x, lamb):
        lamb_bar = lamb - self.H0_inv @ (self.A @ x - self.b)
        q0k = self.calculate_q0k(x, lamb, lamb_bar)
        grad_L_eval = self.lagrangian_grad(x, q0k)
        lmo = linprog(c=grad_L_eval, A_eq=self.A, b_eq=self.b)
        s = lmo.x
        gamma_opt = minimize(lambda g: self.lagrangian(x + g * (s - x), q0k), 0, bounds=[(0, 1)])
        gamma = gamma_opt.x
        x_new = (1 - gamma) * x + gamma * s
        return x_new
