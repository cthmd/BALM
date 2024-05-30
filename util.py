import numpy as np
import pyproximal
from scipy.optimize import minimize

def generate_test_constraint(m, n=None):
    if n == None:
        n = m
    A = np.random.rand(m, n) * 10
    x = np.arange(1, n+1)
    b = A @ x
    return A, b, x

def f_L1(x, M=None):
    if M == None:
        return np.sum(np.abs(x))
    else:
        return np.sum(np.abs(M @ x))

def grad_L1(x, M=None):
    if M == None:
        return np.sign(x)
    else:
        return np.sign(M @ x)

def f_quad(x):
    return 0.5 * x.T @ x

def grad_quad(x):
    return x

def f_linear(x):
    return x

def grad_linear(x):
    return np.ones_like(x)

class MyProx(pyproximal.proximal.Nonlinear):
    def __init__(self, x0, niter=10, warm=True, f=None, grad=None):
        super().__init__(x0, niter=10, warm=True)
        self.f = f
        self.grad = grad
    def fun(self, x):
        return self.f(x)
    def grad(self, x):
        return self.grad(x)
    def optimize(self):
         def callback(x):
            self.solhist.append(x)
        self.solhist = []
        self.solhist.append(self.x0)
        sol = minimize(lambda x: self._funprox(x, self.tau),
                                   x0=self.x0,
                                   jac=lambda x: self._gradprox(x, self.tau),
                                   method='L-BFGS-B', callback=callback,
                                   options=dict(maxiter=15))
        sol = sol.x

        self.solhist = np.array(self.solhist)
        return sol