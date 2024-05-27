import numpy as np

def generate_test_constraint(m, n=None):
    if n == None:
        n = m
    A = np.random.rand(m, n) * 10
    x = np.arange(1, n+1)
    b = A @ x
    return A, b, x

def f_L1(x):
    return np.sum(np.abs(x))

def grad_L1(x):
    return np.sign(x)

def f_quad(x):
    return 0.5 * x.T @ x

def grad_quad(x):
    return x

def f_linear(x):
    return x

def grad_linear(x):
    return np.ones_like(x)