import numpy as np
import matplotlib.pyplot as plt
import util
import functools
from Optimizer import BALM, FW_BALM


A, b, _ = util.generate_test_constraint(4, 4)
n = A.shape[0]
mu = 1.0
c = np.random.randn(n)
d = np.random.randn(n, n)
D = np.dot(d, d.T)

def f(x):
    entropy_term = mu * np.sum(x * np.log(x))
    linear_term = np.dot(c, x)
    quadratic_term = 0.5 * np.dot(x, np.dot(D, x))
    return entropy_term + linear_term + quadratic_term

def grad(x):
    return mu * (1 + np.log(x)) + c + np.dot(D,x)

problem = util.MyProx(x0=np.ones_like(A[0]), f=f, g=grad)

opt_FW_BALM = FW_BALM(problem.fun, problem.grad, A, b)
x, lamb = opt_FW_BALM.optimize()
print(np.allclose(b, A@x))
print(problem.fun(x))
plt.plot(opt_FW_BALM.history)
plt.show()

opt_BALM = BALM(problem, None, A, b)
x, lamb = opt_BALM.optimize()
print(np.allclose(b, A@x))
print(problem.fun(x))
plt.plot(opt_BALM.history)
plt.show()
