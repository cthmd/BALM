import numpy as np
import pyproximal
import matplotlib.pyplot as plt
import util
from Optimizer import BALM, DP_BALM, FW_BALM, FW_DP_BALM, FW_ALM

A = np.array([[1, 2, -3, 4, 5, 6, 7, 8],
            [-11, 12, 13, 14, 15, 16, 17, 18],
            [21, -22, 23, 24, 25, 26, -27, 28],
            [31, 32, 33, -34, 35, 36, 37, 38],
            [41, 42, 43, 44, 45, 46, 47, 48],
            [51, 52, -53, 54, 55, 56, 57, 58],
            [61, 62, 63, 64, -65, 66, 67, 68], 
            [71, 72, 73, 74, 75, 76, 77, 78]])
b = np.array([ 186,  542,  458 ,1012, 1644, 1686, 1714, 2724])

A, b, _ = util.generate_test_constraint(10)
obj_f = pyproximal.L1()

def f(x):
    return np.sum(np.abs(x))
def grad_f(x):
    return np.sign(x)

opt_BALM = BALM(obj_f, None, A, b)
opt_DP_BALM = DP_BALM(obj_f, None, A, b, alpha=1.3)
opt_FW_BALM = FW_BALM(f, grad_f, A, b)
opt_FW_DP_BALM = FW_DP_BALM(f, grad_f, A, b, alpha=1.3)

x, lamb = opt_BALM.optimize()
print(np.allclose(b, A@x))
print(f(x))
x, lamb = opt_FW_BALM.optimize()
print(np.allclose(b, A@x))
print(f(x))
