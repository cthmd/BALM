import numpy as np
import pyproximal
import matplotlib.pyplot as plt
import util
from Optimizer import BALM, DP_BALM, FW_BALM, FW_DP_BALM, FW_ALM

A, b, _ = util.generate_test_constraint(10, 20)
obj_f = pyproximal.L1()
f = util.f_L1
grad = util.grad_L1

opt_BALM = BALM(obj_f, None, A, b)
opt_DP_BALM = DP_BALM(obj_f, None, A, b, alpha=1.3)
opt_FW_BALM = FW_BALM(f, grad, A, b)
opt_FW_DP_BALM = FW_DP_BALM(f, grad, A, b, alpha=1.3)

x, lamb = opt_BALM.optimize()
print(np.allclose(b, A@x))
print(f(x))
x, lamb = opt_FW_BALM.optimize()
print(np.allclose(b, A@x))
print(f(x))
