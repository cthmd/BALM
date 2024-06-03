import numpy as np
import matplotlib.pyplot as plt
import util
import functools
from Optimizer import BALM, DP_BALM, FW_BALM, FW_DP_BALM, FW_ALM



A, b, _ = util.generate_test_constraint(10, 10)
affineL1 = util.MyProx(x0=np.ones_like(A[0]), f=util.f_L1, grad=util.grad_L1)
problem = affineL1

opt_BALM = BALM(problem, None, A, b)
opt_DP_BALM = DP_BALM(problem, None, A, b, alpha=1.3)
opt_FW_BALM = FW_BALM(problem.fun, problem.grad, A, b)
opt_FW_DP_BALM = FW_DP_BALM(problem.fun, problem.grad, A, b, alpha=1.3)

x, lamb = opt_BALM.optimize()
print(np.allclose(b, A@x))
print(problem.fun(x))
x, lamb = opt_FW_BALM.optimize()
print(np.allclose(b, A@x))
print(problem.fun(x))
