import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import util
from Optimizer import BALM, FW_BALM
import cvxpy as cvx

MAX_RUN = 200
X = plt.imread('brain_small.jpg')
X = X[:,:,0]

ny,nx = X.shape

s = 0.5
k = round(nx * ny * s)
ri = np.random.choice(nx * ny, k, replace=False)
b = X.T.flat[ri].astype(float)

A = np.kron(
    scipy.fftpack.idct(np.identity(nx), norm='ortho', axis=0),
    scipy.fftpack.idct(np.identity(ny), norm='ortho', axis=0)
    )
A = A[ri,:]

'''
# do L1 optimization
vx = cvx.Variable(nx * ny)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A*vx == b]
prob = cvx.Problem(objective, constraints, )
result = prob.solve(verbose=False)
Xat2 = np.array(vx.value).squeeze()
print(Xat2.shape)
Xa = util.idct2(Xat2.reshape(nx, ny).T)
plt.imshow(Xa, cmap="gray")
plt.show()
'''

import pylops
import pyproximal
Aop = pylops.MatrixMult(A)
x = np.zeros(nx * ny)
f = pyproximal.AffineSet(Aop, b, niter=1)
g = pyproximal.L1()
xhist=[np.zeros_like(x),]
def callback(x):
    xhist.append(x)
Xat2 = pyproximal.optimization.primal.ADMM(f, g, np.zeros_like(x), 0.1, niter=MAX_RUN, show=True, callback=callback)[0]
Xa_ADMM = util.idct2(Xat2.reshape(nx, ny).T)

problem = util.MyProx(x0=np.ones(nx * ny), f=util.f_L1, g=util.grad_L1)
opt = BALM(problem, None, A=A, b=b, verbose=True, max_run=MAX_RUN)
Xat2, _ = opt.optimize()
Xa_BALM = util.idct2(Xat2.reshape(nx, ny).T)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(X, cmap="gray")
ax1.set_title("Original")
ax2.imshow(Xa_ADMM, cmap="gray")
ax2.set_title("ADMM")
ax3.imshow(Xa_BALM, cmap="gray")
ax3.set_title("BALM")
plt.show()

iters = np.arange(0, MAX_RUN+1, 1)
plt.plot(iters, [opt.obj_f(x) for x in xhist], label="ADMM")
plt.plot(iters, [opt.obj_f(x) for x in opt.history], label="BALM")
plt.ylabel("Objective value")
plt.xlabel("Iteration")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
plt.show()

