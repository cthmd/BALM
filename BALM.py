import numpy as np
import pyproximal
import pylops
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog

def f(x):
    return x.T @ x

def grad_f(x):
    return 2*x

def L_BALM(x, q0k, beta):
    return f(x) + beta/2 * np.linalg.norm(x - q0k, ord=2)**2

def grad_L_BALM(x, q0k, beta):
    return grad_f(x) + beta * np.linalg.norm(x - q0k, ord=2)


def FW_step_BALM(xk, q0k, beta, A, b):
    grad_L_eval = grad_L_BALM(xk, q0k, beta)
    lmo = linprog(c=grad_L_eval, A_eq=A, b_eq=b, method='simplex') # ub
    s = lmo.x
    step = lmo.nit
    print(step)
    gamma_opt = minimize(lambda g: L_BALM(xk + g * (s - xk), q0k, beta), 0, bounds=[(0, 1)])
    gamma = gamma_opt.x
    x_new = (1 - gamma) * xk + gamma * s
    return x_new

def x_step_prox(q0k):
    quad = pyproximal.Quadratic(Op=pylops.MatrixMult(np.identity(H0.shape[0])))
    xk1 = pyproximal.optimization.primal.ProximalPoint(quad, q0k, r, show=False, niter=1)
    return xk1


def BALM(A, b, r, x0, lambda0):
    improvement = []
    xk = x0
    lambdak = lambda0
    
    for i in range(100):
        q0k = xk + (1/r) * A.T @ lambdak

        # x step
        xk1 = x_step_prox(q0k)
        
        # lambda step
        s0k = A @ (2*xk1 - xk) - b
        lambdak1 = lambdak - H0_inv @ s0k
        
        # update
        error = np.sum(np.abs(xk - xk1))
        improvement.append(error)
        if error < 1e-6:
            return xk1, improvement
        xk = xk1
        lambdak = lambdak1
        print(f"iteration {i}: {xk}")
        
    return xk, improvement

def DP_BALM(A, b, alpha, beta, x0, lambda0):
    xk = x0
    lambdak = lambda0
    improvement = []

    for i in range(100):
        # prediction
        lambdak_bar = lambdak - H0_inv @ (A @ xk - b)
        q0k = xk + (1/beta) * A.T @ (2 * lambdak_bar - lambdak)
        xk_bar = x_step_prox(q0k)

        # correction
        xk1 = xk + alpha*(xk_bar - xk)
        lambdak1 = lambdak + alpha*(lambdak_bar - lambdak)

        error = np.linalg.norm(xk - xk1)
        improvement.append(error)
        if error < 1e-6:
            return xk1, improvement
        xk = xk1
        lambdak = lambdak1
        print(f"iteration {i}: {xk}")
    return xk, improvement

def FW_DP_BALM(A, b, alpha, beta, x0, lambda0):
    xk = x0
    lambdak = lambda0
    improvement = []

    for i in range(100):
        # prediction
        lambdak_bar = lambdak - H0_inv @ (A @ xk - b)
        q0k = xk + (1/beta) * A.T @ (2 * lambdak_bar - lambdak)
        xk_bar = FW_step_BALM(xk, q0k, beta, A, b)

        # correction
        xk1 = xk + alpha*(xk_bar - xk)
        lambdak1 = lambdak + alpha*(lambdak_bar - lambdak)

        error = np.linalg.norm(xk - xk1)
        improvement.append(error)
        if error < 1e-6:
            return xk1, improvement
        xk = xk1
        lambdak = lambdak1
        print(f"iteration {i}: {xk}")
    return xk, improvement

'''
A = np.array([[1, 2, 3, 4, 5],
                [-1, 4, 5, 7, 8],
                [4, 6, 2, 6, 7],
                [6, 8, 2, -7, 2],
                [5, 7, -9, 4, 1]])
b = np.array([59, 98, 83, -4,  1])
'''
A = np.array([[1, 2, -3, 4, 5, 6, 7, 8],
            [-11, 12, 13, 14, 15, 16, 17, 18],
            [21, -22, 23, 24, 25, 26, -27, 28],
            [31, 32, 33, -34, 35, 36, 37, 38],
            [41, 42, 43, 44, 45, 46, 47, 48],
            [51, 52, -53, 54, 55, 56, 57, 58],
            [61, 62, 63, 64, -65, 66, 67, 68], 
            [71, 72, 73, 74, 75, 76, 77, 78]])
b = np.array([ 186,  542,  458 ,1012, 1644, 1686, 1714, 2724])

r = beta = 1
alpha = 1
delta = 0.001 
H0 = 1/r * A @ A.T
H0 = H0 + delta * np.identity(H0.shape[0])
H0_inv = np.linalg.inv(H0)
x0 = np.ones_like(b)
lambda0 = np.ones_like(b)

xk, improvement_BALM = BALM(A,b,r,x0,lambda0)
xk, improvement_DPBALM = DP_BALM(A,b,alpha,beta,x0,lambda0)
xk, improvement_FWDPBALM = FW_DP_BALM(A,b,alpha,beta,x0,lambda0)

plt.plot(improvement_DPBALM, label='DP-BALM')
plt.plot(improvement_FWDPBALM, label='FW-DP-BALM')
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.yscale('log')
plt.legend()
plt.show()

'''
#xk, improvement = BALM(A,b,r,x0,lambda0)
for beta in [0.2, 0.6, 1, 1.2, 1.6]:
    xk, improvement = DP_BALM(A,b,alpha,beta,x0,lambda0)
    #plt.plot(improvement, label=f"$\\alpha$={alpha}")
    plt.plot(improvement, label=f"$\\beta$={beta}")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.yscale("log")
plt.title("Dual-Primal BALM $\\beta$")
plt.legend()
plt.show()
'''
'''
xk, improvement = BALM(A,b,r,x0,lambda0)
plt.plot(improvement, label="BALM")
# from the other file
improvement_FW_BALM = np.array([16.132425094159075, 8.29943534311331, 0.39586974130081315, 0.01909126394615053, 0.000674153525626441, 0.0])
plt.plot(improvement_FW_BALM, label="FW-BALM")
for alpha in [0.5, 1, 1.3, 1.5]:
    xk, improvement_DP = DP_BALM(A,b,alpha,beta,x0,lambda0)
    plt.plot(improvement_DP, label=f"DP-BALM $\\alpha$={alpha}")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("DP-BALM vs BALM")
plt.yscale("log")
plt.legend()
plt.show()
'''
'''
for r in [1, 1.2, 1.6, 2]:
    xk, improvement = BALM(A,b,r,x0,lambda0)
    plt.plot(improvement, label=f"r={r}")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Effect of BALM r")
plt.yscale("log")
plt.legend()
plt.show()
'''