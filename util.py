import numpy as np

def generate_test_constraint(dim):

    A = np.random.rand(dim, dim) * 10
    x = np.arange(1, 100)[:dim]

    b = np.dot(A, x)
    return A, b, x
