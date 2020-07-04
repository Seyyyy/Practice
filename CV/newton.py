# def newton1dim(f, df, x0, eps=1e-10, max_iter=1000):
#     x = x0
#     iter = 0
#     while True:
#         x_new = x - f(x)/df(x)
#         if abs(x-x_new) < eps:
#             break
#         x = x_new
#         iter += 1
#         if iter == max_iter:
#             break
#     return x_new


# def f(x):
#     return x**3 - 5*x + 1


# def df(x):
#     return 3*x**2 - 5


# print(newton1dim(f, df, 2))
# print(newton1dim(f, df, 0))
# print(newton1dim(f, df, -3))


import numpy as np
from numpy import linalg


class Newton:
    def __init__(self, f, df, eps=1e-10, max_iter=1000):
        self.f = f
        self.df = df
        self.eps = eps
        self.max_iter = max_iter

    
    def solve(self, x0):
        x = x0
        iter = 0
        self.path_ = x0.reshape(1, -1)
        while True:
            x_new = x - np.dot(linalg.inv(self.df(x)), self.f(x))
            self.path_ = np.r_[self.path_, x_new.reshape(1, -1)]
            if((x - x_new)**2).sum() < self.eps**2:
                break
            x = x_new
            iter += 1
            if iter == self.max_iter:
                break
        return x_new

