import numpy as np

class GradientDescent:
    def __init__(self, f, df, alpha, eps):
        self.f = f
        self.df = df
        self.alpha = alpha
        self.eps = eps


    def solve(self, init):
        x = init
        grad = self.df(x)
        while((grad**2).sum() > self.eps**2):
            x = x - self.alpha * grad
            grad = self.df(x)
        self.x_ = x
        self.opt_ = self.f(x)


def f(xx):
    x = xx
    return x**2 + 3*x + 1


def df(xx):
    x = xx
    return 2*x + 3

def sample():
    grad = GradientDescent(f, df, alpha=0.01, eps=1e-6)
    initial = np.array([2])
    grad.solve(initial)
    print(grad.x_)
    print(grad.opt_)

sample()