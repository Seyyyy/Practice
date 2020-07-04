import numpy as np
import matplotlib.pyplot as plt
import warnings
import linear_regression
import polyreg

def f(x):
    return 1 / (1 + x)


def sample(n):
    x = np.random.random(n) * 5
    y = f(x)
    return x, y


xx = np.arange(0, 5, 0.01)
np.random.seed(0)
y_poly_sum = np.zeros(len(xx))
y_poly_sum_sq = np.zeros(len(xx))
y_lin_sum = np.zeros(len(xx))
y_lin_sum_sq = np.zeros(len(xx))
y_true = f(xx)
n = 100
warnings.filterwarnings("ignore")
for _ in range(n):
    x, y = sample(5)
    poly = polyreg.PolynomialRegression(4)
    poly.fit(x, y)
    lin = linear_regression.LinearRegression()
    lin.fit(x, y)
    y_poly = poly.predict(xx)
    y_poly_sum += y_poly
    y_poly_sum_sq += (y_poly - y_true)**2
    y_lin = lin.predict(xx.reshape(-1, 1))
    y_lin_sum += y_lin
    y_lin_sum_sq += (y_lin - y_true)**2


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 - fig.add_subplot(122)

plt.plot(xx, f(xx), label='truth', color='k', linestyle='solid')
plt.plot(xx, y_poly_sum / n, label='polynomial reg.', color='k', linestyle='dotted')
plt.plot(xx, y_lin_sum / n, label='linear reg.', color='k', linestyle='dashed')
plt.legend()
plt.show()
