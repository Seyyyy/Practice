import numpy as np
import sympy as sp
import math

# class LagrangeMultiplier:
#     def __init__(self, f, dLx, g, dLy, dLl):



def f(x, y, l):
    x = x
    y = y
    return 5*x**2 + 6*x*y + 5*y**2 - 26*x - 26*y

def dLX(x, y, l):
    x = x
    y = y
    l = l
    return 10*x + 6*y - 26 + 2*l*x

def dLY(x, y, l):
    x = x
    y = y
    l = l
    return 10*y + 6*x - 26 + 2*l*y

def dLL(x, y, l):
    x = x
    y = y
    return x**2 + y**2 - 4

x = sp.Symbol('x')
y = sp.Symbol('y')
l = sp.Symbol('l')
dLx = sp.Eq(10*x + 6*y - 26 + 2*l*x ,0)
dLy = sp.Eq(10*y + 6*x - 26 + 2*l*y, 0)
dLl = sp.Eq(x**2 + y**2 - 4, 0)
result = sp.solve([dLx, dLy, dLl], [x, y, l])
print(result)

print(dLX(-math.sqrt(2), -math.sqrt(2), -13*math.sqrt(2)/2 - 8))