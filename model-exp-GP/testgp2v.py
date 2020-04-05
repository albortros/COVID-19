import lsqfitgp2 as lgp
import autograd
from autograd import numpy as np

def fun(params):
    gp = lgp.GP(lgp.ExpQuad(scale=params[0]) + lgp.ExpQuad(scale=params[1]))
    x = np.arange(10)
    gp.addx(x)
    y = np.sin(x)
    return gp.marginal_likelihood(y)

fungrad = autograd.grad(fun)

params = np.array([3, 2], dtype=float)
print(fungrad(params))
