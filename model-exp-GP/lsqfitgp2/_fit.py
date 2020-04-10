from __future__ import division

import gvar
from autograd import numpy as np
from autograd.scipy import linalg
import autograd
from scipy import optimize

from . import _GP

__all__ = [
    'empbayes_fit'
]

def empbayes_fit(prior, gpfactory, data):
    assert isinstance(prior, np.ndarray)
    assert isinstance(data, dict)
    assert callable(gpfactory)
    chol = linalg.cholesky(gvar.evalcov(prior), lower=True)
    priormean = gvar.mean(prior)
    def fun(p):
        gp = gpfactory(p)
        assert isinstance(gp, _GP.GP)
        res = p - priormean
        diagres = linalg.solve_triangular(chol, res, lower=True)
        return -gp.marginal_likelihood(data) + 1/2 * np.sum(diagres ** 2)
    result = optimize.minimize(autograd.value_and_grad(fun), gvar.mean(prior), jac=True)
    return gvar.gvar(result.x, result.hess_inv)
