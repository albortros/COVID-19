from __future__ import division

import gvar
from scipy import linalg
import numpy as np

__doc__ = """Tools to fit gaussian processes with lsqfit."""

class GP:
    
    def __init__(self, xdata, covfun, xpred=None):
        # check xdata
        xdata = np.asarray(xdata)
        assert len(xdata.shape) == 1
        assert len(xdata) >= 1
        
        # check xpred
        if not (xpred is None):
            xpred = np.asarray(xpred)
            assert len(xpred.shape) == 1
            assert len(xpred) >= 1
        
        # overall x vector
        if xpred is None:
            x = xdata
        else:
            x = np.concatenate([xdata, xpred])
        
        # build covariance matrix and check it is positive definite
        cov = covfun(x.reshape(-1, 1), x.reshape(1, -1))
        assert isinstance(cov, np.ndarray)
        assert np.issubdtype(cov.dtype, np.floating)
        assert cov.shape == (len(x), len(x))
        eigv = linalg.eigvalsh(cov)
        assert np.all(eigv > -len(cov) * np.finfo(float).eps)
        # since we are diagonalizing, maybe instead do cholesky and apply
        # automatically the transformation, so that lsqfit does not do it
        # again. the problem I have to do a cholesky on something which is not
        # exactly positive definite due to roundoff. eigenvalue cut then?
        # or can I do something with an LU?
        
        # create prior
        prior = gvar.gvar(np.zeros(len(x)), cov)
        
        # assign instance variables
        self._datarange = len(xdata)
        self._prior = prior
        self._cov = cov
    
    def prior(self):
        return self._prior[:self._datarange]
    
    def pred(self, fxdata):
        # check there are x to predict
        assert self._datarange < len(self._prior)
        
        # check fxdata
        y = np.asarray(fxdata)
        assert len(y.shape) == 1
        assert len(y) == self._datarange
        
        # compute things
        Kxsx = self._cov[self._datarange:, :self._datarange]
        Kxx = self._cov[:self._datarange, :self._datarange]
        yp = self._prior[:self._datarange]
        ysp = self._prior[self._datarange:]
        
        return Kxsx @ gvar.linalg.solve(Kxx, y - yp) + ysp
    
    def predalt(self, fxdata):
        # check there are x to predict
        assert self._datarange < len(self._prior)
        
        # check fxdata
        y = np.asarray(fxdata)
        assert len(y.shape) == 1
        assert len(y) == self._datarange
        
        # compute things
        C = gvar.evalcov(gvar.gvar(y))
        Kxxs = self._cov[:self._datarange, self._datarange:]
        Kxx = self._cov[:self._datarange, :self._datarange]
        Kxsxs = self._cov[self._datarange:, self._datarange:]
        A = linalg.solve(Kxx, Kxxs, assume_a='pos').T
        cov = Kxsxs + A @ (C - Kxx) @ A.T
        mean = A @ gvar.mean(y)
        
        return gvar.gvar(mean, cov)
        
class IsotropicKernel:
    
    def __init__(self, kernel, *, scale=1, ampl=1, **kw):
        for p in scale, ampl:
            assert np.isscalar(p)
            assert np.isfinite(p)
            assert p > 0
        self._kw = kw
        self._scale = scale
        self._ampl = ampl
        self._kernel = kernel
    
    def __call__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return self._ampl * self._kernel(np.abs(x - y) / self._scale, **self._kw)

class Kernel(IsotropicKernel):
    
    def __init__(self, *args, loc=0, **kw):
        assert np.isscalar(loc)
        assert np.isfinite(loc)
        self._loc = loc
        super().__init__(*args, **kw)
    
    def __call__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return self._ampl * self._kernel((x - self._loc) / self._scale, (y - self._loc) / self._scale, **self._kw)

def makekernel(kernel, superclass):
    name = 'Specific' + superclass.__name__
    return type(name, (superclass,), dict(
        __init__=lambda self, *args, **kw: super(self.__class__, self).__init__(kernel, *args, **kw),
        __doc__=kernel.__doc__
    ))

isotropickernel = lambda kernel: makekernel(kernel, IsotropicKernel)
kernel = lambda kernel: makekernel(kernel, Kernel)

ExpQuad = isotropickernel(lambda r: np.exp(-1/2 * r ** 2))

Matern12 = isotropickernel(lambda r: np.exp(-r))
Matern32 = isotropickernel(lambda r: (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r))
Matern52 = isotropickernel(lambda r: (1 + np.sqrt(5) * r + 5/3 * r**3) * np.exp(-np.sqrt(5) * r))

@isotropickernel
def GammaExp(r, gamma=None):
    """Gamma exponential"""
    assert np.isscalar(gamma)
    assert 0 < gamma <= 2
    return np.exp(-(r ** gamma))

@isotropickernel
def RatQuad(r, alpha=None):
    assert np.isscalar(alpha)
    assert alpha > 0
    return (1 + r ** 2 / (2 * alpha)) ** (-alpha)

@kernel
def NNKernel(x, y, q=None):
    assert np.isscalar(q)
    assert q >= 1
    q2 = q ** 2
    return 2/np.pi * np.arcsin(2 * (q2 + x * y) / ((1 + 2 * (q2 + x**2)) * (1 + 2 * (q2 + y**2))))
