from __future__ import division

import gvar
from scipy import linalg, special
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
        assert np.all(eigv > -len(cov) * np.finfo(float).eps * np.max(eigv))
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
    
    def predraw(self, fxdata_mean, fxdata_cov=None):
        # check there are x to predict
        assert self._datarange < len(self._prior)
        
        # check fxdata_mean
        y = np.asarray(fxdata_mean)
        assert len(y.shape) == 1
        assert len(y) == self._datarange
        
        # check fxdata_cov
        if fxdata_cov is None:
            C = 0
        else:
            C = np.asarray(fxdata_cov)
            assert C.shape == (len(y), len(y))
            assert np.allclose(C, C.T)
        
        # compute things
        Kxxs = self._cov[:self._datarange, self._datarange:]
        Kxx = self._cov[:self._datarange, :self._datarange]
        Kxsxs = self._cov[self._datarange:, self._datarange:]
        A = linalg.solve(Kxx, Kxxs, assume_a='pos').T
        cov = Kxsxs + A @ (C - Kxx) @ A.T
        mean = A @ y
        
        return mean, cov

    def predalt(self, fxdata):
        fxdata_mean = gvar.mean(fxdata)
        fxdata_cov = gvar.evalcov(gvar.gvar(fxdata))
        mean, cov = self.predraw(fxdata_mean, fxdata_cov)
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
        np.broadcast(x, y)
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
        np.broadcast(x, y)
        return self._ampl * self._kernel((x - self._loc) / self._scale, (y - self._loc) / self._scale, **self._kw)

def makekernel(kernel, superclass):
    name = 'Specific' + superclass.__name__
    return type(name, (superclass,), dict(
        __init__=lambda self, *args, **kw: super(self.__class__, self).__init__(kernel, *args, **kw),
        __doc__=kernel.__doc__
    ))

isotropickernel = lambda kernel: makekernel(kernel, IsotropicKernel)
kernel = lambda kernel: makekernel(kernel, Kernel)

Linear = kernel(lambda x, y: x * y)
ExpQuad = isotropickernel(lambda r: np.exp(-1/2 * r ** 2))

@kernel
def Polynomial(x, y, exponent=None, sigma=None):
    for p in exponent, sigma:
        assert np.isscalar(p)
        assert p > 0
    return (x * y + sigma ** 2) ** exponent
    
@isotropickernel
def Matern(r, nu=None):
    assert np.isscalar(nu)
    x = np.sqrt(2 * nu) * r
    xpos = x > 0
    out = np.ones_like(x, dtype=float)
    out[xpos] = 2 ** (1 - nu) / special.gamma(nu) * x[xpos] ** nu * special.kv(nu, x[xpos])
    return out

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

@kernel
def Wiener(x, y):
    assert np.all(x >= 0)
    assert np.all(y >= 0)
    return np.minimum(x, y)

@kernel
def VarScale(x, y, scalefun=None):
    sx = scalefun(x)
    sy = scalefun(y)
    denom = sx ** 2 + sy ** 2
    factor = np.sqrt(2 * sx * sy / denom)
    return factor * np.exp(-(x - y) ** 2 / denom)
