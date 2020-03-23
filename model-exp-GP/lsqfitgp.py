from __future__ import division

import gvar
from scipy import linalg, special
import numpy as np

__doc__ = """Tools to fit gaussian processes with lsqfit."""

class GP:
    
    def __init__(self, xdata, covfun, xpred=None):
        # check covfun
        assert isinstance(covfun, Kernel)
        
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
        mineigv = np.min(eigv)
        if mineigv < 0:
            assert mineigv > -len(cov) * np.finfo(float).eps * np.max(eigv)
            cov[np.diag_indices(len(cov))] += -mineigv
        # since we are diagonalizing, maybe save the transformation and apply
        # it so that lsqfit does not diagonalize it again for the fit.
        
        # assign instance variables
        self._datarange = len(xdata)
        self._prior = None
        self._cov = cov
    
    def _makeprior(self):
        if self._prior is None:
            self._prior = gvar.gvar(np.zeros(len(self._cov)), self._cov)
    
    def prior(self):
        self._makeprior()
        return self._prior[:self._datarange]
    
    def predprior(self):
        self._makeprior()
        return self._prior[self._datarange:]
    
    def pred(self, fxdata):
        # This function is tipically used when fxdata has been obtained from
        # a fit using the prior. However, since the prior-posterior
        # correlations actually cancel in the normal approximation, all this
        # works even if the posterior was obtained "manually" and is not
        # keeping track of correlations with the prior. So, it makes sense to
        # allow the prior gvar to not have been generated explicitly.
        self._makeprior()
        
        # check there are x to predict
        assert self._datarange < len(self._cov)
        
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
        assert self._datarange < len(self._cov)
        
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
        
    def fitpred(self, y):
        self._makeprior()
        
        # check there are x to predict
        assert self._datarange < len(self._cov)
        
        # check y
        y = np.asarray(y)
        assert len(y.shape) == 1
        assert len(y) == self._datarange
        S = gvar.evalcov(y)
        
        # compute things
        Kxsx = self._cov[self._datarange:, :self._datarange]
        Kxx = self._cov[:self._datarange, :self._datarange]
        yp = self._prior[:self._datarange]
        ysp = self._prior[self._datarange:]
        U = Kxx + S
        
        return Kxsx @ gvar.linalg.solve(U, y - yp) + ysp
    
    def fitpredraw(self, y, yerr=None):
        # check there are x to predict
        assert self._datarange < len(self._cov)
        
        # check y
        y = np.asarray(y)
        assert len(y.shape) == 1
        assert len(y) == self._datarange
        
        # check yerr
        if yerr is None:
            S = 0
        else:
            yerr = np.asarray(yerr)
            assert len(yerr.shape) <= 2
            if len(yerr.shape) <= 1:
                S = np.diag(yerr ** 2 * np.ones(len(y)))
            else:
                assert yerr.shape == (len(y), len(y))
                assert np.allclose(yerr, yerr.T)
                S = yerr
        
        # compute things
        Kxxs = self._cov[:self._datarange, self._datarange:]
        Kxx = self._cov[:self._datarange, :self._datarange]
        Kxsxs = self._cov[self._datarange:, self._datarange:]
        U = Kxx + S
        B = linalg.solve(U, Kxxs, assume_a='pos').T
        cov = Kxsxs - Kxxs.T @ B.T
        mean = B @ y
        
        return mean, cov
    
    def fitpredalt(self, y):
        y_mean = gvar.mean(y)
        y_cov = gvar.evalcov(gvar.gvar(y))
        mean, cov = self.fitpredraw(y_mean, y_cov)
        return gvar.gvar(mean, cov)

class Kernel:
    
    def __init__(self, kernel, *, scale=1, loc=0, **kw):
        assert np.isscalar(scale)
        assert np.isscalar(loc)
        assert np.isfinite(scale)
        assert np.isfinite(loc)
        assert scale > 0
        self._kernel = lambda x, y: kernel((x - loc) / scale, (y - loc) / scale, **kw)
    
    def __call__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        np.broadcast(x, y)
        return self._kernel(x, y)
    
    def __add__(self, value):
        if isinstance(value, Kernel):
            return Kernel(lambda x, y: self._kernel(x, y) + value._kernel(x, y))
        elif np.isscalar(value):
            return Kernel(lambda x, y: self._kernel(x, y) + value)
        else:
            return NotImplemented
    
    __radd__ = __add__
    
    def __mul__(self, value):
        if isinstance(value, Kernel):
            return Kernel(lambda x, y: self._kernel(x, y) * value._kernel(x, y))
        elif np.isscalar(value):
            assert np.isfinite(value)
            assert value >= 0
            return Kernel(lambda x, y: value * self._kernel(x, y))
        else:
            return NotImplemented
    
    __rmul__ = __mul__
    
    def __pow__(self, value):
        if np.isscalar(value):
            assert np.isfinite(value)
            assert value >= 0
            return Kernel(lambda x, y: self._kernel(x, y) ** value)
        else:
            return NotImplemented

class IsotropicKernel(Kernel):
    
    def __init__(self, kernel, *, scale=1, **kw):
        super().__init__(lambda x, y: kernel(np.abs(x - y), **kw), scale=scale)
    
def makekernel(kernel, superclass):
    name = 'Specific' + superclass.__name__
    newclass = type(name, (superclass,), dict(
        __doc__=kernel.__doc__
    ))
    newclass.__init__ = lambda self, *args, **kw: super(newclass, self).__init__(kernel, *args, **kw)
    return newclass

isotropickernel = lambda kernel: makekernel(kernel, IsotropicKernel)
kernel = lambda kernel: makekernel(kernel, Kernel)

Constant = isotropickernel(lambda r: np.ones_like(r))
White = isotropickernel(lambda r: np.where(r == 0, 1, 0))
Linear = kernel(lambda x, y: x * y)
ExpQuad = isotropickernel(lambda r: np.exp(-1/2 * r ** 2))

@kernel
def Polynomial(x, y, exponent=None, sigma0=1):
    for p in exponent, sigma0:
        assert np.isscalar(p)
        assert p >= 0
    return (x * y + sigma0 ** 2) ** exponent
    
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
def GammaExp(r, gamma=1):
    """Gamma exponential"""
    assert np.isscalar(gamma)
    assert 0 < gamma <= 2
    return np.exp(-(r ** gamma))

@isotropickernel
def RatQuad(r, alpha=2):
    assert np.isscalar(alpha)
    assert alpha > 0
    return (1 + r ** 2 / (2 * alpha)) ** (-alpha)

@kernel
def NNKernel(x, y, sigma0=1):
    assert np.isscalar(sigma0)
    assert sigma0 > 0
    q = sigma0 ** 2
    return 2/np.pi * np.arcsin(2 * (q + x * y) / ((1 + 2 * (q + x**2)) * (1 + 2 * (q + y**2))))

@kernel
def Wiener(x, y):
    assert np.all(x >= 0)
    assert np.all(y >= 0)
    return np.minimum(x, y)

@kernel
def VarScale(x, y, scalefun=None):
    sx = scalefun(x)
    sy = scalefun(y)
    assert np.all(sx > 0)
    assert np.all(sy > 0)
    denom = sx ** 2 + sy ** 2
    factor = np.sqrt(2 * sx * sy / denom)
    return factor * np.exp(-(x - y) ** 2 / denom)

@isotropickernel
def Periodic(r, outerscale=1):
    assert np.isscalar(outerscale)
    assert outerscale > 0
    return np.exp(-2 * np.sin(r / 2) ** 2 / outerscale ** 2)
