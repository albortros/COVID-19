from __future__ import division

import gvar
import autograd
from autograd import numpy as np
from autograd.scipy import linalg, special
import collections

__doc__ = """Module to fit gaussian processes with lsqfit."""

class GP:
    
    def __init__(self, covfun):
        if not isinstance(covfun, Kernel):
            raise TypeError('covariance function must be of class Kernel')
        self._covfun = covfun
        self._x = collections.defaultdict(lambda: collections.defaultdict(list))
        # self._x: label -> (derivative order -> list of arrays)
        self._canaddx = True
    
    def _checkderiv(self, deriv):
        if deriv != int(deriv):
            raise ValueError('derivative order {} is not an integer'.format(deriv))
        deriv = int(deriv)
        if deriv < 0:
            raise ValueError('derivative order must be >= 0')
        return deriv
    
    def addx(self, x, key=None, deriv=0):
        if not self._canaddx:
            raise RuntimeError('can not add x any more to this process because it has been used')
        
        deriv = self._checkderiv(deriv)
        
        if isinstance(x, (list, np.ndarray)):
            if None in self._x and not (key is None):
                raise ValueError("previous x is array, can't add key")
            if key is None and (len(self._x) >= 2 or self._x and not (None in self._x)):
                raise ValueError("previous x is dictionary, can't append array")
                
            x = {key: x}
                    
        elif isinstance(x, (dict, gvar.BufferDict)):
            if not (key is None):
                raise ValueError('can not specify key if x is a dictionary')
            if None in x:
                raise ValueError('`None` key in x not allowed')
            if len(self._x) == 1 and None in self._x:
                raise ValueError("previous x is array, can't append dictionary")
        
        else:
            raise TypeError('x must be array or dict')
            
        for key in x:
            gx = x[key]
            if not isinstance(gx, (list, np.ndarray)):
                raise TypeError('object for key `{}` in x is not array or list'.format(key))
            
            gx = np.asarray(gx)
            if len(gx.shape) != 1 or len(gx) == 0:
                raise ValueError('array for key `{}` in x is not 1D and nonempty'.format(key))
            
            self._x[key][deriv].append(gx)
    
    # def _totlenx(self):
    #     return sum(sum(sum(map(len, l)) for l in d.values()) for d in self._x.values())
    
    def _makeslices(self):
        slices = collections.defaultdict(dict)
        xlist = []
        # slices: label -> (derivative order -> slice)
        i = 0
        for key, d in self._x.items():
            for deriv, l in d.items():
                length = sum(map(len, l))
                slices[key][deriv] = slice(i, i + length)
                i += length
                xlist += l
        return slices, xlist
    
    def _buildcov(self):
        assert self._x
        self._canaddx = False
        
        # this must be changed because I have to derivate kernel function for
        # derivatives
        self._slices, xlist = self._makeslices()
        x = np.concatenate(xlist)
        
        # build covariance matrix and check it is positive definite
        cov = self._covfun(x.reshape(-1, 1), x.reshape(1, -1))
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
        return cov
    
    @property 
    def _cov(self):
        if not hasattr(self, '_covmatrix'):
            self._covmatrix = self._buildcov()
            self._covmatrix.flags['WRITEABLE'] = False
        return self._covmatrix
    
    @property
    def _prior(self):
        # use gvar.BufferDict instead of dict, otherwise pickling is a mess
        if not hasattr(self, '_priordict'):
            flatprior = gvar.gvar(np.zeros(len(self._cov)), self._cov)
            self._priordict = gvar.BufferDict({
                (key, deriv): flatprior[s]
                for key, d in self._slices.items()
                for deriv, s in d.items()
            })
        return self._priordict

    def prior(self, key=None, deriv=None, stripderiv0=None):
        if not (deriv is None):
            deriv = self._checkderiv(deriv)
        
        if stripderiv0 is None:
            stripderiv0 = deriv is None
        stripderiv0 = bool(stripderiv0)
        
        if not (key is None):
            if None in self._x:
                raise ValueError('you have given key but x is array')
            if not key in self._x:
                raise KeyError(key)
        
        if key is None and deriv is None:
            if None in self._x:
                if len(self._x[None]) == 1 and 0 in self._x[None] and stripderiv0:
                    return self._prior[None, 0]
                else:
                    return gvar.BufferDict({
                        deriv: self._prior[None, deriv]
                        for deriv in self._x[None]
                    })
            elif stripderiv0:
                return gvar.BufferDict({
                    (key, deriv) if deriv else key: obj
                    for (key, deriv), obj in self._prior.items()
                })
            else:
                return self._prior
                
        elif key is None and not (deriv is None):
            if None in self._x:
                return self._prior[None, deriv]
            else:
                return gvar.BufferDict({
                    key: self._prior[key, deriv]
                    for key in self._x
                })
        
        elif not (key is None) and deriv is None:
            if len(self._x[key]) == 1 and 0 in self._x[key] and stripderiv0:
                return self._prior[key, 0]
            else:
                return gvar.BufferDict({
                    deriv: self._prior[key, deriv]
                    for deriv in self._x[key]
                })
            
        elif not (key is None) and not (deriv is None):
            return self._prior[key, deriv]
            
        else:
            raise 'wtf??'
    
    def predprior(self):
        return self._prior[self._datarange:]
    
    def pred(self, fxdata):
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
    supername = 'Specific' + superclass.__name__
    name = getattr(kernel, '__name__', supername)
    if name == '<lambda>':
        name = supername
    newclass = type(name, (superclass,), dict(
        __doc__=kernel.__doc__
    ))
    newclass.__init__ = lambda self, *args, **kw: super(newclass, self).__init__(kernel, *args, **kw)
    return newclass

isotropickernel = lambda kernel: makekernel(kernel, IsotropicKernel)
kernel = lambda kernel: makekernel(kernel, Kernel)

@isotropickernel
def Constant(r):
    return np.ones_like(r)
    
@isotropickernel
def White(r):
    return np.where(r == 0, 1, 0)

@kernel
def Linear(x, y):
    return x * y

@isotropickernel
def ExpQuad(r):
    return np.exp(-1/2 * r ** 2)

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

@isotropickernel
def Matern12(r):
    return np.exp(-r)

@isotropickernel
def Matern32(r):
    return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)

@isotropickernel
def Matern52(r):
    return (1 + np.sqrt(5) * r + 5/3 * r**3) * np.exp(-np.sqrt(5) * r)

@isotropickernel
def GammaExp(r, gamma=1):
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
def Gibbs(x, y, scalefun=None):
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
