from __future__ import division

import gvar
import autograd
from autograd import numpy as np
from autograd.scipy import linalg, special
import collections
import itertools

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
        
        if isinstance(key, tuple):
            raise TypeError('key can not be tuple')
        
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
        slices = dict()
        # slices: (key, derivative order) -> slice
        i = 0
        for key, d in self._x.items():
            for deriv, l in d.items():
                length = sum(map(len, l))
                slices[key, deriv] = slice(i, i + length)
                i += length
        return slices, i
    
    def _covfunderiv(self, xderiv, yderiv):
        assert isinstance(xderiv, int)
        assert isinstance(yderiv, int)
        fun = self._covfun
        for _ in range(xderiv):
            fun = autograd.elementwise_grad(fun, 0)
        for _ in range(yderiv):
            fun = autograd.elementwise_grad(fun, 1)
        return fun
    
    def _buildcov(self):
        if not self._x:
            raise ValueError('process is empty, add values with `addx`')
        self._canaddx = False
        
        self._slices, length = self._makeslices()
        cov = np.empty((length, length))
        for kdkd in itertools.product(self._slices, repeat=2):
            xy = [
                np.concatenate(self._x[key][deriv])
                for key, deriv in kdkd
            ]
            xy[0] = xy[0].reshape(-1, 1) * np.ones(len(xy[0])).reshape(1, -1)
            xy[1] = xy[1].reshape(1, -1) * np.ones(len(xy[1])).reshape(-1, 1)
            assert len(xy) == 2
            kernel = self._covfunderiv(kdkd[0][1], kdkd[1][1])
            slices = [self._slices[k, d] for k, d in kdkd]
            cov[slices[0], slices[1]] = kernel(xy[0], xy[1])
        
        # check covariance matrix is positive definite
        if not np.allclose(cov, cov.T):
            raise ValueError('covariance matrix is not symmetric')
        eigv = linalg.eigvalsh(cov)
        mineigv = np.min(eigv)
        if mineigv < 0:
            if mineigv < -len(cov) * np.finfo(float).eps * np.max(eigv):
                raise ValueError('covariance matrix is not positive definite')
            cov[np.diag_indices(len(cov))] += -mineigv
            # this is a fast but strong regularization, maybe we could just do
            # an svd cut
        # since we are diagonalizing, maybe save the transformation and apply
        # it so that lsqfit does not diagonalize it again for the fit.
        
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
            flatprior.flags['WRITEABLE'] = False
            self._priordict = gvar.BufferDict({
                (key, deriv): flatprior[s]
                for (key, deriv), s in self._slices.items()
            })
        return self._priordict
    
    def _checkkeyderiv(self, key, deriv):
        # this method not to be used by addx, and not to check keys in
        # dictionaries
        if not (key is None):
            if isinstance(key, tuple):
                raise TypeError('key can not be tuple')
            if None in self._x:
                raise ValueError('you have given key but x is array')
            if not key in self._x:
                raise KeyError(key)
        
        if not (deriv is None):
            deriv = self._checkderiv(deriv)
            if key is None:
                for k, d in self._slices:
                    if deriv == d:
                        break
                else:
                    raise ValueError("there's no derivative {} in process".format(deriv))
            elif not (deriv in self._x[key]):
                raise ValueError('no derivative {} for key {}'.format(deriv, key))
        
        return key, deriv
    
    def _getkeyderivlist(self, key, deriv):
        if key is None and deriv is None:
            return list(self._slices)
        elif key is None and not (deriv is None):
            return [(k, deriv) for k in self._x if deriv in self._x[k]]
        elif not (key is None) and deriv is None:
            return [(key, d) for d in self._x[key]]
        elif not (key is None) and not (deriv is None):
            return [(key, deriv)]
        assert False
    
    def _stripkeyderiv(self, kdlist, key, deriv, strip0):
        if strip0 is None:
            strip0 = deriv is None
        strip0 = bool(strip0)
        
        if None in self._x or not (key is None) and deriv is None:
            outlist = [d for _, d in kdlist]
            return [] if outlist == [0] and strip0 else outlist
        if not (key is None) and not (deriv is None):
            return []
        if key is None and not (deriv is None):
            return [k for k, _ in kdlist]
        if key is None and deriv is None:
            return [(k, d) if d else k for k, d in kdlist] if strip0 else kdlist
        assert False

    def prior(self, key=None, deriv=None, strip0=None):
        self._prior
        key, deriv = self._checkkeyderiv(key, deriv)
        kdlist = self._getkeyderivlist(key, deriv)
        assert kdlist
        strippedkd = self._stripkeyderiv(kdlist, key, deriv, strip0)
        
        if strippedkd:
            return gvar.BufferDict({
                strippedkd[i]: self._prior[kdlist[i]]
                for i in range(len(kdlist))
            })
        else:
            assert len(kdlist) == 1
            return self._prior[kdlist[0]]
        
    def _flatgiven(self, given):
        if isinstance(given, (list, np.ndarray)):
            if None in self._x and len(self._x[None]) == 1:
                given = {(None, *self._x[None]): given}
            else:
                raise ValueError('`given` is an array but x has keys and/or multiple derivatives, provide a dictionary')
            
        elif not isinstance(given, (dict, gvar.BufferDict)):
            raise TypeError('`given` must be array or dict')
        
        ylist = []
        yslices = []
        kdlist = []
        for k, l in given.items():
            if isinstance(k, tuple):
                if len(k) != 2:
                    raise ValueError('key `{}` from `given` is a tuple but has not length 2')
                key, deriv = k
            elif k is None:
                raise KeyError('None key in `given` not allowed')
            elif None in self._x:
                key = None
                deriv = k
            else:
                key = k
                deriv = 0
                
            if not (key in self._x):
                raise KeyError(key)
            if deriv != int(deriv) or int(deriv) < 0:
                raise ValueError('supposed deriv order `{}` is not a nonnegative integer'.format(key, deriv, deriv))
            if not (deriv in self._x[key]):
                raise KeyError('derivative {} for key {} missing'.format(key, deriv))

            if not isinstance(l, (list, np.ndarray)):
                raise TypeError('element `given[{}]` is not list or array'.format(k))
            s = self._slices[key, deriv]
            xlen = s.stop - s.start
            if len(l) != xlen:
                raise ValueError('`given[{}]` has length {} different from x length {}'.format(k, len(l), xlen))
        
            l = np.asarray(l)
            if not len(l.shape) == 1 and len(l) >= 1:
                raise ValueError('`given[{}]` is not 1D nonempty array'.formay(k))
            
            ylist.append(l)
            yslices.append(self._slices[key, deriv])
            kdlist.append((key, deriv))
            
        return ylist, yslices, kdlist
    
    def _compatslices(self, sliceslist):
        i = 0
        out = []
        for s in sliceslist:
            length = s.stop - s.start
            out.append(slice(i, i + length))
            i += length
        return out
    
    def predfromfit(self, given, key=None, deriv=None, strip0=None):
        key, deriv = self._checkkeyderiv(key, deriv)
        kdlist = self._getkeyderivlist(key, deriv)
        assert kdlist
        strippedkd = self._stripkeyderiv(kdlist, key, deriv, strip0)
        assert strippedkd or len(kdlist) == 1
        
        ylist, yslices, inkdl = self._flatgiven(given)
        cyslices = self._compatslices(yslices)
        yplist = [self._prior[kd] for kd in inkdl]
        
        yspslices = [self._slices[kd] for kd in kdlist]
        cyspslices = self._compatslices(yspslices)
        ysplist = [self._prior[kd] for kd in kdlist]
        
        y = np.concatenate(ylist)
        yp = np.concatenate(yplist)
        ysp = np.concatenate(ysplist)
        
        Kxsx = np.nan * np.empty((len(ysp), len(yp)))
        for ss, css in zip(yspslices, cyspslices):
            for s, cs in zip(yslices, cyslices):
                Kxsx[css, cs] = self._cov[ss, s]
        
        Kxx = np.nan * np.empty((len(yp), len(yp)))
        for s1, cs1 in zip(yslices, cyslices):
            for s2, cs2 in zip(yslices, cyslices):
                Kxx[cs1, cs2] = self._cov[s1, s2]
        assert np.allclose(Kxx, Kxx.T)
        
        flatout = Kxsx @ gvar.linalg.solve(Kxx, y - yp) + ysp
        
        if strippedkd:
            return gvar.BufferDict({
                strippedkd[i]: flatout[self._slices[kdlist[i]]]
                for i in range(len(kdlist))
            })
        else:
            return flatout
    
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
        x = np.array(x, copy=False, dtype=float)
        y = np.array(y, copy=False, dtype=float)
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
            
def softabs(x):
    a = np.finfo(float).eps
    return np.sqrt(x ** 2 + a ** 2)

class IsotropicKernel(Kernel):
    
    def __init__(self, kernel, *, scale=1, **kw):
        super().__init__(lambda x, y: kernel(softabs(x - y), **kw), scale=scale)
    
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
