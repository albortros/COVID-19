from __future__ import division

import autograd
from autograd import numpy as np
from autograd.scipy import special
from scipy import special as special_noderiv
from autograd import extend

class Kernel:
    
    def __init__(self, kernel, *, scale=1, loc=0, **kw):
        assert np.isscalar(scale)
        assert np.isscalar(loc)
        assert np.isfinite(scale)
        assert np.isfinite(loc)
        assert scale > 0
        transf = lambda x: (x - loc) / scale
        self._kernel = lambda x, y: kernel(transf(x), transf(y), **kw)
    
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
    a = np.finfo(x.dtype).eps
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

kv = extend.primitive(special_noderiv.kv)
    
@isotropickernel
def Matern(r, nu=None):
    assert np.isscalar(nu)
    x = np.sqrt(2 * nu) * r
    xpos = x > 0
    out = np.ones_like(x, dtype=float)
    out[xpos] = 2 ** (1 - nu) / special.gamma(nu) * x[xpos] ** nu * kv(nu, x[xpos])
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
def Gibbs(x, y, scalefun=lambda x: 1):
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
    return np.exp(-2 * (np.sin(r / 2) / outerscale) ** 2)
