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
            
class StationaryKernel(Kernel):
    
    def __init__(self, kernel, *, scale=1, **kw):
        super().__init__(lambda x, y: kernel(x - y, **kw), scale=scale)
    
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

stationarykernel = lambda kernel: makekernel(kernel, StationaryKernel)
kernel = lambda kernel: makekernel(kernel, Kernel)

@stationarykernel
def Constant(r):
    return np.ones_like(r)
    
@stationarykernel
def White(r):
    return np.where(r == 0, 1, 0)

@kernel
def Linear(x, y):
    return x * y

@stationarykernel
def ExpQuad(r):
    return np.exp(-1/2 * r ** 2)

@kernel
def Polynomial(x, y, exponent=None, sigma0=1):
    for p in exponent, sigma0:
        assert np.isscalar(p)
        assert p >= 0
    return (x * y + sigma0 ** 2) ** exponent
    
def _softabs(x, eps=None):
    if not eps:
        eps = np.finfo(x.dtype).eps
    return np.sqrt(x ** 2 + eps ** 2)

# This still does not work with derivatives due to the pole of kv. I need a
# direct calculation of x ** nu * kv(nu, x).
_kvp = extend.primitive(special_noderiv.kvp)
extend.defvjp(
    _kvp,
    lambda ans, v, z, n: lambda g: g * _kvp(v, z, n + 1),
    argnums=[1]
)
# _kv = lambda v, z: _kvp(v, z, 0)
_kv = special_noderiv.kv

@extend.primitive # TODO define derivative of this, automatical is not stable
def _maternp(x, p):
    poly = 1
    for k in reversed(range(p)):
        c_kp1_over_ck = (p - k) / ((2 * p - k) * (k + 1))
        poly *= c_kp1_over_ck * 2 * x
        poly += 1
    return np.exp(-x) * poly

@stationarykernel
def Matern(r, nu=None):
    assert np.isscalar(nu)
    assert nu > 0
    x = np.sqrt(2 * nu) * _softabs(r)
    if (2 * nu) % 1 == 0 and nu >= 1/2:
        return _maternp(x, int(nu - 1/2))
    else:
        return 2 ** (1 - nu) / special.gamma(nu) * x ** nu * _kv(nu, x)

@stationarykernel
def Matern12(r):
    r = _softabs(r)
    return np.exp(-r)

@extend.primitive
def _matern32(x):
    return (1 + x) * np.exp(-x)

extend.defvjp(
    _matern32,
    lambda ans, x: lambda g: g * -x * np.exp(-x)
)

@stationarykernel
def Matern32(r):
    r = _softabs(r)
    return _matern32(np.sqrt(3) * r)

@extend.primitive
def _matern52(x):
    return (1 + x * (1 + x/3)) * np.exp(-x)

extend.defvjp(
    _matern52,
    lambda ans, x: lambda g: g * -x/3 * _matern32(x)
)

@stationarykernel
def Matern52(r):
    r = _softabs(r)
    return _matern52(np.sqrt(5) * r)

@stationarykernel
def GammaExp(r, gamma=1):
    r = _softabs(r)
    assert np.isscalar(gamma)
    assert 0 < gamma <= 2
    return np.exp(-(r ** gamma))

@stationarykernel
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

@stationarykernel
def Periodic(r, outerscale=1):
    assert np.isscalar(outerscale)
    assert outerscale > 0
    return np.exp(-2 * (np.sin(r / 2) / outerscale) ** 2)
