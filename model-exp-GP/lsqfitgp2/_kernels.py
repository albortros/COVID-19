from __future__ import division

import autograd
from autograd import numpy as np
from autograd.scipy import special
from scipy import special as special_noderiv
from autograd import extend

class Kernel:
    """
    
    Base class for objects representing covariance kernels. A Kernel object
    is callable, the signature is obj(x, y). Kernel objects can be summed and
    multiplied between them and with scalars, or raised to power with a scalar
    exponent.
    
    This class can be used directly by passing a callable at initialization,
    or it can be subclassed. Subclasses need to assign the member `_kernel` with
    a callable which will be called when the Kernel object is called. `_kernel`
    will be called with two arguments x, y that are two broadcastable float
    numpy arrays. It must return Cov[f(x), f(y)] where `f` is the gaussian
    process.
    
    The decorator `@kernel` can be used to make quickly subclasses.
    
    """
    
    def __init__(self, kernel, *, loc=0, scale=1, **kw):
        """
        
        Initialize the object with callable `kernel`.
        
        Parameters
        ----------
        kernel : callable
            A function with signature f(x, y) where x and y are two
            broadcastable float numpy arrays which computes the covariance of
            f(x) with f(y).
        loc, scale : scalars
            The inputs to `kernel` are transformed as (x - loc) / scale.
        **kw :
            Other keyword arguments are passed to `kernel`: kernel(x, y, **kw).
        
        """
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
    """
    
    Subclass of `Kernel` that represents stationary kernels, i.e. the result
    only depends on x - y. The decorator for making subclasses is
    `@stationarykernel`.
    
    """
    
    def __init__(self, kernel, *, scale=1, **kw):
        """
        
        Parameters
        ----------
        kernel : callable
            A function taking one argument `r = x - y` (so it can be negative),
            plus optionally keyword arguments.
        scale : scalar
            The input to `kernel` is rescaled as `r / scale`.
        **kw :
            Other keyword arguments are passed to `kernel`.
        
        """
        super().__init__(lambda x, y: kernel(x - y, **kw), scale=scale)
    
def makekernel(kernel, superclass):
    supername = 'Specific' + superclass.__name__
    name = getattr(kernel, '__name__', supername)
    if name == '<lambda>':
        name = supername
    newclass = type(name, (superclass,), dict(
        __doc__=kernel.__doc__
    ))
    newclass.__init__ = lambda self, **kw: super(newclass, self).__init__(kernel, **kw)
    return newclass

def stationarykernel(kernel):
    """
    
    Decorator to convert a function to a subclass of `Kernel`. Use it like this:
    
    @stationarykernel
    def MyKernel(r, cippa=1, lippa=42, ...):
        return ... # something computing Cov[f(x), f(y)] where x - y = r
    
    """
    return makekernel(kernel, StationaryKernel)

def kernel(kernel):
    """
    
    Decorator to convert a function to a subclass of `Kernel`. Use it like this:
    
    @kernel
    def MyKernel(x, y, cippa=1, lippa=42, ...):
        return ... # something computing Cov[f(x), f(y)]
    
    """
    return makekernel(kernel, Kernel)

@stationarykernel
def Constant(r):
    """
    Kernel that returns a constant value, so all points are completely
    correlated. Thus it is equivalent to fitting with a horizontal line.
    """
    return np.ones_like(r)
    
@stationarykernel
def White(r):
    """
    Kernel that returns 1 when r == 0, zero otherwise, so it represents white
    noise.
    """
    return np.where(r == 0, 1, 0)

@kernel
def Linear(x, y):
    """
    Kernel which just returns x * y. It is equivalent to fitting with a line
    passing by the origin.
    """
    return x * y

@stationarykernel
def ExpQuad(r):
    """
    Gaussian kernel. It is very smooth, and has a strict typical lengthscale:
    under that the process does not oscillate, and over that it completely
    forgets other points.
    """
    return np.exp(-1/2 * r ** 2)

@kernel
def Polynomial(x, y, exponent=None, sigma0=1):
    """
    Kernel which is equivalent to fitting with a polynomial of degree
    `exponent`. The prior on the horizontal intercept has width `sigma0`.
    """
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
    """
    Matérn kernel of order `nu` > 0. The nearest integer below `nu` indicates
    how many times the gaussian process is derivable: so for `nu` < 1 it
    is continuous but not derivable, for 1 <= `nu` < 2 it is derivable but has
    not a decond derivative, etc. The half-integer case (nu = 1/2, 3/2, ...)
    uses internally a simpler formula so you should prefer it.
    """
    assert np.isscalar(nu)
    assert nu > 0
    x = np.sqrt(2 * nu) * _softabs(r)
    if (2 * nu) % 1 == 0 and nu >= 1/2:
        return _maternp(x, int(nu - 1/2))
    else:
        return 2 ** (1 - nu) / special.gamma(nu) * x ** nu * _kv(nu, x)

@stationarykernel
def Matern12(r):
    """
    Matérn kernel of order 1/2 (not derivable).
    """
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
    """
    Matérn kernel of order 3/2 (derivable one time).
    """
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
    """
    Matérn kernel of order 5/2 (derivable two times).
    """
    r = _softabs(r)
    return _matern52(np.sqrt(5) * r)

@stationarykernel
def GammaExp(r, gamma=1):
    """
    Return exp(-(r ** gamma)), with 0 < `gamma` <= 2. For `gamma` = 2 it is the
    gaussian kernel, for `gamma` = 1 it is the Matérn 1/2 kernel, for `gamma` =
    0 it is the constant kernel. The process is differentiable only for `gamma`
    = 2, however as `gamma` gets closer to 2 the "roughness" decreases.
    """
    r = _softabs(r)
    assert np.isscalar(gamma)
    assert 0 < gamma <= 2
    return np.exp(-(r ** gamma))

@stationarykernel
def RatQuad(r, alpha=2):
    """
    Rational quadratic kernel. It is equivalent to a lengthscale mixture of
    gaussian kernels where the scale distribution is a gamma with shape
    parameter `alpha`. For `alpha` -> infinity, it becomes the gaussian kernel.
    """
    assert np.isscalar(alpha)
    assert alpha > 0
    return (1 + r ** 2 / (2 * alpha)) ** (-alpha)

@kernel
def NNKernel(x, y, sigma0=1):
    """
    Kernel which is equivalent to a neural network with one infinite hidden
    layer with gaussian priors on the weights. In other words, you can think
    of the process as a superposition of sigmoids where `sigma0` sets the
    dispersion of the centers of the sigmoids.
    """
    assert np.isscalar(sigma0)
    assert sigma0 > 0
    q = sigma0 ** 2
    return 2/np.pi * np.arcsin(2 * (q + x * y) / ((1 + 2 * (q + x**2)) * (1 + 2 * (q + y**2))))

@kernel
def Wiener(x, y):
    """
    A kernel representing  non-differentiable random walk. It is defined only
    for x, y >= 0 (the starting point of the random walk).
    """
    assert np.all(x >= 0)
    assert np.all(y >= 0)
    return np.minimum(x, y)

@kernel
def Gibbs(x, y, scalefun=lambda x: 1):
    """
    Kernel which in some sense is like a gaussian kernel where the scale
    changes at every point. The scale is computed by the parameter `scalefun`
    which must be a callable taking the x array and returning a scale for each
    point. By default it returns a constant so it is a gaussian kernel.
    """
    sx = scalefun(x)
    sy = scalefun(y)
    assert np.all(sx > 0)
    assert np.all(sy > 0)
    denom = sx ** 2 + sy ** 2
    factor = np.sqrt(2 * sx * sy / denom)
    return factor * np.exp(-(x - y) ** 2 / denom)

@stationarykernel
def Periodic(r, outerscale=1):
    """
    A gaussian kernel over a transformed periodic space. It represents a
    periodic process. The usual `scale` parameter sets the period, with the
    default `scale` = 1 giving a period of 2π, while the `outerscale` parameter
    sets the scale of the gaussian kernel.
    """
    assert np.isscalar(outerscale)
    assert outerscale > 0
    return np.exp(-2 * (np.sin(r / 2) / outerscale) ** 2)
