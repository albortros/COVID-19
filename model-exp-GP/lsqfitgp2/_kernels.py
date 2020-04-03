from __future__ import division

import autograd
from autograd import numpy as np
from autograd.scipy import special
from scipy import special as special_noderiv
from autograd import extend

def _apply2fields(transf, x):
    if x.dtype.names is not None:
        out = np.empty_like(x)
        for f in x.dtype.names:
            out[f] = transf(x[f])
        return out
    else:
        return transf(x)

class _StructuredArrayWrap(dict):
    pass

def _wrap_structured(x):
    wrap = _StructuredArrayWrap({
        name: x[name] for name in x.dtype.names
    })
    wrap.dtype = x.dtype
    wrap.shape = x.shape
    wrap.size = x.size
    return wrap

class Kernel:
    """
    
    Base class for objects representing covariance kernels. A Kernel object
    is callable, the signature is obj(x, y). Kernel objects can be summed and
    multiplied between them and with scalars, or raised to power with a scalar
    exponent.
    
    This class can be used directly by passing a callable at initialization, or
    it can be subclassed. Subclasses need to assign the member `_kernel` with a
    callable that will be called when the Kernel object is called. `_kernel`
    will be called with two arguments x, y that are two broadcastable numpy
    arrays. It must return Cov[f(x), f(y)] where `f` is the gaussian process.
    
    If `x` and `y` are structured arrays, they represent multidimensional
    input. Kernels can be specified to act only on a field of `x` and `y` or
    on all of them.
        
    The decorator `@kernel` can be used to quickly make subclasses.
    
    Methods
    -------
    diff :
        Derivatives of the kernel.
    
    """
    
    def __init__(self, kernel, *, dim=None, loc=0, scale=1, forcebroadcast=False, dtype=None, forcekron=False, **kw):
        """
        
        Initialize the object with callable `kernel`.
        
        Parameters
        ----------
        kernel : callable
            A function with signature `kernel(x, y)` where x and y are two
            broadcastable numpy arrays which computes the covariance of f(x)
            with f(y) where f is the gaussian process.
        dim : None or str
            When the input arrays are structured arrays, if dim=None the kernel
            will operate on all fields, i.e. it will be passed the whole
            arrays. If `dim` is a string, `kernel` will see only the arrays for
            the field `dim`. If `dim` is a string and the array is not
            structured, an exception is raised.
        loc, scale : scalars
            The inputs to `kernel` are transformed as (x - loc) / scale.
        forcebroadcast : bool
            If True, the inputs to `kernel` will always have the same shape.
        dtype : numpy data type
            If specified, the inputs to `kernel` will be coerced to that type.
        forcekron : bool
            If True, when calling `kernel`, if `x` and `y` are structured
            arrays, i.e. if they represent multidimensional input, `kernel` is
            invoked separately for each dimension, and the result is the
            product. Default False. If `dim` is specified, `forcekron` will
            have no effect.
        **kw :
            Other keyword arguments are passed to `kernel`: kernel(x, y, **kw).
        
        """
        assert isinstance(dim, (str, type(None)))
        assert np.isscalar(scale)
        assert np.isscalar(loc)
        assert np.isfinite(scale)
        assert np.isfinite(loc)
        assert scale > 0
        self._forcebroadcast = bool(forcebroadcast)
        self._dtype = None if dtype is None else np.dtype(dtype)
        forcekron = bool(forcekron)
        
        transf = lambda x: x
        
        if isinstance(dim, str):
            def transf(x):
                if x.dtype.names is not None:
                    return x[dim]
                else:
                    raise ValueError('kernel called on non-structured array but dim="{}"'.format(dim))
        
        if loc != 0:
            transf1 = transf
            transf = lambda x: _apply2fields(lambda x: x - loc, transf1(x))
        
        if scale != 1:
            transf2 = transf
            transf = lambda x: _apply2fields(lambda x: x / scale, transf2(x))
                
        if dim is None and forcekron:
            def _kernel(x, y):
                x = transf(x)
                y = transf(y)
                if x.dtype.names is not None:
                    return np.array([
                        kernel(x[f], y[f], **kw)
                        for f in x.dtype.names
                    ])
                else:
                    return kernel(x, y, **kw)
        else:
            _kernel = lambda x, y: kernel(transf(x), transf(y), **kw)
        
        self._kernel = _kernel
    
    def __call__(self, x, y):
        x = np.array(x, copy=False, dtype=self._dtype)
        y = np.array(y, copy=False, dtype=self._dtype)
        assert x.dtype == y.dtype
        shape = np.broadcast(x, y).shape
        if self._forcebroadcast:
            x, y = np.broadcast_arrays(x, y)
        result = self._kernel(x, y)
        assert isinstance(result, np.ndarray) # TODO use autograd isinstance?
        assert np.issubdtype(result.dtype, np.number)
        assert result.shape == shape
        return result
    
    def __add__(self, value):
        if isinstance(value, Kernel):
            return Kernel(lambda x, y: self._kernel(x, y) + value._kernel(x, y))
        elif np.isscalar(value):
            assert np.isfinite(value)
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
    
    def diff(self, xorder=0, yorder=0, xdim=None, ydim=None):
        """
        
        Return a Kernel object that computes the derivatives of this kernel.
        The derivatives are computed automatically with `autograd`. If `xorder`
        and `yorder` are 0, this is a no-op and returns the object itself.
        
        Parameters
        ----------
        xorder, yorder : int
            How many times the kernel is derived w.r.t the first and second
            arguments respectively.
        xdim, ydim : None or str
            When the inputs are structured arrays, indicate which field to
            derivate.
        
        Returns
        -------
        diffkernel : Kernel
            Another Kernel object representing the derivatives of this one.
        """
        for order in xorder, yorder:
            if not isinstance(order, (int, np.integer)) or order < 0:
                raise ValueError('derivative orders must be nonnegative integers')
        for dim in xdim, ydim:
            assert isinstance(dim, (str, type(None)))
        
        if xorder == yorder == 0:
            return self
        
        kernel = self._kernel
        def fun(x, y):
            if x.dtype.names is not None:
                for order, dim, z in zip((xorder, yorder), (xdim, ydim), (x, y)):
                    if order and dim is None:
                        raise ValueError('can not differentiate w.r.t structured input ({}) if dim not specified'.format(', '.join(z.dtype.names)))
                    if order and dim not in z.dtype.names:
                        raise ValueError('differentiation dimension `{}` missing in fields ({})'.format(dim, ', '.join(z.dtype.names)))
                if xorder:
                    x = _wrap_structured(x)
                if yorder:
                    y = _wrap_structured(y)
                def f(a, b):
                    if xorder:
                        x[xdim] = a
                    if yorder:
                        y[ydim] = b
                    return kernel(x, y)
            else:
                f = kernel
            
            for _ in range(xorder):
                f = autograd.elementwise_grad(f, 0)
            for _ in range(yorder):
                f = autograd.elementwise_grad(f, 1)
            
            return f(x, y)
        
        return Kernel(fun, forcebroadcast=True, dtype=float)
            
class IsotropicKernel(Kernel):
    """
    
    Subclass of `Kernel` that represents isotropic kernels, i.e. the result
    only depends on a distance defined between points. The decorator for
    making subclasses is `isotropickernel`.
    
    """
    
    def __init__(self, kernel, *, input='squared', **kw):
        """
        
        Parameters
        ----------
        kernel : callable
            A function taking one argument `r2` which is the squared distance
            between x and y, plus optionally keyword arguments. `r2` is a 1D
            numpy array.
        input : str
            See "input options" below.
        **kw :
            Other keyword arguments are passed to the `Kernel` init.
        
        Input options
        -------------
        squared :
            Pass the squared distance (default).
        soft :
            Pass the distance, but instead of 0 it yields a very small number.
        
        """
        allowed_input = ('squared', 'soft')
        if not (input in allowed_input):
            raise ValueError('input option `{}` not valid, must be one of {}'.format(input, allowed_input))
        
        def function(x, y, **kwargs):
            if x.dtype.names is not None:
                q = sum((x[f] - y[f]) ** 2 for f in x.dtype.names)
            else:
                q = (x - y) ** 2
            if input == 'soft':
                eps = np.finfo(x.dtype).eps
                q = np.sqrt(q + eps ** 2)
            return kernel(q, **kwargs)
        
        super().__init__(function, **kw)
    
def _makekernelsubclass(kernel, superclass, **prekw):
    assert issubclass(superclass, Kernel)
    
    supername = 'Specific' + superclass.__name__
    name = getattr(kernel, '__name__', supername)
    if name == '<lambda>':
        name = supername
    
    newclass = type(name, (superclass,), {})
    
    def __init__(self, **kw):
        kwargs = prekw.copy()
        kwargs.update(kw)
        return super(newclass, self).__init__(kernel, **kwargs)
    newclass.__init__ = __init__
    newclass.__doc__ = kernel.__doc__
    
    return newclass

def _kerneldecoratorimpl(cls, *args, **kw):
    functional = lambda kernel: _makekernelsubclass(kernel, cls, **kw)
    if len(args) == 0:
        return functional
    elif len(args) == 1:
        return functional(*args)
    else:
        raise ValueError(len(args))

def kernel(*args, **kw):
    """
    
    Decorator to convert a function to a subclass of `Kernel`. Use it like this:
    
    @kernel
    def MyKernel(x, y, cippa=1, lippa=42, ...):
        return ... # something computing Cov[f(x), f(y)]
    
    """
    return _kerneldecoratorimpl(Kernel, *args, **kw)

def isotropickernel(*args, **kw):
    """
    
    Decorator to convert a function to a subclass of `IsotropicKernel`. Use it
    like this:
    
    @isotropickernel
    def MyKernel(rsquared, cippa=1, lippa=42, ...):
        return ...
        # something computing Cov[f(x), f(y)] where rsquared = ||x - y||^2
    
    """
    return _kerneldecoratorimpl(IsotropicKernel, *args, **kw)

@isotropickernel
def Constant(r2):
    """
    Kernel that returns a constant value, so all points are completely
    correlated. Thus it is equivalent to fitting with a horizontal line.
    """
    return np.ones_like(r2)
    
@isotropickernel
def White(r2):
    """
    Kernel that returns 1 when x == y, zero otherwise, so it represents white
    noise.
    """
    return np.where(r2 == 0, 1, 0)

@isotropickernel
def ExpQuad(r2):
    """
    Gaussian kernel. It is very smooth, and has a strict typical lengthscale:
    under that the process does not oscillate, and over that it completely
    forgets other points.
    """
    return np.exp(-1/2 * r2)

def _dot(x, y):
    if x.dtype.names is not None:
        return sum(x[f] * y[f] for f in x.dtype.names)
    else:
        return x * y

@kernel
def Linear(x, y):
    """
    Kernel which just returns x * y. It is equivalent to fitting with a
    line/plane passing by the origin.
    """
    return _dot(x, y)

@kernel
def Polynomial(x, y, exponent=None, sigma0=1):
    """
    Kernel which is equivalent to fitting with a polynomial of degree
    `exponent`. The prior on the horizontal intercept has width `sigma0`.
    """
    assert np.isscalar(exponent)
    assert esponent >= 0
    assert np.isscalar(sigma0)
    assert sigma0 >= 0
    return (_dot(x, y) + sigma0 ** 2) ** exponent
    
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

@isotropickernel(input='soft')
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
    x = np.sqrt(2 * nu) * r
    if (2 * nu) % 1 == 0 and nu >= 1/2:
        return _maternp(x, int(nu - 1/2))
    else:
        return 2 ** (1 - nu) / special.gamma(nu) * x ** nu * _kv(nu, x)

@isotropickernel(input='soft')
def Matern12(r):
    """
    Matérn kernel of order 1/2 (not derivable).
    """
    return np.exp(-r)

@extend.primitive
def _matern32(x):
    return (1 + x) * np.exp(-x)

extend.defvjp(
    _matern32,
    lambda ans, x: lambda g: g * -x * np.exp(-x)
)

@isotropickernel(input='soft')
def Matern32(r):
    """
    Matérn kernel of order 3/2 (derivable one time).
    """
    return _matern32(np.sqrt(3) * r)

@extend.primitive
def _matern52(x):
    return (1 + x * (1 + x/3)) * np.exp(-x)

extend.defvjp(
    _matern52,
    lambda ans, x: lambda g: g * -x/3 * _matern32(x)
)

@isotropickernel(input='soft')
def Matern52(r):
    """
    Matérn kernel of order 5/2 (derivable two times).
    """
    return _matern52(np.sqrt(5) * r)

@isotropickernel(input='soft')
def GammaExp(r, gamma=1):
    """
    Return exp(-(r ** gamma)), with 0 < `gamma` <= 2. For `gamma` = 2 it is the
    gaussian kernel, for `gamma` = 1 it is the Matérn 1/2 kernel, for `gamma` =
    0 it is the constant kernel. The process is differentiable only for `gamma`
    = 2, however as `gamma` gets closer to 2 the "roughness" decreases.
    """
    assert np.isscalar(gamma)
    assert 0 < gamma <= 2
    return np.exp(-(r ** gamma))

@isotropickernel
def RatQuad(r2, alpha=2):
    """
    Rational quadratic kernel. It is equivalent to a lengthscale mixture of
    gaussian kernels where the scale distribution is a gamma with shape
    parameter `alpha`. For `alpha` -> infinity, it becomes the gaussian kernel.
    """
    assert np.isscalar(alpha)
    assert alpha > 0
    return (1 + r2 / (2 * alpha)) ** (-alpha)

@kernel
def NNKernel(x, y, sigma0=1):
    """
    Kernel which is equivalent to a neural network with one infinite hidden
    layer with gaussian priors on the weights. In other words, you can think
    of the process as a superposition of sigmoids where `sigma0` sets the
    dispersion of the centers of the sigmoids.
    """
    assert np.isscalar(sigma0)
    assert np.isfinite(sigma0)
    assert sigma0 > 0
    q = sigma0 ** 2
    denom = (1 + 2 * (q + _dot(x, x))) * (1 + 2 * (q + _dot(y, y)))
    return 2/np.pi * np.arcsin(2 * (q + _dot(x, y)) / denom)

@kernel(forcekron=True)
def Wiener(x, y):
    """
    A kernel representing  non-differentiable random walk. It is defined only
    for x, y >= 0 (the starting point of the random walk).
    """
    assert np.all(x >= 0)
    assert np.all(y >= 0)
    return np.minimum(x, y)

@kernel(forcekron=True)
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

@isotropickernel(input='soft', forcekron=True)
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
