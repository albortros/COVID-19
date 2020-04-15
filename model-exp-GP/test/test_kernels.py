from __future__ import division

import sys
import inspect

import numpy as np
import autograd
from scipy import linalg

sys.path = ['.'] + sys.path
from lsqfitgp2 import _kernels, _Kernel

# Make list of Kernel concrete subclasses.
kernels = []
for obj in _kernels.__dict__.values():
    if inspect.isclass(obj) and issubclass(obj, _Kernel.Kernel):
        if obj is _Kernel.Kernel or obj is _Kernel.IsotropicKernel:
            continue
        kernels.append(obj)

class KernelTestBase:
    """
    Abstract base class to test kernels. Each subclass tests one specific
    kernel.
    """
    
    @property
    def kernel_class(self):
        raise NotImplementedError()
    
    @property
    def kwargs_list(self):
        return [dict()]
    
    def random_x(self, **kw):
        return np.random.uniform(-5, 5, size=100)
    
    def checkpos_cov(self, cov):
        eigv = linalg.eigvalsh(cov)
        assert np.min(eigv) > -len(cov) * np.finfo(float).eps * np.max(eigv)

    def checkpos_kernel(self, kw):
        x = self.random_x(**kw)
        cov = self.kernel_class(**kw)(x[None, :], x[:, None])
        self.checkpos_cov(cov)

    def test_positive(self):
        for kw in self.kwargs_list:
            self.checkpos_kernel(kw)
    
    def check_unit_variance(self, kw):
        x = self.random_x(**kw)
        var = self.kernel_class(**kw)(x, x)
        assert np.allclose(var, 1)
    
    def test_normalized(self):
        if issubclass(self.kernel_class, _Kernel.IsotropicKernel):
            for kw in self.kwargs_list:
                self.check_unit_variance(kw)
    
    @classmethod
    def make_subclass(cls, kernel_class, kwargs_list=None, random_x_fun=None):
        name = 'Test' + kernel_class.__name__
        subclass = type(cls)(name, (cls,), {})
        subclass.kernel_class = property(lambda self: kernel_class)
        if kwargs_list is not None:
            subclass.kwargs_list = property(lambda self: kwargs_list)
        if random_x_fun is not None:
            subclass.random_x = lambda self, **kw: random_x_fun(**kw)
        return subclass

def matrix_square(A):
    return A.T @ A

def random_nd(size, ndim):
    out = np.empty(size, dtype=[('xyz', float, (ndim,))])
    out['xyz'] = np.random.uniform(-5, 5, size=(size, ndim))
    return out
        
# Define concrete subclasses of PosCheckBase for all kernels.
specialized = {
    _kernels.Matern: dict(kwargs_list=[
        dict(nu=0.5), dict(nu=0.6), dict(nu=1.5), dict(nu=2.5)
    ]),
    _kernels.PPKernel: dict(kwargs_list=[
        dict(q=q, D=D) for q in range(4) for D in range(1, 10)
    ], random_x_fun=lambda **kw: random_nd(100, kw['D'])),
    _kernels.Polynomial: dict(kwargs_list=[
        dict(exponent=p) for p in range(10)
    ]),
    _kernels.Wiener: dict(random_x_fun=lambda **kw: np.random.uniform(0, 10, size=100)),
    _kernels.FracBrownian: dict(random_x_fun=lambda **kw: np.random.uniform(0, 10, size=100)),
    _kernels.Categorical: dict(kwargs_list=[
        dict(cov=matrix_square(np.random.randn(10, 10)))
    ], random_x_fun=lambda **kw: np.random.randint(10, size=100))
}
for kernel in kernels:
    factory_kw = specialized.get(kernel, {})
    newclass = KernelTestBase.make_subclass(kernel, **factory_kw)
    exec('{} = newclass'.format(newclass.__name__))

def test_matern_half_integer():
    """
    Check that the formula for half integer nu gives the same result of the
    formula for real nu.
    """
    for p in range(10):
        nu = p + 1/2
        assert nu - 1/2 == p
        nualt = nu * (1 + 4 * np.finfo(float).eps)
        assert nualt > nu
        x, y = 3 * np.random.randn(2, 100)
        r1 = _kernels.Matern(nu=nu)(x, y)
        r2 = _kernels.Matern(nu=nualt)(x, y)
        assert np.allclose(r1, r2)

def test_matern_spec():
    """
    Test implementations of specific cases of nu.
    """
    for p in range(3):
        nu = p + 1/2
        spec = eval('_kernels.Matern{}2'.format(1 + 2 * p))
        x, y = np.random.randn(2, 100)
        r1 = _kernels.Matern(nu=nu)(x, y)
        r2 = spec()(x, y)
        assert np.allclose(r1, r2)

def test_matern_deriv_spec():
    """
    Test derivatives for nu = 1/2, 3/2, 5/2.
    """
    for p in range(3):
        nu = p + 1/2
        spec = eval('_kernels.Matern{}2'.format(1 + 2 * p))
        x, y = np.random.randn(2, 100)
        r1 = autograd.elementwise_grad(_kernels.Matern(nu=nu))(x, y)
        r2 = autograd.elementwise_grad(spec())(x, y)
        assert np.allclose(r1, r2)
