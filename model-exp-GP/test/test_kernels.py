from __future__ import division

import sys
import inspect

import numpy as np

sys.path = ['.'] + sys.path
from lsqfitgp2 import _kernels, _Kernel

# Make list of Kernel concrete subclasses.
kernels = []
for obj in _kernels.__dict__.values():
    if inspect.isclass(obj) and issubclass(obj, _Kernel.Kernel):
        if obj is _Kernel.Kernel or obj is _Kernel.IsotropicKernel:
            continue
        kernels.append(obj)

def test_normalized():
    """
    Check that isotropickernel(x, x) == 1.
    """
    kwargs = {
        _kernels.Matern: [
            dict(nu=0.5),
            dict(nu=0.6)
        ]
    }

    for kernel in kernels:
        if issubclass(kernel, _Kernel.IsotropicKernel):
            arglist = kwargs.get(kernel, [{}])
            for args in arglist:
                x = np.random.randn(100)
                result = kernel(**args)(x, x)
                assert np.allclose(result, 1)

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
    for p in range(2):
        nu = p + 1/2
        spec = eval('_kernels.Matern{}2'.format(1 + 2 * p))
        x, y = np.random.randn(2, 100)
        r1 = _kernels.Matern(nu=nu)(x, y)
        r2 = spec()(x, y)
        assert np.allclose(r1, r2)
