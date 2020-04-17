from ._GP import *
from ._Kernel import *
from ._kernels import *
from ._array import *
from ._fit import *
from ._Deriv import *

__doc__ = """

Module to fit gaussian processes with gvar/lsqfit. It can both be used
standalone to fit data with a gaussian process only, and with lsqfit inside a
possibly nonlinear model with other parameters. In lsqfit style, all the
results will be properly correlated with prior, data, and other non-gaussian
process parameters in the fit, even when doing conditional prediction.

The main class is `GP`, which represents a gaussian process over arbitrary
input. It can be used both autonomously and with lsqfit. The inputs/outputs can
be arrays or dictionaries of arrays. It supports doing inference with the
derivatives of the process, using `autograd` to compute automatically
derivatives of the kernels. Indirectly, this can be used to make inference with
integrals.

The covariance kernels are represented by subclasses of class `Kernel`. There's
also `StationaryKernel` for covariance functions that depend only on the
difference of the arguments. Kernel objects can be summed, multiplied and
raised to a power.

To make a custom kernel, you can instantiate one of the two general classes by
passing them a function, or subclass them. For convenience, decorators are
provided to convert a function to a covariance kernel. Otherwise, use one of
the already available subclasses. Isotropic kernels are normalized to have unit
variance and roughly unit lengthscale.

    Constant :
        Equivalent to fitting with a constant.
    Linear :
        Equivalent to fitting with a line.
    Polynomial :
        Equivalent to fitting with a polynomial.
    ExpQuad :
        Gaussian kernel.
    White :
        White noise, each point is indipendent.
    Matern :
        Matérn kernel, you can set how many times it is differentiable.
    Matern12, Matern32, Matern52 :
        Matérn kernel for the specific cases nu = 1/2, 3/2, 5/2.
    GammaExp :
        Gamma exponential. Not differentiable, but you can set how close it is
        to being differentiable.
    RatQuad :
        Equivalent to a mixture of gaussian kernels with gamma-distributed
        length scales.
    NNKernel :
        Equivalent to training a neural network with one latent infinite layer.
    Wiener :
        Random walk.
    Gibbs :
        A gaussian kernel with a custom variable length scale.
    Periodic :
        A periodic gaussian kernel, represents a periodic function.
    Categorical :
        Arbitrary covariance matrix over a finite set of values.
    Cos:
        A cosine.
    FracBrownian :
        Fractional Brownian motion, like Wiener but with correlations.
    PPKernel :
        Finite support isotropic kernel.

Reference: Rasmussen et al. (2006), "Gaussian Processes for Machine Learning".

"""

# TODO
#
# stabilize Matern kernel near r == 0, then Matern derivatives for real nu
# (quick partial fix: larger eps in _softabs)
#
# Delete the _x as soon as they are not needed any more.
#
# Kronecker optimization: subclass GPKron where addx has a parameter `dim` and
# it accepts only non-structured arrays. Or, more flexible: make a class
# Lattice that is structured-array-like but different shapes for each field,
# and a field _kronok in Kernel update automatically when doing operations with
# kernels. Also, take a look at the pymc3 implementation. Can I use the
# kronecker optimization when the data covariance is non-null? -> Yes with a
# reasonable approximation of the marginal likelihood, but the data covariance
# must be diagonal. Other desiderata: separation along arbitrary subsets of
# the dimensions.
#
# Block matrix solving. Example: solve a subproblem with kronecker, another
# plain. Cache decompositions of blocks.
#
# A way to get the single contribution out of a sum of GPs. I think it is
# sufficient to use the usual formula with the full process for Kxx and the
# component I need for Kxsx and Kxsxs.
#
# sparse algorithms (after adding finite support kernels)
# DiagLowRank for low rank matrix + multiple of the identity (multiple rank-1
# updates to the Cholesky factor?)
#
# Option to compute only the diagonal of the output covariance matrix, and
# allow diagonal-only input covariance for data (will be fundamental for
# kronecker).
#
# Accept xarray.DataSet and pandas.DataFrame as inputs. Probably I can't use
# these as core formats due to autograd.
#
# Make everything opt-in except numpy. There's already a numpy submodule for
# doing this with scipy.linalg (I don't remember the name, it started with 'e').
# autograd can be handled by try-except ImportError and defining a variable
# has_autograd. With gvar maybe I can get through quickly if I define
# gvar.BufferDict = dict and other things NotImplemented. (Low priority).
#
# Decomposition of the posterior covariance matrix, or tool to take samples.
# Maybe a class for matrices? Example: prediction on kronecker data, the
# covariance matrix may be too large to fit in memory.
#
# Check that float32 is respected.
#
# Check that the gradient of the marginal likelihood works with derivatives.
#
# Fourier kernels. Look at Celerite's algorithms.
#
# Implement _array.broadcast and _array.broadcast_arrays that work both
# with np.ndarray and StructuredArray, and use them in _KernelBase.__call__,
# then make an example script with empbayes_fit on a GP with a derivative.
#
# Matrix transformation of inputs.
#
# Apply kernels over a subset of the fields, accepting a list for `dim` in
# Kernel.__init__.
#
# Support taking derivatives in arbitrarily nested dtypes.
#
# New kernels:
# is there a smooth version of the wiener process? like, softmin(x, y)?
# non-real input kernels (there are some examples in GPML)
