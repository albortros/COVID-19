from ._GP import *
from ._kernels import *

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

Reference: Rasmussen et al. (2006), "Gaussian Processes for Machine Learning".

"""

# TODO
#
# Matern derivatives for half-integer nu
# stabilize Matern kernel near r == 0, then Matern derivatives for real nu
# (quick fix: larger eps in _softabs)
# delete the _x as soon as they are not needed any more
#
# Kronecker optimization: subclass GPKron where addx has a parameter `dim` and
# it accepts only non-structured arrays. Or, more flexible: make a class
# Lattice that is structured-array-like but different shapes for each field,
# and a field _kronok in Kernel update automatically when doing operations with
# kernels. Also, take a look at the pymc3 implementation.
#
# Structured arrays are not autograd-friendly. Convert _StructuredArrayWrap
# to a public class and accept it as input in GP. Make functions _isarray
# and _isdict to avoid confusion, also do not make _StructuredArrayWrap a
# subclass of dict. Modify _concatenate_noop to handle correctly
# _StructuredArrayWrap. Adapt explicit broadcasting in Kernel.__call__.
# Possible names: ArrayDict, StructuredArray. Unluckily I can't use
# gvar.BufferDict for this.
#
# sparse algorithms (after adding finite support kernels)
# DiagLowRank for low rank matrix + multiple of the identity (multiple rank-1
# updates to the Cholesky factor?)
# option to compute only the diagonal of the output covariance matrix
#
# Decomposition of the posterior covariance matrix, or tool to take samples.
# maybe a class for matrices?
#
# Fourier kernels. Look at Celerite's algorithms.
#
# kernel rescaling
# apply isotropic kernels to multivalued fields
# multidim support in Gibbs kernel
#
# New kernels:
# finite support
# fractional brownian motion
# is there a smooth version of the wiener process? like, softmin(x, y)?
# non-real input kernels (there are some examples in GPML)
# fixed covariance matrix over list of values, and version for two values only
# kernel with given basis functions
