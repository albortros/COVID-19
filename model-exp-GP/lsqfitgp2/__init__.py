from ._GP import *
from ._kernels import *

__doc__ = """

Module to fit gaussian processes with gvar/lsqfit. It can both be used
standalone to fit data with a gaussian process only, and with lsqfit inside a
possibly nonlinear model with other parameters. In lsqfit style, all the
results will be properly correlated with prior, data, and other non-gaussian
process parameters in the fit, even when doing conditional prediction.

The main class is `GP`, which represents a gaussian process over 1D real input.
It can be used both autonomously and with lsqfit. The inputs/outputs can be
arrays or dictionaries of arrays. It supports doing inference with the
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
the already available subclasses. They are normalized to have unit variance
and roughly unit lengthscale.

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
# Multidimensional input, both multidim and structured arrays. The first
# axis is the dimesion axis: this way caching makes sense with separated
# kernels, and it works also in exactly the same way with structured arrays
# by using strings instead of numbers. What dimension the kernel operates on
# is specified by the `dim` keyword at construction, dim=None means all
# dimensions and can accept 1D x without trying to extract something (I want
# to avoid having to use 2D x in any case like pymc3). On its part, GP will
# look at the shape/dtype of the first x it receives and enforce it for all
# other xs.
#
# Matern derivatives for half-integer nu
# stabilize Matern kernel near r == 0, then Matern derivatives for real nu
# (quick fix: larger eps in _softabs)
# GP._prior stored flat?
# compute only half of the covariance matrices (checksym=True to compute full)
# GP._cov stored 1D? (lower triangular)
# non-real input kernels (there are some examples in GPML)
# marginal likelihood derivatives
# method GP._covblock to get covariance matrix that builds it one piece at a
# time as required
# delete the _x as soon as they are not needed any more
# kronecker optimization
# sparse algorithms (after adding finite support kernels)
#
# Question: when I do the svdcut, should I do a diagonalization or really an
# SVD? Is there a numerical stability difference?
