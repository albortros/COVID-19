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
# Stabilize Matern kernel near r == 0, then Matern derivatives for real nu
# (quick partial fix: larger eps in IsotropicKernel.__init__).
#
# Kronecker optimization: subclass GPKron where addx has a parameter `dim` and
# it accepts only non-structured arrays. Or, more flexible: make a class
# Lattice that is structured-array-like but different shapes for each field,
# and a field _kronok in Kernel update automatically when doing operations with
# kernels. Also, take a look at the pymc3 implementation. Can I use the
# kronecker optimization when the data covariance is non-null? -> Yes with a
# reasonable approximation of the marginal likelihood, but the data covariance
# must be diagonal. Other desiderata: separation along arbitrary subsets of
# the dimensions, it would be important when combining different keys with
# addtransf.
#
# Block matrix solving. Example: solve a subproblem with kronecker, another
# plain. Cache decompositions of blocks. Caching is effective with data if
# I can reuse the decomposition of Kxx to compute the decomposition of
# Kxx + ycov, i.e. it works in all cases if ycov is scalar, and in some cases
# if ycov is diagonal. First make an interface where _solver instead of being
# a variable is a method taking a list of keys over which to compute Kxx and
# optionally a matrix ycov to be added to Kxx. For caching: is there an
# efficient way to update a Cholesky decomposition if I add a diagonal matrix?
#
# A way to get the single contribution out of a sum of GPs. I think it is
# sufficient to use the usual formula with the full process for Kxx and the
# component I need for Kxsx and Kxsxs. This is handled as a specific case of
# the `addtransf` method if I use multidimensional input, see below.
#
# New method GP.addtransf to add a finite transformation over one or more keys.
# Interface: addtransf(keys, mat, key). `key` is the key under which the
# transformation is placed. `keys` is a key or a list of keys over which the
# transformation is applied. `mat` is an array or a list of arrays that
# represents the matrix that transforms the process over the given keys. It
# is applied tensordot-like to all axes of the array for the keys. The
# transformed keys can be primary keys or other transforms. This also allows
# to build vector-valued processes explicitly.
#
# Can I also apply non-linear transformations by implicitly taking the
# derivative? For prediction this means computing the derivative on the
# predicted mean. This is equivalent to applying a gvar ufunc to the result,
# really, so it does not make sense to add it to GP. For data it would mean
# I need the inverse transformation. But again this is equivalent to
# transforming the data prior to feeding it to the GP, so it is not useful.
# The useful cases would be non-elementwise transformations that I can not
# implement, so the conclusion is non-linear fits really need an explicit
# latent GP like I'm doing now.
#
# Sparse algorithms. I think autograd already supports sparse matrices, check
# autograd support before doing anything. First make a simple implementation
# that checks for zeros in the matrix built by the kernel, when it works I
# can think about how to let the kernel know it can return directly a sparse
# matrix if appropriate.
#
# DiagLowRank for low rank matrix + multiple of the identity (multiple rank-1
# updates to the Cholesky factor?)
#
# Option to compute only the diagonal of the output covariance matrix, and
# allow diagonal-only input covariance for data (will be fundamental for
# kronecker). For the output it already works implicitly when using gvars.
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
# Fourier kernels. Look at Celerite's algorithms.
#
# Check that the gradient of the marginal likelihood works with derivatives.
# Implement _array.broadcast and _array.broadcast_arrays that work both
# with np.ndarray and StructuredArray, and use them in _KernelBase.__call__,
# then make an example script with empbayes_fit on a GP with a derivative.
#
# Matrix transformation of inputs. Should work with arbitrarily nested dtypes.
#
# Apply kernels over a subset of the fields, accepting a list for `dim` in
# Kernel.__init__.
#
# Support taking derivatives in arbitrarily nested dtypes.
#
# Is there a smooth version of the Wiener process? like, softmin(x, y)?
