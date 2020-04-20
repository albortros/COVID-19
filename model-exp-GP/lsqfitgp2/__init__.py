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
# Sparse algorithms. Make a custom minimal CSR class that allows an autograd
# box as values buffer with only kernel operations implemented (addition,
# multiplication, matrix multiplication, power). Make two decompositions
# specifically for sparse matrices, sparselu and sparselowrank. Finite support
# kernels have a parameter sparse=True to return a sparse matrix. Operations
# between a sparse and a dense object should raise an error while computing
# the kernel if the result is dense, but not while making prediction.
# Alternative: make pydata/sparse work with autograd. I hope I can inject the
# code into the module so I don't have to rely on a fork. Probably I have to
# define some missing basic functions and define the vjp of the constructors.
#
# DiagLowRank for low rank matrix + multiple of the identity (multiple rank-1
# updates to the Cholesky factor? Would it be useful anyway?)
#
# Long-term: move to a variable-oriented approach like gvar instead of the
# monolithic GP object I'm doing now. It should be doable because what I'm
# doing now with the keys is quite similar to a set of variables, but I have
# not clear ideas on the interface. It could be based on underlying default
# GP object, like gvar does with its hidden covariance matrix of all primary
# gvars.
#
# Option to compute only the diagonal of the output covariance matrix, and
# allow diagonal-only input covariance for data (will be fundamental for
# kronecker). For the output it already works implicitly when using gvars.
#
# Accept xarray.DataSet and pandas.DataFrame as inputs. Probably I can't use
# these as core formats due to autograd.
#
# Make everything opt-in except numpy. There's already a numpy submodule for
# doing this with scipy.linalg (numpy.dual).
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
# Check that second derivatives work with decompositions (it should currently
# be broken by the non-recursive unpacking of autograd boxes).
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
