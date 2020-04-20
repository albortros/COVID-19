from autograd import numpy as np
from autograd.scipy import linalg
from autograd import extend
from scipy.sparse import linalg as slinalg

__doc__ = """

Decompositions of positive definite matrices. A decomposition object is
initialized with a matrix and then can solve linear systems for that matrix.
These classes never check for infs/nans in the matrices.

Classes
-------
DecompMeta :
    Metaclass that adds autograd support.
Decomposition :
    Abstract base class.
Diag :
    Diagonalization.
EigCutFullRank :
    Diagonalization rounding up small eigenvalues.
EigCutLowRank :
    Diagonalization removing small eigenvalues.
ReduceRank :
    Partial diagonalization with higher eigenvalues only.
Chol :
    Cholesky decomposition.
CholMaxEig :
    Cholesky regularized using the maximum eigenvalue.
CholGersh :
    Cholesky regularized using an estimate of the maximum eigenvalue.

"""

def _noautograd(x):
    """
    Unpack an autograd numpy array.
    """
    if isinstance(x, np.numpy_boxes.ArrayBox):
        return _noautograd(x._value)
    else:
        return x

class DecompMeta(type):
    
    def __new__(cls, name, bases, dct):
        subclass = super().__new__(cls, name, bases, dct)
        
        old__init__ = subclass.__init__
        def __init__(self, K, **kw):
            old__init__(self, _noautograd(K), **kw)
            self._K = K
        subclass.__init__ = __init__
        
        oldsolve = subclass.solve
        if not hasattr(oldsolve, '_autograd'):
            @extend.primitive
            def solve_autograd(self, K, b):
                return oldsolve(self, b)
            def solve_vjp(ans, self, K, b):
                assert ans.shape == b.shape
                assert b.shape[0] == K.shape[0] == K.shape[1]
                def vjp(g):
                    assert g.shape[-len(b.shape):] == b.shape
                    A = solve_autograd(self, K, np.moveaxis(g, -len(b.shape), 0))
                    B = np.moveaxis(ans, 0, -1)
                    AB = np.tensordot(A, B, len(b.shape) - 1)
                    AB = np.moveaxis(AB, 0, -2)
                    assert AB.shape == g.shape[:-len(b.shape)] + K.shape
                    return -AB
                return vjp
            extend.defvjp(
                solve_autograd,
                solve_vjp,
                argnums=[1]
            )
            def solve(self, b):
                return solve_autograd(self, self._K, b)
            solve._autograd = True
            subclass.solve = solve
        
        # TODO write vjp optimized for quad instead of relying on solve
        oldquad = subclass.quad
        if not hasattr(oldquad, '_autograd'):
            def quad(self, b):
                if isinstance(self._K, np.numpy_boxes.ArrayBox):
                    return b.T @ self.solve(b)
                else:
                    return oldquad(self, b)
            quad._autograd = True
            subclass.quad = quad
        
        oldlogdet = subclass.logdet
        if not hasattr(oldlogdet, '_autograd'):
            # TODO I'm actually assuming that also solve has been overwritten,
            # otherwise solve_autograd is not defined. Maybe _autograd should
            # be the autograd version of the function instead of just a flag.
            @extend.primitive
            def logdet_autograd(self, K):
                return oldlogdet(self)
            def logdet_vjp(ans, self, K):
                assert ans.shape == ()
                assert K.shape[0] == K.shape[1]
                def vjp(g):
                    invK = solve_autograd(self, K, np.eye(len(K)))
                    return g[..., None, None] * invK
                return vjp
            extend.defvjp(
                logdet_autograd,
                logdet_vjp,
                argnums=[1]
            )
            # TODO the vjp is bad because it makes a matrix inversion. I would
            # like to do forward propagation just for the logdet plus the
            # preceding step, but I don't think autograd supports this.
            # def logdet_jvp(ans, self, K):
            #     assert ans.shape == ()
            #     assert K.shape[0] == K.shape[1]
            #     def jvp(g):
            #         assert g.shape[:2] == K.shape
            #         return np.trace(solve_autograd(self, K, g))
            #     return jvp
            # extend.defjvp(
            #     logdet_autograd,
            #     logdet_jvp,
            #     argnums=[1]
            # )
            def logdet(self):
                return logdet_autograd(self, self._K)
            logdet._autograd = True
            subclass.logdet = logdet
        
        return subclass

class Decomposition(metaclass=DecompMeta):
    """
    
    Abstract base class for positive definite symmetric matrices decomposition.
    
    Methods
    -------
    solve
    usolve
    quad
    logdet
    
    """
    
    def __init__(self, K):
        """
        Decompose matrix K.
        """
        raise NotImplementedError()
        
    def solve(self, b):
        """
        Solve the linear system K @ x = b.
        """
        raise NotImplementedError()
    
    def usolve(self, ub):
        """
        Solve the linear system K @ x = b where b is possibly an array of
        `gvar`s.
        """
        raise NotImplementedError()
    
    def quad(self, b):
        """
        Compute the quadratic form b.T @ inv(K) @ b.
        """
        return b.T @ self.solve(b)
    
    def logdet(self):
        """
        Compute log(det(K)).
        """
        raise NotImplementedError()

class Diag(Decomposition):
    """
    Diagonalization.
    """
    
    def __init__(self, K):
        self._w, self._V = linalg.eigh(K, check_finite=False)
    
    def solve(self, b):
        return (self._V / self._w) @ (self._V.T @ b)
    
    usolve = solve
    
    def quad(self, b):
        VTb = self._V.T @ b
        return (VTb.T / self._w) @ VTb
    
    def logdet(self):
        return np.sum(np.log(self._w))
    
    def _eps(self, eps):
        # TODO error checking on `eps`
        w = self._w
        if eps is None:
            eps = len(w) * np.finfo(w.dtype).eps
        return eps * np.max(w)

class EigCutFullRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` are set to `eps`, where `eps` is
    relative to the largest eigenvalue.
    """
    
    def __init__(self, K, eps=None, **kw):
        super().__init__(K, **kw)
        eps = self._eps(eps)
        self._w[self._w < eps] = eps
            
class EigCutLowRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` are removed, where `eps` is
    relative to the largest eigenvalue.
    """
    
    def __init__(self, K, eps=None, **kw):
        super().__init__(K, **kw)
        eps = self._eps(eps)
        subset = slice(np.sum(self._w < eps), None) # w is sorted ascending
        self._w = self._w[subset]
        self._V = self._V[:, subset]
        
class ReduceRank(Diag):
    """
    Keep only the first `rank` higher eigenmodes.
    """
    
    def __init__(self, K, rank=1):
        # TODO error checking on `rank`
        self._w, self._V = slinalg.eigsh(K, k=rank, which='LM')

def solve_triangular(a, b, lower=False):
    """
    Pure python implementation of scipy.linalg.solve_triangular for when
    a or b are object arrays. b must be 1D or a column vector.
    """
    x = np.copy(b)
    a = a.reshape(a.shape + x.shape[1:])
    if lower:
        x[0] /= a[0, 0]
        for i in range(1, len(x)):
            x[i:] -= x[i - 1] * a[i:, i - 1]
            x[i] /= a[i, i]
    else:
        x[-1] /= a[-1, -1]
        for i in range(len(x) - 1, 0, -1):
            x[:i] -= x[i] * a[:i, i]
            x[i - 1] /= a[i - 1, i - 1]
    return x

def grad_chol(L):
    n = len(L)
    I = np.eye(n)
    s1 = I[:, None, :, None] * L[None, :, None, :]
    s2 = I[None, :, :, None] * L[:, None, None, :]
    return (s1 + s2).reshape(2 * (n**2,))
        
class Chol(Decomposition):
    """
    Cholesky decomposition.
    """
    
    def __init__(self, K):
        self._L = linalg.cholesky(K, lower=True, check_finite=False)
    
    def solve(self, b):
        invLb = linalg.solve_triangular(self._L, b, lower=True)
        return linalg.solve_triangular(self._L.T, invLb, lower=False)
    
    def usolve(self, b):
        invLb = solve_triangular(self._L, b, lower=True)
        return solve_triangular(self._L.T, invLb, lower=False)
    
    def quad(self, b):
        invLb = linalg.solve_triangular(self._L, b, lower=True)
        return invLb.T @ invLb
    
    def logdet(self):
        return 2 * np.sum(np.log(np.diag(self._L)))
    
    # TODO _eps function here like Diag with error checking

class CholMaxEig(Chol):
    """
    Cholesky decomposition. The matrix is corrected for numerical roundoff
    by adding to the diagonal a small number relative to the maximum eigenvalue.
    `eps` multiplies this number.
    """
    
    def __init__(self, K, eps=None, **kw):
        w = slinalg.eigsh(K, k=1, which='LM', return_eigenvectors=False)
        if not eps:
            eps = len(K) * np.finfo(K.dtype).eps
        super().__init__(K + np.diag(np.full(len(K), eps * w[0])), **kw)


class CholGersh(Chol):
    """
    Cholesky decomposition. The matrix is corrected for numerical roundoff
    by adding to the diagonal a small number relative to the maximum eigenvalue.
    `eps` multiplies this number. The maximum eigenvalue is estimated
    with the Gershgorin theorem.
    """
    
    def __init__(self, K, eps=None, **kw):
        maxeigv = _gershgorin_eigval_bound(K)
        if not eps:
            eps = len(K) * np.finfo(K.dtype).eps
        super().__init__(K + np.diag(np.full(len(K), eps * maxeigv)), **kw)

def _gershgorin_eigval_bound(K):
    return np.max(np.sum(np.abs(K), axis=1))
