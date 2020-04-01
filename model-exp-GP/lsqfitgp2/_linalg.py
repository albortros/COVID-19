import numpy as np
from scipy import linalg
from scipy.sparse import linalg as slinalg

__doc__ = """

Decompositions of positive definite matrices. A decomposition object is
initialized with a matrix and then can solve linear systems for that matrix.

Classes
-------
Decomposition :
    Abstract base class.
SVD :
    Singular value decomposition.
SVDFullRank :
    SVD rounding up small singular values.
SVDLowRank :
    SVD removing small singular values.
DiagLowRank :
    Partial diagonalization with higher eigenvalues only.
Chol :
    Cholesky decomposition.
CholMaxEig :
    Cholesky regularized using the maximum eigenvalue.
CholGersh :
    Cholesky regularized using an estimate of the maximum eigenvalue.

"""

class Decomposition:
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
        inv = self.solve(np.eye(len(ub)))
        return inv @ ub
    
    def quad(self, b):
        """
        Compute the quadratic form b.T @ K**(-1) @ b.
        """
        return b.T @ self.solve(b)
    
    def logdet(self):
        """
        Compute log(det(K)).
        """
        raise NotImplementedError()

class SVD(Decomposition):
    """
    Singular value decomposition.
    """
    
    def __init__(self, K):
        self._U, self._s, self._VT = linalg.svd(K, hermitian=True)
    
    def solve(self, b):
        if b.dtype != object and (len(b.shape) == 1 or b.shape[1] < b.shape[0]):
            UTb = self._U.T @ b
            UTb /= self._s.reshape(-1, 1) if len(UTb.shape) == 2 else self._s
            return self._VT.T @ UTb
        else:
            return ((self._VT.T / self._s) @ self._U.T) @ b
    
    usolve = solve
    
    def logdet(self):
        return np.sum(np.log(self._s))
    
    def _default_svdcut(self):
        s = self._s
        return 2 * len(s) * np.finfo(s.dtype).eps * np.max(s)

class SVDFullRank(SVD):
    """
    Singular value decomposition. Singular values below `svdcut` are set to
    `svdcut`.
    """
    
    def __init__(self, K, svdcut=None):
        super().__init__(K)
        if svdcut is None:
            svdcut = self._default_svdcut()
        self._s[self._s < svdcut] = svdcut
            
class SVDLowRank(SVD):
    """
    Singular value decomposition. Singular values below `svdcut` are removed.
    """
    
    def __init__(self, K, svdcut=None):
        super().__init__(K)
        if svdcut is None:
            svdcut = self._default_svdcut()
        subset = self._s >= svdcut
        self._U = self._U[:, subset]
        self._s = self._s[subset]
        self._VT = self._VT[subset, :]
        
class DiagLowRank(Decomposition):
    """
    Keep only the first `rank` higher eigenmodes.
    """
    
    def __init__(self, K, rank=1):
        self._w, self._V = slinalg.eigsh(K, k=rank, which='LM')
        self._trace = np.sum(np.diag(K))
        
    def solve(self, b):
        if b.dtype != object and (len(b.shape) == 1 or b.shape[1] < b.shape[0]):
            VTb = self._V.T @ b
            VTb /= self._w.reshape(-1, 1) if len(VTb.shape) == 2 else self._w
            return self._V @ VTb
        else:
            return ((self._V / self._w) @ self._V.T) @ b
    
    usolve = solve
    
    def quad(self, b):
        VTb = self._V.T @ b
        return (VTb.T / self._w) @ VTb
    
    def logdet(self):
        nmissing = len(self._V) - len(self._w)
        tracemissing = self._trace - np.sum(self._w)
        return np.sum(np.log(self._w)) + nmissing * np.log(tracemissing / nmissing)

def Chol(Decomposition):
    """
    Cholesky decomposition.
    """
    
    def __init__(self, K):
        self._L = linalg.cholesky(K)
    
    def solve(self, b):
        invLb = linalg.solve_triangular(self._L, b)
        return linalg.solve_triangular(self._L.T, invLb)
    
    def usolve(self, b):
        invL = linalg.solve_triangular(self._L, np.eye(len(self._L)))
        return (invL.T @ invL) @ b
    
    def quad(self, b):
        invLb = linalg.solve_triangular(self._L, b)
        return invLb.T @ invLb
    
    def logdet(self):
        return 2 * np.sum(np.log(np.diag(self.L)))

class CholMaxEig(Chol):
    """
    Cholesky decomposition. The matrix is corrected for numerical roundoff
    by adding to the diagonal a small number relative to the maximum eigenvalue.
    `epsfactor` multiplies this number.
    """
    
    def __init__(self, K, epsfactor=1):
        w = slinalg.eigvalsh(K, k=1, which='LM')
        eps = 2 * epsfactor * len(K) * np.finfo(K.dtype).eps * w[0]
        super().__init__(K + np.diag(np.full(len(K), eps)))


class CholGersh(Chol):
    """
    Cholesky decomposition. The matrix is corrected for numerical roundoff
    by adding to the diagonal a small number relative to the maximum eigenvalue.
    `epsfactor` multiplies this number. The maximum eigenvalue is estimated
    with the Gershgorin theorem.
    """
    
    def __init__(self, K, epsfactor=1):
        maxeigv = _gershgorin_eigval_bound(K)
        eps = 2 * epsfactor * len(K) * np.finfo(K.dtype).eps * maxeigv
        super().__init__(K + np.diag(np.full(len(K), eps)))

def _gershgorin_eigval_bound(K):
    return np.max(np.sum(np.abs(K), axis=1))
