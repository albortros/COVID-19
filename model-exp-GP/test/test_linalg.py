import sys

import numpy as np
from scipy import linalg
import gvar

sys.path = ['.'] + sys.path
from lsqfitgp2 import _linalg

class DecompTestBase:
    
    @property
    def decompclass(self):
        raise NotImplementedError()
        
    def randsize(self):
        return np.random.randint(1, 20)
        
    def randsymmat(self, n=None):
        if not n:
            n = self.randsize()
        A = np.random.randn(n, n)
        return A.T @ A
    
    def randvec(self, n=None):
        if not n:
            n = self.randsize()
        return np.random.randn(n)
    
    def randmat(self, m=None, n=None):
        if not m:
            m = self.randsize()
        if not n:
            n = self.randsize()
        return np.random.randn(m, n)
    
    def solve(self, K, b):
        return linalg.solve(K, b)
            
    def test_solve_vec(self):
        for _ in range(100):
            K = self.randsymmat()
            b = self.randvec(len(K))
            sol = self.solve(K, b)
            result = self.decompclass(K).solve(b)
            assert np.allclose(sol, result, rtol=1e-4)
    
    def test_solve_matrix(self):
        for _ in range(100):
            K = self.randsymmat()
            b = self.randmat(len(K))
            sol = self.solve(K, b)
            result = self.decompclass(K).solve(b)
            assert np.allclose(sol, result, rtol=1e-4)

    def test_usolve_vec_gvar(self):
        for _ in range(100):
            K = self.randsymmat()
            mean = self.randvec(len(K))
            xcov = np.linspace(0, 3, len(K))
            cov = np.exp(-(xcov.reshape(-1, 1) - xcov.reshape(1, -1)) ** 2)
            b = gvar.gvar(mean, cov)
            invK = self.solve(K, np.eye(len(K)))
            sol = invK @ b
            result = self.decompclass(K).usolve(b)
            diff = result - sol
            
            diffmean = gvar.mean(diff)
            solcov = gvar.evalcov(gvar.svd(sol))
            q = diffmean @ linalg.solve(solcov, diffmean, assume_a='pos')
            assert np.allclose(q, 0, atol=1e-7)
            
            diffcov = gvar.evalcov(diff)
            solmax = np.max(linalg.eigvalsh(solcov))
            diffmax = np.max(linalg.eigvalsh(diffcov))
            assert np.allclose(diffmax / solmax, 0)
    
    def test_quad_vec(self):
        for _ in range(100):
            K = self.randsymmat()
            b = self.randvec(len(K))
            sol = b @ self.solve(K, b)
            result = self.decompclass(K).quad(b)
            assert np.allclose(sol, result)
    
    def test_logdet(self):
        for _ in range(100):
            K = self.randsymmat()
            sol = np.sum(np.log(linalg.eigvalsh(K)))
            result = self.decompclass(K).logdet()
            assert np.allclose(sol, result)

class TestDiag(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.Diag

class TestEigCutFullRank(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.EigCutFullRank

class TestEigCutLowRank(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.EigCutLowRank

class TestChol(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.Chol

class TestCholMaxEig(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.CholMaxEig

class TestCholGersh(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.CholGersh

class TestReduceRank(DecompTestBase):
    
    @property
    def decompclass(self):
        return lambda K: _linalg.ReduceRank(K, rank=self._rank)
    
    def solve(self, K, b):
        invK, rank = linalg.pinv(K, return_rank=True)
        assert rank == self._rank
        return invK @ b
    
    def randsymmat(self, n=None):
        if not n:
            n = self.randsize()
        self._rank = np.random.randint(1, n + 1)
        A = np.random.randn(self._rank, n)
        return A.T @ A
        
    def test_logdet(self):
        for _ in range(100):
            K = self.randsymmat()
            sol = np.sum(np.log(np.sort(linalg.eigvalsh(K))[-self._rank:]))
            result = self.decompclass(K).logdet()
            assert np.allclose(sol, result)
