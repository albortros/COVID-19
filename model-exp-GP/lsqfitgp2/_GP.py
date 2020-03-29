from __future__ import division

import collections
import itertools

import gvar
import autograd
import numpy as np
from scipy import linalg

from . import _kernels

class GP:
    
    def __init__(self, covfun):
        if not isinstance(covfun, _kernels.Kernel):
            raise TypeError('covariance function must be of class Kernel')
        self._covfun = covfun
        self._x = collections.defaultdict(lambda: collections.defaultdict(list))
        # self._x: label -> (derivative order -> list of arrays)
        self._canaddx = True
    
    def _checkderiv(self, deriv):
        if deriv != int(deriv):
            raise ValueError('derivative order {} is not an integer'.format(deriv))
        deriv = int(deriv)
        if deriv < 0:
            raise ValueError('derivative order must be >= 0')
        return deriv
    
    def addx(self, x, key=None, deriv=0):
        if not self._canaddx:
            raise RuntimeError('can not add x any more to this process because it has been used')
        
        if isinstance(key, tuple):
            raise TypeError('key can not be tuple')
        
        deriv = self._checkderiv(deriv)
        
        if isinstance(x, (list, np.ndarray)):
            if None in self._x and not (key is None):
                raise ValueError("previous x is array, can't add key")
            if key is None and (len(self._x) >= 2 or self._x and not (None in self._x)):
                raise ValueError("previous x is dictionary, can't append array")
                
            x = {key: x}
                    
        elif isinstance(x, (dict, gvar.BufferDict)):
            if not (key is None):
                raise ValueError('can not specify key if x is a dictionary')
            if None in x:
                raise ValueError('`None` key in x not allowed')
            if len(self._x) == 1 and None in self._x:
                raise ValueError("previous x is array, can't append dictionary")
        
        else:
            raise TypeError('x must be array or dict')
        
        for k in x:
            if isinstance(k, tuple):
                if not (deriv is None):
                    raise ValueError('key `{}` in x is tuple but derivative is specified'.format(k))
                if len(k) != 2:
                    raise ValueError('key `{}` in x is tuple but not length 2'.format(k))
                key, d = k
            else:
                key, d = k, deriv
            
            gx = x[key]
            if not isinstance(gx, (list, np.ndarray)):
                raise TypeError('`x[{}]` is not array or list'.format(k))
            
            gx = np.asarray(gx)
            if len(gx.shape) != 1 or len(gx) == 0:
                raise ValueError('`x[{}]` is not 1D and nonempty'.format(k))
            
            self._x[key][d].append(gx)
    
    @property
    def _length(self):
        return sum(sum(sum(map(len, l)) for l in d.values()) for d in self._x.values())
    
    def _makeslices(self):
        slices = dict()
        # slices: (key, derivative order) -> slice
        i = 0
        for key, d in self._x.items():
            for deriv, l in d.items():
                length = sum(map(len, l))
                slices[key, deriv] = slice(i, i + length)
                i += length
        return slices
    
    @property
    def _slices(self):
        if not hasattr(self, '_slicesdict'):
            self._slicesdict = self._makeslices()
        return self._slicesdict
    
    def _covfunderiv(self, xderiv, yderiv):
        assert isinstance(xderiv, int)
        assert isinstance(yderiv, int)
        fun = self._covfun
        for _ in range(xderiv):
            fun = autograd.elementwise_grad(fun, 0)
        for _ in range(yderiv):
            fun = autograd.elementwise_grad(fun, 1)
        return fun
    
    def _buildcov(self):
        if not self._x:
            raise ValueError('process is empty, add values with `addx`')
        self._canaddx = False
        
        cov = np.empty((self._length, self._length))
        for kdkd in itertools.product(self._slices, repeat=2):
            xy = [
                np.concatenate(self._x[key][deriv])
                for key, deriv in kdkd
            ]
            xy[0] = xy[0].reshape(-1, 1) * np.ones(len(xy[1])).reshape(1, -1)
            xy[1] = xy[1].reshape(1, -1) * np.ones(len(xy[0])).reshape(-1, 1)
            assert len(xy) == 2
            kernel = self._covfunderiv(kdkd[0][1], kdkd[1][1])
            slices = [self._slices[k, d] for k, d in kdkd]
            cov[slices[0], slices[1]] = kernel(xy[0], xy[1])
        
        # check covariance matrix is positive definite
        if not np.allclose(cov, cov.T):
            raise ValueError('covariance matrix is not symmetric')
        eigv = linalg.eigvalsh(cov)
        mineigv = np.min(eigv)
        if mineigv < 0:
            if mineigv < -len(cov) * np.finfo(float).eps * np.max(eigv):
                raise ValueError('covariance matrix is not positive definite')
            cov[np.diag_indices(len(cov))] += -mineigv
            # this is a fast but strong regularization, maybe we could just do
            # an svd cut
        # since we are diagonalizing, maybe save the transformation and apply
        # it so that lsqfit does not diagonalize it again for the fit.
        
        return cov
    
    @property 
    def _cov(self):
        if not hasattr(self, '_covmatrix'):
            self._covmatrix = self._buildcov()
            self._covmatrix.flags['WRITEABLE'] = False
        return self._covmatrix
    
    @property
    def _prior(self):
        # use gvar.BufferDict instead of dict, otherwise pickling is a mess
        if not hasattr(self, '_priordict'):
            flatprior = gvar.gvar(np.zeros(len(self._cov)), self._cov)
            flatprior.flags['WRITEABLE'] = False
            self._priordict = gvar.BufferDict({
                (key, deriv): flatprior[s]
                for (key, deriv), s in self._slices.items()
            })
        return self._priordict
    
    def _checkkeyderiv(self, key, deriv):
        # this method not to be used by addx, and not to check keys in
        # dictionaries
        if not (key is None):
            if isinstance(key, tuple):
                raise TypeError('key can not be tuple')
            if None in self._x:
                raise ValueError('you have given key but x is array')
            if not key in self._x:
                raise KeyError(key)
        
        if not (deriv is None):
            deriv = self._checkderiv(deriv)
            if key is None:
                for k, d in self._slices:
                    if deriv == d:
                        break
                else:
                    raise ValueError("there's no derivative {} in process".format(deriv))
            elif not (deriv in self._x[key]):
                raise ValueError('no derivative {} for key {}'.format(deriv, key))
        
        return key, deriv
    
    def _getkeyderivlist(self, key, deriv):
        if key is None and deriv is None:
            return list(self._slices)
        elif key is None and not (deriv is None):
            return [(k, deriv) for k in self._x if deriv in self._x[k]]
        elif not (key is None) and deriv is None:
            return [(key, d) for d in self._x[key]]
        elif not (key is None) and not (deriv is None):
            return [(key, deriv)]
        assert False
    
    def _stripkeyderiv(self, kdlist, key, deriv, strip0):
        if strip0 is None:
            strip0 = deriv is None
        strip0 = bool(strip0)
        
        if None in self._x or not (key is None) and deriv is None:
            outlist = [d for _, d in kdlist]
            return [] if outlist == [0] and strip0 else outlist
        if not (key is None) and not (deriv is None):
            return []
        if key is None and not (deriv is None):
            return [k for k, _ in kdlist]
        if key is None and deriv is None:
            return [(k, d) if d else k for k, d in kdlist] if strip0 else kdlist
        assert False

    def prior(self, key=None, deriv=None, strip0=None):
        key, deriv = self._checkkeyderiv(key, deriv)
        kdlist = self._getkeyderivlist(key, deriv)
        assert kdlist
        strippedkd = self._stripkeyderiv(kdlist, key, deriv, strip0)
        
        if strippedkd:
            return gvar.BufferDict({
                strippedkd[i]: self._prior[kdlist[i]]
                for i in range(len(kdlist))
            })
        else:
            assert len(kdlist) == 1
            return self._prior[kdlist[0]]
        
    def _flatgiven(self, given):
        if isinstance(given, (list, np.ndarray)):
            if None in self._x and len(self._x[None]) == 1:
                given = {(None, *self._x[None]): given}
            else:
                raise ValueError('`given` is an array but x has keys and/or multiple derivatives, provide a dictionary')
            
        elif not isinstance(given, (dict, gvar.BufferDict)):
            raise TypeError('`given` must be array or dict')
        
        ylist = []
        yslices = []
        kdlist = []
        for k, l in given.items():
            if isinstance(k, tuple):
                if len(k) != 2:
                    raise ValueError('key `{}` from `given` is a tuple but has not length 2')
                key, deriv = k
            elif k is None:
                raise KeyError('None key in `given` not allowed')
            elif None in self._x:
                key = None
                deriv = k
            else:
                key = k
                deriv = 0
                
            if not (key in self._x):
                raise KeyError(key)
            if deriv != int(deriv) or int(deriv) < 0:
                raise ValueError('supposed deriv order `{}` is not a nonnegative integer'.format(key, deriv, deriv))
            if not (deriv in self._x[key]):
                raise KeyError('derivative `{}` for key `{}` missing'.format(deriv, key))

            if not isinstance(l, (list, np.ndarray)):
                raise TypeError('element `given[{}]` is not list or array'.format(k))
            s = self._slices[key, deriv]
            xlen = s.stop - s.start
            if len(l) != xlen:
                raise ValueError('`given[{}]` has length {} different from x length {}'.format(k, len(l), xlen))
        
            l = np.asarray(l)
            if not len(l.shape) == 1 and len(l) >= 1:
                raise ValueError('`given[{}]` is not 1D nonempty array'.formay(k))
            
            ylist.append(l)
            yslices.append(self._slices[key, deriv])
            kdlist.append((key, deriv))
            
        return ylist, yslices, kdlist
    
    def _compatslices(self, sliceslist):
        i = 0
        out = []
        for s in sliceslist:
            length = s.stop - s.start
            out.append(slice(i, i + length))
            i += length
        return out
    
    def pred(self, given, key=None, deriv=None, strip0=None, fromdata=None, raw=False, keepcorr=None):
        if fromdata is None:
            raise ValueError('you must specify if `given` is data or fit result')
        fromdata = bool(fromdata)
        raw = bool(raw)
        if keepcorr is None:
            keepcorr = not raw
        if keepcorr and raw:
            raise ValueError('both keepcorr=True and raw=True')
        
        key, deriv = self._checkkeyderiv(key, deriv)
        kdlist = self._getkeyderivlist(key, deriv)
        assert kdlist
        strippedkd = self._stripkeyderiv(kdlist, key, deriv, strip0)
        assert strippedkd or len(kdlist) == 1
        
        ylist, yslices, inkdl = self._flatgiven(given)
        cyslices = self._compatslices(yslices)
        yplist = [self._prior[kd] for kd in inkdl]
        
        yspslices = [self._slices[kd] for kd in kdlist]
        cyspslices = self._compatslices(yspslices)
        ysplist = [self._prior[kd] for kd in kdlist]
        
        y = np.concatenate(ylist)
        yp = np.concatenate(yplist)
        ysp = np.concatenate(ysplist)
        
        Kxsx = np.full((len(ysp), len(yp)), np.nan)
        for ss, css in zip(yspslices, cyspslices):
            for s, cs in zip(yslices, cyslices):
                Kxsx[css, cs] = self._cov[ss, s]
        
        Kxx = np.full((len(yp), len(yp)), np.nan)
        for s1, cs1 in zip(yslices, cyslices):
            for s2, cs2 in zip(yslices, cyslices):
                Kxx[cs1, cs2] = self._cov[s1, s2]
        assert np.allclose(Kxx, Kxx.T)
        
        if (fromdata or raw or not keepcorr) and y.dtype == object:
            S = gvar.evalcov(gvar.gvar(y))
        else:
            S = 0
        
        if raw or not keepcorr:
            
            Kxsxs = np.nan * np.empty((len(ysp), len(ysp)))
            for s1, cs1 in zip(yspslices, cyspslices):
                for s2, cs2 in zip(yspslices, cyspslices):
                    Kxsxs[cs1, cs2] = self._cov[s1, s2]
            assert np.allclose(Kxsxs, Kxsxs.T)

            if fromdata:
                B = linalg.solve(Kxx + S, Kxsx.T, assume_a='pos').T
                cov = Kxsxs - Kxsx @ B.T
                mean = B @ gvar.mean(y)
            else:
                A = linalg.solve(Kxx, Kxsx.T, assume_a='pos').T
                cov = Kxsxs + A @ (S - Kxx) @ A.T
                mean = A @ gvar.mean(y)
            
        else: # (keepcorr and not raw)        
            flatout = Kxsx @ gvar.linalg.solve(Kxx + S, y - yp) + ysp
        
        if raw and strippedkd:
            meandict = gvar.BufferDict({
                strippedkd[i]: mean[cyspslices[i]]
                for i in range(len(kdlist))
            })
            
            covdict = gvar.BufferDict({
                (strippedkd[i], strippedkd[j]):
                cov[cyspslices[i], cyspslices[j]]
                for i in range(len(kdlist))
                for j in range(len(kdlist))
            })
            
            return meandict, covdict
            
        elif raw:
            return mean, cov
        
        elif not keepcorr:
            flatout = gvar.gvar(mean, cov)
        
        if strippedkd:
            return gvar.BufferDict({
                strippedkd[i]: flatout[cyspslices[i]]
                for i in range(len(kdlist))
            })
        else:
            return flatout
        
    def predfromfit(self, *args, **kw):
        return self.pred(*args, fromdata=False, **kw)
    
    def predfromdata(self, *args, **kw):
        return self.pred(*args, fromdata=True, **kw)
