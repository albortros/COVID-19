from __future__ import division

import collections
import itertools
import sys

import gvar
import numpy as np
from scipy import linalg

from . import _kernels
from . import _linalg

def _concatenate_noop(alist, **kw):
    """
    Like np.concatenate, but does not make a copy when concatenating only one
    array.
    """
    if len(alist) == 1:
        return np.array(alist[0], copy=False)
    else:
        return np.concatenate(alist, **kw)

class GP:
    """
    
    Object that represents a gaussian process over 1D real input.
    
    Methods that accept array/dictionary input also recognize lists and
    gvar.BufferDict. The output is always a np.ndarray or gvar.BufferDict.
    
    Methods
    -------
    addx :
        Add points where the gaussian process is evaluated.
    prior :
        Return a collection of unique `gvar`s representing the prior.
    pred :
        Compute the posterior for the process.
    predfromfit, predfromdata :
        Convenience wrappers for `pred`.
    marginal_likelihood :
        Compute the "marginal likelihood" also known as "bayes factor".
    
    """
    
    def __init__(self, covfun, solver='eigcut+', checkpos=True, checksym=True, checkfinite=True, **kw):
        """
        
        Parameters
        ----------
        covfun : Kernel
            An instance of `Kernel` representing the covariance kernel.
        solver : str
            A solver used to invert the covariance matrix. See list below for
            the available solvers. Default is `eigcut+` which is slow but
            robust.
        checkpos : bool
            If True (default), raise a `ValueError` if the covariance matrix
            turns out non positive within numerical error. The exception will
            be raised the first time you call `prior`, `pred` or
            `marginal_likelihood`. Setting `checkpos=False` can give a large
            speed benefit if you have more than ~1000 points in the GP.
        checksym : bool
            If True (default), check the covariance matrix is symmetric. If
            False, only half of the matrix is computed.
        checkfinite : bool
            If True (default), check that the covariance matrix does not
            contain infs or nans.
        
        Solvers
        -------
        eigcut+ :
            Promote small eigenvalues to a minimum value (default). What
            `lsqfit` does by default.
        eigcut- :
            Remove small eigenvalues.
        lowrank :
            Reduce the rank of the matrix. The complexity is O(n^2 r) where
            `n` is the matrix size and `r` the required rank, while other
            algorithms are O(n^3). Slow for small sizes.
        gersh :
            Cholesky decomposition after regularizing the matrix with a
            Gershgorin estimate of the maximum eigenvalue. The fastest of the
            O(n^3) algorithms.
        maxeigv :
            Cholesky decomposition regularizing the matrix with the maximum
            eigenvalue. Slow for small sizes.
        
        Keyword arguments
        -----------------
        eps : positive float
            For solvers `eigcut+`, `eigcut-`, `gersh` and `maxeigv`. Specifies
            the threshold for considering small the eigenvalues, relative to
            the maximum eigenvalue. The default is matrix size * float epsilon.
        rank : positive integer
            For the `lowrank` solver, the target rank. It should be much
            smaller than the matrix size for the method to be convenient.
        
        """
        if not isinstance(covfun, _kernels.Kernel):
            raise TypeError('covariance function must be of class Kernel')
        self._covfun = covfun
        self._x = collections.defaultdict(lambda: collections.defaultdict(list))
        # self._x: label -> (derivative order -> list of arrays)
        self._canaddx = True
        self._checkpositive = bool(checkpos)
        decomp = {
            'eigcut+': _linalg.EigCutFullRank,
            'eigcut-': _linalg.EigCutLowRank,
            'lowrank': _linalg.ReduceRank,
            'gersh'  : _linalg.CholGersh,
            'maxeigv': _linalg.CholMaxEig
        }[solver]
        self._solver = lambda K, **kwargs: decomp(K, **kwargs, **kw)
        self._checkfinite = bool(checkfinite)
        self._checksym = bool(checksym)
        self._derivatives = False
            
    def _checkderiv(self, deriv):
        if not isinstance(deriv, (int, np.integer)):
            raise ValueError('derivative order {} is not an integer'.format(deriv))
        deriv = int(deriv)
        if deriv < 0:
            raise ValueError('derivative order must be >= 0')
        return deriv
    
    def addx(self, x, key=None, deriv=0):
        """
        
        Add points where the gaussian process is evaluated. The points can be
        added in two ways: "array mode" or "dictionary mode". The mode is
        decided the first time you call `addx`: if you just pass an array,
        `addx` expects to receive again only an array in eventual subsequent
        calls, and concatenates the arrays along the first axis. If you either
        pass a dictionary or an array and a key, `addx` will organize arrays of
        points in an internal dictionary, and when you give an array for an
        already used key, the old array and the new one will be concatenated.
        
        You can specify if the points are used to evaluate the gaussian process
        itself or its derivatives by passing a nonzero `deriv` argument.
        Array of points for different differentiation orders are kept separate,
        both in array and in dictionary mode.
        
        Once `prior` or `pred` have been called, `addx` raises a RuntimeError,
        unless they were called with `raw=True` or `keepcorr=False`.
        
        `addx` never copies the input arrays if they are numpy arrays, so if
        you change their contents before doing something with the GP, the
        change will be reflected on the result. However, after the GP has
        computed internally its covariance matrix, the x are ignored.
        
        Parameters
        ----------
        x : 1D/2D array or dictionary of 1D/2D arrays
            The points to be added. If a dictionary, the keys can be anything
            except None. Tuple keys must be length 2 and are unpacked in
            key, deriv.
        key : hashable type, not None or tuple
            `addx(array, key)` is equivalent to `addx({key: array})`.
        deriv : int >= 0
            The derivative order.
        
        """
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
                d = self._checkderiv(d)
            else:
                key, d = k, deriv
            
            gx = x[k]
            if not isinstance(gx, (list, np.ndarray)):
                raise TypeError('`x[{}]` is not array or list'.format(k))
            
            gx = np.array(gx, copy=False)
            if not gx.size:
                raise ValueError('`x[{}]` is empty'.format(k))
            if d and not np.issubdtype(gx.dtype, np.number):
                raise ValueError('`x[{}]` has non-numeric type `{}`, but derivatives ({}) are taken'.format(k, gx.dtype, d))
            if len(gx.shape) == 0:
                raise ValueError('`x[{}]` is 0d'.format(k))
            
            if hasattr(self, '_dtype'):
                if not all(np.issubdtype(t, np.number) for t in (self._dtype, gx.dtype)):
                    if gx.dtype != self._dtype:
                        raise ValueError('`x[{}]` has dtype {} but previous dtype used was {}'.format(k, gx.dtype, self._dtype))
            else:
                self._dtype = gx.dtype
            
            prev = self._x.get(key, {}).get(d, [])
            if prev:
                shape = prev[0].shape
                if gx.shape[1:] != shape[1:]:
                    raise ValueError("`x[{}]` with shape {} does not concatenate with shape {} along first axis".format(k, gx.shape, shape))
            
            self._derivatives = self._derivatives or d > 0
            if self._derivatives and self._dtype.fields:
                raise ValueError('derivatives supported only with non-structured data')
            
            self._x[key][d].append(gx)
    
    @property
    def _length(self):
        return sum(sum(sum(x.size for x in l) for l in d.values()) for d in self._x.values())
    
    def _makeslices(self):
        slices = dict()
        # slices: (key, derivative order) -> slice
        i = 0
        for key, d in self._x.items():
            for deriv, l in d.items():
                length = sum(x.size for x in l)
                slices[key, deriv] = slice(i, i + length)
                i += length
        return slices
    
    @property
    def _slices(self):
        if not hasattr(self, '_slicesdict'):
            self._slicesdict = self._makeslices()
        return self._slicesdict
    
    def _makeshapes(self):
        shapes = dict()
        # shapes: (key, derivative order) -> shape of x
        for key, d in self._x.items():
            for deriv, l in d.items():
                shape = (sum(x.shape[0] for x in l),) + l[0].shape[1:]
                shapes[key, deriv] = shape
        return shapes

    @property
    def _shapes(self):
        if not hasattr(self, '_shapesdict'):
            self._shapesdict = self._makeshapes()
        return self._shapesdict
    
    def _buildcovblock(self, kdkd, cov):
        xy = [
            _concatenate_noop(self._x[key][deriv], axis=0)
            for key, deriv in kdkd
        ]
        kernel = self._covfun.diff(kdkd[0][1], kdkd[1][1])
        
        slices = [self._slices[kd] for kd in kdkd]
        shape = tuple(s.stop - s.start for s in slices)

        if slices[0] == slices[1] and not self._checksym:
            indices = np.triu_indices(shape[0])
            xy = [x.reshape(-1)[i] for x, i in zip(xy, indices)]
            thiscov = kernel(*xy)
            cov[indices] = thiscov
            cov[tuple(reversed(indices))] = thiscov
        else:
            xy = [x.reshape(-1)[t] for x, t in zip(xy, itertools.permutations([slice(None), None]))]
            thiscov = kernel(*xy)
            cov[:] = thiscov
        
        if self._checkfinite and not np.all(np.isfinite(thiscov)):
            raise RuntimeError('covariance block ({}, {}) is not finite'.format(*kdkd))
        if self._checksym and slices[0] == slices[1] and not np.allclose(thiscov, thiscov.T):
            raise RuntimeError('covariance block ({}, {}) is not symmetric'.format(*kdkd))
            
    def _buildcov(self):
        if not self._x:
            raise ValueError('process is empty, add values with `addx`')
        
        cov = np.empty((self._length, self._length))
        for kdkd in itertools.combinations_with_replacement(self._slices, 2):
            slices = tuple(self._slices[kd] for kd in kdkd)
            self._buildcovblock(kdkd, cov[slices])
            
            if slices[0] != slices[1]:
                revslices = tuple(reversed(slices))
                if not self._checksym:
                    cov[revslices] = cov[slices].T
                else:
                    self._buildcovblock(tuple(reversed(kdkd)), cov[revslices])
                    if not np.allclose(cov[slices], cov[revslices].T):
                        raise ValueError('covariance block ({}, {}) is not symmetric'.format(*kdkd))

        if self._checkpositive:
            eigv = linalg.eigvalsh(cov)
            mineigv = np.min(eigv)
            if mineigv < 0:
                bound = -len(cov) * np.finfo(float).eps * np.max(eigv)
                if mineigv < bound:
                    msg = 'covariance matrix is not positive definite: '
                    msg += 'mineigv = {:.4g} < {:.4g}'.format(mineigv, bound)
                    raise ValueError(msg)
        
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
            self._canaddx = False
            flatprior = gvar.gvar(np.zeros(len(self._cov)), self._cov)
            flatprior.flags['WRITEABLE'] = False
            self._priordict = gvar.BufferDict({
                kd: flatprior[s].reshape(self._shapes[kd])
                for kd, s in self._slices.items()
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

    def prior(self, key=None, deriv=None, strip0=None, raw=False):
        """
        
        Return an array or a dictionary of arrays of `gvar`s representing the
        prior for the gaussian process. The returned object is not unique but
        the `gvar`s stored inside are, so all the correlations are kept between
        objects returned by different calls to `prior`.
        
        Calling without arguments returns an array or dictionary, depending
        on the mode set by the first call to `addx`, representing the complete
        prior. If you have specified nonzero derivatives, the returned object
        is a dictionary where the keys are the derivative orders if in array
        mode, and the pairs `(key, deriv)` if in dictionary mode.
        
        By specifying the `key` and `deriv` parameters you can get a subset
        of the prior.
        
        Parameters
        ----------
        key :
            A key corresponding to one passed to `addx`. None for all keys.
        deriv : int >= 0
            One of the derivatives passed to `addx`. None for all derivatives.
        strip0 : None or bool
            By default (None), 0 order derivatives (so, no derivative taken at
            all) are stripped (example: `{0: array}` becomes just `array`, and
            `{(A, 0): array1, (B, 1): array2}` becomes `{A: array1, (B, 1):
            array2}`), unless `deriv` is specified explicitly.
        raw : bool
            If True, instead of returning a collection of `gvar`s it returns
            their covariance matrix as would be returned by `gvar.evalcov`.
        
        Returns
        -------
        If raw=False (default):
        
        prior : np.ndarray or gvar.BufferDict
            A collection of `gvar`s representing the prior.
        
        If raw=True:
        
        cov : np.ndarray or gvar.BufferDict
            The covariance matrix of the prior.
        """
        raw = bool(raw)
        
        key, deriv = self._checkkeyderiv(key, deriv)
        kdlist = self._getkeyderivlist(key, deriv)
        assert kdlist
        strippedkd = self._stripkeyderiv(kdlist, key, deriv, strip0)
        assert strippedkd or len(kdlist) == 1
        
        if raw and strippedkd:
            return gvar.BufferDict({
                (strippedkd[i], strippedkd[j]):
                self._cov[self._slices[kdlist[i]], self._slices[kdlist[j]]]
                for i in range(len(kdlist))
                for j in range(len(kdlist))
            })
        elif raw:
            s = self._slices[kdlist[0]]
            return self._cov[s, s]
        elif strippedkd:
            return gvar.BufferDict({
                strippedkd[i]: self._prior[kdlist[i]]
                for i in range(len(kdlist))
            })
        else:
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
            
            l = np.array(l, copy=False)
            shape = self._shapes[key, deriv]
            if l.shape != shape:
                raise ValueError('`given[{}]` has shape {} different from x shape {}'.format(k, l.shape, shape))
            if l.dtype != object and not np.issubdtype(l.dtype, np.number):
                    raise ValueError('`given[{}]` has non-numerical dtype {}'.format(k, l.dtype))
            
            ylist.append(l.reshape(-1))
            kdlist.append((key, deriv))
            
        return ylist, kdlist
    
    def _compatslices(self, sliceslist):
        i = 0
        out = []
        for s in sliceslist:
            length = s.stop - s.start
            out.append(slice(i, i + length))
            i += length
        return out
    
    def pred(self, given, key=None, deriv=None, strip0=None, fromdata=None, raw=False, keepcorr=None):
        """
        
        Compute the posterior for the gaussian process, either on all points,
        on a subset of points, or conditionally from a subset of points on
        another subset; and either directly from data or from a posterior
        obtained with a fit. The latter case is for when the gaussian process
        was used in a fit with other parameters.
        
        The output is a collection of `gvar`s, either an array or a dictionary
        of arrays. They are properly correlated with `gvar`s returned by
        `prior` and with the input data/fit.
        
        The input is an array or dictionary of arrays, `given`. You can pass an
        array only if the GP is in array mode as set by `addx`. The contents of
        `given` are either the input data or posterior. If a dictionary, the
        keys in given must follow the same convention of the output from
        `prior()`, i.e. `(key, deriv)`, or just `key` with implicitly `deriv =
        0` when in dictionary mode, and `deriv` in array mode.
        
        The parameters `key`, `deriv` and `strip0` control what is the output
        in the same way as in `prior()`.
        
        Parameters
        ----------
        given : array or dictionary of arrays
            The data or fit result for some/all of the points in the GP.
            The arrays can contain either `gvar`s or normal numbers, the latter
            being equivalent to zero-uncertainty `gvar`s.
        key, deriv :
            If None, compute the posterior for all points in the GP (also those
            used in `given`). Otherwise only those specified by key and/or
            deriv.
        strip0 : bool
            By default, 0th order derivatives are stripped from returned
            dictionary keys, unless `deriv` is explicitly specified to be 0.
        fromdata : bool
            Mandatory. Specify if the contents of `given` are data or already
            a posterior.
        raw : bool (default False)
            If True, instead of returning a collection of `gvar`s, return
            the mean and the covariance. When the mean is a dictionary, the
            covariance is a dictionary whose keys are pairs of keys of the
            mean (the same format used by `gvar.evalcov`).
        keepcorr : bool
            If True (default), the returned `gvar`s are correlated with the
            prior and the data/fit. If False, they have the correct covariance
            between themselves, but are independent from all other preexisting
            `gvar`s.
        
        Returns
        -------
        If raw=False (default):
        
        posterior : array or dictionary of arrays
            A collections of `gvar`s representing the posterior.
        
        If raw=True:
        
        pmean : array or dictionary of arrays
            The mean of the posterior. Equivalent to `gvar.mean(posterior)`.
        pcov : 2D array or dictionary of 2D arrays
            The covariance matrix of the posterior. If `pmean` is a dictionary,
            the keys of `pcov` are pairs of keys of `pmean`. Equivalent to
            `gvar.evalcov(posterior)`.
        
        """
        
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
        
        ylist, inkdl = self._flatgiven(given)
        yslices = [self._slices[kd] for kd in inkdl]
        cyslices = self._compatslices(yslices)
        
        yspslices = [self._slices[kd] for kd in kdlist]
        cyspslices = self._compatslices(yspslices)
        ysplen = sum(s.stop - s.start for s in cyspslices)
        
        y = _concatenate_noop(ylist)
        
        Kxsx = np.full((ysplen, len(y)), np.nan)
        for ss, css in zip(yspslices, cyspslices):
            for s, cs in zip(yslices, cyslices):
                Kxsx[css, cs] = self._cov[ss, s]
        
        Kxx = np.full((len(y), len(y)), np.nan)
        for s1, cs1 in zip(yslices, cyslices):
            for s2, cs2 in zip(yslices, cyslices):
                Kxx[cs1, cs2] = self._cov[s1, s2]
        if self._checksym:
            assert np.allclose(Kxx, Kxx.T)
                
        if (fromdata or raw or not keepcorr) and y.dtype == object:
            ycov = gvar.evalcov(gvar.gvar(y)) ## TODO use evalcov_block?
            if self._checkfinite and not np.all(np.isfinite(ycov)):
                raise ValueError('covariance matrix of `given` is not finite')
        else:
            ycov = 0
        
        if raw or not keepcorr:
            
            Kxsxs = np.full((ysplen, ysplen), np.nan)
            for s1, cs1 in zip(yspslices, cyspslices):
                for s2, cs2 in zip(yspslices, cyspslices):
                    Kxsxs[cs1, cs2] = self._cov[s1, s2]
            if self._checksym:
                assert np.allclose(Kxsxs, Kxsxs.T)
            
            ymean = gvar.mean(y)
            if self._checkfinite and not np.all(np.isfinite(ymean)):
                raise ValueError('mean of `given` is not finite')
            
            if fromdata:
                Kxx += ycov
                B = self._solver(Kxx, overwrite=True).solve(Kxsx.T).T
                cov = Kxsxs - Kxsx @ B.T
                mean = B @ ymean
            else:
                A = self._solver(Kxx, overwrite=False).solve(Kxsx.T).T
                Kxx -= ycov
                cov = Kxsxs - A @ Kxx @ A.T
                mean = A @ ymean
            
        else: # (keepcorr and not raw)        
            yplist = [self._prior[kd].reshape(-1) for kd in inkdl]
            ysplist = [self._prior[kd].reshape(-1) for kd in kdlist]
            yp = _concatenate_noop(yplist)
            ysp = _concatenate_noop(ysplist)
        
            Kxx += ycov
            flatout = Kxsx @ self._solver(Kxx, overwrite=True).usolve(y - yp) + ysp
        
        if raw and strippedkd:
            meandict = gvar.BufferDict({
                strippedkd[i]: mean[cyspslices[i]].reshape(self._shapes[kdlist[i]])
                for i in range(len(kdlist))
            })
            
            covdict = gvar.BufferDict({
                (strippedkd[i], strippedkd[j]):
                cov[cyspslices[i], cyspslices[j]].reshape(self._shapes[kdlist[i]] + self._shapes[kdlist[j]])
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
                strippedkd[i]:
                flatout[cyspslices[i]].reshape(self._shapes[kdlist[i]])
                for i in range(len(kdlist))
            })
        else:
            return flatout.reshape(self._shapes[kdlist[0]])
        
    def predfromfit(self, *args, **kw):
        """
        Like `pred` with `fromdata=False`.
        """
        return self.pred(*args, fromdata=False, **kw)
    
    def predfromdata(self, *args, **kw):
        """
        Like `pred` with `fromdata=True`.
        """
        return self.pred(*args, fromdata=True, **kw)

    def marginal_likelihood(self, given):
        """
        
        Compute (the logarithm of) the marginal likelihood given data, i.e. the
        probability of the data conditioned on the gaussian process prior and
        data error.
        
        Unlike `pred()`, you can't compute this with a fit result instead of
        data. If you used the gaussian process as latent variable in a fit,
        use the whole fit to compute the marginal likelihood. E.g. `lsqfit`
        always computes the logGBF (it's the same thing).
        
        The input is an array or dictionary of arrays, `given`. You can pass an
        array only if the GP is in array mode as set by `addx`. The contents of
        `given` represent the input data. If a dictionary, the keys in given
        must follow the same convention of the output from `prior()`, i.e.
        `(key, deriv)`, or just `key` with implicitly `deriv = 0` when in
        dictionary mode, and `deriv` in array mode.
                
        Parameters
        ----------
        given : array or dictionary of arrays
            The data for some/all of the points in the GP. The arrays can
            contain either `gvar`s or normal numbers, the latter being
            equivalent to zero-uncertainty `gvar`s.
        
        Returns
        -------
        marglike : scalar
            The logarithm of the marginal likelihood.
            
        """        
        ylist, inkdl = self._flatgiven(given)
        yslices = [self._slices[kd] for kd in inkdl]
        cyslices = self._compatslices(yslices)
                
        y = _concatenate_noop(ylist)
                
        Kxx = np.full((len(y), len(y)), np.nan)
        for s1, cs1 in zip(yslices, cyslices):
            for s2, cs2 in zip(yslices, cyslices):
                Kxx[cs1, cs2] = self._cov[s1, s2]
        if self._checksym:
            assert np.allclose(Kxx, Kxx.T)
        
        if y.dtype == object:
            gvary = gvar.gvar(y)
            ycov = gvar.evalcov(gvary)
            ymean = gvar.mean(gvary)
        else:
            ycov = 0
            ymean = y
        
        if self._checkfinite and not np.all(np.isfinite(ymean)):
            raise ValueError('mean of `given` is not finite')
        if self._checkfinite and not np.all(np.isfinite(ycov)):
            raise ValueError('covariance matrix of `given` is not finite')
        
        Kxx += ycov
        decomp = self._solver(Kxx, overwrite=True)
        return -1/2 * (decomp.quad(ymean) + decomp.logdet() + len(y) * np.log(2 * np.pi))
