from __future__ import division

import collections
import itertools
import sys

import gvar
import numpy as np
from scipy import linalg

from . import _kernels

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
    
    def __init__(self, covfun, checkpos=True):
        """
        
        Parameters
        ----------
        covfun : Kernel
            An instance of `Kernel` representing the covariance kernel.
        checkpos : bool
            If True (default), raise a `ValueError` if the covariance matrix
            turns out non positive within numerical error. The exception will
            be raised the first time you call `prior` or `pred`.
        
        """
        if not isinstance(covfun, _kernels.Kernel):
            raise TypeError('covariance function must be of class Kernel')
        self._covfun = covfun
        self._x = collections.defaultdict(lambda: collections.defaultdict(list))
        # self._x: label -> (derivative order -> list of arrays)
        self._canaddx = True
        self._checkpositive = checkpos
    
    def _checkderiv(self, deriv):
        if deriv != int(deriv):
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
        calls, and concatenates the arrays. If you either pass a dictionary or
        an array and a key, `addx` will organize arrays of points in an
        internal dictionary, and when you give an array for an already used
        key, the old array and the new one will be concatenated.
        
        You can specify if the points are used to evaluate the gaussian process
        itself or its derivatives by passing a nonzero `deriv` argument.
        Array of points for different differentiation orders are kept separate,
        both in array and in dictionary mode.
        
        Once `prior` or `pred` or have been called, `addx` raises a
        RuntimeError, unless they were called with `raw=True` or
        `keepcorr=False`.
        
        `addx` never copies the input arrays if they are numpy arrays, so if
        you change their contents before doing something with the GP, the
        change will be reflected on the result. However, after the GP has
        computed internally its covariance matrix, the x are ignored.
        
        Parameters
        ----------
        x : 1D array or dictionary of 1D arrays
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
    
    def _buildcov(self):
        if not self._x:
            raise ValueError('process is empty, add values with `addx`')
        
        cov = np.empty((self._length, self._length))
        for kdkd in itertools.product(self._slices, repeat=2):
            xy = [
                np.concatenate(self._x[key][deriv])
                for key, deriv in kdkd
            ]
            assert len(xy) == 2
            slices = [self._slices[kd] for kd in kdkd]
            kernel = self._covfun.diff(kdkd[0][1], kdkd[1][1])
            thiscov = kernel(xy[0].reshape(-1, 1), xy[1].reshape(1, -1))
            if not np.all(np.isfinite(thiscov)):
                raise ValueError('covariance block ({}, {}) is not finite'.format(*kdkd))
            cov[slices[0], slices[1]] = thiscov
        
        if not np.allclose(cov, cov.T):
            raise ValueError('covariance matrix is not symmetric')

        # check covariance matrix is positive definite
        eigv = linalg.eigvalsh(cov)
        mineigv = np.min(eigv)
        if mineigv < 0:
            bound = -len(cov) * np.finfo(float).eps * np.max(eigv)
            if mineigv < bound:
                message = 'covariance matrix is not positive definite: '
                message += f'mineigv = {mineigv:.4g} < {bound:.4g}'
                if self._checkpositive:
                    raise ValueError(message)
                else:
                    print(message, file=sys.stderr)
            cov[np.diag_indices(len(cov))] += -mineigv
        
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
            s = self._slices[key, deriv]
            xlen = s.stop - s.start
            if len(l) != xlen:
                raise ValueError('`given[{}]` has length {} different from x length {}'.format(k, len(l), xlen))
        
            l = np.asarray(l)
            if not len(l.shape) == 1 and len(l) >= 1:
                raise ValueError('`given[{}]` is not 1D nonempty array'.formay(k))
            
            ylist.append(l)
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
            The mean of the posterior.
        pcov : 2D array or dictionary of 2D arrays
            The covariance matrix of the posterior. If `pmean` is a dictionary,
            the keys of `pcov` are pairs of keys of `pmean`.
        
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
        
        y = np.concatenate(ylist)
        
        Kxsx = np.full((ysplen, len(y)), np.nan)
        for ss, css in zip(yspslices, cyspslices):
            for s, cs in zip(yslices, cyslices):
                Kxsx[css, cs] = self._cov[ss, s]
        
        Kxx = np.full((len(y), len(y)), np.nan)
        for s1, cs1 in zip(yslices, cyslices):
            for s2, cs2 in zip(yslices, cyslices):
                Kxx[cs1, cs2] = self._cov[s1, s2]
        assert np.allclose(Kxx, Kxx.T)
        
        if (fromdata or raw or not keepcorr) and y.dtype == object:
            S = gvar.evalcov(gvar.gvar(y))
        else:
            S = 0
        
        if raw or not keepcorr:
            
            Kxsxs = np.nan * np.empty((ysplen, ysplen))
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
            yplist = [self._prior[kd] for kd in inkdl]
            ysplist = [self._prior[kd] for kd in kdlist]
            yp = np.concatenate(yplist)
            ysp = np.concatenate(ysplist)
        
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
            The marginal likelihood.
            
        """        
        ylist, inkdl = self._flatgiven(given)
        yslices = [self._slices[kd] for kd in inkdl]
        cyslices = self._compatslices(yslices)
                
        y = np.concatenate(ylist)
                
        Kxx = np.full((len(y), len(y)), np.nan)
        for s1, cs1 in zip(yslices, cyslices):
            for s2, cs2 in zip(yslices, cyslices):
                Kxx[cs1, cs2] = self._cov[s1, s2]
        assert np.allclose(Kxx, Kxx.T)
        
        if y.dtype == object:
            gvary = gvar.gvar(y)
            ycov = gvar.evalcov(gvary)
            ymean = gvar.mean(gvary)
        else:
            ycov = 0
            ymean = y
        
        Kxx += ycov
        try:
            L = linalg.cholesky(Kxx)
            logdet = 2 * np.sum(np.log(np.diag(L)))
            res = linalg.solve_triangular(L, ymean)
            return -1/2 * (np.sum(res ** 2) + logdet + len(L) * np.log(2 * np.pi))
        except linalg.LinAlgError:
            w, v = linalg.eigh(Kxx)
            bound = len(w) * np.finfo(Kxx.dtype).eps * np.max(w)
            w[w < bound] = bound
            logdet = np.sum(np.log(w))
            res = v.T @ ymean
            return -1/2 + (np.sum(res ** 2 / w) + logdet + len(w) * np.log(2 * np.pi))
            # maybe LU decomposition is a faster solution, but would it give
            # positive determinant?
