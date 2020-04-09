from __future__ import division

import collections
import itertools
import sys
import builtins

import gvar
from autograd import numpy as np
from autograd.scipy import linalg
from autograd.builtins import isinstance

from . import _kernels
from . import _linalg
from . import _array

__all__ = [
    'GP'
]

def _concatenate_noop(alist, **kw):
    """
    Like np.concatenate, but does not make a copy when concatenating only one
    array.
    """
    if len(alist) == 1:
        return np.array(alist[0], copy=False)
    else:
        return np.concatenate(alist, **kw)

def _unpackif1item(*tup):
    return tup[0] if len(tup) == 1 else tup

def _triu_indices_and_back(n):
    indices = np.triu_indices(n)
    q = np.empty((n, n), int)
    a = np.arange(len(indices[0]))
    q[indices] = a
    q[tuple(reversed(indices))] = a
    return indices, q

def _block_matrix(blocks):
    return _concatenate_noop([_concatenate_noop(row, axis=1) for row in blocks], axis=0)

def _noautograd(x):
    if builtins.isinstance(x, np.numpy_boxes.ArrayBox):
        return x._value
    else:
        return x

def _isarraylike(x):
    return isinstance(x, (list, np.ndarray, _array.StructuredArray))

def _asarray(x):
    if isinstance(x, _array.StructuredArray):
        return x
    else:
        return np.array(x, copy=False)

def _isdictlike(x):
    return isinstance(x, (dict, gvar.BufferDict))

class GP:
    """
    
    Object that represents a gaussian process over arbitrary input.
    
    Methods that accept arrays/dictionaries also recognize lists,
    StructuredArray and gvar.BufferDict. The output is always a np.ndarray or
    gvar.BufferDict.
    
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
        Compute the "marginal likelihood", also known as "bayes factor".
    
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
        self._x = collections.defaultdict(dict)
        # self._x: label -> (derivative order, derivative dim -> array)
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
        
    def _checkderiv(self, deriv):
        if not isinstance(deriv, (int, np.integer)):
            raise ValueError('derivative order {} is not an integer'.format(deriv))
        deriv = int(deriv)
        if deriv < 0:
            raise ValueError('derivative order {} < 0'.format(deriv))
        return deriv
    
    def _unpackderiv(self, deriv):
        if isinstance(deriv, tuple):
            if len(deriv) != 2:
                raise ValueError('deriv={} is a tuple but not length 2'.format(deriv))
            deriv, dim = deriv
            if not isinstance(dim, str):
                raise ValueError('second item in deriv={} must be str'.format((deriv, dim)))
        else:
            dim = None
        return self._checkderiv(deriv), dim
        
    def addx(self, x, key=None, deriv=0):
        """
        
        Add points where the gaussian process is evaluated. The points can be
        added in two ways: "array mode" or "dictionary mode". The mode is
        decided the first time you call `addx`: if you just pass an array,
        `addx` expects to receive again only an array in eventual subsequent
        calls. If you either pass a dictionary or an array and a key, `addx`
        will organize arrays of points in an internal dictionary.
        
        You can specify if the points are used to evaluate the gaussian process
        itself or its derivatives by passing a nonzero `deriv` argument.
        Array of points for different differentiation orders are kept separate,
        both in array and in dictionary mode.
        
        If the input is structured arrays and you are taking a derivative, you
        have to specify the field to differentiate against, by passing a tuple
        (deriv, field) as `deriv`.
        
        If `x` is a dictionary, derivatives can also be specified directly in
        the keys of `x` by using 2- or 3-tuples (key, deriv) or (key, deriv,
        field) respectively.
        
        `addx` never copies the input arrays if they are numpy arrays, so if
        you change their contents before doing something with the GP, the
        change will be reflected on the result. However, after the GP has
        computed internally its covariance matrix, the x are ignored.
        
        Once `prior`, `pred` or `marginal_likelihood` have been called, `addx`
        raises a RuntimeError, because the covariance matrix has already been
        computed.
        
        Parameters
        ----------
        x : array or dictionary of arrays
            The points to be added. If a dictionary, the keys can be anything
            except None. Tuple keys must have length 2 or 3 and are unpacked in
            (key, deriv) or (key, deriv, dim) respectively.
        key : *not* a tuple
            If `x` is an array, the dictionary key under which `x` is added.
            Can not be specified if `x` is a dictionary.
        deriv : int or tuple
            The derivative order. If a tuple, it is unpacked as (deriv, dim)
            where `dim` is the field of the structured array along which to
            take the derivative. Can not be specified if `x` is a dictionary
            with derivative indications in the keys.
        
        """
        if not self._canaddx:
            raise RuntimeError('can not add x any more to this process because it has been used')
        
        if isinstance(key, tuple):
            raise TypeError('key can not be tuple')
        
        deriv, dim = self._unpackderiv(deriv)
        
        if _isarraylike(x):
            if None in self._x and key is not None:
                raise ValueError("previous x is array, can't add key")
            if key is None and (len(self._x) >= 2 or self._x and None not in self._x):
                raise ValueError("previous x is dictionary, can't append array")
                
            x = {key: x}
                    
        elif _isdictlike(x):
            if key is not None:
                raise ValueError('can not specify key if x is a dictionary')
            if None in x:
                raise ValueError('`None` key in x not allowed')
            if len(self._x) == 1 and None in self._x:
                raise ValueError("previous x is array, can't append dictionary")
        
        else:
            raise TypeError('x must be array or dict')
        
        for k in x:
            
            # If the key is a tuple, unpack it.
            if isinstance(k, tuple):
                if len(k) not in (2, 3):
                    raise ValueError('key {} in x is tuple but not length 2 or 3'.format(k))
                if deriv is not None:
                    raise ValueError('key {} in x containts derivative already specified to be {}'.format(k, deriv))
                if len(k) == 2:
                    key, d = k
                elif dim is not None:
                    raise ValueError('key {} in x contains dimension already specified to be {}'.format(k, dim))
                else:
                    key, d, f = k
                    if not isinstance(f, str):
                        raise ValueError('Third item in x key {} must be str'.format(k))
                d = self._checkderiv(d)
            else:
                key, d, f = k, deriv, dim
            
            prev = self._x.get(key, {}).get((d, f), None)
            if prev is not None:
                msg = 'there is already an array for key={}, deriv={}, field={}'
                msg = msg.format(key, d, f)
                raise RuntimeError(msg)
            
            gx = x[k]
            
            # Convert to numpy array or StructuredArray.
            if not _isarraylike(gx):
                raise TypeError('`x[{}]` is not array or list'.format(k))
            gx = _asarray(gx)

            # Check it is not empty or 0d.
            if not gx.size:
                raise ValueError('`x[{}]` is empty'.format(k))
            if len(gx.shape) == 0:
                raise ValueError('`x[{}]` is 0d'.format(k))

            # Check that, if it has fields, they are the same fields of
            # previous arrays added.
            if hasattr(self, '_dtype'):
                if self._dtype.names != gx.dtype.names:
                    raise TypeError('`x[{}]` has fields {} but previous array(s) had {}'.format(self._dtype.names, gx.dtype.names))
            else:
                self._dtype = gx.dtype

            # Check that the derivative specifications are compatible with the
            # array data type.
            if gx.dtype.names is None:
                if f is not None:
                    raise ValueError('`x[{}]` is not structured but field "{}" specified'.format(k, f))
                if d and not np.issubdtype(gx.dtype, np.number):
                    raise ValueError('`x[{}]` has non-numeric type `{}`, but derivatives ({}) are taken'.format(k, gx.dtype, d))
            else:
                if f is not None and f not in gx.dtype.names:
                    raise ValueError('field "{}" not in array fields {}'.format(f, gx.dtype.names))
                if f is None and d:
                    raise ValueError('{} derivative(s) taken on array with fields {} but the field is not specified'.format(d, gx.dtype.names))
            
            self._x[key][d, f] = gx
    
    def _makeshapes(self):
        shapes = dict()
        # shapes: (key, derivative order, derivative dim) -> shape of x
        for key, d in self._x.items():
            for (deriv, dim), x in d.items():
                shapes[key, deriv, dim] = x.shape
        return shapes

    @property
    def _shapes(self):
        if not hasattr(self, '_shapesdict'):
            self._canaddx = False
            self._shapesdict = self._makeshapes()
        return self._shapesdict
    
    def _makecovblock(self, kdkd):
        xy = [self._x[key][deriv, dim] for key, deriv, dim in kdkd]
        kernel = self._covfun.diff(kdkd[0][1], kdkd[1][1], kdkd[0][2], kdkd[1][2])
        
        shape = tuple(np.prod(self._shapes[kd]) for kd in kdkd)

        if kdkd[0] == kdkd[1] and not self._checksym:
            indices, back = _triu_indices_and_back(shape[0])
            xy = [x.reshape(-1)[i] for x, i in zip(xy, indices)]
            halfcov = kernel(*xy)
            cov = halfcov[back]
        else:
            xy = [x.reshape(-1)[t] for x, t in zip(xy, itertools.permutations([slice(None), None]))]
            cov = kernel(*xy)
        
        if self._checkfinite and not np.all(np.isfinite(cov)):
            raise RuntimeError('covariance block ({}, {}) is not finite'.format(*kdkd))
        if self._checksym and kdkd[0] == kdkd[1] and not np.allclose(cov, cov.T):
            raise RuntimeError('covariance block ({}, {}) is not symmetric'.format(*kdkd))
        
        return cov
    
    def _covblock(self, row, col):
        if not self._x:
            raise RuntimeError('process is empty, add values with `addx`')

        if not hasattr(self, '_covblocks'):
            self._covblocks = {}
            # _covblocks : ((key, deriv, field), (key, deriv, field)) -> mat
        
        if (row, col) not in self._covblocks:
            block = self._makecovblock([row, col])
            if row != col:
                if self._checksym:
                    blockT = self._makecovblock([col, row])
                    if not np.allclose(block.T, blockT):
                        raise RuntimeError('covariance block ({}, {}) is not symmetric'.format(*kdkd))
                self._covblocks[col, row] = block.T
            self._covblocks[row, col] = block
        
        return self._covblocks[row, col]
        
    def _assemblecovblocks(self, kdlistrow, kdlistcol=None):
        if kdlistcol is None:
            kdlistcol = kdlistrow
        blocks = [[self._covblock(row, col) for col in kdlistcol] for row in kdlistrow]
        return _block_matrix(blocks)
        
    def _checkpos(self, cov):
        eigv = linalg.eigvalsh(_noautograd(cov))
        mineigv = np.min(eigv)
        if mineigv < 0:
            bound = -len(cov) * np.finfo(float).eps * np.max(eigv)
            if mineigv < bound:
                msg = 'covariance matrix is not positive definite: '
                msg += 'mineigv = {:.4g} < {:.4g}'.format(mineigv, bound)
                raise ValueError(msg)
        
    @property
    def _prior(self):
        if not hasattr(self, '_priordict'):
            if self._checkpositive:
                fullcov = self._assemblecovblocks(list(self._shapes))
                self._checkpos(fullcov)
            mean = {
                kd: np.zeros(s)
                for kd, s in self._shapes.items()
            }
            cov = {
                (row, col): self._covblock(row, col).reshape(rs + ls)
                for row, rs in self._shapes.items()
                for col, ls in self._shapes.items()
            }
            self._priordict = gvar.gvar(mean, cov)
        return self._priordict
    
    def _checkkeyderiv(self, key, deriv):
        # this method not to be used by addx, and not to check keys in
        # dictionaries
        if key is not None:
            if isinstance(key, tuple):
                raise TypeError('key can not be tuple')
            if None in self._x:
                raise ValueError('you have given key but x is array')
            if not key in self._x:
                raise KeyError(key)
        
        if deriv is not None:
            deriv, dim = self._unpackderiv(deriv)
            if deriv and dim is None and self._dtype.names is not None:
                raise ValueError('x have fields {} but field not specified for derivative order {}'.format(self._dtype.names, deriv))
            if key is None:
                for k, d, f in self._shapes:
                    if (deriv, dim) == (d, f):
                        break
                else:
                    raise ValueError("there's no derivative {} on field \"{}\" in process".format(deriv, dim))
            elif (deriv, dim) not in self._x[key]:
                raise ValueError('no derivative {} on field "{}" for key {}'.format(deriv, dim, key))
        else:
            dim = None
        
        return key, deriv, dim
    
    def _getkeyderivlist(self, key, deriv, dim):
        if key is None and deriv is None:
            return list(self._shapes)
        elif key is None and deriv is not None:
            return [(k, deriv, dim) for k in self._x if (deriv, dim) in self._x[k]]
        elif key is not None and deriv is None:
            return [(key, d, f) for d, f in self._x[key]]
        elif key is not None and deriv is not None:
            return [(key, deriv, dim)]
        assert False
    
    def _stripkeyderiv(self, kdlist, key, deriv, strip0, stripdim):
        if strip0 is None:
            strip0 = deriv is None
        strip0 = bool(strip0)
        stripdim = bool(stripdim)
        
        names = self._dtype.names is not None
        
        if stripdim:
            stripf = lambda f: () if f is None else (f,)
        else:
            stripf = lambda f: (f,)
        
        if None in self._x or key is not None and deriv is None:
            outlist = [
                _unpackif1item(d, *stripf(f)) if names else d
                for _, d, f in kdlist
            ]
            return [] if outlist in ([0], [(0, None)]) and strip0 else outlist
        if key is not None and deriv is not None:
            return []
        if key is None and deriv is not None:
            return [k for k, _, _ in kdlist]
        if key is None and deriv is None:
            return [
                (k, d, *stripf(f)) if d else k
                for k, d, f in kdlist
            ] if strip0 else [
                (k, d, *stripf(f))
                for k, d, f in kdlist
            ]
        assert False

    def prior(self, key=None, deriv=None, strip0=None, stripdim=True, raw=False):
        """
        
        Return an array or a dictionary of arrays of `gvar`s representing the
        prior for the gaussian process. The returned object is not unique but
        the `gvar`s stored inside are, so all the correlations are kept between
        objects returned by different calls to `prior`.
        
        Calling without arguments returns an array or dictionary, depending
        on the mode set by the first call to `addx`, representing the complete
        prior. If you have specified nonzero derivatives, the returned object
        is a dictionary where the keys are the derivative orders if in array
        mode, and the pairs `(key, deriv)` if in dictionary mode, and
        (key, deriv, field) if the arrays are structured to specify along which
        field the derivative is taken.
        
        By specifying the `key` and `deriv` parameters you can get a subset
        of the prior.
        
        Parameters
        ----------
        key :
            A key corresponding to one passed to `addx`. None for all keys.
        deriv : int >= 0 or tuple
            One of the derivatives passed to `addx`. None for all derivatives.
            If x has fields, `deriv` must be a tuple (deriv, field).
        strip0 : None or bool
            By default (None), 0 order derivatives (so, no derivative taken at
            all) are stripped (example: `{0: array}` becomes just `array`, and
            `{(A, 0): array1, (B, 1): array2}` becomes `{A: array1, (B, 1):
            array2}`), unless `deriv` is specified explicitly.
        stripdim : bool
            If True (default), when the output is a dictionary with derivatives
            in the keys, there are 0 order derivatives, and the arrays are
            structured, the `None` field specification is dropped.
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
        
        key, deriv, dim = self._checkkeyderiv(key, deriv)
        kdlist = self._getkeyderivlist(key, deriv, dim)
        assert kdlist
        strippedkd = self._stripkeyderiv(kdlist, key, deriv, strip0, stripdim)
        assert strippedkd or len(kdlist) == 1
        
        if raw and strippedkd:
            return gvar.BufferDict({
                (strippedkd[i], strippedkd[j]):
                self._covblock(kdlist[i], kdlist[j])
                for i in range(len(kdlist))
                for j in range(len(kdlist))
            })
        elif raw:
            return self._covblock(kdlist[0], kdlist[0])
        elif strippedkd:
            return gvar.BufferDict({
                strippedkd[i]: self._prior[kdlist[i]]
                for i in range(len(kdlist))
            })
        else:
            return self._prior[kdlist[0]]
        
    def _flatgiven(self, given, givencov):
        if _isarraylike(given):
            if len(self._x.get(None, {})) == 1:
                given = {(None, *df): given for df in self._x[None]}
                assert len(given) == 1
                if givencov is not None:
                    givencov = {(k, k): givencov for k in given}
            else:
                raise ValueError('`given` is an array but x has keys and/or multiple derivatives, provide a dictionary')
            
        elif not _isdictlike(given):
            raise TypeError('`given` must be array or dict')
        
        ylist = []
        kdlist = []
        keylist = []
        for k, l in given.items():
            checkdim = False
            if isinstance(k, tuple):
                if len(k) not in (2, 3):
                    raise ValueError('key `{}` from `given` is a tuple but has not length 2 or 3')
                if len(k) == 2:
                    key, deriv = k
                    dim = None
                else:
                    key, deriv, dim = k
                    checkdim = True
            elif k is None:
                raise KeyError('None key in `given` not allowed')
            elif None in self._x:
                key = None
                if self._dtype.names is not None:
                    deriv, dim = k
                    checkdim = True
                else:
                    deriv = k
                    dim = None
            else:
                key = k
                deriv = 0
                dim = None
            
            if checkdim and deriv and not isinstance(dim, str):
                raise ValueError('derivative field specification `{}` is not a string'.format(dim))
            if checkdim and not deriv and dim is not None:
                raise ValueError('derivative is 0 but field "{}" specified, must be None'.format(dim))  
            if key not in self._x:
                raise KeyError(key)
            if not isinstance(deriv, (int, np.integer)) or deriv < 0:
                raise ValueError('supposed derivative order `{}` is not a nonnegative integer'.format(deriv))
            if (deriv, dim) not in self._x[key]:
                raise KeyError('derivative {} on field "{}" for key `{}` missing'.format(deriv, dim, key))

            if not isinstance(l, (list, np.ndarray)):
                raise TypeError('element `given[{}]` is not list or array'.format(k))
            
            l = np.array(l, copy=False)
            shape = self._shapes[key, deriv, dim]
            if l.shape != shape:
                raise ValueError('`given[{}]` has shape {} different from x shape {}'.format(k, l.shape, shape))
            if l.dtype != object and not np.issubdtype(l.dtype, np.number):
                    raise ValueError('`given[{}]` has non-numerical dtype {}'.format(k, l.dtype))
            
            ylist.append(l.reshape(-1))
            kdlist.append((key, deriv, dim))
            keylist.append(k)
        
        if givencov is not None:
            covblocks = [
                [
                    givencov[keylist[i], keylist[j]].reshape(ylist[i].shape + ylist[j].shape)
                    for j in range(len(kdlist))
                ]
                for i in range(len(kdlist))
            ]
        else:
            covblocks = None
            
        return ylist, kdlist, covblocks
    
    def _slices(self, kdlist):
        sizes = [np.prod(self._shapes[kd]) for kd in kdlist]
        stops = np.concatenate([[0], np.cumsum(sizes)])
        return [slice(stops[i - 1], stops[i]) for i in range(1, len(stops))]
    
    def pred(self, given, key=None, deriv=None, givencov=None, strip0=None, stripdim=True, fromdata=None, raw=False, keepcorr=None):
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
            deriv. If x is a structured array, deriv must be a tuple
            (deriv, field).
        givencov : array or dictionary of arrays
            Covariance matrix of `given`. If not specified, the covariance
            is extracted from `given` with `gvar.evalcov(given)`.
        strip0 : bool
            By default, 0th order derivatives are stripped from returned
            dictionary keys, unless `deriv` is explicitly specified to be 0.
        stripdim : bool
            If True (default), drop the `None` field specification for 0-th
            order derivatives.
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
        
        key, deriv, dim = self._checkkeyderiv(key, deriv)
        outkd = self._getkeyderivlist(key, deriv, dim)
        assert outkd
        strippedkd = self._stripkeyderiv(outkd, key, deriv, strip0, stripdim)
        assert strippedkd or len(outkd) == 1
        outslices = self._slices(outkd)
        
        ylist, inkd, ycovblocks = self._flatgiven(given, givencov)
        y = _concatenate_noop(ylist)
        
        Kxsx = self._assemblecovblocks(outkd, inkd)
        Kxx = self._assemblecovblocks(inkd)
        
        # TODO remove
        assert np.allclose(Kxx, Kxx.T)
        assert np.allclose(Kxsx, self._assemblecovblocks(inkd, outkd).T)
        
        if ycovblocks is not None:
            ycov = _block_matrix(ycovblocks)
        elif (fromdata or raw or not keepcorr) and y.dtype == object:
            ycov = gvar.evalcov(gvar.gvar(y)) ## TODO use evalcov_block?
            if self._checkfinite and not np.all(np.isfinite(ycov)):
                raise ValueError('covariance matrix of `given` is not finite')
        else:
            ycov = 0
        
        if raw or not keepcorr:
            
            Kxsxs = self._assemblecovblocks(outkd)

            assert np.allclose(Kxsxs, Kxsxs.T) # TODO remove
            
            ymean = gvar.mean(y)
            if self._checkfinite and not np.all(np.isfinite(ymean)):
                raise ValueError('mean of `given` is not finite')
            
            if fromdata:
                B = self._solver(Kxx + ycov).solve(Kxsx.T).T
                cov = Kxsxs - Kxsx @ B.T
                mean = B @ ymean
            else:
                A = self._solver(Kxx).solve(Kxsx.T).T
                cov = Kxsxs - A @ (Kxx - ycov) @ A.T
                mean = A @ ymean
            
        else: # (keepcorr and not raw)        
            yplist = [self._prior[kd].reshape(-1) for kd in inkd]
            ysplist = [self._prior[kd].reshape(-1) for kd in outkd]
            yp = _concatenate_noop(yplist)
            ysp = _concatenate_noop(ysplist)
        
            flatout = Kxsx @ self._solver(Kxx + ycov).usolve(y - yp) + ysp
        
        if raw and strippedkd:
            meandict = gvar.BufferDict({
                strippedkd[i]:
                mean[outslices[i]].reshape(self._shapes[outkd[i]])
                for i in range(len(outkd))
            })
            
            covdict = gvar.BufferDict({
                (strippedkd[i], strippedkd[j]):
                cov[outslices[i], outslices[j]].reshape(self._shapes[outkd[i]] + self._shapes[outkd[j]])
                for i in range(len(outkd))
                for j in range(len(outkd))
            })
            
            return meandict, covdict
            
        elif raw:
            mean = mean.reshape(self._shapes[outkd[0]])
            cov = cov.reshape(2 * self._shapes[outkd[0]])
            return mean, cov
        
        elif not keepcorr:
            flatout = gvar.gvar(mean, cov)
        
        if strippedkd:
            return gvar.BufferDict({
                strippedkd[i]:
                flatout[outslices[i]].reshape(self._shapes[outkd[i]])
                for i in range(len(outkd))
            })
        else:
            return flatout.reshape(self._shapes[outkd[0]])
        
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

    def marginal_likelihood(self, given, givencov=None):
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
        givencov : array or dictionary of arrays
            Covariance matrix of `given`. If not specified, the covariance
            is extracted from `given` with `gvar.evalcov(given)`.
        
        Returns
        -------
        marglike : scalar
            The logarithm of the marginal likelihood.
            
        """        
        ylist, inkd, ycovblocks = self._flatgiven(given, givencov)
        y = _concatenate_noop(ylist)

        Kxx = self._assemblecovblocks(inkd)

        assert np.allclose(Kxx, Kxx.T) # TODO remove
        
        if ycovblocks is not None:
            ycov = _block_matrix(ycovblocks)
            ymean = gvar.mean(y)
        elif y.dtype == object:
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
        
        decomp = self._solver(Kxx + ycov)
        return -1/2 * (decomp.quad(ymean) + decomp.logdet() + len(y) * np.log(2 * np.pi))
