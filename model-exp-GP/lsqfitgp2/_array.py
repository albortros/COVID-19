import builtins

from autograd import numpy as np
from autograd.builtins import isinstance

__all__ = [
    'StructuredArray'
]

def _readonlyview(x):
    if builtins.isinstance(x, np.numpy_boxes.ArrayBox):
        a = x._value
    else:
        a = x
    
    b = a.view()
    b.flags['WRITEABLE'] = False
    
    if a is x:
        return b
    else:
        x._value = b
        return x

class StructuredArray:
    """
    Autograd-friendly imitation of a numpy structured array.
    """
    
    @classmethod
    def _fromarrayanddict(cls, x, d):
        out = super().__new__(cls)
        out.dtype = x.dtype
        out._dict = d
        f0 = x.dtype.names[0]
        a0 = d[f0]
        subshape = x.dtype.fields[f0][0].shape
        out.shape = a0.shape[:len(a0.shape) - len(subshape)]
        out.size = np.prod(out.shape)
        return out
    
    def __new__(cls, array):
        assert isinstance(array, (np.ndarray, cls))
        assert array.dtype.names is not None
        d = {
            name: _readonlyview(array[name])
            for name in array.dtype.names
        }
        return cls._fromarrayanddict(array, d)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self._dict[key]
        elif isinstance(key, list) and all(isinstance(k, str) for k in key):
            d = {
                name: self._dict[name]
                for name in key
            }
        else:
            d = {
                name: x[key]
                for name, x in self._dict.items()
            }
        return type(self)._fromarrayanddict(self, d)
    
    def __setitem__(self, key, val):
        assert key in self.dtype.names
        assert isinstance(val, np.ndarray)
        prev = self._dict[key]
        assert prev.dtype == val.dtype
        assert prev.shape == val.shape
        self._dict[key] = _readonlyview(val)
    
    def reshape(self, *shape):
        d = {
            name: x.reshape(shape + self.dtype.fields[name][0].shape)
            for name, x in self._dict.items()
        }
        return type(self)._fromarrayanddict(self, d)
