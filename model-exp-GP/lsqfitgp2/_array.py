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
        a0 = next(iter(d.values()))
        out.shape = a0.shape
        out.size = a0.size
        return out
    
    def __new__(cls, array, dtype=None):
        assert isinstance(array, (np.ndarray, cls))
        assert array.dtype.names is not None
        
        if dtype is None:
            dtype = array.dtype
        else:
            dtype = np.dtype(dtype)
        assert dtype.names == array.dtype.names
        
        d = {
            name:
            _readonlyview(np.array(array[name], copy=False, dtype=dtype.fields[name][0]))
            for name in array.dtype.names
        }
        
        return cls._fromarrayanddict(array, d)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self._dict[key]
        else:
            d = {
                name: x[key]
                for name, x in self._dict.items()
            }
            return type(self)._fromarrayanddict(self, d)
    
    def __setitem__(self, key, val):
        assert key in self.dtype.names
        assert isinstance(val, np.ndarray)
        assert self.dtype.fields[key][0] == val.dtype
        assert val.shape == self.shape
        self._dict[key] = _readonlyview(val)
    
    def reshape(self, *shape):
        d = {
            name: x.reshape(*shape)
            for name, x in self._dict.items()
        }
        return type(self)._fromarrayanddict(self, d)
