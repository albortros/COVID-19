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
    
    def __new__(cls, array, dtype=None):
        assert isinstance(array, (np.ndarray, cls))
        assert array.dtype.names is not None
        
        if dtype is None:
            dtype = array.dtype
        else:
            dtype = np.dtype(dtype)
        assert dtype.names == array.dtype.names
        
        _dict = {
            name:
            np.array(array[name], copy=False, dtype=dtype.fields[name][0])
            for name in array.dtype.names
        }
        
        for x in _dict.values():
            x.flags['WRITEABLE'] = False
        
        self = super().__new__(cls)
        self._dict = _dict
        self.dtype = array.dtype
        self.shape = array.shape
        self.size = array.size
        
        return self
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __setitem__(self, key, val):
        assert key in self.dtype.names
        assert isinstance(val, np.ndarray)
        assert self.dtype.fields[key][0] == val.dtype
        assert val.shape == self.shape
        self._dict[key] = _readonlyview(val)
