import gvar
import numpy as np
from scipy import special

@np.vectorize
def relu(x, scale=1):
    """
    softmax(0, x) = log(1 + exp(x))
    
    if scale != 1:
    scale * softmax(0, x / scale)
    """
    if isinstance(x, gvar.GVar):
        m = x.mean / scale
        f = scale * special.logsumexp([0, m])
        dfdx = 1 / (1 + np.exp(-m))
        return gvar.gvar_function(x, f, dfdx)
    else:
        return scale * special.logsumexp([0, x / scale])

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    fig = plt.figure('relu')
    fig.clf()
    ax = fig.subplots(1, 1)
    
    x = np.linspace(-5, 5, 100)
    scale = 2
    ax.plot(x, relu(x, scale=scale))

    ux = gvar.gvar(x, np.full_like(x, 0.4))
    uy = relu(ux, scale=scale)
    ymean = gvar.mean(uy)
    ysdev = gvar.sdev(uy)
    ax.fill_between(x, ymean - ysdev, ymean + ysdev, alpha=0.3)
    
    fig.show()
