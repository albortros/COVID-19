import gvar
import numpy as np
import pandas as pd

# Differential equation.
def SIR(SI, t, p):
    S = SI[0]
    I = SI[1]
    
    R0 = p[0]
    lamda = p[1]
    beta = R0 * lamda
    
    dS = -beta * S * I
    dI = -dS - lamda * I
    return [dS, dI]

# Model function.
def fcn(args, p):
    pop = args['min_pop'] + p['_population']
    
    I0 = p['I0_pop'] / pop
    S0 = 1 - I0

    def deriv(t, SI):
        return np.array(SIR(SI, t, [p['R0'], p['lambda']]))
    integrator = gvar.ode.Integrator(deriv=deriv, tol=1e-4)
    SIfun = integrator.solution(-1, [S0, I0])
    
    SI = [SIfun(t) for t in args['times']]
    R = np.array([1 - si[0] - si[1] for si in SI])
    I = np.array([si[1] for si in SI])
    return gvar.BufferDict(R=R * pop, I=I * pop)

def time_to_number(times):
    try:
        times = pd.to_numeric(times).values
    except TypeError:
        pass
    times = np.array(times, dtype=float)
    times -= times[0]
    times /= 1e9 * 60 * 60 * 24 # ns -> days
    return times

def make_poisson_data(v):
    v = np.asarray(v)
    assert len(v.shape) <= 1
    assert np.all(v >= 0)
    return gvar.gvar(v, np.where(v > 0, np.sqrt(v), 1))
