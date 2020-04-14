import gvar
import numpy as np
import pandas as pd

# Differential equation.
def SIRH(SIH, t, p):
    S = SIH[0]
    I = SIH[1]
    H = SIH[2]
    
    R0 = p[0]
    gamma = p[1]
    beta = R0 * gamma
    yupsilon = p[2]
    
    dS = -beta * S * I
    dI = -dS - gamma * I
    dH = yupsilon * I
    return [dS, dI, dH]

# Model function.
def fcn(args, p):
    pop = args['min_pop'] + p['_population']
    
    I0 = p['I0_pop'] / pop
    S0 = 1 - I0
    H0 = 0

    def deriv(t, SIH):
        return np.array(SIRH(SIH, t, [p['R0'], p['gamma'], p['yupsilon']]))
    integrator = gvar.ode.Integrator(deriv=deriv, tol=1e-4)
    SIHfun = integrator.solution(-1, [S0, I0, H0])
    
    SIH = [SIHfun(t) for t in args['times']]
    I = np.array([sih[1] for sih in SIH])
    R = np.array([1 - sih[0] - sih[1] for sih in SIH])
    H = np.array([sih[2] for sih in SIH])
    D = R - H
    return gvar.BufferDict(D=D * pop, I=I * pop, H=H * pop)

def time_to_number(times):
    try:
        times = pd.to_numeric(times).values
    except TypeError:
        pass
    times = np.array(times, dtype=float)
    times /= 1e9 * 60 * 60 * 24 # ns -> days
    return times

def make_poisson_data(v):
    v = np.asarray(v)
    assert len(v.shape) <= 1
    assert np.all(v >= 0)
    return gvar.gvar(v, np.where(v > 0, np.sqrt(v), 1))

def rescale_sdev(x, factor):
    return x * factor + gvar.mean(x) * (1 - factor)
