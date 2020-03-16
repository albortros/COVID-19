import pandas as pd
# import pymc3 as pm
import lsqfit
import gvar
import numpy as np
from matplotlib import pyplot as plt
import pickle
import namedate

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Read region data.
data = pd.read_csv(
    '../pcm-dpc-COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv',
    parse_dates=['data'],
    infer_datetime_format=True
)
gdata = data.groupby('denominazione_regione')
# use the name to group because problems with south tirol

# Times.
table = gdata.get_group('Lombardia')
times = np.array(pd.to_numeric(table['data']).values, dtype=float)
times -= times[0]
times /= 1e9 * 60 * 60 * 24 # ns -> days

# Data.
I_data = table['totale_attualmente_positivi'].values
I_data = gvar.gvar(I_data, np.sqrt(I_data))
R_data = table['totale_casi'].values - table['totale_attualmente_positivi'].values
R_data = gvar.gvar(R_data, np.sqrt(R_data))
fitdata = gvar.BufferDict(I=I_data, R=R_data)

# Prior.
prior = gvar.BufferDict({
    'log(R0)': gvar.gvar('0(2)'),
    'log(lambda)': gvar.gvar('0(2)'),
    'log(population)': gvar.gvar(np.log(1e6), np.log(1e2)),
    'log(I0)': gvar.gvar(np.log(1e2), 1)
})

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
def fcn(p):
    def deriv(t, SI):
        return np.array(SIR(SI, t, [p['R0'], p['lambda']]))
    integrator = gvar.ode.Integrator(deriv=deriv, tol=1e-4)
    S0 = p['population'] - p['I0']
    SIfun = integrator.solution(-1, [S0, p['I0']])
    
    SI = [SIfun(t) for t in times]
    R = np.array([p['population'] - si[0] - si[1] for si in SI])
    I = np.array([si[1] for si in SI])
    return gvar.BufferDict(R=R, I=I)

# Run fit.
fit = lsqfit.nonlinear_fit(data=fitdata, prior=prior, fcn=fcn)
fitlog = fit.format(maxline=True)
print(fitlog)

# Save results.
pickle_dict = dict(
    data=fit_data,
    p=fit.p,
    prior=prior,
    log=fitlog,
    chi2=fit.chi2,
    dof=fit.dof,
    pvalue=fit.Q
)
pickle_file = 'fitlsq_' + namedate.file_timestamp() + '.pickle'
print(f'Saving to {pickle_file}...')
pickle.dump(pickle_dict, open(pickle_file, 'wb'))
