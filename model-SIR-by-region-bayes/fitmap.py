import pandas as pd
import pymc3 as pm
import numpy as np
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

sir_model = pm.ode.DifferentialEquation(
    func=SIR,
    times=times,
    n_states=2,
    n_theta=2,
    t0=-1
)

I_data = table['totale_attualmente_positivi'].values
R_data = table['totale_casi'].values - I_data

model = pm.Model()
with model:
    R0 = pm.InverseGamma('R0', alpha=1, beta=1)
    lamda = pm.Lognormal('lambda', mu=0, sigma=2)
    beta = pm.Deterministic('beta', lamda * R0)
    
    popdistr = pm.Bound(pm.Lognormal, lower=np.max(R_data + I_data))
    population = popdistr('population', mu=np.log(1e6), sigma=np.log(20))
    I0 = pm.Lognormal('I0', mu=np.log(1e2), sigma=1)
    S0 = pm.Deterministic('S0', population - I0)

    SI = sir_model(y0=[S0, I0], theta=[R0, lamda])
    S = pm.Deterministic('S', SI[:, 0])
    I = pm.Deterministic('I', SI[:, 1])
    R = pm.Deterministic('R', population - S - I)
    pm.Poisson('I_data', mu=I, observed=I_data)
    pm.Poisson('R_data', mu=R, observed=R_data)

    mp = pm.find_MAP()

pickle_file = 'fitmap_' + namedate.file_timestamp() + '.pickle'
pickle_dict = dict(
    model=model,
    mp=mp,
    table=table,
    sir_model=sir_model
)
print(f'Saving to {pickle_file}...')
pickle.dump(pickle_dict, open(pickle_file, 'wb'))
