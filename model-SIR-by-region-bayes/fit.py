import pandas as pd
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import pickle

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

# Get times.

# Fit.
def SIR(y, t, p):
    S = y[0]
    I = y[1]
    
    R0 = p[0]
    lamda = p[1]
    beta = R0 * lamda
    
    dS = -beta * S * I
    dI = -dS - lamda * I
    return [dS, dI]

table = gdata.get_group('Lombardia')
times = np.array(pd.to_numeric(table['data']).values, dtype=float)
times -= times[0]
times /= 1e9 * 60 * 60 * 24
sir_model = pm.ode.DifferentialEquation(
    func=SIR,
    times=times,
    n_states=2,
    n_theta=2,
    t0=0
)

I_data = table['totale_attualmente_positivi']
R_data = table['totale_casi'] - I_data

model = pm.Model()
with model:
    R0 = pm.InverseGamma('R0', alpha=1, beta=1)
    lamda = pm.Lognormal('lambda', mu=0, sigma=2)
    beta = pm.Deterministic('beta', lamda * R0)
    
    population = pm.Lognormal('population', mu=np.log(1e6), sigma=np.log(1e2))
    I0 = pm.Lognormal('I0', mu=np.log(1e2), sigma=1)
    S0 = pm.Deterministic('S0', population - I0)
    
    SI = sir_model(y0=[S0, I0], theta=[R0, lamda])
    R = population - SI[:, 0] - SI[:, 1]
    pm.Poisson('I', mu=SI[:, 1], observed=I_data)
    pm.Poisson('R', mu=R, observed=R_data)
    
    mp = pm.find_MAP()

# Plot.
# fig = plt.figure('fit')
# fig.clf()
# axs = fig.subplots(1, 3)
#
# # sort by max total cases
# sorted_region = gdata['totale_casi'].max().sort_values().keys()[::-1]
#
# # iterate over regions
# for key in sorted_region:
#     table = gdata.get_group(key)
#     line, = axs[0].plot(table['data'], table['totale_casi'], '-', label=key)
#     color = line.get_color()
#     axs[1].plot(table['data'], table['dimessi_guariti'].values, color=color)
#     axs[2].plot(table['data'], table['deceduti'].values, color=color)
#
# axs[0].legend(loc='upper left', fontsize='small')
# axs[0].set_title('totale_casi')
# axs[1].set_title('dimessi_guariti')
# axs[2].set_title('deceduti')
# for ax in axs:
#     ax.set_yscale('symlog', linthreshy=1)
# fig.autofmt_xdate()
#
# plt.show()
