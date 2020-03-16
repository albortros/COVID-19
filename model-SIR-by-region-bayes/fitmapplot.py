import pickle
from matplotlib import pyplot as plt
import sys
import pandas
import pymc3 as pm
import numpy as np

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Differential equation.
# this definition is copied from fitmap.py and is needed by pickle
def SIR(SI, t, p):
    S = SI[0]
    I = SI[1]
    
    R0 = p[0]
    lamda = p[1]
    beta = R0 * lamda
    
    dS = -beta * S * I
    dI = -dS - lamda * I
    return [dS, dI]

# Load fit result.
fit = pickle.load(open(sys.argv[1], 'rb'))

# Extract data.
table = fit['table']
mp = fit['mp']
model = fit['model']

# Plot data and fit.
fig = plt.figure('fitmapplot')
fig.clf()
ax = fig.subplots(1, 1)

x = table['data']
for label in 'I', 'R':
    if label == 'I':
        y = table['totale_attualmente_positivi']
    else:
        y = table['totale_casi'] - table['totale_attualmente_positivi']
    line, = ax.plot(x, y.values, '.', label=label) # data
    ax.plot(x, mp[label] * mp['population'], '-', color=line.get_color()) # fit

ax.legend(loc='best')

fig.autofmt_xdate()

# Plot priors.
fig = plt.figure('fitmapplot-priors')
fig.clf()
var_names = ['R0', 'lambda', 'I0', 'population']
axs = fig.subplots(len(var_names), 1)

with model:
    samples = pm.sample_prior_predictive(var_names=var_names, samples=1000)

for i, ax in enumerate(axs):
    sample = samples[var_names[i]]
    bins = np.logspace(np.log10(np.min(sample)), np.log10(np.max(sample)), int(np.ceil(np.sqrt(len(sample)))))
    ax.hist(sample, bins=bins, density=True, histtype='stepfilled')
    ax.set_title(var_names[i])
    ax.set_xscale('log')

fig.tight_layout()

plt.show()
