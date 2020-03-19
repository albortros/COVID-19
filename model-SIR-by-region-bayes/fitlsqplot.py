import pickle
from matplotlib import pyplot as plt
import sys
import pandas as pd
import numpy as np
import gvar
import fitlsqdefs
import os

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Load fit result.
fits = pickle.load(open(sys.argv[1], 'rb'))

# Prepare figure.
fig = plt.figure('fitlsqplot')
fig.clf()
ax = fig.subplots(1, 1)

# Make directory for saving figures.
savedir = 'fitlsqplot'
os.makedirs(savedir, exist_ok=True)

def log10(x):
    return gvar.log(x) / gvar.log(10)

# Iterate over regions.
for i, region in enumerate(fits):
    print(f'\n------------- {region} --------------')
    fit = fits[region]
    
    # Print results.
    p = fit['p']
    population = p['_population'] + fit['min_pop']
    print(f'population = {population}')
    print(f'log10(population) = {log10(population)}')
    print(f'I0_pop = {p["I0_pop"]}')
    print(f'R0 = {p["R0"]}')
    print(f'lambda = {p["lambda"]}')
    
    ax.cla()
    ax.set_yscale('symlog', linthreshy=1)
    ax.set_title(region)
    
    x = fit['table']['data']
    left = np.min(x).value
    right = np.max(x).value
    xfit = pd.to_datetime(np.linspace(left, right + (right - left), 100))
    xfit_num = fitlsqdefs.time_to_number(xfit)
    yfit = fitlsqdefs.fcn(dict(times=xfit_num, min_pop=fit['min_pop']), fit['p'])
    for label in 'I', 'R':
        # data
        y = fit['y'][label]
        rt = ax.errorbar(x, gvar.mean(y), yerr=gvar.sdev(y), fmt='.', label=label)
        color = rt[0].get_color()
    
        # fit
        ym, ys = gvar.mean(yfit[label]), gvar.sdev(yfit[label])
        ax.fill_between(xfit, ym - ys, ym + ys, color=color)

    ax.legend(loc='upper left')
    ax.grid(linestyle=':')

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f'{savedir}/{region}.pdf')

del fig, ax
