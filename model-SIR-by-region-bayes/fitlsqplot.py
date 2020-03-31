import pickle
from matplotlib import pyplot as plt
import sys
import pandas as pd
import numpy as np
import gvar
import fitlsqdefs
import os
import tqdm

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Load fit result.
fits = pickle.load(open(sys.argv[1], 'rb'))
fits.pop('prior_option')

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
print(f'Writing plots in {savedir}/...')
for region, fit in tqdm.tqdm(fits.items()):    
    # Prepare plot.
    ax.cla()
    ax.set_title(region)
    
    # Compute times sequence to plot curve.
    x = fit['table']['data']
    left = np.min(x).value
    right = np.max(x).value
    xfit = pd.to_datetime(np.linspace(left, right + (right - left), 100))
    xfit_num = fitlsqdefs.time_to_number(xfit) - fit['time_zero']
    
    # Compute I, R at given times.
    p = fitlsqdefs.rescale_sdev(fit['p'], np.sqrt(fit['chi2'] / fit['dof']))
    yfit = fitlsqdefs.fcn(dict(times=xfit_num, min_pop=fit['min_pop']), p)
    
    # Plot.
    for label in 'I', 'R':
        # data
        y = fit['y'][label]
        rt = ax.errorbar(x, gvar.mean(y), yerr=gvar.sdev(y), fmt='.', label=label)
        color = rt[0].get_color()
    
        # fit
        ym, ys = gvar.mean(yfit[label]), gvar.sdev(yfit[label])
        ax.fill_between(xfit, ym - ys, ym + ys, color=color, alpha=0.3)

    # Embellishments.
    top = max(np.max(gvar.mean(fit['y'][label])) for label in ['I', 'R'])
    top *= 1.2
    top = max(1, top)
    ax.set_yscale('symlog', linthreshy=top, subsy=np.arange(2, 10))
    ax.axhline(top, color='black', linestyle='--', label='logscale boundary')
    ax.legend(loc='upper left')
    ax.grid(linestyle=':')
    
    # Box with fit results.
    population = p['_population'] + fit['min_pop']
    brief = f"""population = {population} people
$log_{{10}}$(population) = {log10(population)}
initial I = {p["I0_pop"]} people
$R_0$ = {p["R0"]}
$\\gamma^{{-1}}$ = {1 / p["lambda"]} days
$\\sqrt{{\\chi^2 / \\mathrm{{dof}}}}$ = {np.sqrt(fit["chi2"] / fit["dof"]):.1f}"""
    ax.annotate(
        brief, (1, 0), xytext=(-8, 8),
        va='bottom',
        ha='right',
        xycoords='axes fraction',
        textcoords='offset points',
        bbox=dict(
            facecolor='white',
            alpha=0.8,
            edgecolor='gray',
            boxstyle='round'
        )
    )

    # Save figure.
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f'{savedir}/{region}.png')

del fig, ax
