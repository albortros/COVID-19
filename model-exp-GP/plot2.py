import pickle
from matplotlib import pyplot as plt
import sys
import numpy as np
import pandas as pd
import gvar
import os
import tqdm
import relu

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Load fit result.
fits = pickle.load(open('fit2.pickle', 'rb'))

# Create figure.
fig = plt.figure('plot2')
fig.clf()
fig.set_size_inches(11, 6)
fig.set_tight_layout(True)

# Make directory for saving figures.
savedir = 'plots'
os.makedirs(savedir, exist_ok=True)

# Iterate over regions.
print(f'Writing plots in {savedir}/...')
for region, fit in tqdm.tqdm(fits.items()):
    
    # Prepare plot.
    fig.clf()
    axs = fig.subplots(1, len(fit['y']))
    
    # Plot.
    for ax, label in zip(axs, fit['y']):
        # data
        x = fit['table']['data']
        y = fit['y'][label]
        rt = ax.errorbar(x, gvar.mean(y), yerr=gvar.sdev(y), fmt='.', zorder=10)
        color = rt[0].get_color()
    
        # fit
        xfit = fit['dates']['plot']
        yfit = fit['plot'][label]
        ym, ys = gvar.mean(yfit), gvar.sdev(yfit)
        ax.fill_between(xfit, ym - ys, ym + ys, color=color, alpha=0.5)
        ax.plot(xfit, ym, color=color)
        
        # fit samples
        cov = gvar.evalcov(yfit)
        samples = np.random.multivariate_normal(ym, cov, size=1)
        ax.plot(xfit, samples.T, '-', color=color, alpha=0.2)

        # Embellishments.
        # ax.set_yscale('symlog', linthreshy=1, subsy=np.arange(2, 10), linscaley=0.3)
        if ax.get_ylim()[0] < 0:
            ax.set_ylim(0, ax.get_ylim()[1])
        ax.grid(linestyle=':')
        ax.set_title(f'{region}—{label.replace("_", " ")}')
        
    # Box with fit results.
    brief = '\n'.join(
        f'{label} = {value}'
        for label, value in fit['params'].items()
    )
    axs[0].annotate(
        brief, (1, 1), xytext=(-8, -8),
        va='top',
        ha='right',
        xycoords='axes fraction',
        textcoords='offset points',
        # fontsize='small',
        bbox=dict(
            facecolor='white',
            alpha=0.8,
            edgecolor='gray',
            boxstyle='round'
        )
    )

    # Save figure.
    fig.autofmt_xdate(rotation=70)
    fig.savefig(f'{savedir}/{region}.png')
