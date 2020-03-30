import pickle
from matplotlib import pyplot as plt
import sys
import pandas as pd
import numpy as np
import gvar
import os
import tqdm

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Load fit result.
fits = pickle.load(open(sys.argv[1], 'rb'))

# Prepare figure.
fig = plt.figure('plot')
fig.clf()
ax = fig.subplots(1, 1)

# Make directory for saving figures.
savedir = 'plots'
os.makedirs(savedir, exist_ok=True)

# Iterate over regions.
print(f'Writing plots in {savedir}/...')
for region, fit in tqdm.tqdm(fits.items()):
    
    # Prepare plot.
    ax.cla()
    ax.set_title(region)
    
    # Plot.
    for label in fit['y']:
        # data
        x = fit['table']['data']
        y = fit['y'][label]
        rt = ax.errorbar(x, gvar.mean(y), yerr=gvar.sdev(y), fmt='.', label=label)
        color = rt[0].get_color()
    
        # fit
        xfit = fit['dates']['plot']
        yfit = fit['plot'][label]
        ym, ys = gvar.mean(yfit), gvar.sdev(yfit)
        ax.fill_between(xfit, ym - ys, ym + ys, color=color, alpha=0.5)
        
        # fit samples
        cov = gvar.evalcov(yfit)
        samples = np.random.multivariate_normal(ym, cov, size=20)
        ax.plot(xfit, samples.T, '-', color=color, alpha=0.2)

    # Embellishments.
    # ax.set_yscale('symlog', linthreshy=1, subsy=np.arange(2, 10), linscaley=0.3)
    if ax.get_ylim()[0] < 0:
        ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend(loc='upper left')
    ax.grid(linestyle=':')
    
    # Box with fit results.
    # brief = '\n'.join(
    #     f'{label} = {value}'
    #     for label, value in fit['p'].items()
    #     if label.startswith('t0')
    # )
    # ax.annotate(
    #     brief, (1, 0), xytext=(-8, 8),
    #     va='bottom',
    #     ha='right',
    #     xycoords='axes fraction',
    #     textcoords='offset points',
    #     bbox=dict(
    #         facecolor='white',
    #         alpha=0.8,
    #         edgecolor='gray',
    #         boxstyle='round'
    #     )
    # )

    # Save figure.
    fig.autofmt_xdate(rotation=70)
    fig.tight_layout()
    fig.savefig(f'{savedir}/{region}.png')

fig.show()
