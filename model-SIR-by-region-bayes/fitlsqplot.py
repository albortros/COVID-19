import pickle
from matplotlib import pyplot as plt
import sys
import pandas as pd
import numpy as np
import gvar
import fitlsqdefs

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Load fit result.
fit = pickle.load(open(sys.argv[1], 'rb'))

# Print report.
p = fit['p']
population = p['_population'] + fit['min_pop']
print(f'population = {population}')
print(f'I0_pop = {p["I0_pop"]}')
print(f'R0 = {p["R0"]}')
print(f'lambda = {p["lambda"]}')

# Plot data and fit.
fig = plt.figure('fitlsqplot')
fig.clf()
ax = fig.subplots(1, 1)

x = fit['table']['data']
xfit = pd.to_datetime(np.linspace(np.min(x).value, np.max(x).value, 100))
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

ax.legend(loc='best')

fig.autofmt_xdate()
fig.tight_layout()

plt.show()
