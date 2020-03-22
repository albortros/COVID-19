import lsqfitgp
import lsqfit
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-15, 25, 1000)
y = np.sin(xdata)
yerr = 0.05
gp = lsqfitgp.GP(xdata, lsqfitgp.ExpQuad(scale=3), xpred=xpred)

m, cov = gp.fitpredraw(y, yerr)
s = np.sqrt(np.diag(cov))

fig = plt.figure('testgp')
fig.clf()
ax = fig.subplots(1, 1)

patch = ax.fill_between(xpred, m - s, m + s, label='pred', alpha=0.5)
color = patch.get_facecolor()[0]
simulated_lines = np.random.multivariate_normal(m, cov, size=10)
ax.plot(xpred, simulated_lines.T, '-', color=color)
ax.errorbar(xdata, y, yerr=yerr, fmt='k.', label='data')
ax.legend(loc='best')

fig.show()
