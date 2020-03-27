import lsqfitgp2 as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-5, 15, 100)
y = np.sin(xdata)
yerr = 0

gp = lgp.GP(lgp.ExpQuad(scale=3))
gp.addx(xdata, 'data')
gp.addx(xpred, 'pred')

u = gp.pred({'data': y}, 'pred')
m = gvar.mean(u)
s = gvar.sdev(u)
cov = gvar.evalcov(u)

fig = plt.figure('testgp2a')
fig.clf()
ax = fig.subplots(1, 1)

patch = ax.fill_between(xpred, m - s, m + s, label='pred', alpha=0.5)
color = patch.get_facecolor()[0]
simulated_lines = np.random.multivariate_normal(m, cov, size=10)
ax.plot(xpred, simulated_lines.T, '-', color=color)
ax.errorbar(xdata, y, yerr=yerr, fmt='k.', label='data')
ax.legend(loc='best')

fig.show()
