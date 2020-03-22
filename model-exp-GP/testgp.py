import lsqfitgp
import lsqfit
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-1, 11, 300)

gp = lsqfitgp.GP(xdata, lsqfitgp.Matern32(), xpred=xpred)

fx = np.sin(xdata)

pred1 = gp.pred(fx)
pred2 = gp.predalt(fx)

fig = plt.figure('testgp')
fig.clf()
ax = fig.subplots(1, 1)

ax.plot(xdata, fx, '.', label='data')
for label in 'pred1', 'pred2':
    pred = eval(label)
    m = gvar.mean(pred)
    s = gvar.sdev(pred)
    cov = gvar.evalcov(pred)
    patch = ax.fill_between(xpred, m - s, m + s, label=label, alpha=0.5)
    color = patch.get_facecolor()[0]
    simulated_lines = np.random.multivariate_normal(m, cov, size=10)
    ax.plot(xpred, simulated_lines.T, '-', color=color)
ax.legend(loc='best')

fig.show()
