import lsqfitgp2 as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-15, 25, 200)
y = np.sin(xdata)

print('make GP...')
gp = lgp.GP(lgp.ExpQuad(scale=3))
gp.addx(xdata, 'data')
gp.addx(xpred, 'pred', 0)
gp.addx(xpred, 'pred', 1)

print('fit...')
umean, ucov = gp.predfromdata({'data': y}, 'pred', raw=True)

print('figure...')
fig = plt.figure('testgp2g')
fig.clf()
ax = fig.subplots(1, 1)

colors = dict()
for deriv in 0, 1:
    m = umean[deriv]
    s = np.sqrt(np.diag(ucov[deriv, deriv]))
    patch = ax.fill_between(xpred, m - s, m + s, label=f'deriv {deriv}', alpha=0.5)
    colors[deriv] = patch.get_facecolor()[0]
    
print('samples...')
for deriv in 0, 1:
    m = umean[deriv]
    cov = ucov[deriv, deriv]
    samples = np.random.multivariate_normal(m, cov, size=10)
    ax.plot(xpred, samples.T, '-', color=colors[deriv])

ax.plot(xdata, y, 'k.', label='data')
ax.legend(loc='best')

fig.show()
