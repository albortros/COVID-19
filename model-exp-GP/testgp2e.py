import lsqfitgp2 as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(-5, 5, 10)
xpred = np.linspace(-15, 25, 200)
y = np.sin(xdata)
y[1::2] = np.cos(xdata[1::2])

print('make GP...')
gp = lgp.GP(lgp.ExpQuad(scale=3))
gp.addx(xdata[0::2], 'data', 0)
gp.addx(xdata[1::2], 'data', 1)
gp.addx(xpred, 'pred', 0)
gp.addx(xpred, 'pred', 1)

print('fit...')
u = gp.pred({('data', d): y[d::2] for d in [0, 1]}, 'pred')

print('figure...')
fig = plt.figure('testgp2e')
fig.clf()
ax = fig.subplots(1, 1)

colors = dict()
for deriv in 0, 1:
    m = gvar.mean(u[deriv])
    s = gvar.sdev(u[deriv])
    patch = ax.fill_between(xpred, m - s, m + s, label=f'deriv {deriv}', alpha=0.5)
    colors[deriv] = patch.get_facecolor()[0]
    
print('samples...')
for i, sample in zip(range(30), gvar.raniter(u)):
    for deriv in 0, 1:
        ax.plot(xpred, sample[deriv], '-', color=colors[deriv])

for deriv, marker in (0, '+'), (1, 'x'):
    ax.plot(xdata[deriv::2], y[deriv::2], f'k{marker}', label=f'data deriv {deriv}')
ax.legend(loc='best')

fig.show()
