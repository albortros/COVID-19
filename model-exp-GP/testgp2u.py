import lsqfitgp2 as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

time = np.linspace(-5, 5, 10)
x = np.empty(len(time), dtype=[
    ('time', float),
    ('label', int)
])
x['time'] = time
x['label'] = 0

data_error = 0.05
data_mean = np.cos(time)
data_mean += data_error * np.random.randn(*data_mean.shape)
data = gvar.gvar(data_mean, np.full_like(data_mean, data_error))

label_scale = 0.8
corr = lgp.ExpQuad(scale=label_scale)(0, 1)
print(f'corr = {corr:.3g}')

gp = lgp.GP(lgp.ExpQuad(scale=3, dim='time') * lgp.ExpQuad(scale=label_scale, dim='label'))
gp.addx(x, 'A')

time_pred = np.linspace(-10, 10, 100)
xpred = np.empty((2, len(time_pred)), dtype=x.dtype)
xpred['time'] = time_pred
xpred['label'][0] = 0
xpred['label'][1] = 1
gp.addx(xpred[0], 0)
gp.addx(xpred[1], 1, deriv=(1, 'time'))

pred = gp.predfromdata({'A': data}, [0, 1])

fig = plt.figure('testgp2u')
fig.clf()
ax = fig.subplots(1, 1)

colors = dict()
for deriv in pred:
    m = gvar.mean(pred[deriv])
    s = gvar.sdev(pred[deriv])
    polys = ax.fill_between(time_pred, m - s, m + s, alpha=0.5, label=f'deriv {deriv}')
    colors[deriv] = polys.get_facecolor()[0]

for _, sample in zip(range(3), gvar.raniter(pred)):
    for deriv in pred:
        ax.plot(time_pred, sample[deriv], color=colors[deriv])

ax.errorbar(time, gvar.mean(data), yerr=gvar.sdev(data), fmt='.', color=colors[0], alpha=1)

ax.legend(loc='best')
ax.set_xlabel('time')

fig.show()
