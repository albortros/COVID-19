import lsqfitgp2 as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar
from scipy import optimize

time = np.arange(21)
x = np.empty((2, len(time)), dtype=[
    ('time', float),
    ('label', int)
])
x['time'][0] = time
delay = 20
x['time'][1] = time - delay
x['label'][0] = 0
x['label'][1] = 1
label_names = ['gatti_comprati', 'gatti_morti']

function = lambda x: np.exp(-1/2 * ((x - 10) / 5)**2)
data = function(x['time'])

def makegp(params):
    time_scale, label_scale = np.exp(params)
    return lgp.GP(lgp.ExpQuad(scale=time_scale, dim='time') * lgp.ExpQuad(scale=label_scale, dim='label'))

def fun(params):
    gp = makegp(params)
    gp.addx(x.reshape(-1))
    return -gp.marginal_likelihood(data.reshape(-1))

result = optimize.minimize(fun, np.log([3, 3]))
print(result)

gp = makegp(result.x)
gp.addx(x.reshape(-1), 'A')

xpred = np.empty((2, 100), dtype=x.dtype)
time_pred = np.linspace(np.min(time), np.max(time) + 1.5 * (np.max(time) - np.min(time)), xpred.shape[1])
xpred['time'][0] = time_pred
xpred['time'][1] = time_pred - delay
xpred['label'][0] = 0
xpred['label'][1] = 1
gp.addx(xpred.reshape(-1), 'B')

pred = gp.predfromdata({'A': data.reshape(-1)}, 'B')
pred = pred.reshape(xpred.shape)

fig = plt.figure('testgp2t')
fig.clf()
ax = fig.subplots(1, 1)

colors = []
for i in range(2):
    m = gvar.mean(pred[i])
    s = gvar.sdev(pred[i])
    polys = ax.fill_between(time_pred, m - s, m + s, alpha=0.5, label=label_names[i])
    colors.append(polys.get_facecolor()[0])

for _, sample in zip(range(3), gvar.raniter(pred)):
    for i in range(2):
        ax.plot(time_pred, sample[i], color=colors[i])

for i in range(2):
    ax.plot(time, data[i], '.', color=colors[i], alpha=1)

ax.legend(loc='best')
ax.set_xlabel('time')

fig.show()
