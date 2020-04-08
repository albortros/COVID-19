import lsqfitgp2 as lgp
from matplotlib import pyplot as plt
from autograd import numpy as np
import gvar
from scipy import optimize
import autograd

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
data_error = 0.05
data_mean = function(x['time']) + data_error * np.random.randn(*x.shape)
data_mean[1] += 0.02 * time
data = gvar.gvar(data_mean, np.full_like(data_mean, data_error))

x = lgp.StructuredArray(x)
def makegp(params):
    time_scale, label_scale = np.exp(params[:2])
    delay = params[2]
    gp = lgp.GP(lgp.RatQuad(scale=time_scale, dim='time', alpha=1) * lgp.ExpQuad(scale=label_scale, dim='label'))
    x['time'] = np.array([time, time - delay])
    gp.addx(x, 'A')
    return gp

def fun(params):
    gp = makegp(params)
    return -gp.marginal_likelihood({'A': data})

result = optimize.minimize(autograd.value_and_grad(fun), [2, 2, 10], jac=True)
params = gvar.gvar(result.x, result.hess_inv)
print(result)
print('time scale = {}'.format(gvar.exp(params[0])))
corr = lgp.ExpQuad(scale=np.exp(result.x[1]))(0, 1)
print('correlation = {:.3g} (equiv. scale = {})'.format(corr, params[1]))
print('delay = {}'.format(params[2]))

gp = makegp(result.x)

xpred = np.empty((2, 100), dtype=x.dtype)
time_pred = np.linspace(np.min(time), np.max(time) + 1.5 * (np.max(time) - np.min(time)), xpred.shape[1])
xpred['time'][0] = time_pred
xpred['time'][1] = time_pred - result.x[2]
xpred['label'][0] = 0
xpred['label'][1] = 1
gp.addx(xpred, 'B')

pred = gp.predfromdata({'A': data}, 'B')

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
    ax.errorbar(time, gvar.mean(data[i]), yerr=gvar.sdev(data[i]), fmt='.', color=colors[i], alpha=1)

ax.legend(loc='best')
ax.set_xlabel('time')

fig.show()
