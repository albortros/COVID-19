import lsqfitgp
import lsqfit
from matplotlib import pyplot as plt
import numpy as np
import gvar

plot_simulated_lines = False

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-5, 15, 300)

gp = lsqfitgp.GP(xdata, lsqfitgp.ExpQuad(scale=3) + 0.05**2 * lsqfitgp.Matern12(), xpred=xpred)

true_par = dict(
    phi=np.sin(xdata),
    y0=10
)

def fcn(data_or_pred, p):
    if data_or_pred == 'data':
        phi = p['phi']
    elif data_or_pred == 'pred':
        phi = gp.pred(p['phi'])
    elif data_or_pred == 'predalt':
        phi = gp.predalt(p['phi'])
    else:
        raise KeyError(data_or_pred)
    
    return gvar.tanh(1 + phi) + p['y0']

yerr = 0.05
ysdev = yerr * np.ones(len(xdata))
ymean = fcn('data', true_par) + ysdev * np.random.randn(len(ysdev))
y = gvar.gvar(ymean, ysdev)

prior = dict(
    phi=gp.prior(),
    y0=gvar.gvar(0, 1000)
)

p0=dict(
    phi=np.random.multivariate_normal(np.zeros(len(xdata)), gvar.evalcov(prior['phi']))
)

fit = lsqfit.nonlinear_fit(data=('data', y), prior=prior, fcn=fcn, p0=p0)
print(fit.format(maxline=True))

ypred = fcn('pred', fit.p)
ypredalt = fcn('pred', fit.palt)

phipred = gp.pred(fit.p['phi'])
phipredalt = gp.predalt(fit.palt['phi'])

fig = plt.figure('testgp2')
fig.clf()
axs = fig.subplots(1, 2)

for ax, variable in zip(axs, ['y', 'phi']):
    ax.set_title(variable)
    
    for label in 'pred', 'predalt':
        pred = eval(variable + label)

        m = gvar.mean(pred)
        s = gvar.sdev(pred)

        patch = ax.fill_between(xpred, m - s, m + s, label=label, alpha=0.5)
        color = patch.get_facecolor()[0]
    
        if plot_simulated_lines:
            cov = gvar.evalcov(pred)
            simulated_lines = np.random.multivariate_normal(m, cov, size=5)
            ax.plot(xpred, simulated_lines.T, '-', color=color)

axs[0].errorbar(xdata, gvar.mean(y), yerr=gvar.sdev(y), fmt='k.', label='data')

for ax in axs:
    ax.legend(loc='best')

fig.tight_layout()
fig.show()
