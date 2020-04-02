import lsqfitgp2 as lgp
from lsqfitgp2 import _linalg
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import gvar

xdata1d = np.linspace(-4, 4, 10)
xpred1d = np.linspace(-10, 10, 50)
xdata = np.array(np.meshgrid(xdata1d, xdata1d)).reshape(2, -1)
xpred = np.array(np.meshgrid(xpred1d, xpred1d)).reshape(2, -1)
y = np.cos(xdata[0]) * np.cos(xdata[1])

gp = lgp.GP(lgp.ExpQuad(scale=3, dim=0) * lgp.ExpQuad(scale=3, dim=1), checkpos=False, solver='gersh')
gp.addx(xdata, 'pere')
gp.addx(xpred, 'banane')

print('fit...')
m, cov = gp.predfromdata({'pere': y}, 'banane', raw=True)

print('samples...')
# samples = np.random.multivariate_normal(m, cov)
# dec = _linalg.LowRank(cov, rank=300)
# samples = m + dec._V @ (np.random.randn(len(dec._w)) * dec._w)
samples = m + _linalg.CholGersh(cov, eps=1e-5)._L @ np.random.randn(len(cov))

print('plot...')
fig = plt.figure('testgp2q')
fig.clf()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(*xdata, y, color='black')
ax.plot_surface(*xpred.reshape(2, len(xpred1d), len(xpred1d)), samples.reshape(len(xpred1d), len(xpred1d)), alpha=0.8)

fig.show()
