import lsqfitgp2 as lgp
from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure('testgp2v')
fig.clf()
ax = fig.subplots(1, 1)

for label, kernel in [('expquad', lgp.ExpQuad()), ('cos', lgp.ExpQuad() * lgp.Cos())]:
    gp = lgp.GP(kernel)
    x = np.linspace(-10, 10, 1000)
    gp.addx(x, 'x')
    cov = gp.prior(raw=True)['x', 'x']
    samples = np.random.multivariate_normal(np.zeros_like(x), cov, size=1)
    ax.plot(x, samples.T, alpha=0.3, label=label)

ax.legend(loc='best')
fig.show()
