import re
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

with open('table2.txt') as file:
    txt = file.read()

txt = txt.replace(',', '')

linereg = re.compile(r'([a-zA-Z ]+?)\s*([0-9\.]+)\s*([0-9\.]+)\s*\[([0-9\.]+)\s*-\s*([0-9\.]+)\]\s*([0-9\.]+)\s*\[([0-9\.]+)\s*-\s*([0-9\.]+)\]\s*([0-9\.]+)\s*\[([0-9\.]+)\s*-\s*([0-9\.]+)\]\s*([0-9\.]+)\s*\[([0-9\.]+)\s*-\s*([0-9\.]+)\]')

table = []
names = []
for match in linereg.finditer(txt):
    values = match.groups()
    names.append(values[0])
    numbers = values[1:]
    assert len(numbers) == 13
    table.append(list(map(float, numbers)))
table = np.array(table).T

obs, model, modellow, modelup = table[:4]
errdown = (model - modellow) / 2
errup = (modelup - model) / 2
    
fig = plt.figure('plottable2')
fig.clf()
ax = fig.subplots(1, 1)

ax.plot(obs, 'xk', label='data', markersize=15, zorder=-1)
ax.errorbar(names, model, yerr=(errdown, errup), fmt='.', capsize=4, label='forecast ($\\pm 1 \\sigma$)', color='red')

residuals = (obs - model) / np.where(obs < model, errdown, errup)
chisq = np.sum(residuals ** 2)
dof = len(obs) - 1
p = stats.chi2(dof).sf(chisq)
print(f'chisq/dof [dof] (p) = {chisq/dof:.1f} [{dof}] ({p:.2g})')

ax.legend(loc='best')
ax.set_yscale('log')
ax.grid(linestyle=':')
ax.set_ylabel('Dead people')
ax.set_title('Predictions and actual data for March 28th\n(from Flaxman et al., Table 2)')

fig.autofmt_xdate()
fig.tight_layout()
fig.show()
