# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:08:00 2020

@author: jacop
"""

import numpy as np
import pylab as pl
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x, a, b, c):
    return a * x *x + b*x + c

# test data and error
x = np.linspace(-10, 10, 100)
y0 = -0.07 * x * x + 0.5 * x + 2.
noise = np.random.normal(0.0, 1.0, len(x))
y = y0 + noise

# curve fit [with only y-error]
popt, pcov = curve_fit(func, x, y, sigma=1./(noise*noise))
print(popt,'\n', pcov)
perr = np.sqrt(np.diag(pcov))

#print fit parameters and 1-sigma estimates
print('fit parameter 1-sigma error')
print('———————————–')
for i in range(len(popt)):
    print(str(popt[i])+' +- '+str(perr[i]))

# prepare confidence level curves
nstd = 5. # to draw 5-sigma intervals
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

fit = func(x, *popt)
fit_up = func(x, *popt_up)
fit_dw = func(x, *popt_dw)

#plot
fig, ax = plt.subplots(1)
pl.rcParams['xtick.labelsize'] = 18
pl.rcParams['ytick.labelsize'] = 18
pl.rcParams['font.size']= 20
plt.errorbar(x, y0, yerr=noise, xerr=0, hold=True, ecolor='k', fmt='none', label='data')

plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.title('fit with only Y-error', fontsize=18)
plt.plot(x, fit, 'r', lw=2, label='best fit curve')
plt.plot(x, y0, 'k-', lw=2, label='True curve')
ax.fill_between(x, fit_up, fit_dw, alpha=.25, label='5-sigma interval')
plt.legend(loc='lower right',fontsize=18)
plt.show()