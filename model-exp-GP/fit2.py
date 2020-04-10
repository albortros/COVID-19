import pandas as pd
import gvar
from autograd import numpy as np
import autograd
import pickle
import tqdm
import sys
import lsqfitgp2 as lgp
from relu import relu
from scipy import optimize
from autograd.scipy import linalg

# Read command line.
regions = sys.argv[1:]
#regions = ['Abruzzo', 'Basilicata', 'Lombardia', 'Veneto']
labels = ['nuovi_positivi', 'nuovi_deceduti']

pcm_github_url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/"
folder = "/dati-json/dpc-covid19-ita-regioni.json"
url = pcm_github_url + folder
data = pd.read_json(url, convert_dates=['data'])
if not regions:
    regions = data['denominazione_regione'].unique()

gdata = data.groupby('denominazione_regione')
# use the name to group because problems with south tirol

# This dictionary will be saved on file at the end.
pickle_dict = dict()

def time_to_number(times):
    try:
        times = pd.to_numeric(times).values
    except TypeError:
        pass
    times = np.array(times, dtype=float)
    times /= 1e9 * 60 * 60 * 24 # ns -> days
    return times

print('Iterating over regions...')
for region in tqdm.tqdm(regions):
    table = gdata.get_group(region)

    # Times for data.
    times = time_to_number(table['data'])
    time_zero = times[0]
    times -= time_zero

    # Times for prediction.
    lastdate = table['data'].max()
    dates_pred = pd.date_range(lastdate, periods=60, freq='1D')[1:]
    times_pred = time_to_number(dates_pred) - time_zero

    # Times for plot.
    firstdate = table['data'].min()
    dates_plot = pd.date_range(firstdate, dates_pred[-1], 600)
    times_plot = time_to_number(dates_plot) - time_zero
    
    data_list = []
    for label in labels:
        if label == 'nuovi_deceduti':
            # Adding 'nuovi_deceduti' column, first value added "manually"
            first_value = 0
            if region == 'Lombardia':
                first_value = 4
            other_values = np.diff(table['deceduti'].values)
            data_list.append([first_value] + list(other_values))
        else:
            data_list.append(table[label].values)
    data = np.stack(data_list)
    
    def makex(times):
        x = np.empty((len(labels), len(times)), dtype=[
            ('time', float),
            ('label', int)
        ])
        x['label'] = np.arange(len(labels)).reshape(-1, 1)
        x['time'] = times
        return x
        
    x = makex(times)
    assert data.shape == x.shape
    def makegp(hyperparams):
        p = np.exp(hyperparams)
        longscale = p[0]
        shortscale = p[1]
        longvars = p[2:2+len(labels)]
        shortvars = p[2+len(labels):2+2*len(labels)]
        kernel = lgp.ExpQuad(scale=longscale, dim='time') * lgp.Categorical(cov=np.diag(longvars), dim='label')
        kernel += lgp.ExpQuad(scale=shortscale, dim='time') * lgp.Cos(scale=shortscale, dim='time') * lgp.Categorical(cov=np.diag(shortvars), dim='label')
        gp = lgp.GP(kernel)
        gp.addx(x, 'data')
        return gp
    
    def chisq(prior):
        chol = linalg.cholesky(gvar.evalcov(prior), lower=True)
        def f(x):
            res = x - gvar.mean(prior)
            diagres = linalg.solve_triangular(chol, res, lower=True)
            return 1/2 * np.sum(diagres ** 2)
        return f
    
    hyperprior = gvar.log(gvar.gvar([20, 2], [6, 2]))
    hyperchisq = chisq(hyperprior)
    def fun(hyperparams):
        gp = makegp(hyperparams)
        return -gp.marginal_likelihood({'data': data}) + hyperchisq(hyperparams[:2])
    
    p0 = np.log(np.concatenate([
        [14, 1],
        np.max(data, axis=-1) ** 2,
        np.max(data, axis=-1) ** 2
    ]))
    result = optimize.minimize(autograd.value_and_grad(fun), p0, jac=True)
    hyperparams = gvar.exp(gvar.gvar(result.x, result.hess_inv))
    params = gvar.BufferDict(**{
        'longscale': hyperparams[0],
        'shortscale': hyperparams[1]
    }, **{
        f'longstd_{label}': gvar.sqrt(hyperparams[2 + i])
        for i, label in enumerate(labels)
    }, **{
        f'shortstd_{label}': gvar.sqrt(hyperparams[2 + len(labels) + i])
        for i, label in enumerate(labels)
    })
    
    gp = makegp(result.x)
    xpred = makex(times_pred)
    gp.addx(xpred, 'pred')
    xplot = makex(times_plot)
    gp.addx(xplot, 'plot')
    pred = gp.predfromdata({'data': data}, 'pred', keepcorr=False)
    plot = gp.predfromdata({'data': data}, 'plot', keepcorr=False)
    
    def tobufdict(uy):
        return gvar.BufferDict({
            label: uy[i]
            for i, label in enumerate(labels)
        })
    
    # Save results.
    pickle_dict[region] = dict(
        minresult=result,
        params=params,
        y=tobufdict(data),
        table=table,
        time_zero=time_zero,
        pred=tobufdict(pred),
        plot=tobufdict(plot),
        dates=dict(pred=dates_pred, plot=dates_plot)
    )
    
# Save results on file.
# pickle_file = 'fit_' + namedate.file_timestamp() + '.pickle'
pickle_file = 'fit2.pickle'
print(f'Saving to {pickle_file}...')
pickle.dump(pickle_dict, open(pickle_file, 'wb'))
