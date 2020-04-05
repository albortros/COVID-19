import pandas as pd
import lsqfit
import gvar
import numpy as np
import pickle
import tqdm
import sys
import lsqfitgp2 as lgp

# Read command line.
regions = ['Veneto']

data = pd.read_json(
    '../pcm-dpc-COVID-19/dati-json/dpc-covid19-ita-regioni.json',
    convert_dates=['data']
)

gdata = data.groupby('denominazione_regione')
# use the name to group because problems with south tirol

# This dictionary will be saved on file at the end.
pickle_dict = dict()

# Model function.
def fcn(args, p):
    times = args['times']
    gps = args['gps']
    pred = args.get('pred', '')
    
    out = gvar.BufferDict()
    for label in gps:
        phi = p['phi_' + label]
        if pred:
            phi = gps[label].predfromfit({'data': phi}, pred)
        out[label] = phi
    
    return out

def time_to_number(times):
    try:
        times = pd.to_numeric(times).values
    except TypeError:
        pass
    times = np.array(times, dtype=float)
    times /= 1e9 * 60 * 60 * 24 # ns -> days
    return times

def make_poisson_data(v):
    v = np.asarray(v)
    assert len(v.shape) <= 1
    assert np.all(v >= 0)
    return gvar.gvar(v, np.where(v > 0, np.sqrt(v), 1))

print('Iterating over regions...')
for region in regions if regions else tqdm.tqdm(data['denominazione_regione'].unique()):
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
    dates_plot = pd.date_range(firstdate, dates_pred[-1], 300)
    times_plot = time_to_number(dates_plot) - time_zero
    
    # Data.
    fitdata = gvar.BufferDict({
        label: make_poisson_data(table[label].values)
        for label in ['nuovi_positivi']
    })
    
    # Prior.
    gps = dict()
    for label in fitdata:
        gp = lgp.GP(500 ** 2 * lgp.ExpQuad(scale=30) + 50 ** 2 * lgp.ExpQuad())
        gp.addx(times, 'data')
        gp.addx(times_pred, 'pred'),
        gp.addx(times_plot, 'plot')
        gps[label] = gp
    prior = gvar.BufferDict({
        'phi_' + label: gps[label].prior('data')
        for label in fitdata
    })

    # Run fit.
    args = dict(times=times, gps=gps)
    fit = lsqfit.nonlinear_fit(data=(args, fitdata), prior=prior, fcn=fcn)
    
    # Compute prediction.
    predargs = dict(times=times_pred, pred='pred', gps=gps)
    pred = fcn(predargs, fit.palt)
    predargs = dict(times=times_plot, pred='plot', gps=gps)
    plot = fcn(predargs, fit.palt)

    # Save results.
    pickle_dict[region] = dict(
        y=fitdata,
        p=fit.palt,
        log=fit.format(maxline=True),
        chi2=fit.chi2,
        dof=fit.dof,
        pvalue=fit.Q,
        table=table,
        time_zero=time_zero,
        pred=pred,
        plot=plot,
        dates=dict(pred=dates_pred, plot=dates_plot)
    )

# Save results on file.
# pickle_file = 'fit_' + namedate.file_timestamp() + '.pickle'
pickle_file = 'fit.pickle'
print(f'Saving to {pickle_file}...')
pickle.dump(pickle_dict, open(pickle_file, 'wb'))