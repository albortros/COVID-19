import pandas as pd
import lsqfit
import gvar
import numpy as np
import pickle
import namedate
import tqdm
import sys
import lsqfitgp as lgp

# Read command line.
regions = sys.argv[1:]

# Read region data.
data = pd.read_csv(
    '../pcm-dpc-COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv',
    parse_dates=['data']
)
gdata = data.groupby('denominazione_regione')
# use the name to group because problems with south tirol

# This dictionary will be saved on file at the end.
pickle_dict = dict()

# Model function.
def fcn(args, p):
    times = args['times']
    gps = args['gps']
    pred = args.get('pred', False)
    
    out = gvar.BufferDict()
    for label in gps:
        phi = p['phi_' + label]
        if pred:
            phi = gps[label].pred(phi)
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
    dates_pred = pd.date_range(lastdate, periods=15, freq='1D')[1:]
    times_pred = time_to_number(dates_pred) - time_zero

    # Times for plot.
    firstdate = table['data'].min()
    dates_plot = pd.date_range(firstdate, dates_pred[-1], 100)
    times_plot = time_to_number(dates_plot) - time_zero
    
    # Data.
    fitdata = gvar.BufferDict({
        label: np.log(make_poisson_data(table[label].values)) / (10 + times)
        for label in ['totale_casi']
    })
    
    # Prior.
    times_gppred = np.concatenate([times_plot, times_pred])
    gps = {
        label: lgp.GP(times, lgp.NNKernel(scale=3, sigma0=20, loc=-20), times_gppred)
        for label in fitdata
    }
    prior = gvar.BufferDict(**{
        't0_' + label: gvar.gvar('-20(.1)')
        for label in []#fitdata
    }, **{
        'phi_' + label: gps[label].prior()
        for label in fitdata
    })

    # Run fit.
    args = dict(times=times, gps=gps)
    fit = lsqfit.nonlinear_fit(data=(args, fitdata), prior=prior, fcn=fcn)
    
    # Compute prediction.
    predargs = dict(times=times_gppred, pred=True, gps=gps)
    # prediction = fcn(predargs, fit.p)
    prediction = gvar.BufferDict(totale_casi=gps['totale_casi'].fitpredalt(fitdata['totale_casi']))

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
        prediction=prediction,
        dates_plot=dates_plot,
        dates_pred=dates_pred
    )

# Save results on file.
pickle_file = 'fit_' + namedate.file_timestamp() + '.pickle'
print(f'Saving to {pickle_file}...')
pickle.dump(pickle_dict, open(pickle_file, 'wb'))
