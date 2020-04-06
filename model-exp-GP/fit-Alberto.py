import pandas as pd
import lsqfit
import gvar
import numpy as np
import pickle
import tqdm
import sys
import lsqfitgp2 as lgp

# Read command line.
regions = sys.argv[1:]

data = pd.read_json(
    '../pcm-dpc-COVID-19/dati-json/dpc-covid19-ita-regioni.json',
    convert_dates=['data']
)

gdata = data.groupby('denominazione_regione')
# use the name to group because problems with south tirol

# Initialize prediction table for covidcast.it
today = max(data.data.values)
df_temp = data[data.data == today]
df_temp = df_temp[df_temp.denominazione_regione != '']
columns_we_keep = ['data', 'codice_regione', 'denominazione_regione']
df_forecast = df_temp[columns_we_keep].copy()
new_columns = ['casi_domani', 'std_casi_domani',
               'morti_domani', 'std_morti_domani',
               'casi_dopodomani', 'std_casi_dopodomani',
               'morti_dopodomani', 'std_morti_dopodomani']
for c in new_columns:
    df_forecast[c] = 0

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
    #assert np.all(v >= 0)
    return gvar.gvar(v, np.where(v > 0, np.sqrt(abs(v)), 1))

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
    
    # Adding 'nuovi_deceduti' column, first value added "manually"
    first_value = 0
    if region == 'Lombardia':
        first_value = 4
    other_values = np.diff(table['deceduti'].values)
    table['nuovi_deceduti'] = [first_value] + list(other_values)
    
    # Data.
    fitdata = gvar.BufferDict({
        label: make_poisson_data(table[label].values)
        for label in ['nuovi_positivi', 'nuovi_deceduti']
    })
    
    # Prior.
    gps = dict()
    for label in fitdata:
        yy = table[label].values
        # Set the amplitude for two Gaussian Processes
        # amplitude1 is the amplitude of the long-scale GP
        # amplitude2 is the amplitude of the short-scale GP
        amplitude1 = np.sqrt((yy**2).mean())
        amplitude2 = np.diff(yy).std()
        gp = lgp.GP(amplitude1 ** 2 * lgp.ExpQuad(scale=30) + amplitude2 ** 2 * lgp.ExpQuad())
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
    
    # Save in csv for covidcast.it
    columns_dict = {'casi_domani': pred['nuovi_positivi'][0].mean,
                    'std_casi_domani': pred['nuovi_positivi'][0].sdev,
                    'morti_domani': pred['nuovi_deceduti'][0].mean,
                    'std_morti_domani': pred['nuovi_deceduti'][0].sdev,
                    'casi_dopodomani': pred['nuovi_positivi'][1].mean,
                    'std_casi_dopodomani': pred['nuovi_positivi'][1].sdev,
                    'morti_dopodomani': pred['nuovi_deceduti'][1].mean,
                    'std_morti_dopodomani': pred['nuovi_deceduti'][1].sdev}
    
    for c in new_columns:
        value = np.round(columns_dict[c])
        df_forecast.loc[df_forecast.denominazione_regione == region, c] = value
    
# Save csv for covidcast.it
saving_path = f'../predictions/for_covidcast.it/{str(today)[:10]}_GP.csv'
df_forecast.to_csv(saving_path, index=False)

# Save results on file.
# pickle_file = 'fit_' + namedate.file_timestamp() + '.pickle'
pickle_file = 'fit.pickle'
print(f'Saving to {pickle_file}...')
#pickle.dump(pickle_dict, open(pickle_file, 'wb'))

