import pandas as pd
import lsqfit
import gvar
import numpy as np
import pickle
import tqdm
import sys
import lsqfitgp2 as lgp
from relu import relu
from matplotlib import pyplot as plt

# Read command line.
regions = sys.argv[1:]
#regions = ['Valle d\'Aosta']

pcm_github_url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/"
folder = "/dati-json/dpc-covid19-ita-regioni.json"
url = pcm_github_url + folder
data = pd.read_json(url, convert_dates=['data'])

gdata = data.groupby('denominazione_regione')
# use the name to group because problems with south tirol

# Initialize prediction table for covidcast.it
today = max(data.data.values)
df_temp = data[data.data == today]
df_temp = df_temp[df_temp.denominazione_regione != '']
columns_we_keep = ['data', 'codice_regione', 'denominazione_regione']
df_forecast = df_temp[columns_we_keep].copy()
new_columns = ['casi_oggi', 'morti_oggi',
               'casi_domani', 'std_casi_domani',
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
    relu_scales = args['relu_scales']
    
    out = gvar.BufferDict()
    for label in gps:
        phi = p['phi_' + label]
        if pred:
            phi = gps[label].predfromfit({'data': phi}, pred)
        out[label] = relu(phi, scale=relu_scales[label])
    return out

def time_to_number(times):
    try:
        times = pd.to_numeric(times).values
    except TypeError:
        pass
    times = np.array(times, dtype=float)
    times /= 1e9 * 60 * 60 * 24 # ns -> days
    return times

def moving_average(v, n):
    v = np.array(v)
    return np.convolve(v, np.ones(n)/n, mode='same')

def moving_errors(v, moving_average=7):
    v = np.array(v)
    d = np.diff(v, prepend=0)
    errors = np.convolve(d**2, np.ones(moving_average)/moving_average, mode='same')
    return np.where(errors != 0, np.sqrt(errors), 1)

def poisson_errors(v, moving_average=7):
    v = np.array(v)
    errors = np.where(v != 0, np.sqrt(abs(v)), 1)
    return np.convolve(errors, np.ones(moving_average)/moving_average, mode='same')

def make_poisson_data(v):
    v = np.asarray(v)
    assert len(v.shape) <= 1
    #assert np.all(v >= 0)
    return gvar.gvar(v, np.where(v != 0, np.sqrt(abs(v)), 1))

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
    
    #fig, ax = plt.subplots()
    #xx = table['nuovi_positivi'].values
    #ax.plot(moving_errors(xx))
    #ax.plot(poisson_errors(xx))
    #ratio = moving_errors(xx) / np.maximum(moving_average(xx, 5), np.ones(len(xx)))
    #ratio = moving_errors(xx) / poisson_errors(xx)
    #ax.plot(ratio)
    #m = ratio.mean()
    #ax.plot(m * np.ones(len(xx)), '--', color='tab:red', label=f'{m:.2g}')
    #ax.legend(loc=2)
    #ax.set_title(f'{region}')
    #fig.savefig(f'plots/1error_test_{region}.png', dpi=300)
    
    # Prior.
    amplitude1 = dict()
    relu_scales = dict()
    for label in fitdata:
        yy = table[label].values
        # Set the amplitude for two Gaussian Processes
        amplitude1[label] = np.sqrt((yy**2)[-30:].mean()) # quadratic mean
        #amplitude2 = np.diff(yy)[-long_scale:].std() # standard deviation of first derivative
        length = len(yy)
        #fitdata[label] += 0.3 * gvar.gvar(np.zeros(length), moving_average(yy, 7))
        
        relu_scales[label] = amplitude1[label] / 4

    # Run fit.
    
    def fdata(hyper_params):
        f = hyper_params[0]
        fdata = gvar.BufferDict(fitdata)
        for label in fitdata:
            yyy = table[label].values
            fdata[label] += f * gvar.gvar(np.zeros(length), moving_average(yyy, 6))
        return fdata
    
    def make_gp(hyper_params):
        long_scale = np.exp(hyper_params[1])
        gps = dict()
        for label in fitdata:
            gp = lgp.GP(amplitude1[label] ** 2 * lgp.ExpQuad(scale=long_scale))
            gp.addx(times, 'data')
            gp.addx(times_pred, 'pred'),
            gp.addx(times_plot, 'plot')
            gps[label] = gp
        return gps
    
    def make_args(hyper_params, **kw):
        args = dict(times=times, gps=make_gp(hyper_params), relu_scales=relu_scales)
        args.update(kw)
        return args
    
    def fitargs(hyper_params, **kw):
        args = make_args(hyper_params, **kw)
        prior = gvar.BufferDict({
            'phi_' + label: args['gps'][label].prior('data')
            for label in fitdata
        })
        return dict(data=(args, fdata(hyper_params)), prior=prior, fcn=fcn)
        
    #fit = lsqfit.nonlinear_fit(data=(args, fitdata), prior=prior, fcn=fcn)
    fit, hyper_params = lsqfit.empbayes_fit([0.3, np.log(15)], fitargs)
    
    # Compute prediction.
    predargs = make_args(hyper_params, times=times_pred, pred='pred')
    pred = fcn(predargs, fit.palt)
    predargs = make_args(hyper_params, times=times_plot, pred='plot')
    plot = fcn(predargs, fit.palt)
    
    # Save in csv for covidcast.it
    columns_dict = {'casi_oggi': table['nuovi_positivi'].values[-1],
                    'morti_oggi': table['nuovi_deceduti'].values[-1],
                    'casi_domani': pred['nuovi_positivi'][0].mean,
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
    
        # Save results.
    pickle_dict[region] = dict(
        hyper_params=hyper_params,
        y=fdata(hyper_params),
        #p=fit.palt,
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
    
# Save csv for covidcast.it
saving_path = f'../predictions/for_covidcast.it/{str(today)[:10]}_GP.csv'
df_forecast.to_csv(saving_path, index=False)

# Save results on file.
# pickle_file = 'fit_' + namedate.file_timestamp() + '.pickle'
pickle_file = 'fit.pickle'
print(f'Saving to {pickle_file}...')
pickle.dump(pickle_dict, open(pickle_file, 'wb'))