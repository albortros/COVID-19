import pandas as pd
import lsqfit
import gvar
import numpy as np
import pickle
import namedate
import fitlsqdefs
import tqdm
import sys

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Read command line.
regions = sys.argv[1:]

# Read region data.
data = pd.read_csv(
    '../pcm-dpc-COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv',
    parse_dates=['data'],
    infer_datetime_format=True
)
gdata = data.groupby('denominazione_regione')
# use the name to group because problems with south tirol

# Read additional csv to know the population of each region.
regioninfo = pd.read_csv('../shared_data/dati_regioni.csv')

# This dictionary will be saved on file at the end.
pickle_dict = dict()

print('Iterating over regions...')
for region in regions if regions else tqdm.tqdm(data['denominazione_regione'].unique()):
    table = gdata.get_group(region)

    # Times.
    times = fitlsqdefs.time_to_number(table['data'])

    # Data.
    I_data = table['totale_attualmente_positivi'].values
    R_data = table['totale_casi'].values - I_data
    I_data = fitlsqdefs.make_poisson_data(I_data)
    R_data = fitlsqdefs.make_poisson_data(R_data)
    fitdata = gvar.BufferDict(I=I_data, R=R_data)
    
    # Population prior.
    totpop = regioninfo[regioninfo['denominazione_regione'] == region]['popolazione'].values[0]
    min_pop = np.max(gvar.mean(R_data + I_data))
    _totpop = totpop - min_pop

    # Prior.
    prior = gvar.BufferDict({
        'log(R0)': gvar.gvar(np.log(1), np.log(10)),
        'log(lambda)': gvar.gvar(np.log(1), np.log(10)),
        'log(_population)': gvar.gvar(np.log(_totpop), np.log(20)),
        'log(I0_pop)': gvar.gvar(np.log(10), np.log(100))
    })

    # Run fit.
    args = dict(times=times, min_pop=min_pop)
    fit = lsqfit.nonlinear_fit(data=(args, fitdata), prior=prior, fcn=fitlsqdefs.fcn)

    # Save results.
    pickle_dict[region] = dict(
        y=fitdata,
        p=fit.p,
        prior=prior,
        log=fit.format(maxline=True),
        chi2=fit.chi2,
        dof=fit.dof,
        pvalue=fit.Q,
        table=table,
        min_pop=min_pop
    )

# Save results on file.
pickle_file = 'fitlsq_' + namedate.file_timestamp() + '.pickle'
print(f'Saving to {pickle_file}...')
pickle.dump(pickle_dict, open(pickle_file, 'wb'))
