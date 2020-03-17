import pandas as pd
# import pymc3 as pm
import lsqfit
import gvar
import numpy as np
from matplotlib import pyplot as plt
import pickle
import namedate
import fitlsqdefs
import params

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Read region data.
data = pd.read_csv(
    '../pcm-dpc-COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv',
    parse_dates=['data'],
    infer_datetime_format=True
)
gdata = data.groupby('denominazione_regione')
# use the name to group because problems with south tirol

# Times.
regione = params.regione
print (regione)
table = gdata.get_group(regione)
times = fitlsqdefs.time_to_number(table['data'])

# Data.
I_data = table['totale_attualmente_positivi'].values
I_data = gvar.gvar(I_data, np.sqrt(I_data))
R_data = table['totale_casi'].values - table['totale_attualmente_positivi'].values
R_data = gvar.gvar(R_data, np.sqrt(R_data))
fitdata = gvar.BufferDict(I=I_data, R=R_data)

# Prior.
prior = gvar.BufferDict({
    'log(R0)': gvar.gvar('0(2)'),
    'log(lambda)': gvar.gvar('0(2)'),
    'log(_population)': gvar.gvar(np.log(1e6), np.log(20)),
    'log(I0_pop)': gvar.gvar(np.log(1e2), 1)
})
min_pop = np.max(gvar.mean(R_data + I_data))

# Run fit.
args = dict(times=times, min_pop=min_pop)
fit = lsqfit.nonlinear_fit(data=(args, fitdata), prior=prior, fcn=fitlsqdefs.fcn)
fitlog = fit.format(maxline=True)
print(fitlog)

# Save results.
pickle_dict = dict(
    y=fitdata,
    p=fit.p,
    prior=prior,
    log=fitlog,
    chi2=fit.chi2,
    dof=fit.dof,
    pvalue=fit.Q,
    table=table,
    min_pop=min_pop
)
pickle_file = 'fitlsq_' + namedate.file_timestamp() + '.pickle'
#print(f'Saving to {pickle_file}...')
pickle.dump(pickle_dict, open(pickle_file, 'wb'))
