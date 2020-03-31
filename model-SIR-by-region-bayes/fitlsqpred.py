import pickle
import sys
import pandas as pd
import numpy as np
import gvar
import fitlsqdefs
import os
import collections
import tqdm

# Load fit result.
fits = pickle.load(open(sys.argv[1], 'rb'))
prior_option = fits.pop('prior_option')

# Output dictionary that will be converted to dataframe.
output = collections.defaultdict(list)
# (a dictionary where virtually non-existing items are an empty list)

# Iterate over regions.
print('Iterating over regions...')
for region, fit in tqdm.tqdm(fits.items()):

    # Dates at which to predict.
    lastdate = fit['table']['data'].max()
    futuredates = pd.date_range(start=lastdate, periods=15, freq='1D')[1:]
    x = fitlsqdefs.time_to_number(futuredates) - fit['time_zero']
    
    # Compute prediction.
    p = fitlsqdefs.rescale_sdev(fit['p'], np.sqrt(fit['chi2'] / fit['dof']))
    y = fitlsqdefs.fcn(dict(times=x, min_pop=fit['min_pop']), p)
    
    # Write prediction in table.
    output['denominazione_regione'] += [region] * len(futuredates)
    output['data'] += list(futuredates)
    
    totale_casi = y['R'] + y['I']
    output['totale_casi'] += list(gvar.mean(totale_casi))
    output['std_totale_casi'] += list(gvar.sdev(totale_casi))

    output['totale_positivi'] += list(gvar.mean(y['I']))
    output['std_totale_positivi'] += list(gvar.sdev(y['I']))
    
    output['guariti_o_deceduti'] += list(gvar.mean(y['R']))
    output['std_guariti_o_deceduti'] += list(gvar.sdev(y['R']))

# Convert to dataframe.
regional = pd.DataFrame(output)

# Sum predictions to make national forecast.
output = collections.defaultdict(list)
grouped_by_date = regional.groupby('data')
for date in regional['data'].unique():
    table = grouped_by_date.get_group(date)
    output['data'].append(date)
    for label in 'totale_casi', 'totale_positivi', 'guariti_o_deceduti':
        output[label].append(np.sum(table[label]))
        stdlabel = 'std_' + label
        output[stdlabel].append(np.sqrt(np.sum(table[stdlabel] ** 2)))
national = pd.DataFrame(output)

for df, subdir in [(regional, 'dati-regioni'), (national, 'dati-andamento-nazionale')]:
    # Make directory where file has to be saved.
    directory = f'../predictions/{lastdate.year:04d}-{lastdate.month:02d}-{lastdate.day:02d}/{subdir}'
    os.makedirs(directory, exist_ok=True)

    # Save to file.
    if prior_option == 'weakpop':
        filename = 'model-SIR-by-region-bayes.csv'
    elif prior_option == 'truepop':
        filename = 'model-SIR-region-truepop.csv'
    filepath = f'{directory}/{filename}'
    print(f'Saving to {filepath}...')
    df.to_csv(filepath, index=False)
