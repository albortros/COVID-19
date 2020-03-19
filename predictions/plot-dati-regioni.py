from matplotlib import pyplot as plt
import glob
import pandas as pd
import tqdm
import numpy as np
import os

# yes
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Get list of directories with a date format and which contain `dati-regioni`.
directories = glob.glob('????-??-??/dati-regioni')

# Read region data.
data = pd.read_csv(
    '../pcm-dpc-COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv',
    parse_dates=['data'],
    infer_datetime_format=True
)
regions = data['denominazione_regione'].unique()

# Prepare figure.
fig = plt.figure('predictions-plot')
fig.clf()
ax = fig.subplots(1, 1)

for directory in directories:
    print(f'--------- Predictions made on {directory} ---------')
    
    # Read all csv files.
    files = glob.glob(f'{directory}/*.csv')
    tables = [pd.read_csv(file, parse_dates=['data']) for file in files]
    
    # Make directory for saving figures.
    savedir = f'{directory}/plots'
    os.makedirs(savedir, exist_ok=True)
    
    # Iterate over regions.
    print('Iterating over regions...')
    for region in tqdm.tqdm(regions):
        
        # Clear plot.
        ax.cla()
        ax.set_yscale('symlog', linthreshy=1)
        ax.set_title(region)
        
        # Plot data.
        condition = data['denominazione_regione'] == region
        regiondata = data[condition]
        x = regiondata['data']
        ys = {
            'total': regiondata['totale_casi'],
            'infected': regiondata['totale_attualmente_positivi']
        }
        for label in ys:
            y = ys[label].values
            yerr = np.where(y > 0, np.sqrt(y), 1)
            ax.errorbar(x, y, yerr=yerr, label=label + ' (data)', marker='.', capsize=0)
        
        # Plot predictions.
        for filename, table in zip(files, tables):
            condition = table['denominazione_regione'] == region
            regiontable = table[condition]
            x = regiontable['data']
            ys = {
                'total': regiontable['totale_casi'],
                'infected': regiontable['totale_attualmente_positivi']
            }
            yserr = {
                'total': regiontable['std_totale_casi'],
                'infected': regiontable['std_totale_attualmente_positivi']
            }
            for label in ys:
                y = ys[label].values
                yerr = yserr[label].values
                ax.errorbar(x, y, yerr=yerr, label=label + f' {filename}', marker='', capsize=2)
        
        # Embellishments.
        ax.grid(linestyle=':')
        ax.legend(loc='upper left')
        fig.autofmt_xdate()
        fig.tight_layout()
        
        # Save figure
        fig.savefig(f'{savedir}/{region}.pdf')