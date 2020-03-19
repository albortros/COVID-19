from matplotlib import pyplot as plt
import glob
import pandas as pd
import tqdm
import numpy as np
import os

# yes
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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
axs = fig.subplots(1, 2, sharex=True, sharey=True)

# Iterate over directories with a date format and which contain `dati-regioni`.
directories = glob.glob('????-??-??/dati-regioni')
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
        
        # Prepare plot.
        labels = ['total', 'infected']
        for ax, label in zip(axs, labels):
            ax.cla()
            ax.grid(linestyle=':')
            ax.set_yscale('symlog', linthreshy=1)
            ax.set_title(f'{region} ({label})')
        
        # Plot data.
        condition = data['denominazione_regione'] == region
        regiondata = data[condition]
        x = regiondata['data']
        ys = {
            'total': regiondata['totale_casi'],
            'infected': regiondata['totale_attualmente_positivi']
        }
        for ax, label in zip(axs, ys):
            y = ys[label].values
            yerr = np.where(y > 0, np.sqrt(y), 1)
            ax.errorbar(x, y, yerr=yerr, label='data', marker='.', capsize=0)
        
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
            for ax, label in zip(axs, ys):
                y = ys[label].values
                yerr = yserr[label].values
                nicename = os.path.splitext(os.path.split(filename)[-1])[0]
                ax.errorbar(x, y, yerr=yerr, label=nicename, marker='', capsize=2)
        
        # Embellishments.
        axs[1].legend(loc='lower right', fontsize='small')
        axs[0].set_ylabel('people')
        fig.autofmt_xdate()
        fig.tight_layout()
        
        # Save figure
        fig.savefig(f'{savedir}/{region}.pdf')
