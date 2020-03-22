from matplotlib import pyplot as plt
import glob
import pandas as pd
import tqdm
import numpy as np
import os
import sys

# yes
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Set the labels to be plotted. You can use the following: `total`, `infected`,
# `removed`, `deceased`
labels = ['infected', 'removed']

# Read region data.
data = pd.read_csv(
    '../pcm-dpc-COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv',
    parse_dates=['data'],
    infer_datetime_format=True
)
regions = data['denominazione_regione'].unique()
# regions = ['Emilia Romagna', 'Basilicata', 'Valle d\'Aosta']

# Prepare figure.
fig = plt.figure('plot-dati-regioni')
fig.clf()
fig.set_size_inches((12, 7))
axs = fig.subplots(1, len(labels), sharex=True)
axsl = {l: a for l, a in zip(labels, axs)}

# Iterate over directories with a date format and which contain `dati-regioni`.
cmdline = sys.argv[1:]
if cmdline:
    directories = [f'{d}/dati-regioni' for d in cmdline]
else:
    directories = glob.glob('????-??-??/dati-regioni')
    directories.sort()
for directory in directories:
    print(f'--------- Predictions made on {directory} ---------')
    
    # Read all csv files.
    files = glob.glob(f'{directory}/*.csv')
    files.sort()
    if not files:
        print('No csv files here.')
        continue
    tables = [pd.read_csv(file, parse_dates=['data']) for file in files]
    
    # Make directory for saving figures.
    savedir = f'{directory}/plots'
    os.makedirs(savedir, exist_ok=True)
    
    # Iterate over regions.
    print('Iterating over regions...')
    for region in tqdm.tqdm(regions):
        
        # Prepare plot.
        for label, ax in axsl.items():
            ax.cla()
            ax.set_title(f'{region} ({label})')
        
        # Plot data.
        condition = data['denominazione_regione'] == region
        regiondata = data[condition]
        x = regiondata['data']
        ys_data = {
            'total': regiondata['totale_casi'],
            'infected': regiondata['totale_attualmente_positivi'],
            'removed': regiondata['totale_casi'] - regiondata['totale_attualmente_positivi'],
            'deceased': regiondata['deceduti']
        }
        for label, ax in axsl.items():
            y = ys_data[label].values
            yerr = np.where(y > 0, np.sqrt(y), 1)
            ax.errorbar(x, y, yerr=yerr, label='data', marker='.', capsize=0, linestyle='')
        
        # Plot predictions.
        for filename, table in zip(files, tables):
            condition = table['denominazione_regione'] == region
            regiontable = table[condition]
            
            # times
            x = regiontable['data']
            
            # total and infected
            ys = {
                'total': regiontable['totale_casi'],
                'infected': regiontable['totale_attualmente_positivi']
            }
            yserr = {
                'total': regiontable['std_totale_casi'],
                'infected': regiontable['std_totale_attualmente_positivi']
            }
            
            # removed
            try:
                ys['removed'] = regiontable['guariti_o_deceduti']
                yserr['removed'] = regiontable['std_guariti_o_deceduti']
            except KeyError:
                try:
                    ys['removed'] = regiontable['dimessi_guariti'] + regiontable['deceduti']
                    yserr['removed'] = np.hypot(regiontable['std_dimessi_guariti'], regiontable['std_deceduti'])
                except KeyError:
                    ys['removed'] = ys['total'] - ys['infected']
                    yserr['removed'] = np.hypot(yserr['total'], yserr['infected'])
            
            # deceased
            try:
                ys['deceased'] = regiontable['deceduti']
                yserr['deceased'] = regiontable['std_deceduti']
            except KeyError:
                try:
                    ys['deceased'] = regiontable['guariti_o_deceduti'] - regiontable['dimessi_guariti']
                    yserr['deceased'] = np.hypot(regiontable['std_guariti_o_deceduti'], regiontable['std_dimessi_guariti'])
                except:
                    pass
        
            # plot
            for label, ax in axsl.items():
                if label in ys:
                    y = ys[label].values
                    yerr = yserr[label].values
                    nicename = os.path.splitext(os.path.split(filename)[-1])[0].replace('model-', '')
                    ax.errorbar(x, y, yerr=yerr, label=nicename, marker='', capsize=2, linestyle='')
        
        
        # Embellishments.
        for ax in axs:
            ax.set_yscale('symlog', linthreshy=1, linscaley=0.3, subsy=np.arange(2, 9 + 1))
            ax.grid(linestyle=':')
            if ax.get_ylim()[0] < 0:
                ax.set_ylim(0, ax.get_ylim()[1])
        axs[0].legend(loc='best')
        axs[0].set_ylabel('people')
        fig.autofmt_xdate(rotation=70)
        fig.tight_layout()
        
        # Save figure
        fig.savefig(f'{savedir}/{region}.png')
