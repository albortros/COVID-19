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
labels = ['infected', 'removed', 'deceased']

# Read region data.
data = pd.read_csv(
    '../pcm-dpc-COVID-19/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv',
    parse_dates=['data'],
    infer_datetime_format=True
)

# Prepare figure.
fig = plt.figure('plot-dati-andamento-nazionale')
fig.clf()
fig.set_size_inches((12, 7))
axs = fig.subplots(1, len(labels), sharex=True, sharey=True)
axsl = {l: a for l, a in zip(labels, axs)}

# Iterate over directories with a date format and which contain
# `dati-andamento-nazionale`.
cmdline = sys.argv[1:]
if cmdline:
    directories = [f'{d}/dati-andamento-nazionale' for d in cmdline]
else:
    directories = glob.glob('????-??-??/dati-andamento-nazionale')
    directories.sort()
for directory in directories:
    print(f'--------- Predictions made on {directory} ---------')
    
    # Read all csv files.
    files = glob.glob(f'{directory}/*.csv')
    files.sort()
    if not files:
        print('No csv files here.')
        continue
    tables = []
    for file in files:
        print(f'Reading {file}...')
        tables.append(pd.read_csv(file, parse_dates=['data']))
    
    # Prepare plot.
    for label, ax in axsl.items():
        ax.cla()
        ax.set_title(f'Italia ({label})')
    
    # Plot data.
    x = data['data']
    ys_data = {
        'total': data['totale_casi'],
        'infected': data['totale_attualmente_positivi'],
        'removed': data['totale_casi'] - data['totale_attualmente_positivi'],
        'deceased': data['deceduti']
    }
    for label, ax in axsl.items():
        y = ys_data[label].values
        yerr = np.where(y > 0, np.sqrt(y), 1)
        ax.errorbar(x, y, yerr=yerr, label='data', marker='.', capsize=0, linestyle='')
    
    # Plot predictions.
    fit_colors = dict()
    for filename, table in zip(files, tables):
        # times
        x = table['data']
        
        # total and infected
        ys = {
            'total': table['totale_casi'],
            'infected': table['totale_attualmente_positivi']
        }
        yserr = {
            'total': table['std_totale_casi'],
            'infected': table['std_totale_attualmente_positivi']
        }
        
        # removed
        try:
            ys['removed'] = table['guariti_o_deceduti']
            yserr['removed'] = table['std_guariti_o_deceduti']
        except KeyError:
            try:
                ys['removed'] = table['dimessi_guariti'] + table['deceduti']
                yserr['removed'] = np.hypot(table['std_dimessi_guariti'], table['std_deceduti'])
            except KeyError:
                ys['removed'] = ys['total'] - ys['infected']
                yserr['removed'] = np.hypot(yserr['total'], yserr['infected'])
        
        # deceased
        try:
            ys['deceased'] = table['deceduti']
            yserr['deceased'] = table['std_deceduti']
        except KeyError:
            try:
                ys['deceased'] = table['guariti_o_deceduti'] - table['dimessi_guariti']
                yserr['deceased'] = np.hypot(table['std_guariti_o_deceduti'], table['std_dimessi_guariti'])
            except:
                pass
        
        # plot
        for label, ax in axsl.items():
            if label in ys:
                y = ys[label].values
                arguments = dict(
                    yerr=yserr[label].values,
                    label=os.path.splitext(os.path.split(filename)[-1])[0].replace('model-', ''),
                    marker='',
                    capsize=2,
                    linestyle=''
                )
                if filename in fit_colors:
                    arguments['color'] = fit_colors[filename]
                rt = ax.errorbar(x, y, **arguments)
                fit_colors[filename] = rt[0].get_color()
    
    # Set logarithmic scale.
    axs[0].set_yscale('symlog', linthreshy=1, linscaley=0.3, subsy=np.arange(2, 9 + 1))
    if axs[0].get_ylim()[0] < 0:
        axs[0].set_ylim(0, ax.get_ylim()[1])
    
    # Embellishments.
    for ax in axs:
        ax.grid(linestyle=':')
    axs[0].legend(loc='best')
    axs[0].set_ylabel('people')
    fig.autofmt_xdate(rotation=70, ha='right')
    fig.tight_layout()
    
    # Save figure.
    plotfile = f'{directory}/{directory.replace("/", "-Giacomo-")}.png'
    print(f'Saving plot in {plotfile}...')
    fig.savefig(plotfile)
