from matplotlib import pyplot as plt
import glob
import pandas as pd
import numpy as np
import os
import sys
from extractor import extractor
import autocolor

# Object to assign colors.
colorset = autocolor.AutoColor(['data'])

# Error logging function.
errors = 0
def eprint(*args):
    print('##### ERROR:', *args, file=sys.stderr)
    errors += 1

# Thing for plotting with dates.
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Read region data.
data = pd.read_csv(
    '../pcm-dpc-COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv',
    parse_dates=['data']
)
regions = data['denominazione_regione'].unique()

# Check data is updated.
lastdate = data['data'].max()
today = pd.Timestamp.today()
if today - lastdate > pd.Timedelta(1, 'D'):
    raise ValueError(f'Data is not updated, last date in data in {lastdate}')

# Prepare figure.
fig = plt.figure('plot-dati-regioni')
fig.clf()
fig.set_size_inches((6, 7))
ax = fig.subplots(1, 1)

# Iterate over directories with a date format and which contain `dati-regioni`.
cmdline = sys.argv[1:]
if cmdline:
    directories = [f'{d}/dati-regioni' for d in cmdline]
else:
    directories = glob.glob('????-??-??/dati-regioni')
    directories.sort()
for directory in directories:
    print(f'--------- Predictions made on {directory} ---------')
    
    # Check it is a directory.
    if not os.path.isdir(directory):
        eprint('This is not a directory!')
        continue
    
    # Read all csv files.
    files = glob.glob(f'{directory}/*.csv')
    files.sort()
    if not files:
        eprint('No csv files here.')
        continue
    tables = []
    for file in files:
        print(f'Reading {file}...')
        tables.append(pd.read_csv(file, parse_dates=['data']))
    
    # Make directory for saving figures.
    savedir = f'{directory}/plots'
    os.makedirs(savedir, exist_ok=True)
    
    # Iterate over regions.
    print('Iterating over regions...')
    for region in regions:
        
        models_who_did_something = set()
        
        # Iterate over labels.
        for label in sorted(extractor.labels):
        
            # Prepare plot.
            ax.cla()
            ax.set_title(f'{region} ({label})')
        
            # Iterate over models.
            plotted_something = False
            for filename, table in zip(files, tables):
                
                # Extract data for this region.
                condition = table['denominazione_regione'] == region
                regiontable = table[condition]
            
                # Get model name from file name for legend and color cache.
                name = os.path.splitext(os.path.split(filename)[-1])[0]
                name = name.replace('model-', '')

                # Times, artificially shifted to avoid overlapping with data
                # points.
                x = regiontable['data'] + pd.Timedelta(2, 'H')
                
                # People.
                y, yerr = extractor.extract(regiontable, label)
                
                # Keyword arguments for plotting.
                kw = dict(label=name, linestyle='', color=colorset.colorfor(name))
                
                # Plot predictions.
                if not (y is None) and not (yerr is None):
                    ax.errorbar(x, y, yerr=yerr, marker='', capsize=2, elinewidth=1, **kw)
                elif not (y is None) and yerr is None:
                    ax.plot(x, y, marker='.', **kw)
                    eprint(f'region `{region}`, label `{label}`, model `{name}`: no uncertainties, plot without errorbars')
                elif y is None and not (yerr is None):
                    eprint(f'region `{region}`, label `{label}`, model `{name}`: found uncertainties but not values (??)')
                    continue
                else:
                    continue
                
                # Record we actually plotted something.
                plotted_something = True
                models_who_did_something.add(filename)
            
            # Avoid plotting data if there are no models for this label.
            if not plotted_something:
                continue
        
            # Get data for the current region.
            condition = data['denominazione_regione'] == region
            regiondata = data[condition]
            
            # Times.
            x = regiondata['data']
            
            # People.
            y, yerr = extractor.extract(regiondata, label)
            if not (yerr is None):
                eprint(f'region `{region}`, label `{label}`: got uncertainties for real data, ignoring them and using poisson errors')
            yerr = np.where(y > 0, np.sqrt(y), 1)
            
            # Plot data.
            kw = dict(label='data', marker='.', capsize=0, linestyle='', color=colorset.colorfor('data'))
            ax.errorbar(x, y, yerr=yerr, **kw)
        
            # Embellishments.
            ax.grid(linestyle=':')
            if ax.get_ylim()[0] < 0:
                ax.set_ylim(0, ax.get_ylim()[1])
            ax.legend(loc='best')
            ax.set_ylabel('People')
            fig.autofmt_xdate(rotation=70)
            fig.tight_layout()
        
            # Save figure.
            for scale in 'linear', 'log':
                if scale == 'log':
                    ax.set_yscale('symlog', linthreshy=1, linscaley=0.3, subsy=np.arange(2, 9 + 1))
                    if ax.get_ylim()[0] < 0:
                        ax.set_ylim(0, ax.get_ylim()[1])
                
                plotfile = f'{savedir}/{directory.replace("/", "-Giacomo-")}-{region}-{label}-{scale}.png'
                print(f'Writing {plotfile}...')
                fig.savefig(plotfile)
        
        lazy_models = set(files) - models_who_did_something
        for file in lazy_models:
            eprint(f'region {region}: model {model}: no prediction for any labels')
