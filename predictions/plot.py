# Start counting time.
import time
start = time.time()

# Parse command line.
import arguments
cmdargs = arguments.parseargs()

from matplotlib import pyplot as plt
import glob
import pandas as pd
import numpy as np
import os
from extractor import extractor
import autocolor
import re
import sys

# Thing for plotting with dates.
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Object to assign colors.
colorset = autocolor.AutoColor(['data'])
# (the first color is assigned to label `data`)

# Error logging function.
errors = 0
def eprint(*args):
    global errors
    print('##### ERROR #####', *args, file=sys.stderr)
    errors += 1
    
# Create output directory.
if cmdargs.outputdir:
    savedir = cmdargs.outputdir[0]
else:
    savedir = 'plots-' + str(pd.Timestamp.today()).replace(':', ';')
    os.makedirs(savedir, exist_ok=False)
os.makedirs(savedir, exist_ok=True)
print(f'Will save plots in {savedir}')

# Determine date directories to be visited.
if cmdargs.date:
    directories = cmdargs.date
else:
    directories = []
    dirs = next(os.walk('.'))[1]
    regexp = re.compile(r'^\d\d\d\d-\d\d-\d\d$')
    for d in dirs:
        if regexp.match(d):
            directories.append(d)
directories.sort()
for d in directories:
    if not os.path.isdir(d):
        raise ValueError(f'{d} is not a directory')
assert directories
print('Will scan directories', ', '.join(directories))

# Determine data types to process.
if cmdargs.type:
    dtypes = cmdargs.type
else:
    dtypes = [
        'dati-regioni',
        'dati-andamento-nazionale'
    ]
assert dtypes
print('Will process data subdirectories', ', '.join(dtypes))

# Read data.
data = dict()
for dtype in dtypes:
    file = f'../pcm-dpc-COVID-19/{dtype}/dpc-covid19-ita-{dtype[5:]}.csv'
    if not os.path.isfile(file):
        raise ValueError(f'file `{file}` does not exist')
    print(f'Reading {file}...')
    data[dtype] = pd.read_csv(
        file,
        parse_dates=['data']
    )
assert data

# Determine list of regions to be plotted.
if 'dati-regioni' in data:
    if cmdargs.region:
        regions = cmdargs.region
    else:
        regions = data['dati-regioni']['denominazione_regione'].unique()
    regions = sorted(regions)
    assert regions
    print('Will process regions', ', '.join(regions))
elif cmdargs.region:
    raise ValueError('You specified regions {", ".join(cmdargs.region)} but not to process regions')

# Check data is updated.
for table in data.values():
    lastdate = table['data'].max()
    today = pd.Timestamp.today()
    if today - lastdate > pd.Timedelta(1, 'D'):
        raise ValueError(f'Data is not update, last date in data is {lastdate}')
    
# Determine labels to plot.
if cmdargs.label:
    labels = cmdargs.label
else:
    labels = extractor.labels
labels = sorted(labels)
for label in labels:
    if not label in extractor.labels:
        raise ValueError(f'label `{label}` not recognized, if new add it in predictions/extractor.py')
assert labels
print(f'Will process labels', ', '.join(labels))

# Determine models to use.
if cmdargs.model:
    print(f'Will process only models', ', '.join(cmdargs.model))
        
# Prepare figure.
fig = plt.figure('plot-dati-regioni')
fig.clf()
fig.set_size_inches((6, 7))
ax = fig.subplots(1, 1)

# Iterate over date directories.
nplots = 0
for date_directory in directories:
    
  # Iterate over data type subdirectories.
  for dtype in dtypes:
    
    directory = f'{date_directory}/{dtype}'
    print(f'--------- Predictions made on {directory} ---------')
    
    # Check it is a directory.
    if not os.path.isdir(directory):
        eprint('Directory does not exist')
        continue
    
    # Find all CSV files.
    files = glob.glob(f'{directory}/*.csv')
    files.sort()
    if not files:
        eprint('No csv files here')
        continue
    print('Found csv files', ', '.join(files))
    
    # Filter files by requested models.
    if cmdargs.model:
        selfiles = [
            f for f in files
            if any(F.lower() in f.lower() for F in cmdargs.model)
        ]
        assert files
        if not selfiles:
            print('No file correspond to requested models', ', '.join(cmdargs.model))
            continue
        files = selfiles
    
    # Read all files.
    tables = []
    for file in files:
        print(f'Reading {file}...')
        tables.append(pd.read_csv(file, parse_dates=['data']))
    
    # Iterate over regions.
    for region in regions if dtype == 'dati-regioni' else ['Italia']:
        
        models_who_did_something = set()
        
        # Iterate over labels.
        for label in labels:
        
            # Prepare plot.
            ax.cla()
            ax.set_title(f'{region}—{label.replace("_", " ")}')
        
            # Iterate over models.
            plotted_something = False
            for filename, table in zip(files, tables):
                
                # Extract data for this region.
                if dtype == 'dati-regioni':
                    condition = table['denominazione_regione'] == region
                    regiontable = table[condition]
                else:
                    regiontable = table
            
                # Get model name from file name for legend and color cache.
                name = os.path.splitext(os.path.split(filename)[-1])[0]
                name = name.replace('model-', '')

                # Times, artificially shifted to avoid overlapping with data
                # points.
                x = regiontable['data'] + pd.Timedelta(2, 'H')
                
                # People.
                y, yerr = extractor.extract(regiontable, label)
                
                # Keyword arguments for plotting.
                if not (y is None):
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
            if dtype == 'dati-regioni':
                condition = data[dtype]['denominazione_regione'] == region
                regiondata = data[dtype][condition]
            else:
                regiondata = data[dtype]
            
            # Times.
            x = regiondata['data']
            
            # People.
            y, yerr = extractor.extract(regiondata, label)
            if y is None:
                eprint(f'region `{region}`, label `{label}`: no data (??)')
            if not (yerr is None):
                eprint(f'region `{region}`, label `{label}`: got uncertainties for real data, ignoring them and using poisson errors')
            
            # Plot data.
            if not (y is None):
                yerr = np.where(y > 0, np.sqrt(y), 1)
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
                nplots += 1
        
        lazy_models = set(files) - models_who_did_something
        for file in lazy_models:
            eprint(f'region {region}: model {model}: no prediction for any labels')

# Final report.
end = time.time()
interval = pd.Timedelta(end - start, 'sec')
print(f'Saved {nplots} plots in {interval}. There were {errors} errors.')
