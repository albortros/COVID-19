import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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

# Plot.
fig = plt.figure('fit')
fig.clf()
axs = fig.subplots(1, 3)

# sort by max total cases
sorted_region = gdata['totale_casi'].max().sort_values().keys()[::-1]

# iterate over regions
for key in sorted_region:
    table = gdata.get_group(key)
    line, = axs[0].plot(table['data'], table['totale_casi'], '-', label=key)
    color = line.get_color()
    axs[1].plot(table['data'], table['dimessi_guariti'].values, color=color)
    axs[2].plot(table['data'], table['deceduti'].values, color=color)

axs[0].legend(loc='upper left', fontsize='small')
axs[0].set_title('totale_casi')
axs[1].set_title('dimessi_guariti')
axs[2].set_title('deceduti')
for ax in axs:
    ax.set_yscale('symlog', linthreshy=1)
fig.autofmt_xdate()

plt.show()
