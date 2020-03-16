###Script which plots the death rate by country (under)estimated as deaths/total cases
###from the JH database Time Series csvs, which need to be saved in the same directory 
###(from https://github.com/CSSEGISandData/COVID-19.)
###In the future we'll hopefully set a baseline with precise filepaths, but for now we do
###stuff by hand.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##Minimum number of cases and minimun number of days that have to be passed since that
##number was recorded for a country to be plotted. The first restraint is for statistical
##significance, the second one mostly for plot clarity.
MIN_CASES = 100
MIN_DAYS = 10

###Input and removal of useless features
infected = pd.read_csv("time_series_19-covid-Confirmed.csv")
deaths = pd.read_csv("time_series_19-covid-Deaths.csv")

infected.drop(['Province/State', 'Lat', 'Long'], axis = 1, inplace = True)
deaths.drop(['Province/State', 'Lat', 'Long'], axis = 1, inplace = True)

###We sum the data for each province of a given state and put them in two new DFs
all_death = []
all_inf = []

for country in deaths['Country/Region'].unique():
    subdeaths = deaths[deaths['Country/Region'] == country].iloc[:,1:].sum(axis = 0)
    subdeaths['Country/Region'] = country
    all_death.append(subdeaths)
    subconf = infected[infected['Country/Region'] == country].iloc[:,1:].sum(axis = 0)
    subconf['Country/Region'] = country
    all_inf.append(subconf)

group_deaths = pd.DataFrame(all_death, columns = all_death[0].index)
group_conf = pd.DataFrame(all_inf, columns = all_inf[0].index)

rel_dead = group_deaths[group_conf.iloc[:,-2] > MIN_CASES]
rel_conf = group_conf[group_conf.iloc[:,-2] > MIN_CASES]

###We plot the data. Colors are randomly drawn, the linspaced rainbow colormap was a mess.
for country in rel_conf['Country/Region'].unique():
    temp_conf = rel_conf[rel_conf['Country/Region'] == country].iloc[:,:-1]
    temp_dead = rel_dead[rel_dead['Country/Region'] == country].iloc[:,:-1]
    
    mask = temp_conf > MIN_CASES
    temp_conf = temp_conf[mask].dropna(axis = 1)
    temp_dead = temp_dead[mask].dropna(axis = 1)
    
    temp = temp_dead.div(temp_conf)
    
    if len(temp.keys()) > MIN_DAYS:
        plt.plot(np.arange(0, len(temp.keys()), 1), temp.iloc[0], '.', linestyle = '-',
                 c = np.random.rand(3,), label = country)
        
plt.title('Death Rate by Country')
plt.ylabel('Mortality Rate')
plt.xlabel(''.join(['Days since ', str(MIN_CASES), 'th Case']))
plt.legend()