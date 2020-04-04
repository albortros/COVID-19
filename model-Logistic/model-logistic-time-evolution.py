# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:32:49 2020

@author: jacop
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 02:22:58 2020

@author: Jacopo Busatto
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

#Quality for plot saving
DPI=450
#Averaging N for fits errors
SIZE = 100
##days of prediction
#NumberOfDaysPredicted=14
#Plot name format:
path='Time_Evolution/'
name='-Jacopo'
model='-model-logistic-time-evolution'
# cosa vogliamo analizzare: mi servono per dare nomi alle cose
TypeOfData=['totale_casi','deceduti','dimessi_guariti']
NomeIng=['infected','dead','recovered']
cases=['-infected','-deaths','-recovered']
types=['-log-','-derivative-']
region='Italia'
ext='.png'

# Define analytical model

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)*a))

def logistic(x,Param):
    return logistic_model(x,Param[0],Param[1],Param[2])

def logistic_model_derivative(x,a,b,c):
    return a*c*np.exp(-a*(x-b))/(1+np.exp(-(x-b)*a))**2

def logistic_derivative(x,Par):
    return Par[0]*Par[2]*np.exp(-Par[0]*(x-Par[1]))/(1+np.exp(-(x-Par[1])*Par[0]))**2


# Prendiamo i dati da Github
url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(url, parse_dates=['data'])
dF = df.loc[:,['data','totale_casi']]
FMT = '%Y-%m-%dT%H:%M:%S'
date = dF['data']
dF['data'] = date.map(lambda x : (x - datetime.strptime("2020-01-01T00:00:00", FMT)).days  )
x = list(dF.iloc[:,0])

#Prediction lists
Prediction_curves   = []
Prediction_std      = []
Prediction_curves_D = []
Prediction_std_D    = []
last_day            = []
dates_list          = []

# I parametri di partenza per i fit per i tre casi
InitialParams=[[0.2,70,6000000],[0.2,70,40000],[0.2,70,40000]]

#quanti grafici vogliamo fare
Num_plots=4 #considera that the real number is this +1
start=18
delta = (len(x)-start)/Num_plots

for num in range(Num_plots+1):
    iteration=0
    # dopo quanti giorni mi fermo
    LIM=int(start + delta * num +0.1)
    
    for TYPE in TypeOfData:

        dF = df.loc[:LIM,['data',TYPE]]
        # Formato dati csv
        FMT = '%Y-%m-%dT%H:%M:%S'
        # Formato dati salvataggio
        FMD = '%Y-%m-%d'
        date = dF['data']
        DATE = dF['data'][len(date)-1].strftime(FMD)
        dF['data'] = date.map(lambda x : (x - datetime.strptime("2020-01-01T00:00:00", FMT)).days  )
        namefile=path+DATE+name+model
        
        # tiene conto di che iterazione stiamo facendo
        x = list(dF.iloc[:LIM,0])
        y = list(dF.iloc[:LIM,1])
        YERR = np.sqrt(y)
        
        # Fitting logistic
        P0=InitialParams[iteration]
        temp_Par,temp_Cov = curve_fit(logistic_model,x,y,P0, sigma=YERR, absolute_sigma=False)

        #quanti punti considero: fino a che la differenza tra l'asintoto e la funzione non è < 1
        sol = int(fsolve(lambda x : logistic(x,temp_Par) - int(temp_Par[2]),temp_Par[1]))
        pred_x = list(range(max(x)+1,120))
        xTOT= x+pred_x
       
        # Calcoliamo SIZE funzioni estraendo parametri a caso e facendo la std
        simulated_par= np.random.multivariate_normal(temp_Par, temp_Cov, size=SIZE)
        simulated_curve=[[logistic(ii,par) for ii in xTOT] for par in simulated_par]
        std_fit=np.std(simulated_curve, axis=0)
        
        Prediction_curves = Prediction_curves + [[logistic(i,temp_Par) for i in xTOT]]
        Prediction_std    = Prediction_std + [std_fit]

        # differences
        Y2=np.array(y)
        Y1=np.array(y)
        Y1=np.delete(Y1,-1)
        Y1=np.insert(Y1,0,Y1[0])
        Y3=Y2-Y1
        ERRY3=np.sqrt(Y2+Y1)
        
        temp_ParD, temp_CovD = curve_fit(logistic_model_derivative,x,Y3,p0=[0.2,70,64000],sigma=ERRY3, absolute_sigma=False)
        simulated_par_D= np.random.multivariate_normal(temp_ParD, temp_CovD, size=SIZE)
        simulated_curve_D=[[logistic_derivative(ii,par) for ii in xTOT] for par in simulated_par_D]
        std_fit_D=np.std(simulated_curve_D, axis=0)
        #when MED method needed
    
        Ymin= np.array([logistic_derivative(i,temp_ParD) for i in xTOT])-np.array(std_fit_D)
        Ymax= np.array([logistic_derivative(i,temp_ParD) for i in xTOT])+np.array(std_fit_D)
        
        Prediction_curves_D = Prediction_curves_D + [[logistic_derivative(i,temp_ParD) for i in xTOT]]
        Prediction_std_D    = Prediction_std_D + [std_fit_D]

        iteration = iteration+1
#        #FINE CICLO
    dates_list=dates_list+[DATE]
print('DAJE')    
#        
#        
#PLOTS
##Plot with log prictions
#Plot with log predictions
for Iter in range(len(TypeOfData)):
    plt.figure('time evolution '+TypeOfData[Iter]+types[0])
    for time in range(Num_plots+1):
        Ymin = Prediction_curves[Iter+3*time] - Prediction_std[Iter+3*time]
        Ymax = Prediction_curves[Iter+3*time] + Prediction_std[Iter+3*time]
        plt.fill_between(xTOT,Ymin,Ymax,facecolor=str(1-(time+1)/(Num_plots+2.1)), alpha = 0.3 )
        plt.semilogy(xTOT,Prediction_curves[Iter+3*time], label='up to '+dates_list[time], c = str(1-(time+1)/(Num_plots+2.1)))
    plt.xlabel("Days since 1 January 2020")
    plt.ylabel('Total number of '+cases[Iter]+' people')
    plt.ylim((0.9*min(Prediction_curves[-3+Iter]),max(Prediction_curves[-3+Iter])*20))
    plt.legend()
    plt.grid(linestyle='--',which='both')
    plt.savefig(namefile+cases[Iter]+types[0]+region+ext, dpi=DPI)
    plt.gcf().show()
    
    
for Iter in range(len(TypeOfData)):
    plt.figure('time evolution '+TypeOfData[Iter]+types[1])
    for time in range(Num_plots+1):
        Ymin = Prediction_curves_D[Iter+3*time] - Prediction_std_D[Iter+3*time]
        Ymax = Prediction_curves_D[Iter+3*time] + Prediction_std_D[Iter+3*time]
        plt.fill_between(xTOT,Ymin,Ymax,facecolor=str(1-(time+1)/(Num_plots+2.1)), alpha = 0.3 )
        plt.plot(xTOT,Prediction_curves_D[Iter+3*time], label='up to '+dates_list[time], c = str(1-(time+1)/(Num_plots+2.1)))
    plt.xlabel("Days since 1 January 2020")
    plt.ylabel('New '+cases[Iter]+' people')
    plt.ylim((0,max(Prediction_curves_D[-3+Iter])*1.5))
    plt.legend()
    plt.grid(linestyle='--',which='both')
    plt.savefig(namefile+cases[Iter]+types[1]+region+ext, dpi=DPI)
    plt.gcf().show()






#plt.figure('time evolution infected')
#for time in range(Num_plots+1):
#    Ymin = Prediction_curves[3*time] - Prediction_std[3*time]
#    Ymax = Prediction_curves[3*time] + Prediction_std[3*time]
#    plt.fill_between(xTOT,Ymin,Ymax,facecolor=str(1-(time+1)/(Num_plots+2.1)), alpha = 0.3 )
##    plt.semilogy(xTOT,Prediction_curves[3*time], label='up to '+dates_list[time], c = str(1-(time+1)/(Num_plots+2.1)))
#    plt.plot(xTOT,Prediction_curves[3*time], label='up to '+dates_list[time], c = str(1-(time+1)/(Num_plots+2.1)))
#plt.xlabel("Days since 1 January 2020")
#plt.ylabel('Total number of infected people')
#plt.ylim((100,max(Prediction_curves[-3])*1.1))
#plt.legend()
#plt.grid(linestyle='--',which='both')
##plt.savefig(namefile+cases[iteration]+types[0]+region+ext, dpi=DPI)
#plt.gcf().show()
#
#plt.figure('time evolution dead')
#for time in range(Num_plots+1):
#    Ymin = Prediction_curves[1+3*time] - Prediction_std[1+3*time]
#    Ymax = Prediction_curves[1+3*time] + Prediction_std[1+3*time]
#    plt.fill_between(xTOT,Ymin,Ymax,facecolor=str(1-(time+1)/(Num_plots+2.1)), alpha = 0.3 )
#    plt.plot(xTOT,Prediction_curves[1+3*time], label='up to '+dates_list[time], c = str(1-(time+1)/(Num_plots+2.1)))
#plt.xlabel("Days since 1 January 2020")
#plt.ylabel('Total number of dead people')
#plt.ylim((1,max(Prediction_curves[-2])*1.1))
#plt.legend()
#plt.grid(linestyle='--',which='both')
##plt.savefig(namefile+cases[iteration]+types[0]+region+ext, dpi=DPI)
#plt.gcf().show()
#
#plt.figure('time evolution recovered')
#for time in range(Num_plots+1):
#    Ymin = Prediction_curves[2+3*time] - Prediction_std[2+3*time]
#    Ymax = Prediction_curves[2+3*time] + Prediction_std[2+3*time]
#    plt.fill_between(xTOT,Ymin,Ymax,facecolor=str(1-(time+1)/(Num_plots+2.1)), alpha = 0.3 )
#    plt.plot(xTOT,Prediction_curves[2+3*time], label='up to '+dates_list[time], c = str(1-(time+1)/(Num_plots+2.1)))
#plt.xlabel("Days since 1 January 2020")
#plt.ylabel('Total number of recovered people people')
#plt.ylim((1,max(Prediction_curves[-1])*1.1))
#plt.legend()
#plt.grid(linestyle='--',which='both')
##plt.savefig(namefile+cases[iteration]+types[0]+region+ext, dpi=DPI
#plt.gcf().show()
#
#
#
#
#
#        
#        
## Real data
#plt.figure('derivatives_')
#for time in range(Num_plots +1):
#    Ymin = Prediction_curves_D[3*time] - Prediction_std_D[3*Num_plots]
#    Ymax = Prediction_curves_D[3*time] + Prediction_std_D[3*Num_plots]
#    plt.fill_between(xTOT,Ymin,Ymax,facecolor=str(1-(time+1)/(Num_plots+2.1)), alpha = 0.3, )
#    plt.plot(xTOT,Prediction_curves_D[3*time], label='up to '+dates_list[time], c = str(1-(time+1)/(Num_plots+2.1)))
#plt.legend()
#plt.xlabel("Days since 1 January 2020")
#plt.ylabel('Increase of infected people')
#plt.ylim((0,max([max(Prediction_curves_D[3*ii]) for ii in range(Num_plots+1)])*1.1))
#plt.grid(linestyle='--',which='both')
##plt.savefig(namefile+cases[iteration]+types[1]+region+ext, dpi=DPI)
#plt.gcf().show()
#
#
#
## Real data
#plt.figure('derivatives_deaths')
#for time in range(Num_plots +1):
#    Ymin = Prediction_curves_D[1+3*time] - Prediction_std_D[1+3*time]
#    Ymax = Prediction_curves_D[1+3*time] + Prediction_std_D[1+3*time]
#    plt.fill_between(xTOT,Ymin,Ymax,facecolor=str(1-(time+1)/(Num_plots+2.1)), alpha = 0.3 )
#    plt.plot(xTOT,Prediction_curves_D[1+3*time], label='up to '+dates_list[time], c = str(1-(time+1)/(Num_plots+2.1)))
#plt.legend()
#plt.xlabel("Days since 1 January 2020")
#plt.ylabel('Increase of dead people')
#plt.ylim((0,max([max(Prediction_curves_D[1+3*ii]) for ii in range(Num_plots+1)])*1.1))
#plt.grid(linestyle='--',which='both')
##plt.savefig(namefile+cases[iteration]+types[1]+region+ext, dpi=DPI)
#plt.gcf().show()
#
#
#
## Real data
#plt.figure('derivatives_recovered')
#for time in range(Num_plots +1):
#    Ymin = Prediction_curves_D[2+3*time] - Prediction_std_D[2+3*time]
#    Ymax = Prediction_curves_D[2+3*time] + Prediction_std_D[2+3*time]
#    plt.fill_between(xTOT,Ymin,Ymax,facecolor=str(1-(time+1)/(Num_plots+2.1)), alpha = 0.3 )
#    plt.plot(xTOT,Prediction_curves_D[2+3*time], label='up to '+dates_list[time], c = str(1-(time+1)/(Num_plots+2.1)))
#plt.legend()
#plt.xlabel("Days since 1 January 2020")
#plt.ylabel('Increase of recovered people')
#plt.ylim((0,max([max(Prediction_curves_D[2+3*ii]) for ii in range(Num_plots+1)])*1.1))
#plt.grid(linestyle='--',which='both')
##plt.savefig(namefile+cases[iteration]+types[1]+region+ext, dpi=DPI)
#plt.gcf().show()