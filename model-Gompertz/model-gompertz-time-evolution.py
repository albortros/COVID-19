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
path='Plots/'
name='-Jacopo'
model='-model-gompertz'
# cosa vogliamo analizzare: mi servono per dare nomi alle cose
TypeOfData=['totale_casi','deceduti','dimessi_guariti']
NomeIng=['infected','dead','recovered']
cases=['-infected','-deaths','-recovered']
types=['-log-','-derivative-']
region='Italia'
ext='.png'

# Define analytical model
def gompertz_model(x,a,b,c):
    return c*np.exp(-np.exp(-a*(x-b)))

def gompertz(x,Par):
    return Par[2]*np.exp(-np.exp(-Par[0]*(x-Par[1])))

def gompertz_model_derivative(x,a,b,c):
    return a*c*np.exp(-a*(x-b)-np.exp(-a*(x-b)))

def gompertz_derivative(x,Par):
    return Par[0]*Par[2]*np.exp(-Par[0]*(x-Par[1])-np.exp(-Par[0]*(x-Par[1])))

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
df = df.loc[:,['data','totale_casi']]
FMT = '%Y-%m-%dT%H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x : (x - datetime.strptime("2020-01-01T00:00:00", FMT)).days  )
x = list(df.iloc[:,0])

#Prediction lists
Prediction_curves   = []
Prediction_std      = []
Prediction_curves_D = []
Prediction_std_D    = []
last_day            = []

# I parametri di partenza per i fit per i tre casi
InitialParams=[[0.2,70,6000000],[0.2,70,40000],[0.2,70,40000]]

#quanti grafici vogliamo fare
Num_plots=5
start=25
delta = (len(x)-start)/Num_plots

for num in range(Num_plots):
    iteration=0
    # dopo quanti giorni mi fermo
    LIM=int(start + delta * num)
    for TYPE in TypeOfData:
        df = pd.read_csv(url, parse_dates=['data'])
        df = df.loc[:LIM,['data',TYPE]]
        # Formato dati csv
        FMT = '%Y-%m-%dT%H:%M:%S'
        # Formato dati salvataggio
        FMD = '%Y-%m-%d'
        date = df['data']
        DATE = df['data'][len(date)-1].strftime(FMD)
        df['data'] = date.map(lambda x : (x - datetime.strptime("2020-01-01T00:00:00", FMT)).days  )
        namefile=path+DATE+name+model
        
        # tiene conto di che iterazione stiamo facendo
        x = list(df.iloc[:LIM,0])
        y = list(df.iloc[:LIM,1])
        YERR = np.sqrt(y)
        
        # Fitting gompertzo
        P0=InitialParams[iteration]
        temp_Par,temp_Cov = curve_fit(gompertz_model,x,y,P0, sigma=YERR, absolute_sigma=False)

        #quanti punti considero: fino a che la differenza tra l'asintoto e la funzione non è < 1
        sol = int(fsolve(lambda x : gompertz(x,temp_Par) - int(temp_Par[2]),temp_Par[1]))
        pred_x = list(range(max(x)+1,min(int(sol/1.5),160)))
        xTOT= x+pred_x
       
        # Calcoliamo SIZE funzioni estraendo parametri a caso e facendo la std
        simulated_par= np.random.multivariate_normal(temp_Par, temp_Cov, size=SIZE)
        simulated_curve=[[gompertz(ii,par) for ii in xTOT] for par in simulated_par]
        std_fit=np.std(simulated_curve, axis=0)
        
        Prediction_curves = Prediction_curves + [[gompertz(i,temp_Par) for i in xTOT]]
        Prediction_std    = Prediction_std + [std_fit]
#
#        
#
#        
#        # differences
#        Y2=np.array(y)
#        Y1=np.array(y)
#        Y1=np.delete(Y1,-1)
#        Y1=np.insert(Y1,0,Y1[0])
#        Y3=Y2-Y1
#        ERRY3=np.sqrt(Y2+Y1)
#        X=np.array(x)
#        X=np.delete(X,15,axis=0)
#        Y3=np.delete(Y3,15,axis=0)
#        ERRY3=np.delete(ERRY3,15,axis=0)
#        
#        temp_ParD, temp_CovD = curve_fit(gompertz_model_derivative,X,Y3,p0=[0.2,70,64000],sigma=ERRY3, absolute_sigma=False)
#        simulated_par_D= np.random.multivariate_normal(temp_ParD, temp_CovD, size=SIZE)
#        simulated_curve_D=[[gompertz_derivative(ii,par) for ii in list(X)+pred_x] for par in simulated_par_D]
#        std_fit_D=np.std(simulated_curve_D, axis=0)
#        #when MED method needed
#    
#        Ymin= np.array([gompertz_derivative(i,temp_ParD) for i in list(X)+pred_x])-np.array(std_fit_D)
#        Ymax= np.array([gompertz_derivative(i,temp_ParD) for i in list(X)+pred_x])+np.array(std_fit_D)
#        
#        Prediction_curves_D = Prediction_curves + [[gompertz_derivative(i,temp_ParD) for i in xTOT]]
#        Prediction_std_D    = Prediction_std + [std_fit_D]
#
#        
#
#        
        iteration = iteration+1
#        #FINE CICLO
        
        
#        
#        
#PLOTS
#Plot with log predictions
plt.figure('time evolution infected')
for time in range(Num_plots):
#    Ymin = Prediction_curves[3*time] - Prediction_std[time]
#    Ymax = Prediction_curves[3*time] + Prediction_std[time]
#    plt.fill_between(xTOT,Ymin,Ymax,facecolor='blue', alpha = 0.3 )
    plt.semilogy(xTOT,Prediction_curves[3*time], alpha = time/Num_plots)
plt.xlabel("Days since 1 January 2020")
#plt.ylabel('Total number of '+NomeIng[iteration]+' people')
plt.ylim((100,max(Prediction_curves[-3])*20))
#plt.legend()
plt.grid(linestyle='--',which='both')
#plt.savefig(namefile+cases[iteration]+types[0]+region+ext, dpi=DPI)
plt.gcf().show()

plt.figure('time evolution dead')
for time in range(Num_plots):
#    Ymin = Prediction_curves[1+3*time] - Prediction_std[time]
#    Ymax = Prediction_curves[1+3*time] + Prediction_std[time]
#    plt.fill_between(xTOT,Ymin,Ymax,facecolor='blue', alpha = 0.3 )
    plt.semilogy(xTOT,Prediction_curves[1+3*time], alpha = time/Num_plots)
plt.xlabel("Days since 1 January 2020")
#plt.ylabel('Total number of '+NomeIng[iteration]+' people')
plt.ylim((1,max(Prediction_curves[-2])*2))
#plt.legend()
plt.grid(linestyle='--',which='both')
#plt.savefig(namefile+cases[iteration]+types[0]+region+ext, dpi=DPI)
plt.gcf().show()

plt.figure('time evolution recovered')
for time in range(Num_plots):
#    Ymin = Prediction_curves[2+3*time] - Prediction_std[time]
#    Ymax = Prediction_curves[2+3*time] + Prediction_std[time]
#    plt.fill_between(xTOT,Ymin,Ymax,facecolor='blue', alpha = 0.3 )
    plt.semilogy(xTOT,Prediction_curves[2+3*time], alpha = time/Num_plots)
plt.xlabel("Days since 1 January 2020")
#plt.ylabel('Total number of '+NomeIng[iteration]+' people')
plt.ylim((1,max(Prediction_curves[-1])*2))
#plt.legend()
plt.grid(linestyle='--',which='both')
#plt.savefig(namefile+cases[iteration]+types[0]+region+ext, dpi=DPI)
plt.gcf().show()





#        
#        
## Real data
#plt.figure('derivatives_'+NomeIng[iteration])
#for time in range(len(Prediction_curves)):
#    Ymin = Prediction_curves_D[time] - Prediction_std_D[time]
#    Ymin = Prediction_curves_D[time] + Prediction_std_D[time]
#    plt.fill_between(xTOT,Ymin,Ymax,facecolor='blue', alpha = 0.3 )
#    plt.semilogy(xTOT,Prediction_curves_D[time], 'r')
#plt.legend()
#plt.xlabel("Days since 1 January 2020")
#plt.ylabel('Increase of '+NomeIng[iteration]+' people')
#plt.ylim((min(Y3)*0.9,max([gompertz_derivative(i,temp_ParD) for i in list(X)+pred_x])*1.3))
#plt.grid(linestyle='--',which='both')
#plt.savefig(namefile+cases[iteration]+types[1]+region+ext, dpi=DPI)
#plt.gcf().show()
#        