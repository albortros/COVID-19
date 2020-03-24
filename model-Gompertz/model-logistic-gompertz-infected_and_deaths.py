# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:23:34 2020

@author: jacop
"""

import pandas as pd
import numpy as np
from datetime import datetime
import datetime as dt
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import csv 

#Quality for saving plots
DPI=300
#Averaging N for std
SIZE=100

#when MED method needed
alphasigma=0.67449

NumberOfDaysPredicted=14

# Define analytical model
def gompertz_model(x,a,b,c):
    return c*np.exp(-np.exp(-a*(x-b)))

def gompertz(x,Par):
    return Par[2]*np.exp(-np.exp(-Par[0]*(x-Par[1])))

def gompertz_model_derivative(x,a,b,c):
    return a*c*np.exp(a*(x-b)*np.exp(-a*(x-b)))

def gompertz_derivative(x,Par):
    return Par[0]*Par[2]*np.exp(Par[0]*(x-Par[1])*np.exp(-Par[0]*(x-Par[1])))

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)*a))

def logistic(x,Par):
    return logistic_model(x,Par[0],Par[1],Par[2])

def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))

def logistic_model_derivative(x,a,b,c):
    return a*c*np.exp(-a*(x-b))/(1+np.exp(-(x-b)*a))**2

def logistic_derivative(x,Par):
    return Par[0]*Par[2]*np.exp(-Par[0]*(x-Par[1]))/(1+np.exp(-(x-Par[1])*Par[0]))**2


#Plot name format:
path='Plots/'
tday=dt.date.today()
DATE = tday.strftime("%d-%m-%Y")
name='-jacopo'
model='-model-gompertz'
case=['-infected','-deaths','-recovered']
types=['-log','-derivative']
region='-italia'
ext='.png'

namefile=path+DATE+name+model



# Prendiamo i dati da Github
url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(url)
df = df.loc[:,['data','totale_casi']]
FMT = '%Y-%m-%d %H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days  )

x = list(df.iloc[:,0])
y = list(df.iloc[:,1])
YERR = np.sqrt(y)

# Fitting Logistico
P0=[0.2,70,60000]
fit_logistic = curve_fit(logistic_model,x,y,P0, sigma=YERR, absolute_sigma=False)
fit_gompertz = curve_fit(gompertz_model,x,y,P0, sigma=YERR, absolute_sigma=False)
logistic_Par = [ii for ii in fit_logistic[0]]
gompertz_Par = [ii for ii in fit_gompertz[0]]
logistic_Cov = fit_logistic[1]
gompertz_Cov = fit_gompertz[1]
ErrPar = [np.sqrt(fit_gompertz[1][ii][ii]) for ii in range(3)]

print('\n Gompertz fit parameters for total infected people:')
print('Rate of growth           =', gompertz_Par[0])
print('Peak day from Jan 1st    =', gompertz_Par[1])
print('Final number of infected =', gompertz_Par[2])

sol = int(fsolve(lambda x : gompertz(x,gompertz_Par) - int(gompertz_Par[2]),gompertz_Par[1]))

# Fitting esponenziale (Dummy, solo per grafico)
exp_fit = curve_fit(exponential_model,x,y,p0=[0.005,0.17,50])

#pred_x = list(range(max(x),sol))         #Qui sto cercando la fine dell'epidemia (in questo caso è troppo lontana con gompletz)
pred_x = list(range(max(x),int(1.3*gompertz_Par[1])))
xTOT= x+pred_x

simulated_par_gompertz   = np.random.multivariate_normal(gompertz_Par, gompertz_Cov, size=SIZE)
simulated_curve_gompertz = [[gompertz(ii,par) for ii in xTOT] for par in simulated_par_gompertz]
std_fit_gompertz         = np.std(simulated_curve_gompertz, axis=0)

simulated_par_logistic   = np.random.multivariate_normal(logistic_Par, logistic_Cov, size=SIZE)
simulated_curve_logistic = [[logistic(ii,par) for ii in xTOT] for par in simulated_par_logistic]
std_fit_logistic         = np.std(simulated_curve_logistic, axis=0)


# Mean square errors
print('\nMSE logistic curve:    ',mean_squared_error(y,[logistic(i,logistic_Par) for i in x]))
print('MSE Gompertz curve:    ',mean_squared_error(y,[gompertz(i,gompertz_Par) for i in x]))

DOF=float(len(x)-len(gompertz_Par))
chi2_gompertz_infected = chisquare(y, [gompertz(ii,gompertz_Par) for ii in x])[0]
chi2_logistic_infected = chisquare(y, [logistic(ii,logistic_Par) for ii in x])[0]
print('chi2r gompertz = ', chi2_gompertz_infected/DOF)
print('chi2r logistic = ', chi2_logistic_infected/DOF)

#Fit with error bands
Ymin_gompertz=np.array([gompertz(i,gompertz_Par) for i in xTOT])-np.array(std_fit_gompertz)
Ymax_gompertz=np.array([gompertz(i,gompertz_Par) for i in xTOT])+np.array(std_fit_gompertz)
Ymin_logistic=np.array([logistic(i,logistic_Par) for i in xTOT])-np.array(std_fit_logistic)
Ymax_logistic=np.array([logistic(i,logistic_Par) for i in xTOT])+np.array(std_fit_logistic)

# Predictions
Xpredicted=[ii+max(x) for ii in range(1,NumberOfDaysPredicted+1)]
Ypredicted_gompertz=[gompertz(ii,gompertz_Par) for ii in Xpredicted]
Ypredicted_logistic=[logistic(ii,logistic_Par) for ii in Xpredicted]
YPERR_gompertz = np.array([std_fit_gompertz[i] for i in range(len(x),len(x)+NumberOfDaysPredicted)])
YPERR_logistic = np.array([std_fit_logistic[i] for i in range(len(x),len(x)+NumberOfDaysPredicted)])

Ypredicted_gompertz_infected=Ypredicted_gompertz
YPERR_gompertz_infected=YPERR_gompertz

#Plot with predictions
plt.figure('predictions_infected')
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red", alpha=0.75,label="Data" )
plt.errorbar(Xpredicted,Ypredicted_gompertz, yerr=YPERR_gompertz, fmt='o',color="orange", alpha=0.5,label="Gompertz predictions ({} days)".format(NumberOfDaysPredicted) )
plt.errorbar(Xpredicted,Ypredicted_logistic, yerr=YPERR_logistic, fmt='o',color="green", alpha=0.5,label="Logistic predictions ({} days)".format(NumberOfDaysPredicted) )
plt.fill_between(xTOT,Ymin_gompertz,Ymax_gompertz,facecolor='blue', alpha = 0.3 )
plt.fill_between(xTOT,Ymin_logistic,Ymax_logistic,facecolor='blue', alpha = 0.3 )
plt.semilogy(xTOT, [gompertz(i,gompertz_Par) for i in xTOT], 'b',label="Gompertz model" )
plt.semilogy(xTOT, [logistic(i,logistic_Par) for i in xTOT], 'green',label="Logistic model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,gompertz_Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig(namefile+case[0]+types[0]+region+ext, dpi=DPI)
plt.show()



# differences: derivatives
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])
Y3=Y2-Y1
ERRY3=np.sqrt(Y2+Y1)

#fittiamo i rapporti incrementali con le derivate dei modelli
gompertz_fit_derivative    = curve_fit(gompertz_model_derivative,x,Y3,gompertz_Par,sigma=ERRY3, absolute_sigma=True)
gompertz_ParD              = gompertz_fit_derivative[0]
gompertz_CovD              = gompertz_fit_derivative[1]
gompertz_simulated_par_D   = np.random.multivariate_normal(gompertz_ParD, gompertz_CovD, size=SIZE)
gompertz_simulated_curve_D = np.array([[gompertz_derivative(ii,par) for ii in xTOT] for par in gompertz_simulated_par_D])
gompertz_std_fit_D         = np.std(gompertz_simulated_curve_D, axis=0)
gompertz_YminD             = np.array([gompertz_derivative(i,gompertz_ParD) for i in xTOT])-np.array(gompertz_std_fit_D)
gompertz_YmaxD             = np.array([gompertz_derivative(i,gompertz_ParD) for i in xTOT])+np.array(gompertz_std_fit_D)

logistic_fit_derivative    = curve_fit(logistic_model_derivative,x,Y3,logistic_Par,sigma=ERRY3, absolute_sigma=False)
logistic_ParD              = logistic_fit_derivative[0]
logistic_CovD              = logistic_fit_derivative[1]
logistic_simulated_par_D   = np.random.multivariate_normal(logistic_ParD, logistic_CovD, size=SIZE)
logistic_simulated_curve_D = [[logistic_derivative(ii,par) for ii in xTOT] for par in logistic_simulated_par_D]
logistic_std_fit_D         = np.std(logistic_simulated_curve_D, axis=0)
logistic_YminD             = np.array([logistic_derivative(i,logistic_ParD) for i in xTOT])-np.array(logistic_std_fit_D)
logistic_YmaxD             = np.array([logistic_derivative(i,logistic_ParD) for i in xTOT])+np.array(logistic_std_fit_D)

#plt.figure('prova1')
#for par in gompertz_simulated_par_D:
#    plt.plot(xTOT,[gompertz_derivative(ii,par) for ii in xTOT])
#plt.show()

plt.figure('derivatives_infected')
plt.errorbar(x, Y3, yerr=ERRY3, fmt='o',color="red", alpha=0.75,label="Data" )
plt.plot(xTOT, [logistic_derivative(i,logistic_ParD) for i in xTOT], label="Logistic model derivative" )
plt.fill_between(xTOT,logistic_YminD,logistic_YmaxD,facecolor='blue', alpha = 0.3 )
plt.plot(xTOT, [gompertz_derivative(i,gompertz_ParD) for i in xTOT], label="Gompertz model derivative" )
plt.fill_between(xTOT,gompertz_YminD,gompertz_YmaxD,facecolor='blue', alpha = 0.3 )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Increase of infected people per day")
plt.ylim((min(Y3)*0.9,3*max([logistic_derivative(i,logistic_ParD) for i in xTOT])*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig(namefile+case[0]+types[1]+region+ext, dpi=DPI)
plt.show()



###############################################################################
#                       FACCIO LA STESSA COSA PER LE MORTI
###############################################################################


df = pd.read_csv(url)
df = df.loc[:,['data','deceduti']]
FMT = '%Y-%m-%d %H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days  )

x = list(df.iloc[:,0])
y = list(df.iloc[:,1])
YERR = np.sqrt(y)

# Fitting Logistico
P0=[0.2,70,60000]
fit_logistic = curve_fit(logistic_model,x,y,P0, sigma=YERR, absolute_sigma=False)
fit_gompertz = curve_fit(gompertz_model,x,y,P0, sigma=YERR, absolute_sigma=False)
logistic_Par = [ii for ii in fit_logistic[0]]
gompertz_Par = [ii for ii in fit_gompertz[0]]
logistic_Cov = fit_logistic[1]
gompertz_Cov = fit_gompertz[1]
ErrPar = [np.sqrt(fit_gompertz[1][ii][ii]) for ii in range(3)]

print('\n Gompertz fit parameters for dead people:')
print('Rate of growth           =', gompertz_Par[0])
print('Peak day from Jan 1st    =', gompertz_Par[1])
print('Final number of infected =', gompertz_Par[2])

sol = int(fsolve(lambda x : gompertz(x,gompertz_Par) - int(gompertz_Par[2]),gompertz_Par[1]))

# Fitting esponenziale (Dummy, solo per grafico)
exp_fit = curve_fit(exponential_model,x,y,p0=[0.005,0.17,50])

#pred_x = list(range(max(x),sol))         #Qui sto cercando la fine dell'epidemia (in questo caso è troppo lontana con gompletz)
pred_x = list(range(max(x),int(1.3*gompertz_Par[1])))
xTOT= x+pred_x

simulated_par_gompertz   = np.random.multivariate_normal(gompertz_Par, gompertz_Cov, size=SIZE)
simulated_curve_gompertz = [[gompertz(ii,par) for ii in xTOT] for par in simulated_par_gompertz]
std_fit_gompertz         = np.std(simulated_curve_gompertz, axis=0)

simulated_par_logistic   = np.random.multivariate_normal(logistic_Par, logistic_Cov, size=SIZE)
simulated_curve_logistic = [[logistic(ii,par) for ii in xTOT] for par in simulated_par_logistic]
std_fit_logistic         = np.std(simulated_curve_logistic, axis=0)


# Mean square errors
print('\nMSE logistic curve:    ',mean_squared_error(y,[logistic(i,logistic_Par) for i in x]))
print('MSE Gompertz curve:    ',mean_squared_error(y,[gompertz(i,gompertz_Par) for i in x]))

DOF=float(len(x)-len(gompertz_Par))
chi2_gompertz_infected = chisquare(y, [gompertz(ii,gompertz_Par) for ii in x])[0]
chi2_logistic_infected = chisquare(y, [logistic(ii,logistic_Par) for ii in x])[0]
print('chi2r gompertz = ', chi2_gompertz_infected/DOF)
print('chi2r logistic = ', chi2_logistic_infected/DOF)

#Fit with error bands
Ymin_gompertz=np.array([gompertz(i,gompertz_Par) for i in xTOT])-np.array(std_fit_gompertz)
Ymax_gompertz=np.array([gompertz(i,gompertz_Par) for i in xTOT])+np.array(std_fit_gompertz)

Ymin_logistic=np.array([logistic(i,logistic_Par) for i in xTOT])-np.array(std_fit_logistic)
Ymax_logistic=np.array([logistic(i,logistic_Par) for i in xTOT])+np.array(std_fit_logistic)

# Predictions
Xpredicted=[ii+max(x) for ii in range(1,NumberOfDaysPredicted+1)]
Ypredicted_gompertz=[gompertz(ii,gompertz_Par) for ii in Xpredicted]
Ypredicted_logistic=[logistic(ii,logistic_Par) for ii in Xpredicted]
YPERR_gompertz = np.array([std_fit_gompertz[i] for i in range(len(x),len(x)+NumberOfDaysPredicted)])
YPERR_logistic = np.array([std_fit_logistic[i] for i in range(len(x),len(x)+NumberOfDaysPredicted)])

Ypredicted_gompertz_deaths=Ypredicted_gompertz
YPERR_gompertz_deaths=YPERR_gompertz

#Plot with predictions
plt.figure('predictions_deaths')
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red", alpha=0.75,label="Data" )
plt.errorbar(Xpredicted,Ypredicted_gompertz, yerr=YPERR_gompertz, fmt='o',color="orange", alpha=0.5,label="Gompertz predictions ({} days)".format(NumberOfDaysPredicted) )
plt.errorbar(Xpredicted,Ypredicted_logistic, yerr=YPERR_logistic, fmt='o',color="green", alpha=0.5,label="Logistic predictions ({} days)".format(NumberOfDaysPredicted) )
plt.fill_between(xTOT,Ymin_gompertz,Ymax_gompertz,facecolor='blue', alpha = 0.3 )
plt.fill_between(xTOT,Ymin_logistic,Ymax_logistic,facecolor='blue', alpha = 0.3 )
plt.semilogy(xTOT, [gompertz(i,gompertz_Par) for i in xTOT], 'b',label="Gompertz model" )
plt.semilogy(xTOT, [logistic(i,logistic_Par) for i in xTOT], 'green',label="Logistic model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of dead people")
plt.ylim((min(y)*0.9,gompertz_Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig(namefile+case[1]+types[0]+region+ext, dpi=DPI)
plt.show()


# differences: derivatives
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])
Y3=Y2-Y1
ERRY3=np.sqrt(Y2+Y1)

#fittiamo i rapporti incrementali con le derivate dei modelli
gompertz_fit_derivative    = curve_fit(gompertz_model_derivative,x,Y3,gompertz_Par,sigma=ERRY3, absolute_sigma=False)
gompertz_ParD              = gompertz_fit_derivative[0]
gompertz_CovD              = gompertz_fit_derivative[1]
gompertz_simulated_par_D   = np.random.multivariate_normal(gompertz_ParD, gompertz_CovD, size=SIZE)
gompertz_simulated_curve_D = [[gompertz_derivative(ii,par) for ii in xTOT] for par in gompertz_simulated_par_D]
#gompertz_std_fit_D         = np.std(gompertz_simulated_curve_D, axis=0)
gompertz_std_fit_D         = np.median(np.abs([gompertz_simulated_curve_D[ii] - np.median(gompertz_simulated_curve_D,axis=0) for ii in range(SIZE)]))/alphasigma
gompertz_YminD             = np.array([gompertz_derivative(i,gompertz_ParD) for i in xTOT])-np.array(gompertz_std_fit_D)
gompertz_YmaxD             = np.array([gompertz_derivative(i,gompertz_ParD) for i in xTOT])+np.array(gompertz_std_fit_D)

logistic_fit_derivative    = curve_fit(logistic_model_derivative,x,Y3,logistic_Par,sigma=ERRY3, absolute_sigma=False)
logistic_ParD              = logistic_fit_derivative[0]
logistic_CovD              = logistic_fit_derivative[1]
logistic_simulated_par_D   = np.random.multivariate_normal(logistic_ParD, logistic_CovD, size=SIZE)
logistic_simulated_curve_D = [[logistic_derivative(ii,par) for ii in xTOT] for par in logistic_simulated_par_D]
logistic_std_fit_D         = np.std(logistic_simulated_curve_D, axis=0)
logistic_YminD             = np.array([logistic_derivative(i,logistic_ParD) for i in xTOT])-np.array(logistic_std_fit_D)
logistic_YmaxD             = np.array([logistic_derivative(i,logistic_ParD) for i in xTOT])+np.array(logistic_std_fit_D)

#plt.figure('prova')
#for par in gompertz_simulated_par_D:
#    plt.plot(xTOT,[gompertz_derivative(ii,par) for ii in xTOT])
#plt.show()
# Real data

plt.figure('derivatives_deaths')
plt.errorbar(x, Y3, yerr=ERRY3, fmt='o',color="red", alpha=0.75,label="Data" )
plt.plot(xTOT, [logistic_derivative(i,logistic_ParD) for i in xTOT], label="Logistic model derivative" )
plt.fill_between(xTOT,logistic_YminD,logistic_YmaxD,facecolor='blue', alpha = 0.3 )
plt.plot(xTOT, [gompertz_derivative(i,gompertz_ParD) for i in xTOT], label="Gompertz model derivative" )
plt.fill_between(xTOT,gompertz_YminD,gompertz_YmaxD,facecolor='blue', alpha = 0.3 )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Increase of dead people per day")
plt.ylim((min(Y3)*0.9,3*max([logistic_derivative(i,logistic_ParD) for i in xTOT])*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig(namefile+case[1]+types[1]+region+ext, dpi=DPI)
plt.show()









#####################################################
###### Faccio la stessa cosa con i guariti ##########
#####################################################



df = pd.read_csv(url)
df = df.loc[:,['data','dimessi_guariti']]
FMT = '%Y-%m-%d %H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days  )

x = list(df.iloc[:,0])
y = list(df.iloc[:,1])
YERR = np.sqrt(y)

# Fitting Logistico
P0=[0.2,70,60000]
fit_logistic = curve_fit(logistic_model,x,y,P0, sigma=YERR, absolute_sigma=False)
fit_gompertz = curve_fit(gompertz_model,x,y,P0, sigma=YERR, absolute_sigma=False)
logistic_Par = [ii for ii in fit_logistic[0]]
gompertz_Par = [ii for ii in fit_gompertz[0]]
logistic_Cov = fit_logistic[1]
gompertz_Cov = fit_gompertz[1]
ErrPar = [np.sqrt(fit_gompertz[1][ii][ii]) for ii in range(3)]

print('\n Gompertz fit parameters for recovered people:')
print('Rate of growth           =', gompertz_Par[0])
print('Peak day from Jan 1st    =', gompertz_Par[1])
print('Final number of infected =', gompertz_Par[2])

sol = int(fsolve(lambda x : gompertz(x,gompertz_Par) - int(gompertz_Par[2]),gompertz_Par[1]))

# Fitting esponenziale (Dummy, solo per grafico)
exp_fit = curve_fit(exponential_model,x,y,p0=[0.005,0.17,50])

#pred_x = list(range(max(x),sol))         #Qui sto cercando la fine dell'epidemia (in questo caso è troppo lontana con gompletz)
pred_x = list(range(max(x),int(1.3*gompertz_Par[1])))
xTOT= x+pred_x

simulated_par_gompertz   = np.random.multivariate_normal(gompertz_Par, gompertz_Cov, size=SIZE)
simulated_curve_gompertz = [[gompertz(ii,par) for ii in xTOT] for par in simulated_par_gompertz]
std_fit_gompertz         = np.std(simulated_curve_gompertz, axis=0)

simulated_par_logistic   = np.random.multivariate_normal(logistic_Par, logistic_Cov, size=SIZE)
simulated_curve_logistic = [[logistic(ii,par) for ii in xTOT] for par in simulated_par_logistic]
std_fit_logistic         = np.std(simulated_curve_logistic, axis=0)


# Mean square errors
print('\nMSE logistic curve:    ',mean_squared_error(y,[logistic(i,logistic_Par) for i in x]))
print('MSE Gompertz curve:    ',mean_squared_error(y,[gompertz(i,gompertz_Par) for i in x]))

DOF=float(len(x)-len(gompertz_Par))
chi2_gompertz_infected = chisquare(y, [gompertz(ii,gompertz_Par) for ii in x])[0]
chi2_logistic_infected = chisquare(y, [logistic(ii,logistic_Par) for ii in x])[0]
print('chi2r gompertz = ', chi2_gompertz_infected/DOF)
print('chi2r logistic = ', chi2_logistic_infected/DOF)

#Fit with error bands
Ymin_gompertz=np.array([gompertz(i,gompertz_Par) for i in xTOT])-np.array(std_fit_gompertz)
Ymax_gompertz=np.array([gompertz(i,gompertz_Par) for i in xTOT])+np.array(std_fit_gompertz)

Ymin_logistic=np.array([logistic(i,logistic_Par) for i in xTOT])-np.array(std_fit_logistic)
Ymax_logistic=np.array([logistic(i,logistic_Par) for i in xTOT])+np.array(std_fit_logistic)

# Predictions
NumberOfDaysPredicted=14
Xpredicted=[ii+max(x) for ii in range(1,NumberOfDaysPredicted+1)]
Ypredicted_gompertz=[gompertz(ii,gompertz_Par) for ii in Xpredicted]
Ypredicted_logistic=[logistic(ii,logistic_Par) for ii in Xpredicted]
YPERR_gompertz = np.array([std_fit_gompertz[i] for i in range(len(x),len(x)+NumberOfDaysPredicted)])
YPERR_logistic = np.array([std_fit_logistic[i] for i in range(len(x),len(x)+NumberOfDaysPredicted)])

Ypredicted_gompertz_recovered=Ypredicted_gompertz
YPERR_gompertz_recovered=YPERR_gompertz

#Plot with predictions
plt.figure('predictions_recovered')
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red", alpha=0.75,label="Data" )
plt.errorbar(Xpredicted,Ypredicted_gompertz, yerr=YPERR_gompertz, fmt='o',color="orange", alpha=0.5,label="Gompertz predictions ({} days)".format(NumberOfDaysPredicted) )
plt.errorbar(Xpredicted,Ypredicted_logistic, yerr=YPERR_logistic, fmt='o',color="green", alpha=0.5,label="Logistic predictions ({} days)".format(NumberOfDaysPredicted) )
plt.fill_between(xTOT,Ymin_gompertz,Ymax_gompertz,facecolor='blue', alpha = 0.3 )
plt.fill_between(xTOT,Ymin_logistic,Ymax_logistic,facecolor='blue', alpha = 0.3 )
plt.semilogy(xTOT, [gompertz(i,gompertz_Par) for i in xTOT], 'b',label="Gompertz model" )
plt.semilogy(xTOT, [logistic(i,logistic_Par) for i in xTOT], 'green',label="Logistic model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of recovered people")
plt.ylim((min(y)*0.9,gompertz_Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig(namefile+case[2]+types[0]+region+ext, dpi=DPI)
plt.show()





# differences: derivatives
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])
Y3=Y2-Y1
ERRY3=np.sqrt(Y2+Y1)

#fittiamo i rapporti incrementali con le derivate dei modelli
gompertz_fit_derivative    = curve_fit(gompertz_model_derivative,x,Y3,gompertz_Par,sigma=ERRY3, absolute_sigma=True)
gompertz_ParD              = gompertz_fit_derivative[0]
gompertz_CovD              = gompertz_fit_derivative[1]
gompertz_simulated_par_D   = np.random.multivariate_normal(gompertz_ParD, gompertz_CovD, size=SIZE)
#print(gompertz_simulated_par_D,'\n')
gompertz_simulated_curve_D = [[gompertz_derivative(ii,par) for ii in xTOT] for par in gompertz_simulated_par_D]
#gompertz_std_fit_D         = np.std(gompertz_simulated_curve_D, axis=0)
gompertz_std_fit_D         = np.median(np.abs([gompertz_simulated_curve_D[ii] - np.median(gompertz_simulated_curve_D,axis=0) for ii in range(SIZE)]))/alphasigma

gompertz_YminD             = np.array([gompertz_derivative(i,gompertz_ParD) for i in xTOT])-np.array(gompertz_std_fit_D)
gompertz_YmaxD             = np.array([gompertz_derivative(i,gompertz_ParD) for i in xTOT])+np.array(gompertz_std_fit_D)

logistic_fit_derivative    = curve_fit(logistic_model_derivative,x,Y3,logistic_Par,sigma=ERRY3, absolute_sigma=False)
logistic_ParD              = logistic_fit_derivative[0]
logistic_CovD              = logistic_fit_derivative[1]
logistic_simulated_par_D   = np.random.multivariate_normal(logistic_ParD, logistic_CovD, size=SIZE)
logistic_simulated_curve_D = [[logistic_derivative(ii,par) for ii in xTOT] for par in logistic_simulated_par_D]
logistic_std_fit_D         = np.std(logistic_simulated_curve_D, axis=0)
logistic_YminD             = np.array([logistic_derivative(i,logistic_ParD) for i in xTOT])-np.array(logistic_std_fit_D)
logistic_YmaxD             = np.array([logistic_derivative(i,logistic_ParD) for i in xTOT])+np.array(logistic_std_fit_D)

#plt.figure('prova')
#for par in gompertz_simulated_par_D:
#    plt.plot(xTOT,[gompertz_derivative(ii,par) for ii in xTOT])
#plt.show()

# Real data
plt.figure('derivatives_recovered')
plt.errorbar(x, Y3, yerr=ERRY3, fmt='o',color="red", alpha=0.75,label="Data" )
plt.plot(xTOT, [logistic_derivative(i,logistic_ParD) for i in xTOT], label="Logistic model derivative" )
plt.fill_between(xTOT,logistic_YminD,logistic_YmaxD,facecolor='blue', alpha = 0.3 )
plt.plot(xTOT, [gompertz_derivative(i,gompertz_ParD) for i in xTOT], label="Gompertz model derivative" )
plt.fill_between(xTOT,gompertz_YminD,gompertz_YmaxD,facecolor='blue', alpha = 0.3 )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Increase of recovered people per day")
plt.ylim((min(Y3)*0.9,3*max([logistic_derivative(i,logistic_ParD) for i in xTOT])*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig(namefile+case[2]+types[1]+region+ext, dpi=DPI)
plt.show()








# Exporting predictions
#Date of 2020 Jan 1st 


totale_attualmente_positivi=np.array(Ypredicted_gompertz_infected)-np.array(Ypredicted_gompertz_recovered)-np.array(Ypredicted_gompertz_deaths)
std_totale_attualmente_positivi=np.sqrt(np.array(YPERR_gompertz_infected)**2+np.array(YPERR_gompertz_deaths)**2+np.array(YPERR_gompertz_recovered)**2     )


startingDate=737425
Dates=[datetime.fromordinal(startingDate+ii) for ii in Xpredicted]
for ii in range(len(Dates)):
    Dates[ii]= Dates[ii].replace(minute=00, hour=18,second=00)
FirstLine=['denominazione_regione','data','totale_casi','std_totale_casi','totale_attualmente_positivi','std_totale_attualmente_positivi','deceduti','std_deceduti','dimessi_guariti','std_dimessi_guariti']
with open('model-gompertz-national.csv', 'w',newline='') as pred_gompertz_file:
    wr = csv.writer(pred_gompertz_file, quoting=csv.QUOTE_ALL)
    wr.writerow(FirstLine)
    for ii in range(len(Xpredicted)):
        wr.writerow(['Italia',Dates[ii],Ypredicted_gompertz_infected[ii],YPERR_gompertz_infected[ii],totale_attualmente_positivi[ii],std_totale_attualmente_positivi[ii],Ypredicted_gompertz_deaths[ii],YPERR_gompertz_deaths[ii],Ypredicted_gompertz_recovered[ii],YPERR_gompertz_recovered[ii]])