# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 02:22:58 2020

@author: Jacopo Busatto
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import csv 

#Quality for plot saving
DPI=450

# Define analytical model
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
fit = curve_fit(logistic_model,x,y,P0, sigma=YERR, absolute_sigma=False)

Par = [ii for ii in fit[0]]
Cov = fit[1]
ErrPar = [np.sqrt(fit[1][ii][ii]) for ii in range(3)]

print('\nLogistic fit parameters for total infected people')
print('Rate of growth           =', fit[0][0])
print('Peak day from Jan 1st    =', fit[0][1])
print('Final number of infected =', fit[0][2])

sol = int(fsolve(lambda x : logistic(x,Par) - int(Par[2]),Par[1]))

# Fitting esponenziale (Dummy, solo per grafico)
exp_fit = curve_fit(exponential_model,x,y,p0=[0.005,0.17,50])

pred_x = list(range(max(x),sol))
xTOT= x+pred_x

SIZE=100
simulated_par= np.random.multivariate_normal(Par, Cov, size=SIZE)
simulated_curve=[[logistic(ii,par) for ii in xTOT] for par in simulated_par]
std_fit=np.std(simulated_curve, axis=0)
Ymin=np.array([logistic(i,Par) for i in xTOT])-np.array(std_fit)
Ymax=np.array([logistic(i,Par) for i in xTOT])+np.array(std_fit)

y_pred_logistic = [logistic(i,fit[0]) for i in x]
y_pred_exp =  [exponential_model(i,exp_fit[0][0], exp_fit[0][1],exp_fit[0][2]) for i in x]
print('\nMSE logistic curve:    ',mean_squared_error(y,y_pred_logistic))
# print('MSE exponential curve: ',mean_squared_error(y,y_pred_exp))

DOF=float(len(x)-len(Par))
chi2_logistic_infected = chisquare(y, [logistic(ii,Par) for ii in x])[0]/DOF
print('chi2r = ', chi2_logistic_infected)







# Plot andamento
plt.rcParams['figure.figsize'] = [9, 9]
plt.rc('font', size=16)
fig=plt.figure('linear_infected')
#Dati
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red", alpha=0.75,label="Data" )
#Andamenti
plt.plot(xTOT, [logistic(i,Par) for i in xTOT], label="Logistic model" )
plt.fill_between(xTOT,Ymin,Ymax,facecolor='blue', alpha = 0.3 )
plt.plot(xTOT, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in xTOT], label="Exponential model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/linear_infected.png', dpi=DPI)
#plt.show()

#Log Plot
fig=plt.figure('log_infected')
# Real data
plt.grid()
# plt.scatter(x,y,label="Real data",color="red")
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red",label="Data" )
# Predicted logistic curve
plt.semilogy(xTOT, [logistic(i,Par) for i in xTOT], label="Logistic model" )
# Predicted exponential curve
plt.semilogy(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x], label="Exponential model" )
plt.fill_between(xTOT,Ymin,Ymax,facecolor='blue', alpha = 0.3 )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/log_infected.png', dpi=DPI)

#plt.show()




# Ratio and differences
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])

#Plot ratio
fig=plt.figure('ratios_infected')
plt.plot(x,Y2/Y1,"--or",markersize=8,label="Ratio",alpha=0.6)
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Ratio of infected people respect to the day before")
plt.ylim((min(Y2/Y1)*0.9,max(Y2/Y1)*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/ratios_infected.png', dpi=DPI)
#plt.show()


# differences
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])
Y3=Y2-Y1
ERRY3=np.sqrt(Y2+Y1)
X=np.array(x)
X=np.delete(X,15,axis=0)
Y3=np.delete(Y3,15,axis=0)
ERRY3=np.delete(ERRY3,15,axis=0)
fit_derivative = curve_fit(logistic_model_derivative,X,Y3,p0=[0.2,70,64000],sigma=ERRY3, absolute_sigma=False)
ParD=fit_derivative[0]
CovD=fit_derivative[1]
simulated_par_D= np.random.multivariate_normal(ParD, CovD, size=SIZE)
simulated_curve_D=[[logistic_derivative(ii,par) for ii in list(X)+pred_x] for par in simulated_par_D]
std_fit_D=np.std(simulated_curve_D, axis=0)


Ymin= np.array([logistic_derivative(i,ParD) for i in list(X)+pred_x])-np.array(std_fit_D)
Ymax= np.array([logistic_derivative(i,ParD) for i in list(X)+pred_x])+np.array(std_fit_D)









# Real data
fig=plt.figure('derivatives_infected')
#plt.scatter(X,Y3,label="Real data",color="red", alpha=0.6)
plt.errorbar(X, Y3, yerr=ERRY3, fmt='o',color="red", alpha=0.75,label="Data" )
# Predicted logistic curve
plt.plot(list(X)+pred_x, [logistic_derivative(i,fit_derivative[0]) for i in list(X)+pred_x], label="Logistic model derivative" )
plt.fill_between(list(X)+pred_x,Ymin,Ymax,facecolor='blue', alpha = 0.3 )

plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Increase of infected people")
plt.ylim((min(Y3)*0.9,max([logistic_derivative(i,fit_derivative[0]) for i in list(X)+pred_x])*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/derivatives_infected.png', dpi=DPI)

#plt.show()




# Predictions


NumberOfDaysPredicted=14
Xpredicted=[ii+max(x) for ii in range(1,NumberOfDaysPredicted+1)]
Ypredicted=[logistic(ii,Par) for ii in Xpredicted]
Ymin=np.array([logistic(i,Par) for i in xTOT])-np.array(std_fit)
Ymax=np.array([logistic(i,Par) for i in xTOT])+np.array(std_fit)
YPERR=np.array([std_fit[i] for i in range(len(x),len(x)+NumberOfDaysPredicted)])

Ypredicted_logistic_infected=Ypredicted
YPERR_logistic_infected=YPERR






#Plot with predictions
fig=plt.figure('predictions_infected')
# plt.scatter(x,y,label="Real data",color="red",linestyle="None")
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red", alpha=0.75,label="Data" )
plt.errorbar(Xpredicted,Ypredicted, yerr=YPERR, fmt='o',color="orange", alpha=1,label="Predictions ({} days)".format(NumberOfDaysPredicted) )

# Predicted logistic curve
plt.fill_between(xTOT,Ymin,Ymax,facecolor='blue', alpha = 0.3 )
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], 'r',label="Logistic model" )

plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/predictions_infected.png', dpi=DPI)
#plt.show()













##############################################################
######## RIPETO LA STESSA COSA PER LE MORTI              #####



df = pd.read_csv(url)
df = df.loc[:,['data','deceduti']]
FMT = '%Y-%m-%d %H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days  )

x = list(df.iloc[:,0])
y = list(df.iloc[:,1])
YERR = np.sqrt(y)

# Fitting Logistico
P0=[0.3,70,5000]
fit = curve_fit(logistic_model,x,y,P0, sigma=YERR, absolute_sigma=False)


xTOT= x+pred_x

Par = [ii for ii in fit[0]]
Cov= fit[1]
SIZE=100
simulated_par= np.random.multivariate_normal(Par, Cov, size=SIZE)
simulated_curve=[[logistic(ii,par) for ii in xTOT] for par in simulated_par]
std_fit = np.std(simulated_curve, axis=0)
ErrPar = [np.sqrt(fit[1][ii][ii]) for ii in range(3)]

print('\nLogistic fit parameters for total deaths')
print('Rate of growth         =', fit[0][0])
print('Peak day from Jan 1st  =', fit[0][1])
print('Final number of deaths =', fit[0][2])


# Fitting esponenziale (Dummy, solo per grafico)
exp_fit = curve_fit(exponential_model,x,y,p0=[0.005,0.17,50])





y_pred_logistic = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x]
# y_pred_exp =  [exponential_model(i,exp_fit[0][0], exp_fit[0][1],exp_fit[0][2]) for i in x]
print('\nMSE logistic curve:    ',mean_squared_error(y,y_pred_logistic))
# print('MSE exponential curve: ',mean_squared_error(y,y_pred_exp))

chi2_logistic_deaths = chisquare(y, [logistic(ii,Par) for ii in x])[0]/DOF
print('chi2 = ', chi2_logistic_deaths)

Ymin=np.array([logistic(i,Par) for i in xTOT])-np.array(std_fit)
Ymax=np.array([logistic(i,Par) for i in xTOT])+np.array(std_fit)


# Plot andamento
fig=plt.figure('linear_deaths')
#Dati
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red", alpha=0.75,label="Data" )
#Andamenti
plt.plot(xTOT, [logistic(i,Par) for i in xTOT], label="Logistic model" )
plt.fill_between(xTOT, Ymin,Ymax,facecolor='blue', alpha = 0.3 )
#plt.plot(xTOT, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in xTOT], label="Exponential model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of dead people")
plt.ylim((min(y)*0.9,Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/linear_deaths.png', dpi=DPI)
plt.show()

#Log Plot
fig=plt.figure('log_deaths')
#Real data
plt.grid()
# plt.scatter(x,y,label="Real data",color="red")
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red",label="Data" )
# Predicted logistic curve
plt.semilogy(xTOT, [logistic(i,Par) for i in xTOT], label="Logistic model" )
# Predicted exponential curve
plt.semilogy(xTOT, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in xTOT], label="Exponential model" )
plt.fill_between(xTOT, Ymin,Ymax,facecolor='blue', alpha = 0.3 )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of dead people")
plt.ylim((min(y)*0.9,Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/log_deaths.png', dpi=DPI)
#plt.show()




# Ratio and differences
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])

#Plot ratio
fig=plt.figure('ratios_deaths')
plt.plot(x,Y2/Y1,"--or",markersize=8,label="Ratio",alpha=0.6)
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Ratio of dead people respect to the day before")
plt.ylim((min(Y2/Y1)*0.9,max(Y2/Y1)*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/ratios_deaths.png', dpi=DPI)

#plt.show()


# differences
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])
Y3=Y2-Y1
ERRY3=np.sqrt(Y2+Y1)
X=np.array(x)
X=np.delete(X,15,axis=0)
Y3=np.delete(Y3,15,axis=0)
ERRY3=np.delete(ERRY3,15,axis=0)
fit_derivative = curve_fit(logistic_model_derivative,X,Y3,p0=[0.2,70,64000],sigma=ERRY3, absolute_sigma=False)
ParD=fit_derivative[0]
CovD=fit_derivative[1]
simulated_par_D= np.random.multivariate_normal(ParD, CovD, size=SIZE)
simulated_curve_D=[[logistic_derivative(ii,par) for ii in list(X)+pred_x] for par in simulated_par_D]
std_fit_D=np.std(simulated_curve_D, axis=0)


YminD= np.array([logistic_derivative(i,ParD) for i in list(X)+pred_x])-np.array(std_fit_D)
YmaxD= np.array([logistic_derivative(i,ParD) for i in list(X)+pred_x])+np.array(std_fit_D)


# Real data
fig=plt.figure('derivatives_deaths')
#plt.scatter(X,Y3,label="Real data",color="red", alpha=0.6)
plt.errorbar(X, Y3, yerr=ERRY3, fmt='o',color="red", alpha=0.75,label="Data" )
# Predicted logistic curve
plt.plot(list(X)+pred_x, [logistic_derivative(i,fit_derivative[0]) for i in list(X)+pred_x], label="Logistic model derivative" )
plt.fill_between(list(X)+pred_x,YminD,YmaxD,facecolor='blue', alpha = 0.3 )

plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Increase of dead people")
plt.ylim((min(Y3)*0.9,max([logistic_derivative(i,fit_derivative[0]) for i in list(X)+pred_x])*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/derivatives_deaths.png', dpi=DPI)

#plt.show()




# Predictions
NumberOfDaysPredicted=14
Xpredicted=[ii+max(x) for ii in range(1,NumberOfDaysPredicted+1)]
Ypredicted=[logistic(ii,Par) for ii in Xpredicted]
Ymin=np.array([logistic(i,Par) for i in xTOT])-np.array(std_fit)
Ymax=np.array([logistic(i,Par) for i in xTOT])+np.array(std_fit)
YPERR=np.array([std_fit[i] for i in range(len(x),len(x)+NumberOfDaysPredicted)])


#Plot with predictions
fig=plt.figure('predictions_deaths')
# plt.scatter(x,y,label="Real data",color="red",linestyle="None")
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red", alpha=0.75,label="Data" )
plt.errorbar(Xpredicted,Ypredicted, yerr=YPERR, fmt='o',color="orange", alpha=1,label="Predictions ({} days)".format(NumberOfDaysPredicted) )
# Predicted logistic curve
plt.fill_between(xTOT,Ymin,Ymax,facecolor='blue', alpha = 0.3 )
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], 'r',label="Logistic model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of dead people")
plt.ylim((min(y)*0.9,Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/predictions_deaths.png', dpi=DPI)
plt.show()

Ypredicted_logistic_deaths=Ypredicted
YPERR_logistic_deaths=YPERR






#####################################################
##### Faccio la stessa cosa per i guariti ###########
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
P0=[0.2,70,10000]
fit = curve_fit(logistic_model,x,y,P0, sigma=YERR, absolute_sigma=False)

Par = [ii for ii in fit[0]]
Cov = fit[1]
ErrPar = [np.sqrt(fit[1][ii][ii]) for ii in range(3)]

print('\nLogistic fit parameters for recovered people')
print('Rate of growth           =', fit[0][0])
print('Peak day from Jan 1st    =', fit[0][1])
print('Final number of recovered =', fit[0][2])

sol = int(fsolve(lambda x : logistic(x,Par) - int(Par[2]),Par[1]))

# Fitting esponenziale (Dummy, solo per grafico)
exp_fit = curve_fit(exponential_model,x,y,p0=[0.005,0.17,50])

xTOT= x+pred_x

simulated_par= np.random.multivariate_normal(Par, Cov, size=SIZE)
simulated_curve=[[logistic(ii,par) for ii in xTOT] for par in simulated_par]
std_fit=np.std(simulated_curve, axis=0)
Ymin=np.array([logistic(i,Par) for i in xTOT])-np.array(std_fit)
Ymax=np.array([logistic(i,Par) for i in xTOT])+np.array(std_fit)


y_pred_logistic = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x]
y_pred_exp =  [exponential_model(i,exp_fit[0][0], exp_fit[0][1],exp_fit[0][2]) for i in x]
print('\nMSE logistic curve:    ',mean_squared_error(y,y_pred_logistic))
# print('MSE exponential curve: ',mean_squared_error(y,y_pred_exp))

chi2_logistic_recovered = chisquare(y, [logistic(ii,Par) for ii in x])[0]/DOF
print('chi2r = ', chi2_logistic_recovered)






# Plot andamento
plt.rcParams['figure.figsize'] = [9, 9]
plt.rc('font', size=16)
fig=plt.figure('linear_recovered')
#Dati
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red", alpha=0.75,label="Data" )
#Andamenti
plt.plot(xTOT, [logistic(i,Par) for i in xTOT], label="Logistic model" )
plt.fill_between(xTOT,Ymin,Ymax,facecolor='blue', alpha = 0.3 )
plt.plot(xTOT, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in xTOT], label="Exponential model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of recovered people")
plt.ylim((min(y)*0.9,Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/linear_recovered.png', dpi=DPI)
#plt.show()

#Log Plot
fig=plt.figure('log_recovered')
# Real data
plt.grid()
# plt.scatter(x,y,label="Real data",color="red")
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red",label="Data" )
# Predicted logistic curve
plt.semilogy(xTOT, [logistic(i,Par) for i in xTOT], label="Logistic model" )
# Predicted exponential curve
plt.semilogy(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x], label="Exponential model" )
plt.fill_between(xTOT,Ymin,Ymax,facecolor='blue', alpha = 0.3 )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of recovered people")
plt.ylim((min(y)*0.9,Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/log_recovered.png', dpi=DPI)

#plt.show()




# Ratio and differences
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])

#Plot ratio
fig=plt.figure('ratios_recovered')
plt.plot(x,Y2/Y1,"--or",markersize=8,label="Ratio",alpha=0.6)
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Ratio of recovered people respect to the day before")
plt.ylim((min(Y2/Y1)*0.9,max(Y2/Y1)*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/ratios_recovered.png', dpi=DPI)
#plt.show()


# differences
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])
Y3=Y2-Y1
ERRY3=np.sqrt(Y2+Y1)
X=np.array(x)
X=np.delete(X,15,axis=0)
Y3=np.delete(Y3,15,axis=0)
ERRY3=np.delete(ERRY3,15,axis=0)
fit_derivative = curve_fit(logistic_model_derivative,X,Y3,p0=[0.2,70,64000],sigma=ERRY3, absolute_sigma=False)
ParD=fit_derivative[0]
CovD=fit_derivative[1]
simulated_par_D= np.random.multivariate_normal(ParD, CovD, size=SIZE)
simulated_curve_D=[[logistic_derivative(ii,par) for ii in list(X)+pred_x] for par in simulated_par_D]
std_fit_D=np.std(simulated_curve_D, axis=0)


YminD= np.array([logistic_derivative(i,ParD) for i in list(X)+pred_x])-np.array(std_fit_D)
YmaxD= np.array([logistic_derivative(i,ParD) for i in list(X)+pred_x])+np.array(std_fit_D)


# Real data
fig=plt.figure('derivatives_recovered')
#plt.scatter(X,Y3,label="Real data",color="red", alpha=0.6)
plt.errorbar(X, Y3, yerr=ERRY3, fmt='o',color="red", alpha=0.75,label="Data" )
# Predicted logistic curve
plt.plot(list(X)+pred_x, [logistic_derivative(i,fit_derivative[0]) for i in list(X)+pred_x], label="Logistic model derivative" )
plt.fill_between(list(X)+pred_x,YminD,YmaxD,facecolor='blue', alpha = 0.3 )

plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Increase of recovered people")
plt.ylim((min(Y3)*0.9,max([logistic_derivative(i,fit_derivative[0]) for i in list(X)+pred_x])*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/derivatives_recovered.png', dpi=DPI)

#plt.show()




# Predictions
NumberOfDaysPredicted=14
Ymin=np.array([logistic(i,Par) for i in xTOT])-np.array(std_fit)
Ymax=np.array([logistic(i,Par) for i in xTOT])+np.array(std_fit)
Xpredicted=[ii+max(x) for ii in range(1,NumberOfDaysPredicted+1)]
Ypredicted=[logistic(ii,Par) for ii in Xpredicted]
YPERR=np.array([std_fit[i] for i in range(len(x),len(x)+NumberOfDaysPredicted)])

Ypredicted_logistic_recovered=Ypredicted
YPERR_logistic_recovered=YPERR






#Plot with predictions
fig=plt.figure('predictions_recovered')
# plt.scatter(x,y,label="Real data",color="red",linestyle="None")
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red", alpha=0.75,label="Data" )
plt.errorbar(Xpredicted,Ypredicted, yerr=YPERR, fmt='o',color="orange", alpha=1,label="Predictions ({} days)".format(NumberOfDaysPredicted) )

# Predicted logistic curve
plt.fill_between(xTOT,Ymin,Ymax,facecolor='blue', alpha = 0.3 )
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], 'r',label="Logistic model" )

plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of recovered people")
plt.ylim((min(y)*0.9,Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.savefig('Plots/predictions_recovered.png', dpi=DPI)
#plt.show()


















totale_attualmente_positivi=np.array(Ypredicted_logistic_infected)-np.array(Ypredicted_logistic_recovered)-np.array(Ypredicted_logistic_deaths)
std_totale_attualmente_positivi=np.sqrt(np.array(YPERR_logistic_infected)**2+np.array(YPERR_logistic_deaths)**2+np.array(YPERR_logistic_recovered)**2     )

startingDate=737425
Dates=[datetime.fromordinal(startingDate+ii) for ii in Xpredicted]
for ii in range(len(Dates)):
    Dates[ii]= Dates[ii].replace(minute=00, hour=18,second=00)
FirstLine=['denominazione_regione','data','totale_casi','std_totale_casi','totale_attualmente_positivi','std_totale_attualmente_positivi','deceduti','std_deceduti','dimessi_guariti','std_dimessi_guariti']
with open('model-logistic-national.csv', 'w',newline='') as pred_logistic_file:
    wr = csv.writer(pred_logistic_file, quoting=csv.QUOTE_ALL)
    wr.writerow(FirstLine)
    for ii in range(len(Xpredicted)):
        wr.writerow(['Italia',Dates[ii],Ypredicted_logistic_infected[ii],YPERR_logistic_infected[ii],totale_attualmente_positivi[ii],std_totale_attualmente_positivi[ii],Ypredicted_logistic_deaths[ii],YPERR_logistic_deaths[ii],Ypredicted_logistic_recovered[ii],YPERR_logistic_recovered[ii]])






