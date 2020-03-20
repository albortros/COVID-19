# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:23:34 2020

@author: jacop
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
#import csv with open() as pred_gompertz_file


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
fit_logistic = curve_fit(logistic_model,x,y,P0, sigma=YERR, absolute_sigma=True)
fit_gompertz = curve_fit(gompertz_model,x,y,P0, sigma=YERR, absolute_sigma=True)
logistic_Par = [ii for ii in fit_logistic[0]]
gompertz_Par = [ii for ii in fit_gompertz[0]]
ErrPar = [np.sqrt(fit_gompertz[1][ii][ii]) for ii in range(3)]

print('\n Gompertz fit parameters for total infected people:')
print('Rate of growth           =', gompertz_Par[0])
print('Peak day from Jan 1st    =', gompertz_Par[1])
print('Final number of infected =', gompertz_Par[2])

sol = int(fsolve(lambda x : logistic(x,gompertz_Par) - int(gompertz_Par[2]),gompertz_Par[1]))

# Fitting esponenziale (Dummy, solo per grafico)
exp_fit = curve_fit(exponential_model,x,y,p0=[0.005,0.17,50])

#pred_x = list(range(max(x),sol))         #Qui sto cercando la fine dell'epidemia (in questo caso è troppo lontana con gompletz)
pred_x = list(range(max(x),2*int(gompertz_Par[1])))
xTOT= x+pred_x

YMIN=np.array(y)-np.array(YERR)
YMAX=np.array(y)+np.array(YERR)
fitMIN_gompertz = curve_fit(gompertz_model,x,YMIN,p0=gompertz_Par, sigma=YERR, absolute_sigma=True)
fitMAX_gompertz = curve_fit(gompertz_model,x,YMAX,p0=gompertz_Par, sigma=YERR, absolute_sigma=True)
ParMAX_gompertz=[fitMAX_gompertz[0][i] for i in range(3)]
ParMIN_gompertz=[fitMIN_gompertz[0][i] for i in range(3)]

fitMIN_logistic = curve_fit(logistic_model,x,YMIN,p0=gompertz_Par, sigma=YERR, absolute_sigma=True)
fitMAX_logistic = curve_fit(logistic_model,x,YMAX,p0=logistic_Par, sigma=YERR, absolute_sigma=True)
ParMAX_logistic=[fitMAX_logistic[0][i] for i in range(3)]
ParMIN_logistic=[fitMIN_logistic[0][i] for i in range(3)]


y_pred_logistic = [logistic(i,logistic_Par) for i in x]
y_pred_gompertz =  [gompertz(i,gompertz_Par) for i in x]
print('\nMSE logistic curve:    ',mean_squared_error(y,y_pred_logistic))
print('MSE Gompertz curve:    ',mean_squared_error(y,y_pred_gompertz))







##Lo commento perche nel caso di Gompletz l'asintoto_G>>asintoto_logistico, serve la semilog
## Plot andamento
#plt.rcParams['figure.figsize'] = [9, 9]
#plt.rc('font', size=16)
#plt.figure('Infected')
##Dati
#plt.errorbar(x, y, yerr=YERR, fmt='o',color="red", alpha=0.75,label="Data" )
##Andamenti
#plt.plot(xTOT, [logistic(i,logistic_Par) for i in xTOT], label="Logistic model" )
#plt.plot(xTOT, [gompertz(i,gompertz_Par) for i in xTOT], label="Gompertz model" )
#plt.fill_between(xTOT, [gompertz(i,ParMAX_gompertz) for i in xTOT],[gompertz(i,ParMIN_gompertz) for i in xTOT],facecolor='blue', alpha = 0.3 )
##plt.plot(xTOT, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in xTOT], label="Exponential model" )
#plt.legend()
#plt.xlabel("Days since 1 January 2020")
#plt.ylabel("Total number of infected people")
#plt.ylim((min(y)*0.9,gompertz_Par[2]*1.1))
#plt.grid(linestyle='--',which='both')
#plt.show()

#Log Plot
plt.figure('Log Plot')
# Real data
plt.grid()
# plt.scatter(x,y,label="Real data",color="red")
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red",label="Data",alpha=0.6 )
# Predicted logistic curve
plt.semilogy(xTOT, [logistic(i,logistic_Par) for i in xTOT], label="Logistic model" )
plt.semilogy(xTOT, [gompertz(i,gompertz_Par) for i in xTOT], label="Gompertz model" )
# Predicted exponential curve
plt.semilogy(xTOT, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in xTOT], label="Exponential model" )
plt.fill_between(xTOT, [gompertz(i,ParMAX_gompertz) for i in xTOT],[gompertz(i,ParMIN_gompertz) for i in xTOT],facecolor='red', alpha = 0.3 )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,gompertz_Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.show()




# Ratio and differences
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])

#Plot ratio
plt.figure('Ratio')
plt.plot(x,Y2/Y1,"--or",markersize=8,label="Ratio",alpha=0.6)
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Ratio of infected people respect to the day before")
plt.ylim((min(Y2/Y1)*0.9,max(Y2/Y1)*1.1))
plt.grid(linestyle='--',which='both')
plt.show()


# differences
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])
Y3=Y2-Y1
ERRY3=np.sqrt(Y2+Y1)

#Mi pare di aver letto che il dato n 15 fosse sbagliato

#fittiamo i rapporti incrementali con le derivate dei modelli
fit_derivative_logistic = curve_fit(logistic_model_derivative,x,Y3,logistic_Par)
fdM_logistic = curve_fit(logistic_model_derivative,x,Y3+ERRY3,logistic_Par)
fdm_logistic = curve_fit(logistic_model_derivative,x,Y3-ERRY3,logistic_Par)
Ymin_logistic= [logistic_derivative(i,fdm_logistic[0]) for i in xTOT]
Ymax_logistic= [logistic_derivative(i,fdM_logistic[0]) for i in xTOT]

fit_derivative_gompertz = curve_fit(gompertz_model_derivative,x,Y3,gompertz_Par)
fdM_gompertz = curve_fit(gompertz_model_derivative,x,Y3+ERRY3,gompertz_Par)
fdm_gompertz = curve_fit(gompertz_model_derivative,x,Y3-ERRY3,gompertz_Par)
Ymin_gompertz= [gompertz_derivative(i,fdm_gompertz[0]) for i in xTOT]
Ymax_gompertz= [gompertz_derivative(i,fdM_gompertz[0]) for i in xTOT]


# Real data
plt.figure('Increases')
#plt.scatter(X,Y3,label="Real data",color="red", alpha=0.6)
plt.errorbar(x, Y3, yerr=ERRY3, fmt='o',color="red", alpha=0.75,label="Data" )
# Predicted logistic curve
plt.plot(xTOT, [logistic_derivative(i,fit_derivative_logistic[0]) for i in xTOT], label="Logistic model derivative" )
plt.fill_between(xTOT,Ymin_logistic,Ymax_logistic,facecolor='blue', alpha = 0.3 )

plt.plot(xTOT, [gompertz_derivative(i,fit_derivative_gompertz[0]) for i in xTOT], label="Gompertz model derivative" )
plt.fill_between(xTOT,Ymin_gompertz,Ymax_gompertz,facecolor='blue', alpha = 0.3 )

plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Increase of infected people per day")
plt.ylim((min(Y3)*0.9,2*max([logistic_derivative(i,fit_derivative_logistic[0]) for i in xTOT])*1.1))
plt.grid(linestyle='--',which='both')
plt.show()




# Predictions
NumberOfDaysPredicted=14
Xpredicted=[ii+max(x) for ii in range(1,NumberOfDaysPredicted+1)]
Ypredicted_gompertz=[gompertz(ii,gompertz_Par) for ii in Xpredicted]
Ypredicted_logistic=[logistic(ii,logistic_Par) for ii in Xpredicted]
Ymin_gompertz=[gompertz(i,ParMAX_gompertz) for i in xTOT]
Ymax_gompertz=[gompertz(i,ParMIN_gompertz) for i in xTOT]
YPERR_gompertz=[np.absolute(gompertz(i,ParMAX_gompertz)-gompertz(i,ParMIN_gompertz))/2. for i in Xpredicted]
YPERR_logistic=[np.absolute(logistic(i,ParMAX_logistic)-logistic(i,ParMIN_logistic))/2. for i in Xpredicted]






#Plot with predictions
plt.figure('Predictions')
# plt.scatter(x,y,label="Real data",color="red",linestyle="None")
plt.errorbar(x, y, yerr=YERR, fmt='o',color="red", alpha=0.75,label="Data" )
plt.errorbar(Xpredicted,Ypredicted_gompertz, yerr=YPERR_gompertz, fmt='o',color="orange", alpha=1,label="Gompertz predictions ({} days)".format(NumberOfDaysPredicted) )
plt.errorbar(Xpredicted,Ypredicted_logistic, yerr=YPERR_logistic, fmt='o',color="green", alpha=1,label="Logistic predictions ({} days)".format(NumberOfDaysPredicted) )
# Predicted logistic curve
plt.fill_between(x+pred_x, [gompertz(i,fitMAX_gompertz[0]) for i in xTOT],[gompertz(i,fitMIN_gompertz[0]) for i in xTOT],facecolor='blue', alpha = 0.3 )
plt.semilogy(xTOT, [gompertz(i,gompertz_Par) for i in xTOT], 'r',label="Gompertz model" )
plt.semilogy(xTOT, [logistic(i,logistic_Par) for i in xTOT], 'r',label="Logistic model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,gompertz_Par[2]*1.1))
plt.grid(linestyle='--',which='both')
plt.show()

