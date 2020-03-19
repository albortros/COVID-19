import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(url)
df = df.loc[:,['data','totale_casi']]
FMT = '%Y-%m-%d %H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days  )

# Model definitions
def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)*a))

def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))

def logistic_model_derivative(x,a,b,c):
    return a*c*np.exp(-a*(x-b))/(1+np.exp(-(x-b)*a))**2




# Logistic Fit
x = list(df.iloc[:,0])
y = list(df.iloc[:,1])
fit = curve_fit(logistic_model,x,y,p0=[0.2,100,30000])
errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
a=fit[0][0]
b=fit[0][1]
c=fit[0][2]
print('\nLogistic fit parameters for total infected people')
print('a =', fit[0][0])
print('b =', fit[0][1])
print('c =', fit[0][2])

sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))
print('\nExpected end of infection: ',sol,'days after 1st January')

# Exponential Fit
exp_fit = curve_fit(exponential_model,x,y,p0=[0.005,0.17,50])

# Plot
pred_x = list(range(max(x),sol))
plt.figure(0)
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.style.use('seaborn-whitegrid')
plt.scatter(x,y,label="Real data",color="red")
# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], label="Logistic model" )
# Predicted exponential curve
plt.plot(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x], label="Exponential model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))
plt.show()




y_pred_logistic = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x]
y_pred_exp =  [exponential_model(i,exp_fit[0][0], exp_fit[0][1],exp_fit[0][2]) for i in x]
print('\nMSE logistic curve:    ',mean_squared_error(y,y_pred_logistic))
print('MSE exponential curve: ',mean_squared_error(y,y_pred_exp))

# Daily increases
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])
plt.figure(1)
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
plt.plot(x,Y2/Y1,"--or",markersize=8,label="Ratio")
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Ratio of infected people respect to the day before")
plt.ylim((min(Y2/Y1)*0.9,max(Y2/Y1)*1.1))
plt.style.use('seaborn-whitegrid')
plt.show()



# Derivative fit
Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])
Y3=Y2-Y1
X=np.array(x)
X=np.delete(X,15,axis=0)
Y3=np.delete(Y3,15,axis=0)
fit_derivative = curve_fit(logistic_model_derivative,X,Y3,p0=[0.2,70,64000])

pred_x = list(range(max(X),sol))

plt.figure(2)
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(X,Y3,label="Real data",color="red")
# Predicted logistic derivative curve
plt.plot(list(X)+pred_x, [logistic_model_derivative(i,fit_derivative[0][0],fit_derivative[0][1],fit_derivative[0][2]) for i in list(X)+pred_x], label="Logistic model derivative" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Increase of infected people")
plt.ylim((min(Y3)*0.9,max([logistic_model_derivative(i,fit_derivative[0][0],fit_derivative[0][1],fit_derivative[0][2]) for i in list(X)+pred_x])*1.1))
plt.style.use('seaborn-whitegrid')
plt.show()



###################
# Deaths analysis #
###################

df = pd.read_csv(url)
df = df.loc[:,['data','deceduti']]
FMT = '%Y-%m-%d %H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days  )
x = list(df.iloc[:,0])
y = list(df.iloc[:,1])

plt.figure(3)
plt.plot(x,y,"--o",markersize=8,label="Deaths")
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Deaths since 1 January 2020")
plt.ylim((min(y)*0.9,max(y)*1.1))
plt.style.use('seaborn-whitegrid')
plt.show()

fit = curve_fit(logistic_model,x,y,p0=[0.3,70,3000])
a=fit[0][0]
b=fit[0][1]
c=fit[0][2]

print('\nLogistic fit parameters for total dead people')
print('a =', a)
print('b =', b)
print('c =', c)

sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))

# Exponential fit for deaths
exp_fit = curve_fit(exponential_model,x,y,p0=[3,0.28,50])

pred_x = list(range(max(x),sol))

plt.figure(4)
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(x,y,label="Real data",color="red")
# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], label="Logistic model" )
# Predicted exponential curve
plt.plot(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x],label="Exponential model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of dead people")
plt.ylim((min(y)*0.9,c*1.1))
plt.style.use('seaborn-whitegrid')
plt.show()

y_pred_logistic = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x]
y_pred_exp =  [exponential_model(i,exp_fit[0][0], exp_fit[0][1],exp_fit[0][2]) for i in x]
print('\nMSE logistic curve:    ',mean_squared_error(y,y_pred_logistic))
print('MSE exponential curve: ',mean_squared_error(y,y_pred_exp))


plt.figure(5)
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(x,y,label="Real data",color="red")
# Predicted logistic curve
plt.semilogy(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], label="Logistic model" )
# Predicted exponential curve
plt.semilogy(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x],label="Exponential model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of dead people")
plt.ylim((min(y)*0.9,c*1.1))
plt.style.use('seaborn-whitegrid')
plt.show()

Y2=np.array(y)
Y1=np.array(y)
Y1=np.delete(Y1,-1)
Y1=np.insert(Y1,0,Y1[0])
Y3=Y2-Y1
X=np.array(x)
fit_derivative = curve_fit(logistic_model_derivative,X,Y3,p0=[0.2,70,64000])

print('\nLogistic derivative fit parameters for total dead people')
print('a =', fit_derivative[0][0])
print('b =', fit_derivative[0][1])
print('c =', fit_derivative[0][2])

plt.figure(6)
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(X,Y3,label="Real data",color="red")
# Predicted logistic derivative curve
plt.plot(list(X)+pred_x, [logistic_model_derivative(i,fit_derivative[0][0],fit_derivative[0][1],fit_derivative[0][2]) for i in list(X)+pred_x], label="Logistic model derivative" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Increase of dead people")
plt.ylim((min(Y3)*0.9,max([logistic_model_derivative(i,fit_derivative[0][0],fit_derivative[0][1],fit_derivative[0][2]) for i in list(X)+pred_x])*1.1))
plt.style.use('seaborn-whitegrid')
plt.show()
