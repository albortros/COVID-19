# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:03:01 2020

@author: jacop
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#x sono i parametri: x[0]=S_0,x[1]=J,x[2]=rr,x[3]=pp,x[4]=Heff,x[5]=H
def integrate_ode(x, tempo_totale, quarantine_time):
    # variabili importanti
    h = 5.85*(10**7) # numero di abitanti
    S = np.zeros(tempo_totale)
    E = np.zeros(tempo_totale) # lista degli esposti
    I = np.zeros(tempo_totale) # lista degli infetti
    R = np.zeros(tempo_totale) # lista dei guariti

    
    # condizione iniziale
    I[0]=1  # l'infetti bastardi
    S[0]=h-x[2] # numero di sani
    
    #Parametri
    S_0  = x[0] #free growth rate
    J    = x[1] #rate of exposed->infected
    rr   = x[2] #rate of infected->removed
    p    = x[3] #fraction of time in a day spent in danger
    Heff = x[4] #susceptible population after quarantene
#    h    = x[5] #se volessimo fittare pure H

    a    = S_0*(J-rr)
    b    = S_0*(1-J-rr)
    f1a  = (-1+(1+a)**p)*a*(1+a)**p
    f2a  = p*(-1+(1+a)**p)*(a+1-(1+a)**p)
    
    # inizia l'integrazione numerica
#    for step in range(tempo_totale):
#        # controlliamo se e' partita la quarantena
#        if step > quarantine_time:
#            alpha = x[1]
#        # calcolo delta SE 
#        delta_SE.append((s*I[step]*S[step])/(alpha*h))
#        # calcolo delta EI-
#        if step - t < 0: # se non e' passato un ciclo di incubazione e' uguale a zero
#            delta_EI.append(0)
#        else:
#            delta_EI.append(delta_SE[step-t])
#        # calcolo delta IR
#        if step - r < 0: # se non e' passato un ciclo di guarigione e' uguale a 0
#            delta_IR.append(0)
#        else:
#            delta_IR.append(delta_EI[step-r])
#        # aggiorno le quantita
#        S.append(S[step]-delta_SE[step])
#        E.append(E[step]-delta_EI[step]+delta_SE[step])
#        I.append(I[step]-delta_IR[step]+delta_EI[step])
#        R.append(R[step]+delta_IR[step])
    
    for step in range(quarantine_time-1):
        E[step+1]=E[step]+I[step]*S_0*(h-E[step]-I[step]-R[step])*(1-J)/h
        I[step+1]=I[step]+I[step]*S_0*(h-E[step]-I[step]-R[step])*(J-rr)/h
        R[step+1]=R[step]+I[step]*S_0*(h-E[step]-I[step]-R[step])*rr/h
        S[step+1]=S[step]-E[step]-I[step]-R[step]
    for step in range(quarantine_time-1, tempo_totale-1):
        E[step+1]=E[step]+I[step]*S_0*(1-J)*((1+a)**p -1)/a +p*I[step]*(E[step]+R[step])*(1-J)*S_0*(1+a)**(p-1)/Heff +I[step]**2/(Heff*a**2 * (1+a))*(f1a*(1-J)*S_0+f2a*S_0**2 *((1-J)**2+rr*(1-J)))
        I[step+1]=I[step]*(1+a)**p + p*I[step]*(E[step]+R[step])*a*(1+a)**(p-1)/Heff + I[step]**2*((a-b*p)*(1+a)-(a-b*p)*(1+a)**(p-1) + b*p*((1+a)**p-1))/(Heff*a)
        R[step+1]=R[step]+ I[step]*((1+a)**p -1)*rr*S_0/(a) + p*I[step]*(E[step]+R[step])*rr*S_0*(1+a)**(p-1)/Heff + I[step]**2 * (f1a*rr*S_0 + f2a*S_0**2 *(rr**2 + rr*(1-J)))/(Heff*a*a*(1+a))
        S[step+1]=S[step]-E[step]-I[step]-R[step]

    # fine integrazione numerica
    return S, E, I, R

def calc_resid(x, *args):
    total_time = args[0] # tempo totale dell'evoluzione
    time_shift = args[1] # shift oltre il quale iniziare a confrontare i dati
    reference  = args[2] # lista degli infetti confermati

    S, E, I, R = integrate_ode(x, total_time, time_shift+1) # la quarantena inizia dal 23, quindi un giorno dopo
    ref_I = I[time_shift+1:]
    ref_R = R[time_shift+1:]
    cost_function = 0
    for i in range(len(ref_I)):
        cost_function = cost_function+((reference[i]-ref_I[i]-ref_R[i])/reference[i])**2
    return cost_function




with open("../../jhu-csse-COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv","r") as data:
    for line in data:
        if line.split(",")[0] == 'Province/State':
            giorni = [str(item).rstrip("\n") for item in line.split(",")[4:]]
        if line.split(",")[0] == 'Hubei':
            reference = [int(str(item).rstrip("\n")) for item in line.split(",")[4:]]
print('PORCODDIO')
print(reference)
time_shift = 22
total_time=400
#total_time = time_shift + len(reference)


#S_0,J,rr,p,Heff
x0 = np.array([1.2,0.7, 0.8,0.2,70000])

#fitting = minimize(calc_resid, x0, args=(total_time,time_shift,reference),method="Nelder-Mead")
#print(fitting)

#S, E, I, R = integrate_ode(fitting.x,total_time+1000,time_shift+1)

S, E, I, R = integrate_ode(x0,total_time,time_shift+1)

t = [tt for tt in range(total_time)]
fig = plt.figure('dario', facecolor='w')
plt.clf()
#plt.plot(reference)
plt.plot(t, E, alpha=0.5, lw=2, label='esposti')
plt.plot(t, I+R, 'r', alpha=0.5, lw=2, label='MILO')
plt.xlabel('Time [days]')
plt.ylabel('Total')
legend = plt.legend()
plt.title ('MILO Fitting')
plt.show()











#t = np.linspace(1,len(reference),len(reference))
#t2= np.linspace(1,len(reference)+1000, len(reference)+1000)
#fig = plt.figure('dario', facecolor='w')
#plt.clf()
#plt.plot(t2, E[time_shift+1:], alpha=0.5, lw=2, label='esposti')
#plt.plot(t2, np.array(I[time_shift+1:])+np.array(R[time_shift+1:]), 'r', alpha=0.5, lw=2, label='MILO')
#plt.plot(t, reference, 'b', alpha=0.5, lw=2, label='Hubei')
#plt.xlabel('Time [days]')
#plt.ylabel('Total')
#legend = plt.legend()
#plt.title ('MILO Fitting')
#plt.show()



#t = np.linspace(1,len(S),len(S))
#fig = plt.figure(facecolor='w')
##plt.plot(t, S, 'r', alpha=0.5, lw=2, label='Sani')
#plt.plot(t, E, 'b', alpha=0.5, lw=2, label='Exposed')
#plt.plot(t, I, 'g', alpha=0.5, lw=2, label='Infetti')
#plt.plot(t, R, 'y', alpha=0.5, lw=2, label='Guariti')
#plt.xlabel('Time [days]')
#plt.ylabel('Total')
#legend = plt.legend()
#plt.title ('MILO Fitting')
#plt.show()
