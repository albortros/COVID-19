import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def integrate_ode(x, tempo_totale, quarantine_time):
    # variabili importanti
    h = 5.85*(10**7) # numero di abitanti
    r = 74 # 18.5*4 # tempo medio di guarigione, per 4 perche' aggiorniamo ogni 6 ore
    t = 22 #5.5*4 # tempo medio di incubazione, per 4 perche' aggiorniamo ogni 6 ore
    E = [] # lista degli esposti
    I = [] # lista degli infetti
    R = [] # lista dei guariti
    S = []
    delta_SE = []
    delta_EI = []
    delta_IR = []
    s = x[0] # parametro s
    alpha = 1 # si inizia senza quarantena
    # condizione iniziale
    E.append(0)   # nessun esposto
    I.append(1)   # l'infetto bastardo
    R.append(0)   # nessun guarito per ora fra
    S.append(h-1) # numero di sani
    # inizia l'integrazione numerica
    for step in range(tempo_totale):
        # controlliamo se e' partita la quarantena
        if step > quarantine_time:
            alpha = x[1]
        # calcolo delta SE
        delta_SE.append((1-(1-s/h)**(I[step]/alpha))*S[step])
        # calcolo delta EI
        if step - t < 0: # se non e' passato un ciclo di incubazione e' uguale a zero
            delta_EI.append(0)
        else:
            delta_EI.append(delta_SE[step-t])
        # calcolo delta IR
        if step - r < 0: # se non e' passato un ciclo di guarigione e' uguale a 0
            delta_IR.append(0)
        else:
            delta_IR.append(delta_EI[step-r])
        # aggiorno le quantita
        S.append(S[step]-delta_SE[step])
        E.append(E[step]-delta_EI[step]+delta_SE[step])
        I.append(I[step]-delta_IR[step]+delta_EI[step])
        R.append(R[step]+delta_IR[step])

    # fine integrazione numerica
    return S, E, I, R

def calc_resid(x, *args):
    S = []
    E = []
    I = []
    R = []
    total_time = args[0] # tempo totale dell'evoluzione
    time_shift = args[1] # shift oltre il quale iniziare a confrontare i dati
    reference  = args[2] # lista degli infetti confermati
    S, E, I, R = integrate_ode(x, total_time, time_shift+1) # la quarantena inizia dal 23, quindi un giorno dopo
    ref_I = [int(item) for item in I[time_shift+1:]]
    print("INPUT : "+str(x))
    print("calcolated : "+str(ref_I))
    print("reference  : "+str(reference))
    cost_function = 0
    for i in range(len(ref_I)):
        cost_function = cost_function+((reference[i]-ref_I[i])/reference[i])**2
    return cost_function

with open("../../jhu-csse-COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv","r") as data:
    for line in data:
        if line.split(",")[0] == 'Province/State':
            giorni = [str(item).rstrip("\n") for item in line.split(",")[4:]]
        if line.split(",")[0] == 'Hubei':
            reference = [int(str(item).rstrip("\n")) for item in line.split(",")[4:]]

time_shift = 22
total_time = time_shift + len(reference)
x0 = np.array([2,100])

fitting = minimize(calc_resid, x0, args=(total_time,time_shift,reference),method="Nelder-Mead")
print(fitting)

S = []
E = []
I = []
R = []
S, E, I, R = integrate_ode(fitting.x,total_time,time_shift+1)

t = np.linspace(1,len(reference),len(reference))
fig = plt.figure(facecolor='w')
plt.plot(t, I[time_shift+1:], 'r', alpha=0.5, lw=2, label='MILO')
plt.plot(t, reference, 'b', alpha=0.5, lw=2, label='Hubei')
plt.xlabel('Time [days]')
plt.ylabel('Total')
legend = plt.legend()
plt.title ('MILO Fitting')
plt.show()

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
