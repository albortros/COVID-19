import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def check_boundary(data, abitanti):
    if data < 0:
        data = 0
    if data > abitanti:
        data = abitanti
    return data

def integrate_ode(x, tempo_totale, quarantine):
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
    alpha = x[1]
    # condizione iniziale
    E.append(0)   # nessun esposto
    I.append(1)   # l'infetto bastardo
    R.append(0)   # nessun guarito per ora fra
    S.append(h-1) # numero di sani
    # inizia l'integrazione numerica
    for step in range(tempo_totale):
        print("SITUA: "+str(S[step])+" "+str(E[step])+" "+str(I[step])+" "+str(R[step]))
        if quarantine:
            if I[step] > 1000:
                alpha = 5
        # calcolo delta SE
        delta_SE.append((1-(1-s/h)**(I[step]/alpha))*S[step])
        qua   = (1-(1-s/h)**(I[step]/alpha))*S[step]
        noqua = (1-(1-s/h)**(I[step]))*S[step]
        print("alpha "+str(alpha)+" qua "+str(qua)+ " noqua "+str(noqua))
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
        # print
        #print("stampa dei delta")
        #print("DELTA "+str(alpha)+" "+str(delta_SE[step])+" "+str(delta_EI[step])+" "+str(delta_IR[step]))
        # aggiorno le quantita
        S.append(S[step]-delta_SE[step])
        E.append(E[step]-delta_EI[step]+delta_SE[step])
        I.append(I[step]-delta_IR[step]+delta_EI[step])
        R.append(R[step]+delta_IR[step])

    # fine integrazione numerica
    return E, I, R, S

#def calc_least_squares(x, reference):
def calc_least_squares(x, tempo_totale):
    E = []
    I = []
    R = []
    S = []
    E, I, R, S = integrate_ode(x, tempo_totale, False)
    h = 5.85*(10**7) # numero di abitanti
    #print(E)
    #print(I)
    #print(R)
    fig = plt.figure(facecolor='w')
    t = np.linspace(0, tempo_totale, tempo_totale+1)
    #ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
    plt.plot(t, np.array(S)/h, 'b', alpha=0.5, lw=2, label='Fraz. Sani')
    plt.plot(t, np.array(E)/h, 'r', alpha=0.5, lw=2, label='Fraz. Esposti')
    plt.plot(t, np.array(I)/h, 'g', alpha=0.5, lw=2, label='Fraz. Infetti')
    plt.plot(t, np.array(R)/h, 'y', alpha=0.5, lw=2, label='Fraz. Guariti')
    plt.xlabel('Tempo [6 ore]')
    plt.ylabel('Frazione')
    #plt.ylim(0,1.2)
    #ax.yaxis.set_tick_params(length=0)
    #ax.xaxis.set_tick_params(length=0)
    #ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = plt.legend()
    legend.get_frame().set_alpha(0.8)
    #for spine in ('top', 'right', 'bottom', 'left'):
    #    ax.spines[spine].set_visible(False)
    
    plt.title ('MILO - no quarantine')
#    plt.savefig('model_SIR.png', dpi = 300)
    plt.show()
    E = []
    I = []
    R = []
    S = []
    E, I, R, S = integrate_ode(x, tempo_totale, True)
    h = 5.85*(10**7) # numero di abitanti
    #print(E)
    #print(I)
    #print(R)
    fig = plt.figure(facecolor='w')
    t = np.linspace(0, tempo_totale, tempo_totale+1)
    #ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
    plt.plot(t, np.array(S)/h, 'b', alpha=0.5, lw=2, label='Fraz. Sani')
    plt.plot(t, np.array(E)/h, 'r', alpha=0.5, lw=2, label='Fraz. Esposti')
    plt.plot(t, np.array(I)/h, 'g', alpha=0.5, lw=2, label='Fraz. Infetti')
    plt.plot(t, np.array(R)/h, 'y', alpha=0.5, lw=2, label='Fraz. Guariti')
    plt.xlabel('Tempo [6 ore]')
    plt.ylabel('Frazione')
    #plt.ylim(0,1.2)
    #ax.yaxis.set_tick_params(length=0)
    #ax.xaxis.set_tick_params(length=0)
    #ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = plt.legend()
    legend.get_frame().set_alpha(0.8)
    #for spine in ('top', 'right', 'bottom', 'left'):
    #    ax.spines[spine].set_visible(False)
    
    plt.title ('MILO - quarantine')
#    plt.savefig('model_SIR.png', dpi = 300)
    plt.show()

   # sum_squares = 0.0
   # # reference e' una tupla di lunghezza numero di campionamenti (cioe' giorni per 4)
   # for i in range(len(reference)):
   #     sum_squares = sum_squares + (I[i]-reference[i])**2
   # return sum_squares

#print(calc_least_squares(np.array([3,1]),tuple(ref)))
calc_least_squares(np.array([1.1,1]),400)
