import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def integrate_ode(x, tempo_totale):
    # variabili importanti
    h = 5.85*(10**7) # numero di abitanti
    r = 74 # 18.5*4 # tempo medio di guarigione, per 4 perche' aggiorniamo ogni 6 ore
    t = 22 #5.5*4 # tempo medio di incubazione, per 4 perche' aggiorniamo ogni 6 ore
    E = [] # lista degli esposti
    I = [] # lista degli infetti
    R = [] # lista dei guariti
    s = x[0] # parametro s
    alpha = x[1]
    # condizione iniziale
    E.append(0)   # nessun esposto
    I.append(1)   # l'infetto bastardo
    R.append(0)   # nessun guarito per ora fra
    # inizia l'integrazione numerica
    for step in range(tempo_totale):
        print("Contributi a E")
        print("lineare : "+str(E[step]-I[step]))
        print("exp :"+str((-(1-s*alpha/h)**(I[step]/alpha))*(h-E[step]-I[step]-R[step])))
        print("solo exp :"+str((1-s*alpha/h)**(I[step]/alpha)))
        new_E = E[step] - I[step] + (1-(1-s*alpha/h)**(I[step]/alpha))*(h-E[step]-I[step]-R[step])
        if step+1-t < 0: # controlliamo se e' gia' passato un ciclo di incubazione
            new_I = I[step]-R[step]
        else:
            new_I = E[step+1-t]+I[step]-R[step]
        if step+1-r < 0: # controlliamo se qualcuno e' gia' guarito
            new_R = R[step]
        else:
            new_R = I[step+1-r]+R[step]
        print("tempo "+str(step))
        print(str(new_E)+" "+str(new_I)+" "+str(new_R))
        E.append(new_E)
        I.append(new_I)
        R.append(new_R)

    # fine integrazione numerica
    return E, I, R

#def calc_least_squares(x, reference):
def calc_least_squares(x, tempo_totale):
    E = []
    I = []
    R = []
    E, I, R = integrate_ode(x, tempo_totale)
    h = 5.85*(10**7) # numero di abitanti
    #print(E)
    #print(I)
    #print(R)
    fig = plt.figure(facecolor='w')
    t = np.linspace(0, tempo_totale, tempo_totale+1)
    #ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
    plt.plot(t, (h-np.array(E)-np.array(I))/h, 'b', alpha=0.5, lw=2, label='Fraz. Sani')
    plt.plot(t, np.array(E)/h, 'r', alpha=0.5, lw=2, label='Fraz. Esposti')
    plt.plot(t, np.array(I)/h, 'g', alpha=0.5, lw=2, label='Fraz. Infetti')
    plt.plot(t, np.array(R)/h, 'y', alpha=0.5, lw=2, label='Fraz. Guariti')
    plt.xlabel('Tempo [6 ore]')
    plt.ylabel('Totale')
    #plt.ylim(0,1.2)
    #ax.yaxis.set_tick_params(length=0)
    #ax.xaxis.set_tick_params(length=0)
    #ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = plt.legend()
    legend.get_frame().set_alpha(0.8)
    #for spine in ('top', 'right', 'bottom', 'left'):
    #    ax.spines[spine].set_visible(False)
    
    plt.title ('MILO')
#    plt.savefig('model_SIR.png', dpi = 300)
    plt.show()

   # sum_squares = 0.0
   # # reference e' una tupla di lunghezza numero di campionamenti (cioe' giorni per 4)
   # for i in range(len(reference)):
   #     sum_squares = sum_squares + (I[i]-reference[i])**2
   # return sum_squares

#print(calc_least_squares(np.array([3,1]),tuple(ref)))
calc_least_squares(np.array([3,1]),110)
