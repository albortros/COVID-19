import numpy as np

# lettura comuni
def read_comuni():
    nomi = []
    abitanti = []
    with open("comuni.csv","r") as comuni:
        for line in comuni:
            nomi.append(line.split()[1:4])
            abitanti.append(np.float(line.split()[-1]))
    return nomi, np.array(abitanti)

def genera_flusso(ncity):
    betaij = []
    # per il momento siamo pigri, tutti 1
    for i in range(ncity):
        betaij.append([])
        for j in range(ncity):
            betaij[i].append(1)
    return betaij

def initial_conditions(citta):
    output = []
    for i in range(len(citta)):
        if citta[i][1] == 'Lombardia':
            output.append(1.0)
        else:
            output.append(0.0)    
    return np.array(output)

# leggo i comuni
citta = []
abitanti = []
citta, abitanti = read_comuni()
ncitta = len(abitanti)
# leggo i flussi
betaij = []
betaij = genera_flusso(ncitta)

# numero di giorni di evoluzione
time = 100

# condizioni iniziali
vstart = []
vstart = initial_conditions(citta)

total = [list(vstart)]

# evoluzione temporale
for i in range(time): 
    # self interaction
    vnew = vstart + (1 - np.power(1-vstart/abitanti,vstart))*(abitanti-vstart)
    # mutual interaction
    vstart = vnew
    total.append(vnew)


# dabbato da BPP
## In[54]:
#
#
## Initial conditions vector
#y0 = S0, I0, R0
## Integrate the SIR equations over the time grid, t.
#ret = odeint(derivSIR, y0, t, args=(N, beta, gamma))
#S, I, R = ret.T
#
#
## In[55]:
#
#
## Plot the data on three separate curves for S(t), I(t) and R(t)
#fig = plt.figure(facecolor='w')
##ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
#plt.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
#plt.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
#plt.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
#plt.xlabel('Time [days]')
#plt.ylabel('Fraction')
##plt.ylim(0,1.2)
##ax.yaxis.set_tick_params(length=0)
##ax.xaxis.set_tick_params(length=0)
##ax.grid(b=True, which='major', c='w', lw=2, ls='-')
#legend = plt.legend()
#legend.get_frame().set_alpha(0.8)
##for spine in ('top', 'right', 'bottom', 'left'):
##    ax.spines[spine].set_visible(False)
#
#plt.title ('SIR model')
#plt.savefig('model_SIR.png', dpi = 300)
#plt.show()

