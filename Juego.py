# Importa librer\'ia.
import numpy as np
import math as math
import statistics as stat
import matplotlib.pyplot as plt
from decimal import *

# Inicializa semilla.
np.random.seed(20190604)

# Dimensi\'ones.
# d : X, A, Q
# m : U, B, R
# n : V, D, T
d = 2
m = d
n = d

# Turnos.
N = 6

# Matr\'iz identidad.
Id = np.mat(np.diag(np.array([1.00, 1.00])))

# Declara coeficientes de la din\'amica.
A = np.mat(np.diag(np.array([1.07, 1.03])))
B = np.mat(np.diag(np.array([0.02, 0.80])))
D = np.mat(np.diag(np.array([0.05, 0.75])))

# Declara coeficientes del pago.
Q = np.mat( [[1.02,0.10],[0.10,1.00]] ) # np.mat(np.diag(np.array([1.02, 1.00])))
R = np.mat(np.diag(np.array([0.03, 0.50]))) # [0.00, 0.20]
T = np.mat(np.diag(np.array([1.50, 1.80]))) # [2.45, 2.90]

# Declara coeficientes del ruido.
# Sustituir par\'ametros por los comentarios para
# tener la Simulaci\'on 4.
# Multiplicar M * (-1) para la Simulaci\'on 5.
M = Id # np.mat( [[2, 0.5], [-1.3, 0.8]] )
Sig = Id # np.mat( [[.5, 0.01], [0.01, .5]] )

# Declara listas donde se guardar\'an coeficientes ajustados.
# Los subindices estar\'an desfazados para S, Btilde y
# Dtilde. Por ejemplo:
#   Tesis   |   Programa
#   S_{k+1} |   S[k]
S = []
Btilde = []
Dtilde = []

# Comienza calcular coeficientes ajustados.
for k in reversed( range(1,N+1) ):

    if k == N:
        S_k = Q

    else:
        S_k = (
            Q +
            ( np.transpose( A ) * S[0] * A ) -
            (
                ( np.transpose(A) * S[0] * B ) *
                np.linalg.inv(S[0]) *
                np.transpose( np.transpose(A) * S[0] * B )
            )
        )

    S.insert(0,S_k)

    BSB = np.transpose( B ) * S_k * B
    DSD = np.transpose( D ) * S_k * D

    Btilde.insert(0,BSB + R)
    Dtilde.insert(0,DSD - T)

# Quitar comentarios para revisar los coeficientes ajustados de
# la funci\'on de costo.
#print("Coeficientes ajustados ------------------------")
#for k in range(N):
#
#    print("-------Turno ",k,":")
#
#    print("S_", k+1, ":")
#    print(S[k])
#
#    print("Btilde_", k+1, ":")
#    print(Btilde[k])
#
#    print("Dtilde_", k+1, ":")
#    print(Dtilde[k])

# Coeficientes dentro de {F_k} para Y_k.
CoFY =[]
for k in range(N):
    # Fijo k-\'esimo F_k:
    suma = np.mat(np.diag(np.array([0.00, 0.00])))

    for j in range(k,N):

        # En el j-\'esimo sumando
        if j==k:
            CoFY_k = S[j] * M

        else:
            prod = Id
            for i in range(k+1,j):
                factor = (
                    np.transpose(A) * (
                        Id - ( S[i] * B * np.linalg.inv( Btilde[i] ) * np.transpose( B ) )
                    )
                )
                prod_anterior = prod
                prod = prod_anterior * factor
            CoFY_k = prod * S[j] * np.linalg.matrix_power(M, j + 1 - k )

        suma_anterior = suma
        suma = suma_anterior + CoFY_k

    CoFY.append(suma)
    #print(CoFY[k])

# Inversas para controles.
h12=[]
h21=[]
for k in range(N):
    h12_k = Btilde[k] - (
        ( np.transpose( B ) * S[k] * D ) *
        np.linalg.inv( Dtilde[k] ) *
        np.transpose( np.transpose( B ) * S[k] * D )
    )
    h21_k = Dtilde[k] - (
            np.transpose( np.transpose(B) * S[k] * D ) *
            np.linalg.inv( Btilde[k] ) *
            ( np.transpose(B) * S[k] * D )
    )

    h12.append( h12_k )
    h21.append( h21_k )

# Coeficientes en controles.
b0=[]
b1=[]
d0=[]
d1=[]
for k in range(N):
    # En desventaja.
    b0_k = np.linalg.inv( h12[k] ) * np.transpose( B ) * (
        ( S[k] * D * np.linalg.inv( Dtilde[k] ) * np.transpose( D ) ) - Id
    )
    d1_k = np.linalg.inv( h21[k] ) * np.transpose( D ) * (
        ( S[k] * B * np.linalg.inv( Btilde[k] ) * np.transpose( B ) ) - Id
    )

    # En ventaja.
    b1_k = (-1) * np.linalg.inv( Btilde[k] ) * np.transpose( B )
    d0_k = (-1) * np.linalg.inv( Dtilde[k] ) * np.transpose( D )

    b0.append( b0_k )
    b1.append( b1_k )
    d0.append( d0_k )
    d1.append( d1_k )

##############################################################
#----------------------------------------------- Simulaciones.

# Repeticiones.
NRep = 10000

# Pagos de simulaciones.
VecJ = []

# Probabilidades de ventaja
VecP = [ [0]*(N) for c in range(NRep) ]

VecXN0 = []
VecXN1 = []

# Matr\'iz de Estados Simulados.
MatX = [ [0]*(N+1) for c in range(NRep) ]

# Matr\'iz de Controles Simulados.
MatU = [ [0]*(N) for c in range(NRep) ]
MatV = [ [0]*(N) for c in range(NRep) ]

for c in range(NRep):
    # Comienza escenario.
    #print("****************************************")
    #print("Escenario ",c)

    # Se inicializan listas.
    X=[]
    Y=[]

    # <------------------------------------ Estado inicial.
    # En comentario se tiene los valores para la Simulaci\'on 2.
    X_0 = (0.80, 1.20) # np.random.multivariate_normal((0, 0), Id) #
    X.append(
        np.transpose(
            np.mat(
                X_0
            )
        )
    )

    # Costos en cero
    J = 0

    # Ruido: Markov Escondido.
    Z = np.random.multivariate_normal((0, 0), Id, N-1)

    # Evoluciona el juego.
    for k in range(N):

        # Quitar comentario para revisar estado del sistema en cada turno.
        #print("--------------------------------")
        #print("Turno ", k)
        #print("X[",k,"]:",X[k])

        MatX[c][k] = X[k]

        U_k = np.transpose( np.mat( [0,0] ) )
        V_k = np.transpose( np.mat( [0,0] ) )

        # Se crea el ruido del turno y el factor F_k.
        #
        # Quitar comentarios para revisar los valores de F_k en cada turno,
        # as\'i como el ruido.
        F_k = ( S[k] * A * X[k] )
        #print("F_k sin ruido:",F_k)
        if k == 0:
            Y_k = np.transpose(np.mat((0, 0)))
        else:
            Y_k = (M * Y[k - 1]) + (Sig * np.transpose(np.mat(Z[k - 1])))
            F_k_previo = F_k
            F_k = F_k_previo + ( CoFY[k] * Y[k-1] )
        #print("Ruido",Y_k)
        #print("F_k con ruido:", F_k)
        Y.append(Y_k)


        # Se lanza volado.
        # Cambiar p por 1-p en las lineas 247 y 257 para
        # tener la Simulaci\'on 3.
        p = - math.expm1( (-0.5) * math.sqrt(
            float( np.transpose( X[k] ) * Q * X[k] )
        ) )
        unif = np.random.random_sample()
        VecP[c][k] = p


        # Quitar comentario para revisar probabilidad de ventaja de I
        # en cada turno, en cada escenario.
        #print("p(X_",k,"):",p)


        # Se eligen controles.
        # xi = 1
        if unif <= p:
            #print("Xi = 1.")
            V_k = d1[k] * F_k
            U_k = b1[k] * ( F_k + ( S[k] * D * V_k ) )

        # xi = 0
        else:
            #print("Xi = 0.")
            U_k = b0[k] * F_k
            V_k = d0[k] * ( F_k + ( S[k] * B * U_k ) )

        MatU[c][k] = U_k
        MatV[c][k] = V_k

        # Quitar comentario para revisar controles en cada turno, en cada escenario.
        #print("U[", k, "]:", MatU[c][k])
        #print("V[", k, "]:", MatV[c][k])

        # Costo corriente.
        J_k = float(
            ( np.transpose( X[k] ) * Q * X[k] ) +
            ( np.transpose( U_k ) * R * U_k ) -
            ( np.transpose( V_k ) * T * V_k )
        )

        # Cotribuci\'on al pago
        J_previo = J
        J = J_previo + J_k
        #print("Pago del turno:", J_k)
        #print("Pago acumulado:", J)


        # Nuevo estado.
        X_k = (A * X[k]) + (B * U_k) + (D * V_k) + Y[k]
        X.append(X_k)

    #Costo final
    J_previo = J
    J = J_previo + float( np.transpose( X[N] ) * Q * X[N] )

    # Guarda estado y costo final.
    MatX[c][N] = X[N]
    VecJ.append( J )
    VecXN0.append( float(X[N][0]) )
    VecXN1.append( float(X[N][1]) )

    # Quitar comentarios para ver el final del juego en cada escenario.
    #print("--------------------------------")
    #print("Estado final")
    #print("X[",N,"]:",X[N])
    #print("Costo final:", float( np.transpose( X[N] ) * Q * X[N] ))
    #print("Pago del juego: ",VecJ[c])

    #print("--------------------")
    # Antes de seguir a siguiente escenario, se limpian listas.
    X[:] = []
    Y[:] = []

############################################################
#----------------------------------------------- Resultados.


print("****************************************")

print("N\'umero de escenarios:",NRep)

print("----------------------------------------")

Val_J = stat.mean(VecJ)
print("Pago Final promedio: ",Val_J)

# Quitar comentario para revisar otros estad\isticos.

#Med_J = stat.median(VecJ)
#print("Mediana: ",Med_J)


Desv_J = stat.stdev(VecJ)
print("Desviaci\'on: ",Desv_J)

#print("----------------------------------------")
#
#print("Estado Final promedio.")
#print( stat.mean(VecXN0) , stat.mean(VecXN1) )
#
#print("Desviaci\'on.")
#print( stat.stdev(VecXN0) , stat.stdev(VecXN1) )

print("----------------------------------------")

print("Probabilidad de ventaja media.")

VecPmedia = []

# Para revisar la probabilidad de ventaja media en cada
# escenario de la simulaci\'on.
# Por escenario.
for c in range(NRep):
    VecPmedia.append( stat.mean(VecP[c]) )
    #print("Escenario",c,":",VecPmedia[c])

# Total.
print("En todos los ecenarios:",stat.mean(VecPmedia))


#----------------------------------------------- Histograma:

print("----------------------------------------")
print("Creando Histograma.")

titulo = "Pago del juego."

# Descomentar para que el histograma quede centrado en 6 desviaciones alrededor
# de la media.
desv_hist = [Val_J]
for cc in range(1,3):
    desv_hist.append( Val_J + (cc * Desv_J) )
    #desv_hist.insert(0,Val_J - (cc * Desv_J))

caja_de_texto = '\n'.join(("Repeticiones: " + str(NRep),
                           "Media (Pago del Juego):",
                           "      $\mathbb{E}\ J_N^0(X_0)  = $" + str( round( Decimal( Val_J ), 2 ) ),
                           #"Mediana: " + str( round( Decimal( Med_J ), 2 ) ),
                           "Desviación: " + str( round( Decimal( Desv_J ), 2 ) )
                           )
                          )

fig_histo,ax_histo = plt.subplots()

chartBox = ax_histo.get_position()
ax_histo.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.85, chartBox.height])

ax_histo.hist(
    VecJ,
    density=True,
    bins=100,
    # Descomentar para que el histograma quede centrado en 6 desviaciones alrededor
    # de la media.
    #range= ( desv_hist[0] - Desv_J, desv_hist[4] + Desv_J ),
    range= ( 0, desv_hist[len(desv_hist)-1] + Desv_J ), #(min(VecJ),max(VecJ))
)

# Dibuja lineas de dispersi\'on y media
for cc in desv_hist:
    if cc == Val_J:
        linea = 'red'
    else:
        linea = 'k'
    plt.axvline(cc, color=linea, linestyle='dashed', linewidth=0.5 )



plt.title(titulo)
plt.ylabel("Frecuencia.")
plt.xlabel("Costo del juego $J_N^0(X_0)$")

ax_histo.text(1.05,0.5,
              caja_de_texto,
              transform=ax_histo.transAxes,
              bbox = dict( facecolor = 'whitesmoke',
                           edgecolor = 'silver',
                           #boxstyle = 'round,pad=0.5',
                           #alpha = 0.5,
                           ) )



plt.show()

print("Histograma de pago creado.")

#-----------------------------------------------  Para probabilidades:

# Listas donde se guardar\'an los componentes de los estados simulados
# y probabilidades, organizadas por turnos.
XComp0 = [ [0]*NRep for k in range(N+1) ]
XComp1 = [ [0]*NRep for k in range(N+1) ]
UComp0 = [ [0]*NRep for k in range(N) ]
UComp1 = [ [0]*NRep for k in range(N) ]
VComp0 = [ [0]*NRep for k in range(N) ]
VComp1 = [ [0]*NRep for k in range(N) ]
pComp = [ [0]*NRep for k in range(N) ]
pCompAlt = [ [0]*NRep for k in range(N-2) ]

for c in range(NRep):
    for k in range(N):
        XComp0[k][c] = float( MatX[c][k][0] )
        XComp1[k][c] = float( MatX[c][k][1] )

        UComp0[k][c] = float( MatU[c][k][0] )
        UComp1[k][c] = float( MatU[c][k][1] )

        VComp0[k][c] = float( MatV[c][k][0] )
        VComp1[k][c] = float( MatV[c][k][1] )

        pComp[k][c] = VecP[c][k]

        if k >= 2:
            pCompAlt[k-2][c] = VecP[c][k]
    XComp0[N][c] = float(MatX[c][N][0])
    XComp1[N][c] = float(MatX[c][N][1])

# Todos los turnos.
titulo = "Probabilidades de Ventaja."

fig_phisto,ax_phisto = plt.subplots()

chartBox = ax_phisto.get_position()
ax_phisto.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.85, chartBox.height])

for k in range(N):
    ax_phisto.hist(pComp[k],
                   density=True,
                   alpha = 0.3,
                   bins = 50,
                   label = 'Turno '+str(k),
                   range= (0,1),
                  )

plt.legend(bbox_to_anchor=(1.05, 0.5),
           loc='center left',
           borderaxespad=0.,
           frameon=True,
           facecolor = 'whitesmoke',
           )
plt.title(titulo)
plt.ylabel("Frecuencia relativa.")
plt.xlabel("$p(X_k)$")
plt.show()

# Sin los primeros 2.
titulo = "Probabilidades de Ventaja (Turnos 2 en adelante)."

fig_phistoAlt,ax_phistoAlt = plt.subplots()

chartBox = ax_phistoAlt.get_position()
ax_phistoAlt.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.85, chartBox.height])

for k in range(N-2):
    ax_phistoAlt.hist(pCompAlt[k],
                   density=True,
                   alpha = 0.3,
                   bins = 50,
                   label = 'Turno '+str(k+2),
                   range= (0,1),
                  )

plt.legend(bbox_to_anchor=(1.05, 0.5),
           loc='center left',
           borderaxespad=0.,
           frameon=True,
           facecolor = 'whitesmoke',
           )
plt.title(titulo)
plt.ylabel("Frecuencia relativa.")
plt.xlabel("$p(X_k)$")
plt.show()


print("Histograma de probabilidades creado.")

##########################################################################
#----------------------------------------------- Puntos.

#----------------------------------------------- Para estados del sistema:

print("----------------------------------------")
print("Creando Gr\'afica de puntos.")


turno = []
U_turno =[]
V_turno =[]
etiqueta = []
for k in range(N):
    turno.append( ( XComp0[k], XComp1[k] ) )

    U_turno.append( ( UComp0[k], UComp1[k] ) )

    V_turno.append( ( VComp0[k], VComp1[k] ) )

    etiqueta.append( "Turno "+str(k) )
turno.append( ( XComp0[N], XComp1[N] ) )
etiqueta.append( "Turno "+str(N) )

# Paleta de colores determinada. Quitar comentarios
# para tener colores aleatorios. Sirve para N > 6, por ejeplo.
colores =  ("black","green","purple","blue","brown","red","orange")
#hexa = ('0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F')
#colores = ["#"+''.join([np.random.choice( hexa ) for j in range(6)]) for i in range(N+1)]
#colores = sorted(colores)

U_colores = colores
V_colores = colores

U_etiqueta = etiqueta
V_etiqueta = etiqueta


U_turno_ind = U_turno
U_colores_ind = U_colores
U_etiqueta_ind = U_etiqueta

V_turno_ind = V_turno
V_colores_ind = V_colores
V_etiqueta_ind = V_etiqueta

turno_ind = turno
colores_ind = colores
etiqueta_ind = etiqueta


# Gr\'aficas por turno.
print("---------- Estados del sistema:")
print("Individuales.")
nn = 1
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for turno_ind, colores_ind, etiqueta_ind in zip(turno_ind, colores_ind, etiqueta_ind):
    ax = fig.add_subplot( 3, 3, nn )
    x, y = turno_ind
    ax.scatter(x, y,
               alpha=0.6,
               c=colores_ind,
               edgecolors='none',
               s=30,
               )
    ax.title.set_text('Turno '+str(nn-1))

    ax.set_xlabel('1er Componete de $X_'+str(nn-1)+'$' )
    ax.set_ylabel('2da Componete de $X_'+str(nn-1)+'$' )

    nn += 1


plt.suptitle('Estado del Sistema.')
plt.show()

# Gr\'afica conjunta.
print("Conjuntas.")
fig_total = plt.figure()
ax_total = fig_total.add_subplot(1, 1, 1)  # , axisbg="1.0")

chartBox = ax_total.get_position()
ax_total.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.85, chartBox.height])

for turno, colores, etiqueta in zip(turno, colores, etiqueta):

    x_total, y_total = turno
    ax_total.scatter(x_total, y_total,
                     alpha=0.4,
                     c=colores,
                     edgecolors='none',
                     s=30,
                     label=etiqueta)

plt.title('Evolución del juego')
plt.legend(bbox_to_anchor=(1.05, 0.5),
           loc='center left',
           borderaxespad=0.,
           frameon=True,
           facecolor = 'whitesmoke',
           )

plt.xlabel('1er Componete de $X$')
plt.ylabel('2da Componete de $X$')

plt.show()

#####################################################################
#----------------------------------------------- Para controles de I:

# Gr\'aficas por turno.
print("---------- Estrategias de I:")
print("Individuales.")
nn = 0
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for U_turno_ind, U_colores_ind, U_etiqueta_ind in zip(U_turno_ind, U_colores_ind, U_etiqueta_ind):
    ax = fig.add_subplot(2, 3, nn+1)  # , axisbg="1.0")
    x, y = U_turno_ind
    ax.scatter(x, y,
               alpha=0.6,
               c=U_colores_ind,
               edgecolors='none',
               s=30)
    ax.title.set_text('Turno '+str(nn))

    ax.set_xlabel('1er Componete de $U_' + str(nn) + '$')
    ax.set_ylabel('2da Componete de $U_' + str(nn) + '$')

    nn += 1

plt.suptitle('Estrategias del Jugador I.')
plt.show()

# Gr\'afica conjunta.
print("Conjuntas.")
fig_total = plt.figure()
ax_total = fig_total.add_subplot(1, 1, 1)  # , axisbg="1.0")

chartBox = ax_total.get_position()
ax_total.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.85, chartBox.height])

for U_turno, U_colores, U_etiqueta in zip(U_turno, U_colores, U_etiqueta):

    x_total, y_total = U_turno
    ax_total.scatter(x_total, y_total,
                     alpha=0.4,
                     c=U_colores,
                     edgecolors='none',
                     s=30,
                     label=U_etiqueta)

plt.title('Estrategias de I.')
plt.legend(bbox_to_anchor=(1.05, 0.5),
           loc='center left',
           borderaxespad=0.,
           frameon=True,
           facecolor = 'whitesmoke',
           )


plt.xlabel('1er Componete de $U$')
plt.ylabel('2da Componete de $U$')

plt.show()

#-----------------------------------------------  Para controles de II:

# Gr\'aficas por turno.
print("---------- Estrategias de II:")
print("Individuales.")
nn = 0
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for V_turno_ind, V_colores_ind, V_etiqueta_ind in \
        zip(V_turno_ind, V_colores_ind, V_etiqueta_ind):
    ax = fig.add_subplot(2, 3, nn+1)  # , axisbg="1.0")
    x, y = V_turno_ind
    ax.scatter(x, y,
               alpha=0.6,
               c=V_colores_ind,
               edgecolors='none',
               s=30)
    ax.title.set_text('Turno '+str(nn))

    ax.set_xlabel('1er Componete de $V_' + str(nn) + '$')
    ax.set_ylabel('2da Componete de $V_' + str(nn) + '$')

    nn += 1

plt.suptitle('Estrategias del Jugador II.')
plt.show()

# Gr\'afica conjunta.
print("Conjuntas.")
fig_total = plt.figure()
ax_total = fig_total.add_subplot(1, 1, 1)  # , axisbg="1.0")

chartBox = ax_total.get_position()
ax_total.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.85, chartBox.height])

for V_turno, V_colores, V_etiqueta in zip(V_turno, V_colores, V_etiqueta):

    x_total, y_total = V_turno
    ax_total.scatter(x_total, y_total,
                     alpha=0.4,
                     c=V_colores,
                     edgecolors='none',
                     s=30,
                     label=V_etiqueta)

plt.title('Estrategias de II.')
plt.legend(bbox_to_anchor=(1.05, 0.5),
           loc='center left',
           borderaxespad=0.,
           frameon=True,
           facecolor = 'whitesmoke',
           )


plt.xlabel('1er Componete de $V$')
plt.ylabel('2da Componete de $V$')

plt.show()

print("Gr\'afica de puntos creada.")


##############################################################
#----------------------------------------------- C\'odigo para
# revisar partes del programa:

# Para revisar estados por turno:
#print("----------------------------------------")
#print("Escenario ",0,", turno ",1)
#print(MatX[0][1])
#print("Escenario ",2,", turno ",3)
#print(MatX[2][3])
#print("Escenario ",4,", turno ",5)
#print(MatX[4][5])

# Para revisar componentes de Estados por turno:
#print("----------------------------------------")
#print("Componentes NO son flotantes")
#print("Escenario ",0,", turno ",1," componente ",0)
#print(MatX[0][1][0])
#print("Escenario ",2,", turno ",3," componente ",1)
#print(MatX[2][3][1])
#print("Escenario ",4,", turno ",5," componente ",0)
#print(MatX[4][5][0])

# Para revisar estados de un escenario:
#print("----------------------------------------")
#print("Estados")
#print("Escenario ",0,", todos los turnos.")
#print(MatX[0])
#print("Escenario ",2,", todos los turnos.")
#print(MatX[2])
#print("Escenario ",4,", todos los turnos.")
#print(MatX[4])
