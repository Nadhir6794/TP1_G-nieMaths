#BIETTI Olivier
#OFFKIR Nadhir

import numpy as np
import copy
import time as t
import matplotlib.pyplot as plt


#Géneré M inversible
# MT = np.transpose(M)
# A = MTM


## Verification Symétrique / Définie / Positive
#def Vérification

## Décomposition de Cholesky : Question 1

def Cholesky(A):
    n, n = np.shape(A)
    L = np.zeros((n,n))
    for i in range(n):
        s1=0
        for j in range(i):
            s1=s1+L[i,j]**2
        D =A[i,i]-s1
        L[i,i]=np.sqrt(D)
        for j in range(i+1,n):
            s2=0
            for k in range(i):
                s2=s2+L[i,k]*L[j,k]
            L[j,i]=(A[j,i]-s2)/L[i,i]
            
    LT = np.transpose(L)
    return L, LT




## Résolution de systèmes à l'aide de la décomposition de Cholesky : Question 1

def ResolutionCholesky(L,LT,B):
    n, m = np.shape(L)   
    x = np.zeros(n)
    y = np.zeros(n)
    S = 0
    #Ly=b
    for i in range(0,n):
        for k in range(0,i):
            S += L[i,k]*y[k]
        y[i] = B[i] - S
        S = 0
    #Ux=y
    for i in range(n-1,-1,-1):
        for k in range(n-1,i,-1):
            S += LT[i,k]*x[k]
        x[i] = (y[i] - S)/LT[i,i]
        S = 0
    return x

def ResolCholesky(A,B):
    n, n = np.shape(A)
    B = B.reshape(n,1)
    L,LT = Cholesky(A)
    return ResolutionCholesky(L,LT,B)




#Q3

# Gauss

def ReductionGauss(Au):
    n, m = np.shape(Au)
    for i in range(0, n-1):
        if Au[i,i] == 0 :
            Au[i,:] = Au[i+1]
        else :
            for j in range(i+1,n):
                g = Au[j,i] / Au[i,i]
                Au[j,:] = Au[j,:] - g * Au[i,:]
    return Au
#Q2
print('Q2')
def ResolutionSystTriSup(Tu):
    n, m = np.shape(Tu)
    x = np.zeros(n)
    x[n-1] = Tu[n-1,m-1] / Tu[n-1,n-1]

    for i in range(n-2, -1, -1):
        x[i] = Tu[i,m-1]
        for j in range(i+1, n):
            x[i]= x[i] - Tu[i,j] * x[j]
        x[i] =  x[i] / Tu[i,i]   
    return x
#Q3
print('Q3')
def Gauss(A, B):
    n, m = np.shape(A)
    B = B.reshape(n,1)
    R = np.column_stack((A,B))
    return ResolutionSystTriSup(ReductionGauss(R))
'''Gauss(A,B)'''

def linalgCholesky(A):
    L = np.linalg.cholesky(A)
    LT = np.transpose(L)
    return L,LT

def ResolCholesky2(A,B):
    n, m = np.shape(A)
    B = B.reshape(n,1)
    L,LT = linalgCholesky(A)
    return ResolutionCholesky(L,LT,B)

def linalgSolve(A,B):
    n,m = np.shape(A)
    B=B.reshape(n,1)
    x = np.linalg.solve(A,B)
    
    return x


# Graphes 

pas = 10
maxi = 202
mini = 1


def graphiques():
    #DONNEES
    #temps
    taille=[]
    Y_gauss=[]
    Y_cholesky=[]
    Y_cholesky2=[]
    Y_linalg=[]

    #erreur
    taille_=[]
    e_gauss=[]
    e_cholesky=[]
    e_cholesky2=[]
    e_linalg=[]
    

    #CALCUL
    for i in range(mini, maxi, pas):

        print(i,"\n")
        A1 = np.random.rand(i, i)
        A2 = np.transpose(A1)
        A = np.dot(A2,A1)
        B = np.random.rand(1, i)
        C = np.copy(A)

        

        #Gauss
        t1_gauss = t.time()
        x1 = Gauss(A,B)
        print('Solutions :', x1)
        t2_gauss = t.time()
        y1 = np.linalg.norm(np.dot(A,x1)-np.ravel(B))
        print(y1)
        

        #Cholesky
        t1_cholesky = t.time()
        x2 = ResolCholesky(C,B)
        print('Solutions :', x2)
        t2_cholesky = t.time()
        y2 = np.linalg.norm(np.dot(C,x2)-np.ravel(B))
        print(y2)
        
        #Cholesky2
        t1_cholesky2 = t.time()
        x3 = ResolCholesky2(C,B)
        print('Solutions :', x3)
        t2_cholesky2 = t.time()
        y3 = np.linalg.norm(np.dot(C,x3)-np.ravel(B))
        print(y3)
        
        #linalg.solve
        t1_linalg = t.time()
        x4 = linalgSolve(A,B)
        print('Solutions :', x4)
        t2_linalg = t.time()
        y4 = np.linalg.norm(np.dot(C,x4)-np.ravel(B))
        print(y4)
        


        #enregistrement des résultats
        #temps
        taille.append(i)
        Y_gauss.append(t2_gauss-t1_gauss)
        Y_cholesky.append(t2_cholesky-t1_cholesky)
        Y_cholesky2.append(t2_cholesky2-t1_cholesky2)
        Y_linalg.append(t2_linalg-t1_linalg)

        
        #erreur
        taille_.append(i)
        e_gauss.append(y1)
        e_cholesky.append(y2)
        e_cholesky2.append(y3)
        e_linalg.append(y4)

    #GRAPHS

    #temps
    plt.plot(taille,Y_gauss,'-b',label = 'Gauss')   #Gauss
    plt.plot(taille,Y_cholesky,'-g',label = 'Cholesky')   #Cholesky
    plt.plot(taille,Y_cholesky2,'-r',label = 'linalg.cholesky') #Cholesky 2
    plt.plot(taille,Y_linalg,'-y',label = 'linalg.solve') #Linalg
    plt.xlabel("Taille de la matrice - n")
    plt.ylabel("Temps d'éxecution - T(s)")
    plt.title("Temps en fonction de la taille n")
    plt.legend()
    plt.show()


    #erreur
    plt.semilogy(taille_,e_gauss,'-b',label = 'Gauss')   #Gauss
    plt.semilogy(taille_,e_cholesky,'-g',label = 'Cholesky')   #Cholesky
    plt.plot(taille,e_cholesky2,'-r',label = 'linalg.cholesky') #Cholesky 2
    plt.plot(taille,e_linalg,'-y',label = 'linalg.solve') #Linalg
    plt.xlabel("Taille de la matrice - n")
    plt.ylabel("Erreur - ||AX = B||")
    plt.title("Erreur en fonction de la taille n")
    plt.legend()
    plt.show()
    




graphiques()