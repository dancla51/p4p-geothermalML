import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from sympy import *
import matplotlib.pyplot as plt
import time as tm
import random as rand
start_time = tm.time()


def TikhonovNEWSolveRowReduce(filename, k=5, Lambda1=1, Lambda2=0, Lambda3=0, suppress=False):

    k = 3


    # In[150]:


    # Establish number of wells
    #print(filename)
    n = pd.read_excel(filename, sheet_name='n').iloc[0,0]


    # In[151]:


    # Establish number of timepoints
    T = pd.read_excel(filename, sheet_name='T').iloc[0,0]


    # In[152]:


    zReal = pd.read_excel(filename, sheet_name="OnOff").to_numpy().T
    zReal[pd.isna(zReal)] = 0

    zTotal = int(sum(cell for row in zReal  for cell in row ))


    # In[153]:


    # Establish binary matrix for wells being measured or not
    dReal = pd.read_excel(filename, sheet_name="Measured").to_numpy().T
    dReal[pd.isna(zReal)] = 0
    
    dTotal = sum(sum(dReal))


    # In[154]:


    # Establish vector of actual mass flows
    mReal = pd.read_excel(filename, sheet_name="MassFlows").to_numpy().T
    mReal[pd.isna(mReal)] = 0

    # In[155]:


    # Establish matrix / vector of total mass flow
    MReal = pd.read_excel(filename, sheet_name="TotalMassFlow").to_numpy().T
    MReal[pd.isna(MReal)] = 0


    # Compile the y vector (col. vec.)
    y = np.zeros([T*(n+1)+n*(T-1)+n*(T-2), 1])
    for j in range(T):
        # fill in m values
        for i in range(n):
            y[T*i+j, 0] = mReal[i,j] * np.sqrt(Lambda1) * dReal[i,j]  # Seems to have no effect on result but will affect RMSE score
        # fill in M values
        y[T*n+j, 0] = MReal[0,j]


    # Compile the A1 matrix
    A1 = np.zeros([T*(n+1)+n*(T-1)+n*(T-2), n*T])
    for i in range(n):
        for j in range(T):
            # fill in d values
            A1[T*i+j, T*i+j] = np.sqrt(Lambda1) * dReal[i,j]
            # fill in z values
            A1[T*n+j, T*i+j] = zReal[i,j]
            # fill in forward difference formula to approximate the first order derivative
            if j > 0: # Ignore first row to not add a BC
                A1[T*(n+1)+i*(T-1)+j-1, i*(T)+j-1] = -1 * np.sqrt(Lambda2)
                A1[T*(n+1)+i*(T-1)+j-1, i*(T)+j] = 1 * np.sqrt(Lambda2)
            # fill in second order central difference formula to approximate the second order derivative
            if j > 1: # Ignore first two rows to not add BCs
                A1[T*(n+1)+n*(T-1)+i*(T-2)+j-2, i*(T)+j-2] = 1 * np.sqrt(Lambda3)
                A1[T*(n+1)+n*(T-1)+i*(T-2)+j-2, i*(T)+j-1] = -2 * np.sqrt(Lambda3)
                A1[T*(n+1)+n*(T-1)+i*(T-2)+j-2, i*(T)+j] = 1 * np.sqrt(Lambda3)
                
    
    mask=[]
    for row in range(len(y)):
        if y[row,0]==0 and not np.any(A1[row,:]):
            #print('row '+str(row)+' is all 0s')
            mask.append(row)
            

    y = np.delete(y, mask, 0)
    A1 = np.delete(A1, mask, 0)


    

    # Compile the A2 matrix
    # Use t's uniformly spaced 1 apart
    ts = np.array(list(range(T)))+1
    A2 = np.zeros([n*T, n*(k+1)])
    for block in range(n):
        for r in range(T):
            for c in range(k+1):
                A2[block*T + r, block*(k+1) + c] = ts[r]**c


    # Calculate final A matrix
    A = np.matmul(A1,A2)



    res, _, _, _ = lstsq(A, y)
    res = res.reshape(n, k+1)

    # Coerce results into desired format for printing
    t = symbols('t')
    wExpr = Matrix(np.zeros(n))
    for i in range(n):
        for deg in range(k+1):
            #print(res[i,deg]*t**deg)
            wExpr[i] = wExpr[i] + res[i,deg]*t**deg        

    if not suppress:

        # Plot results from all wells
        plt.figure(figsize=(10,6))
        colours=['b', 'y', 'r', 'g', 'k']
        for i in range(n):
            plt.plot(range(T),mReal[i,:][:], colours[i]+'-', lw=2, label="Real flow for well "+str(i))
            # Plot Time-Dependently
            toplot = []
            for time in range(T):
                resplot = wExpr[i].subs(t, time)
                toplot.append(resplot*zReal[i,time])
                #print(time, toplot)
            plt.plot(range(T),toplot, colours[i]+'-.', lw=1.5, label="Estimated flow for well "+str(i))
            # Plot TFT points
            monthplot = []
            tftplot = []
            for month,element in enumerate(dReal[i,:]):
                if element==1:
                    monthplot.append(month)
                    tftplot.append(mReal[i,month])
            plt.plot(monthplot, tftplot, colours[i]+'*', markersize=10, label="TFT measurements for well "+str(i))

        plt.title("Estimated and real mass flows over time by well, with NEW solving method")
        plt.xlabel("Time (month)")
        plt.ylabel("Mass Flow (kg/s)")
        plt.legend()




        # Plot total flow vs found total flow
        totalFlowFound = np.zeros((T,1))
        times = range(T)
        for time in range(T):
            tff = [wExpr[i].subs(t, time)*zReal[i,time] for i in range(n)]
            #print(t, tff)
            totalFlowFound[time] = sum(tff)

        plt.plot(list(Matrix(times).T), list(MReal[0]), 'g-', lw=2, label="Real Total Mass Flow")
        plt.plot(times,totalFlowFound, 'b-.', lw=1.5, label="Estimated Total Mass Flow")

        plt.title("Estimated and real total mass flows over time, with NEW solving method")
        plt.xlabel("Time (month)")
        plt.ylabel("Total Mass Flow (kg/s)")
        plt.legend()


    # Calculate average out of sample squared error (MSE). Then sqrt
    SE = 0
    for i in range(n):
        for t2 in range(T):
            real = mReal[i, t2]
            estimated = zReal[i,t2]*wExpr[i].subs(t, t2)
            #print(estimated)
            RSS = (real - estimated)**2
            SE = SE + RSS

    MSE = SE / n / T
    RMSE = sqrt(MSE)

    return(RMSE, wExpr)




