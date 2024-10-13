#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sympy import *
import matplotlib.pyplot as plt
import random as rand


def TikhonovOLDSolve(filename, k=5, TikLambda1=0, TikLambda2=0, suppress=False):
    """
    pTest = number of testing points to use to evaluate RMSE
    k = degree of polynomial
    ridgeCoef = parameter weighting the ridge regression term (0 = no ridge)
    """
    
    


    # Establish symbols
    n, T, i, j, t, lda = symbols('n T i j t \lambda')       
    # n wells & T timepoints
    # k = highest degree of w polynomial





    # Establish number of wells
    n = len(pd.read_excel(filename, sheet_name="OnOff").to_numpy().T)
    n





    # Establish number of timepoints
    T = len(pd.read_excel(filename, sheet_name="OnOff").to_numpy().T[0])
    T





    # Establish binary matrix for wells being on or off
    z = MatrixSymbol('z', n, T)      # Row 0 = well 0   , Column 3 = Timepoint 3
    #pprint(z.as_explicit())
    zReal = Matrix(pd.read_excel(filename, sheet_name="OnOff").to_numpy().T)
    zReal




    # Establish binary matrix for wells being measured or not
    d = MatrixSymbol('d', n, T)      # Row 0 = well 0   , Column 3 = Timepoint 3
    #pprint(d.as_explicit())
    dReal = Matrix(pd.read_excel(filename, sheet_name="Measured").to_numpy().T)
    dTotal = sum(dReal)
    dReal





    # Establish matrix of actual mass flows
    m = MatrixSymbol('m', n, T)  
    #pprint(m.as_explicit())
    mReal = Matrix(pd.read_excel(filename, sheet_name="MassFlows").to_numpy().T)
    mReal





    # Establish matrix / vector of total mass flow
    M = MatrixSymbol('M', 1, T)
    #pprint(M.as_explicit())
    MReal = Matrix(pd.read_excel(filename, sheet_name="TotalMassFlow").to_numpy().T)
    MReal





    # Set w as an expression based on t
    wMat = MatrixSymbol('w', n, k+1)
    #pprint(wMat.as_explicit())





    # Instantaneous value of wi is wMati * [1, t, t^2, t^3 etc]
    tList = []
    for power in range(k+1):
        tList.append(t**power)
    ts = Matrix(tList)
    wInstant = Matrix([wMat[i,:].as_explicit().dot(ts) for i in range(n)])
    wInstant[0,0]





    S1 = Sum((M[0,t] - Sum(z[i, t]*wInstant[i,0], (i, 0, n-1)))**2 , (t, 0, T-1))
    S1





    S2 = Sum(Sum(d[i,t]*((m[i,t]-wInstant[i,0]))**2 , (t, 0, T-1)), (i, 0, n-1))
    S2





    Tikhonov = Sum(Sum(wInstant[i,0].diff(t)**2 , (t, 0, T-1)), (i, 0, n-1))
    


    Tikhonov2 = Sum(Sum(wInstant[i,0].diff(t).diff(t)**2 , (t, 0, T-1)), (i, 0, n-1))



    I = S1 + lda*S2 + TikLambda1*Tikhonov +TikLambda2*Tikhonov2
    I





    optLda = T*n**2/dTotal
    optLda





    Inac = I.subs({M: MReal, z: zReal, d: dReal, m: mReal, lda: optLda}).doit()
    Inac





    Inac.diff(wMat[0,0])





    DiffA = np.zeros([(k+1)*n,(k+1)*n])
    Diffb = np.zeros((k+1)*n)
    # Fill in matrix with values for Aw=b  
    for i in range(n):
        for j in range(k+1):
            # v = vertical index
            v = i*(k+1)+j
            dwij = Poly(Inac.diff(wMat[i,j]).doit())
            Diffb[v] = -1*dwij.coeffs()[-1]
            for i2 in range(n):
                for j2 in range(k+1):
                    # h = horizontal index
                    h = i2*(k+1)+j2
                    #print(v, h)
                    DiffA[v,h] = dwij.coeff_monomial(wMat[i2, j2])


    # Then solve for w
    DiffMatrixA = Matrix(DiffA)
    DiffMatrixb = Matrix(Diffb)




    results = DiffMatrixA.inv()*DiffMatrixb
    results





    wExpr = results.reshape(n, k+1)*ts
    
    
    
    if not suppress:
        pprint(wExpr)
        
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
            for month,element in enumerate(dTrain[i,:]):
                if element==1:
                    monthplot.append(month)
                    tftplot.append(mReal[i,month])
            plt.plot(monthplot, tftplot, colours[i]+'*', markersize=10, label="TFT measurements for well "+str(i))

        plt.title("Estimated and real mass flows over time by well")
        plt.xlabel("Time (month)")
        plt.ylabel("Mass Flow (kg/s)")
        plt.legend()
        plt.show()

        # Plot total flow vs found total flow
        totalFlowFound = np.zeros((T,1))
        times = range(T)
        for time in range(T):
            tff = [wExpr[i].subs(t, time)*zReal[i,time] for i in range(n)]
            #print(t, tff)
            totalFlowFound[time] = sum(tff)

        plt.plot(list(Matrix(times).T), list(MReal), 'g-', lw=2, label="Real Total Mass Flow")
        plt.plot(times,totalFlowFound, 'b-.', lw=1.5, label="Estimated Total Mass Flow")

        plt.title("Estimated and real total mass flows over time")
        plt.xlabel("Time (month)")
        plt.ylabel("Total Mass Flow (kg/s)")
        plt.legend()
        plt.show()


    return(wExpr)