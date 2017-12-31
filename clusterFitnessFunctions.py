#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:45:19 2017

@author: Ohi Dibua
File that contains the fitness function, in conjunction with the fitness evaluation
function for variable number of clusters and outliers
"""

import numpy as np
import copy
import logging as logger
import collections
import time

#Fitness function for clusters with variable number of clusters.
#The initial function takes in weights for the CS Measure, 
#KMeans Objective, and number of clusters
def variableKpFitness(wCS,wF,wKp):
    def variableKpFitCSMeasFunc(distFunc,Kp,centroids,weights,data): 
        numeratorCS=0;
        denominatorCS=0;
        activeCentroid=0
        eps=1;
        f=0; nData=np.shape(data)[0]
        minList=[];
        activeCentroid=0;
        for k in range(int(Kp)):
            datInCentroid=np.where(weights[k]==1)[0];
            nDatInCentroid=len(datInCentroid)
    
            numeratorTemp=0; numeratorTemp2=0; distList=[];
            if (nDatInCentroid>0):
                f0=0; datMat=np.zeros((nDatInCentroid,nDatInCentroid));#distDict=collections.defaultdict(dict);
                for j in range(nDatInCentroid):
                    if (j==0):
                        vectDiff=data[datInCentroid][0:]-data[datInCentroid[j]][0:];
                        distCalc=distFunc(data[datInCentroid[j]][0:],data[datInCentroid][0:])
                        datMat[j,0:]=copy.deepcopy(distCalc);
                        datMat[j:,j]=copy.deepcopy(datMat[j,j:]); 
                        vectDiff[1:]=vectDiff[1:]/distCalc[1:,None]
                        
                    f0=f0+weights[k][datInCentroid[j]]*distFunc(centroids[k],data[datInCentroid[j]][0:]);
    
                for j in range(nDatInCentroid-1):
                    temp=(vectDiff[j+1:,0:].T*datMat[0,j+1:]).T
    
                    datMat[j+1,0:j+1]=datMat[0:j+1,j+1]
                    datMat[j+1,j+1:]=distFunc(temp,vectDiff[j+1,0:]*datMat[j+1,0])#(np.linalg.norm(temp-vectDiff[j+1,0:]*datMat[j+1,0],axis=1))#abs(datMat[0,1:]-datMat[j+1,0])  
                numeratorTemp=sum(np.max(datMat,axis=1)); 
    
                numeratorCS=numeratorCS+numeratorTemp/nDatInCentroid;
    
                if (Kp>1):
                    denominatorCS=denominatorCS+np.min(distFunc(centroids[k],np.delete(centroids,k,axis=0)));
                else:
                    denominatorCS=eps 
                f=f+f0/nDatInCentroid;
    
        csMeasure=numeratorCS/(denominatorCS);
        return wCS*csMeasure+wF*f+wKp*np.log(Kp) 
    return variableKpFitCSMeasFunc

########################################################################################################
#Cluster evaluation function that takes in, as an argument, a hyper-parameter, eps, that is the required
#change in the objective function when outliers are removed for the number of outliers to to change
def clusterOutlierFitness(eps):           
    def EvaluationFunc(optimType,currentState,localBestState,globalBestState=None):
        if (globalBestState==None):
            currentFitness=currentState[0]
            currentKp=currentState[1]
            currentCentroids=currentState[2]
            currentNumEmptyClusters=currentState[3]
            currentBestl=currentState[4]
            currentBestOutliers=currentState[5]
            currentWeights=currentState[6]
            currentWeightsOutliers=currentState[7];
                                          
            localBestFitness=localBestState[0]
            localBestKp=localBestState[1]
            localBestCentroid=localBestState[2]
            localBestNumEmptyClusters=localBestState[3] 
            localBestl=localBestState[4]
            localBestOutliers=localBestState[5]
            localBestWeights=localBestState[6]
            localBestWeightsOutliers=localBestState[7];
            
            newLocalBool=0;
            if (currentNumEmptyClusters < localBestNumEmptyClusters):
                newLocalBool=1;
            if (currentNumEmptyClusters==localBestNumEmptyClusters):
                if (currentBestl<localBestl and np.abs((currentFitness-localBestFitness))<eps):
                    newLocalBool=1;
                if (currentBestl>localBestl): 
                    if (optimType.lower()=='max'):
                        if ((currentFitness-localBestFitness) > eps):
                            newGlobalBool=1;
                    elif (optimType.lower()=='min'):
                        if (currentFitness-localBestFitness<-eps):
                            newGlobalBool=1;
                elif (currentBestl==localBestl):                           
                    if (optimType.lower()=='max'):
                        if (currentFitness > localBestFitness):
                            newLocalBool=1;
                    elif (optimType.lower()=='min'):
                        if (currentFitness < localBestFitness):
                            newLocalBool=1;
            if (newLocalBool==1):
                localBestState=(currentFitness,currentKp,currentCentroids,currentNumEmptyClusters,currentBestl,currentBestOutliers,currentWeights,currentWeightsOutliers)            
            return localBestState 
        
        elif (globalBestState is not None):
            globalBestFitness=globalBestState[0]
            globalBestKp=globalBestState[1]
            globalBestCentroid=globalBestState[2]
            globalBestNumEmptyClusters=globalBestState[3] 
            globalBestl=globalBestState[4]
            globalBestOutliers=globalBestState[5]
            globalBestWeights=globalBestState[6]
            globalBestWeightsOutliers=globalBestState[7];
            
            localBestFitness=localBestState[0]
            localBestKp=localBestState[1]
            localBestCentroid=localBestState[2]
            localBestNumEmptyClusters=localBestState[3] 
            localBestl=localBestState[4]
            localBestOutliers=localBestState[5]
            localBestWeights=localBestState[6]
            localBestWeightsOutliers=localBestState[7];

            newGlobalBool=0;
            if (localBestNumEmptyClusters < globalBestNumEmptyClusters or globalBestNumEmptyClusters==float("inf")):
                newGlobalBool=1; 
            if (localBestNumEmptyClusters==globalBestNumEmptyClusters):
                if (localBestl<globalBestl and np.abs((globalBestFitness-localBestFitness))<eps):
                    newGlobalBool=1;
                if (localBestl>globalBestl): 
                    if (optimType.lower()=='max'):
                        if ((localBestFitness-globalBestFitness) > eps):
                            newGlobalBool=1; 
                    elif (optimType.lower()=='min'):
                        if (localBestFitness-globalBestFitness<-eps):
                            newGlobalBool=1;
                elif (localBestl==globalBestl):  
                    if (optimType.lower()=='max'):
                        if (localBestFitness > globalBestFitness):
                            newGlobalBool=1;
                    elif (optimType.lower()=='min'):
                        if (localBestFitness < globalBestFitness):
                            newGlobalBool=1;
                            
            if (newGlobalBool==1):
                print("Change global best solution Kp is",localBestKp, " l is",localBestl)
                globalBestState=(localBestFitness,localBestKp,localBestCentroid,localBestNumEmptyClusters,localBestl,localBestOutliers,localBestWeights,localBestWeightsOutliers)
                        
            return globalBestState 
    return EvaluationFunc