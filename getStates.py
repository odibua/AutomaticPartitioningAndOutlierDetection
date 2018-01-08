#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 01:22:53 2017

@author: Ohi Dibua
Get the two interpolated states from two particles states based 
on interpolations proposed in (SITE)
"""
import numpy as np
import copy

def getState1(data,JPState,KPState,globalBestState,xMin,xMax,distFunc,splitCentroids,
                                      deleteCentroids,assignData,calcCentroids,cntEmptyClusters,findOutliers,removeOutliersFromClusters,fitnessFunction):#(JState,KState,globalBestState,nData,xMin,xMax):  
    JState = copy.deepcopy(JPState); KState = copy.deepcopy(KPState)
    nData = np.shape(data)[0]; nFeatures = np.shape(data)[1]
    #Obtain values neccessary for interpolation
    KpMin = xMin[0]; lMin = xMin[1]; featureMin = xMin[2]; KpMax = xMax[0]; lMax = xMax[1]; featureMax = xMax[2]   
    fXJ = JState[0]; KpJ = JState[1];  centroidsJ = JState[2]; numEmptyClustersJ = JState[3]; lJ = JState[4]; outliersJ = JState[5]; 
    fXK = KState[0]; KpK = KState[1];  centroidsK = KState[2]; numEmptyClustersK = KState[3]; lK = KState[4]; outliersK = KState[5]; 
    fXBest = globalBestState[0]; KpBest = globalBestState[1];  centroidsBest = globalBestState[2]; numEmptyClustersBest = globalBestState[3]; lBest = globalBestState[4]; outliersBest = globalBestState[5];          
    
    #Interpolate new number of centroids and number of outliers           
    Kp1 = 0.5*(KpJ**2-KpBest**2-KpK**2)*fXJ*fXK / ((KpJ-KpK)*fXBest+(KpK-KpBest)*fXJ
        + (KpBest-KpJ)*fXK)         
    l1 = 0.5*(lJ**2-lBest**2-lK**2)*fXJ*fXK / ((lJ-lK)*fXBest+(lK-lBest)*fXJ
        + (lBest-lJ)*fXK)    
    #Make sure that the number of centroids and outliers are feasible
    if not np.isnan(Kp1):
        Kp1 = int(min(max(Kp1,KpMin),KpMax));
    else:
        Kp1 = copy.deepcopy(KpJ)
    if not np.isnan(l1):
        l1 = int(min(max(l1,lMin),lMax)); 
    else:
        l1 = copy.deepcopy(lJ)
    
    #Split and delete centroid of JState until it matches interpolated number of clusters 
    weights1 = np.zeros((int(Kp1),nData)); 
    weightsOutliers1 = np.array([0]*nData)
    weights1 = assignData(KpJ,centroidsJ,data);  
    centroids1Star = []
    if (KpJ<Kp1):
        outputSplitCentroids = splitCentroids(Kp1,weights1,centroidsJ,[],data,assignData,cntEmptyClusters);
        weights1 = outputSplitCentroids[1]; centroids1Star = outputSplitCentroids[2]; numEmptyClusters1 = outputSplitCentroids[3];
    elif (KpJ>Kp1): 
        outputDeleteCentroids = deleteCentroids(Kp1,weights1,centroidsJ,[],assignData,cntEmptyClusters,data); 
        weights1 = outputDeleteCentroids[1]; centroids1Star = outputDeleteCentroids[2]; numEmptyClusters1 = outputDeleteCentroids[3];   
    else:    
        centroids1Star = copy.deepcopy(centroidsJ)                 

    #Interpolate the centroid using the centroids in the KState and globalBestState
    #that are closest to the centroid in the JState. 
    centroids1 = [];
    for k in range(int(Kp1)):    
        if (Kp1>1):
            centBest = centroidsBest[np.argmin(distFunc(centroids1Star[k],np.array(centroidsBest)))];
        else:
            centBest = centroidsBest[np.argmin(distFunc(np.array(centroids1Star),np.array(centroidsBest)))];
            
        centK = centroidsK[np.argmin(distFunc(centroids1Star[k],np.array(centroidsK)))];
        centroids1.append(np.array([0.5*(centroids1Star[k][j]**2-centBest[j]**2-centK[j]**2)*fXJ*fXK/((centroids1Star[k][j]-centK[j])*fXBest+(centK[j]-centBest[j])*fXJ
        +(centBest[j]-centroids1Star[k][j])*fXK) for j in range(nFeatures)]))  
    
    #Store centroids, assign data to new centroids, and remove outliers from data
    centroids1 = copy.deepcopy(centroids1);      
    weights1 = assignData(Kp1,centroids1,data);          
    weightsShape1 = np.shape(weights1); 
    outliers1 = np.reshape(findOutliers(centroids1,weights1,distFunc,data),(nData*Kp1,1)); 
    outliers1 = np.flipud(np.argsort(outliers1,axis=0)); 
    outlierData = []   
    outlierResult = removeOutliersFromClusters(outliers1,outlierData,weights1,weightsOutliers1,l1);
    outlierData1 = outlierResult[0]; weights1 = outlierResult[1]; weightsOutliers1 = outlierResult[2];
      
    #Calculate the new centroids after outliers have been removed            
    centroids1 = calcCentroids(Kp1,centroids1,weights1,data); 
    numEmptyClusters1 = cntEmptyClusters(weights1);    
    fitness1 = fitnessFunction(distFunc,Kp1,centroids1,weights1,data)
    state1 = (fitness1,Kp1,centroids1,numEmptyClusters1,l1,outliers1,weights1,weightsOutliers1)
    del JPState; del KPState; del globalBestState;
    return state1

def getState2(data,JPState,KPState,globalBestState,xMin,xMax,distFunc,splitCentroids,
                                      deleteCentroids,assignData,calcCentroids,cntEmptyClusters,findOutliers,removeOutliersFromClusters,fitnessFunction):#(JState,KState,globalBestState,nData,xMin,xMax):  
    JState = copy.deepcopy(JPState); KState = copy.deepcopy(KPState)
    nData = np.shape(data)[0]; nFeatures = np.shape(data)[1]
    #Obtain values neccessary for interpolation
    KpMin = xMin[0]; lMin = xMin[1]; featureMin = xMin[2]; KpMax = xMax[0]; lMax = xMax[1]; featureMax = xMax[2]    
    fXJ = JState[0]; KpJ = JState[1];  centroidsJ = JState[2]; numEmptyClustersJ = JState[3]; lJ = JState[4]; outliersJ = JState[5]; 
    fXK = KState[0]; KpK = KState[1];  centroidsK = KState[2]; numEmptyClustersK = KState[3]; lK = KState[4]; outliersK = KState[5]; 
    fXBest=globalBestState[0]; KpBest = globalBestState[1];  centroidsBest = globalBestState[2]; numEmptyClustersBest = globalBestState[3]; lBest = globalBestState[4]; outliersBest = globalBestState[5];          
    
    #Interpolate new number of centroids and number of outliers            
    Kp2 = 0.5*((KpJ**2-KpK**2)*fXBest + (KpK**2-KpBest**2)*fXJ + (KpBest**2-KpJ**2)*fXK) / ((KpJ-KpK)*fXBest + (KpK-KpBest)*fXJ + (KpBest-KpJ)*fXK)      
    l2 = 0.5*((lJ**2-lK**2)*fXBest + (lK**2-lBest**2)*fXJ + (lBest**2-lJ**2)*fXK) / ((lJ-lK)*fXBest + (lK-lBest)*fXJ + (lBest-lJ)*fXK)  
    #Make sure that the number of centroids and outliers are feasible
    if not np.isnan(Kp2):
        Kp2 = int(min(max(Kp2,KpMin),KpMax));
    else:
        Kp2 = copy.deepcopy(KpJ)
    if not np.isnan(l2):
        l2 = int(min(max(l2,lMin),lMax)); 
    else:
        l2 = copy.deepcopy(lJ)
        
    #Split and delete centroid of JState until it matches interpolated number of clusters     
    weights2 = np.zeros((Kp2,nData));
    weightsOutliers2 = np.array([0]*nData)
    centroids2Star = []
    weights2 = assignData(KpJ,centroidsJ,data);     
    if (KpJ<Kp2):
        outputSplitCentroids = splitCentroids(Kp2,weights2,centroidsJ,[],data,assignData,cntEmptyClusters);
        weights2 = outputSplitCentroids[1]; centroids2Star = outputSplitCentroids[2]; numEmptyClusters2 = outputSplitCentroids[3];
    elif (KpJ>Kp2):  
        outputDeleteCentroids = deleteCentroids(Kp2,weights2,centroidsJ,[],assignData,cntEmptyClusters,data);  
        weights2 = outputDeleteCentroids[1]; centroids2Star = outputDeleteCentroids[2]; numEmptyClusters2 = outputDeleteCentroids[3];                        
    else:    
        centroids2Star = copy.deepcopy(centroidsJ)  

    #Interpolate the centroid using the centroids in the KState and globalBestState
    #that are closest to the centroid in the JState. 
    centroids2=[];
    for k in range(int(Kp2)):   
        if (Kp2>1):
            centBest = centroidsBest[np.argmin(distFunc(centroids2Star[k],np.array(centroidsBest)))];
        else:
            centBest = centroidsBest[np.argmin(distFunc(np.array(centroids2Star),np.array(centroidsBest)))];
        centK = centroidsK[np.argmin(distFunc(centroids2Star[k],np.array(centroidsK)))];
        centroids2.append(np.array([0.5*((centroids2Star[k][j]**2-centK[j]**2)*fXBest + (centK[j]**2-centBest[j]**2)*fXJ + (centBest[j]**2-centroids2Star[k][j]**2)*fXK)
            / ((centroids2Star[k][j]-centK[j])*fXBest + (centK[j]-centBest[j])*fXJ + (centBest[j]-centroids2Star[k][j])*fXK) for j in range(nFeatures)]))    

    #Store centroids, assign data to new centroids, and remove outliers from data
    centroids2 = copy.deepcopy(centroids2);     
    weights2 = assignData(Kp2,centroids2,data);     
    weightsShape2 = np.shape(weights2); 
    outliers2 = np.reshape(findOutliers(centroids2,weights2,distFunc,data),(nData*Kp2,1)); 
    outliers2 = np.flipud(np.argsort(outliers2,axis=0)); 
    outlierData2 = []   
    outlierResult = removeOutliersFromClusters(outliers2,outlierData2,weights2,weightsOutliers2,l2);
    outlierData2 = outlierResult[0]; weights2 = outlierResult[1]; weightsOutliers2 = outlierResult[2]; 
      
    #Calculate the new centroids after outliers have been removed                
    centroids2 = calcCentroids(Kp2,centroids2,weights2,data); 
    numEmptyClusters2 = cntEmptyClusters(weights2);    
    fitness2 = fitnessFunction(distFunc,Kp2,centroids2,weights2,data)
    state2 = (fitness2,Kp2,centroids2,numEmptyClusters2,l2,outliers2,weights2,weightsOutliers2)
    return state2  
 