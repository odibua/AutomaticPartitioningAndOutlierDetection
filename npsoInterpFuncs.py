#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 06:23:14 2017

@author: odibua
"""
import numpy as np
import copy
from getStates import getState1
from getStates import getState2

#Function that interpolates two solutions for cluster and outliers particle and that
#updates the global best based on the fitness evaluatin function    
def npsoClustersAndOutliersInterpFunc(optimType,JState,KState,xMin,xMax,globalBestState,data,distFunc,splitCentroids,
                                      deleteCentroids,assignData,calcCentroids,cntEmptyClusters,findOutliers,removeOutliersFromClusters,fitnessFunction,fitnessEvaluationFunction):

    nData=np.shape(data)[0]; nFeatures=np.shape(data)[1]

    #Get interpolated states based on splitting/deleting the J State and update
    #the global best state according to the fitnessEvaluationFunction
    state1 = getState1(data,JState,KState,globalBestState,xMin,xMax,distFunc,splitCentroids,
                                      deleteCentroids,assignData,calcCentroids,cntEmptyClusters,findOutliers,removeOutliersFromClusters,fitnessFunction); 
    globalBestState = fitnessEvaluationFunction(optimType,[],state1,globalBestState) 

    state2 = getState2(data,JState,KState,globalBestState,xMin,xMax,distFunc,splitCentroids,
                                      deleteCentroids,assignData,calcCentroids,cntEmptyClusters,findOutliers,removeOutliersFromClusters,fitnessFunction);
    globalBestState = fitnessEvaluationFunction(optimType,[],state2,globalBestState)
   
    #Get interpolated states based on splitting/deleting the J State and update
    #the global best state according to the fitnessEvaluationFunction 
    state1 = getState1(data,KState,JState,globalBestState,xMin,xMax,distFunc,splitCentroids,
                                      deleteCentroids,assignData,calcCentroids,cntEmptyClusters,findOutliers,removeOutliersFromClusters,fitnessFunction); 
    globalBestState = fitnessEvaluationFunction(optimType,[],state1,globalBestState) 
    
    state2 = getState2(data,KState,JState,globalBestState,xMin,xMax,distFunc,splitCentroids,
                                      deleteCentroids,assignData,calcCentroids,cntEmptyClusters,findOutliers,removeOutliersFromClusters,fitnessFunction);
    globalBestState = fitnessEvaluationFunction(optimType,[],state2,globalBestState); 
     
    return globalBestState  
