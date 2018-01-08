#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:12:28 2017

@author: odibua
"""

import numpy as np
import random
from psoClusterAndOutlierParticle import psoParticleClustersAndOutliersFunc
from psoMethods import PSO
from clusterFitnessFunctions import variableKpFitness#variableKpFitCSMeasFunc
from clusterFitnessFunctions import clusterOutlierFitness
import matplotlib.pyplot as plt
import copy
from npsoInterpFuncs import npsoClustersAndOutliersInterpFunc
from npsoClusterAndOutlierParticle import npsoParticleClustersAndOutliersFunc
import sys

testString =['shuttle','wdbc','wine','iris','glass','ArtificialOutliers','ArtificialNoOutliers']
methodString = ['PSO','GCPSO','NPSO']

#Choose data for testing algorithm and method of optimization
testUse=sys.argv[1]; methodUse=sys.argv[2]; nTrials=int(sys.argv[3]);
#testUse=testString[3]; methodUse=methodString[0]
 
if testUse not in testString:
    sys.exit('Input test string does not exist. Type in different string');
elif (methodUse not in methodString):
    sys.exit('PSO method does not exist. Type in different string')

##############################################################################################
if (testUse=='shuttle' or testUse=='wdbc' or testUse=='wine' or testUse=='iris' or testUse=='glass'):
    all_data = np.loadtxt(open("./testData/"+testUse+"_data.csv","r"),
        delimiter=",",
        skiprows=0,
        dtype=np.float64
        )    

if (testUse=='shuttle'):

    # load class labels from column 1
    yData = all_data[:,0]
    # conversion of the class labels to integer-type array
    yData = yData.astype(np.int64, copy=False)
    
    # load the 13 features
    XData = all_data[:,1:]
    ##Downsample shuttle
    idxShuttle = np.where(yData!=4)[0]
    yData = yData[idxShuttle];
    XData = XData[idxShuttle,0:]
    
    idx = np.random.choice(len(yData),1000,replace=False)
    XData = XData[idx,0:]#Normalized[idx,0:];
    y = yData[idx] 
    shapeData = np.shape(XData);
    nData = shapeData[0]; nFeatures = shapeData[1];
elif (testUse=='wdbc'):
    # load class labels from column 1
    yData = all_data[:,0]
    # conversion of the class labels to integer-type array
    yData = yData.astype(np.int64, copy=False)
    
    # load the 13 features
    XData = all_data[:,1:]
    ##Downsample malignant
    idxBenign = np.where(yData==0)[0]
    idxMalign = np.where(yData==1)[0]
    idxMalign = idxMalign[np.random.choice(len(idxMalign),10)];
    yData = yData[np.concatenate((idxBenign,idxMalign)).astype(int)];
    XData = XData[np.concatenate((idxBenign,idxMalign)).astype(int),0:];
    y=yData
    shapeData = np.shape(XData);
    nData = shapeData[0]; nFeatures = shapeData[1];
elif (testUse=='wine' or testUse=='glass'):
    # load class labels from column 1
    yData = all_data[:,0]
    # conversion of the class labels to integer-type array
    yData = yData.astype(np.int64, copy=False)   
    y = yData
    XData = all_data[:,1:];
    shapeData = np.shape(XData);
    nData = shapeData[0]; nFeatures = shapeData[1];
elif (testUse=='iris'): 
    # load class labels from column 1
    yData = all_data[:,0]
    # conversion of the class labels to integer-type array
    yData = yData.astype(np.int64, copy=False)
    y = yData
    XData = all_data[:,1:];
    shapeData = np.shape(XData);
    nData = shapeData[0]; nFeatures = shapeData[1];
elif (testUse=='ArtificialOutliers'):
    cov=np.array([[0.05, 0],[0,0.05]])
    XData = np.vstack([np.random.multivariate_normal([0,0],cov,10),np.random.multivariate_normal([0.9,0.9],cov/100,1),np.random.multivariate_normal([2,2],cov,10),np.random.multivariate_normal([2.9,2.9],cov/100,1),np.random.multivariate_normal([4,4],cov,10),np.random.multivariate_normal([4.9,4.9],cov/100,1),np.random.multivariate_normal([8,8],cov,10)])
    shapeData = np.shape(XData);
    nData = shapeData[0]; nFeatures = shapeData[1];
elif (testUse=='ArtificialNoOutliers'):
    cov=np.array([[0.05, 0],[0,0.05]])
    XData = np.vstack([np.random.multivariate_normal([0,0],cov,10),np.random.multivariate_normal([2,2],cov,10),np.random.multivariate_normal([4,4],cov,10),np.random.multivariate_normal([8,8],cov,10)])
    shapeData = np.shape(XData);
    nData = shapeData[0]; nFeatures = shapeData[1];
    
featureMin = np.min(XData,axis=0);
featureMax = np.max(XData,axis=0);
XDataNormalized = copy.deepcopy(XData);
for j in range(nData):
    for k in range(nFeatures):
        XDataNormalized[j][k] = (XData[j][k]-featureMin[k])/(featureMax[k]-featureMin[k]);
featureMin = np.min(XDataNormalized,axis=0);
featureMax = np.max(XDataNormalized,axis=0);

#Define X as data features
X = XDataNormalized

#Defines maximum and minimum number of clusters, number of outliers
#and feature components
nData=np.shape(X)[0]
KpMin=1.0;
KpMax=6.0;
lMin=0.0;
lMax=0
numEvalState=8;
featureMin=np.min(X,axis=0);
featureMax=np.max(X,axis=0);
nFeatures=len(featureMin)
posMin=[KpMin];
posMax=[KpMax];
posMin.append(lMin);
posMax.append(lMax)
posMin.append(featureMin);
posMax.append(featureMax);
velMin=[-(KpMax-KpMin)/2.0];
velMax=[(KpMax-KpMin)/2.0];
velMin.append(-(lMax-lMin)/2.0);
velMax.append((lMax-lMin)/2.0);
velMin.append(posMin[2]/2.0); 
velMax.append(posMax[2]/2.0);

#Define parameters that govern behavior of particle swarm
#optimization
numParticles = 40;
neighborSize = 4#np.ceil(numParticles/5).astype(int);
w = 0.75;
tol = 1e-3;
numIters = 200
kappa = 0.5;
mult = 1;
c1 = 2.0
c2 = c1*mult;
constrict = 1.0
optimType = 'Min';
sigma = np.linalg.norm(featureMax-featureMin)/2.0;

random.seed(0)
#Define list that will store results of optimization for each trial
bestGlobalFitness={}
bestGlobalKp={} 
bestGlobalCentroids={}
bestGlobalNumEmptyClusters={}
bestGloball={}
bestGlobalOutliers={}
weights={}
weightsOutliers={}

#Number of trials
#nTrials=1;
eps=0.05; #Hyper-parameter that determines extent to which objective must 
            #change in order for solution that has different number of outliers 
            #to be accepted.            
wCS=1.0; wF=0.0; wKp=0.0 #Define weights to be used for each component of feature 
evaluationFunc = clusterOutlierFitness(eps); #Defines evaluation function with hyper-parameter eps
variableKpFitCSMeasFunc=variableKpFitness(wCS,wF,wKp); #Define fitness objective function with weights of terms
 
if (methodUse=='PSO' or methodUse=='GCPSO'):
    clusterOutlierParticles = psoParticleClustersAndOutliersFunc(X,sigma);
    if (methodUse=='GCPSO'):
        c1 = 2.05; c2 = c1;
elif (methodUse=='NPSO'):
    clusterOutlierParticles = npsoParticleClustersAndOutliersFunc(X,sigma); #Define npso particle class using the data and sigma
pso=PSO(); 
for j in range(nTrials): 
    print('Trial',j+1)  
    #Execute optimization
    if (methodUse=='PSO'):
        swarm,evalState = pso.executePSO(c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,clusterOutlierParticles,optimType,numEvalState,variableKpFitCSMeasFunc,evaluationFunc)
    elif (methodUse=='GCPSO'):
        swarm,evalState = pso.executeGCPSO(constrict,c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,clusterOutlierParticles,optimType,numEvalState,variableKpFitCSMeasFunc,evaluationFunc)
    elif (methodUse=='NPSO'): 
        swarm,evalState = pso.executeNPSO(neighborSize,w,posMin,posMax,velMin,velMax,numIters,numParticles,clusterOutlierParticles,optimType,numEvalState,variableKpFitCSMeasFunc,evaluationFunc,npsoClustersAndOutliersInterpFunc)
 
    (bestGlobalFitness[str(j+1)],bestGlobalKp[str(j+1)],bestGlobalCentroids[str(j+1)],
    bestGlobalNumEmptyClusters[str(j+1)],bestGloball[str(j+1)],bestGlobalOutliers[str(j+1)],
    weights[str(j+1)],weightsOutliers[str(j+1)]) = evalState



