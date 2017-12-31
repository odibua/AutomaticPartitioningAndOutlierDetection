#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:07:31 2017

@author: Ohi Dibua

This file contains a class that defines a particle used in the problem of finding the optimal
partitioning (number of clusters) of a data set and the number of outliers that this data set uses.
It combines algorithms proposed from [1] and [2]. The cs measure of optimal partitoning proposed 
is used [3], and the method of choosing the best particle is inspired by [4]. The details of
this are discussed in the clusterFitness file which contains the fitnessFunction and fitnessEvaluation 
function for this particle.

The optimization problem is essential of the form:
    
    min w1*CS(K,u,l,y) + w2*f(K,u,l) + w3*log(K) 
    K,u
    
    where K is number of centroids, l is the number of total outliers,
    u are the centroids, and y is the data being partitioned.
     
    subject to the minimization of the number of infeasible centroids and
    number of outlers, l. 
    
The particle class initializes the following arguments:
    optimType = optimization type ('Max' or 'Min')
    w: intertial constant (Recommended range is between 0.5 and 1.0)
    c1: cognitive constant (Recommended range is between 1.00 and 2.05)
    c2: social constant (Recommended range is between 1.00 and 2.05)
    posMin: minimum values of elements of x
    posMax: maximum values of elements of x
    velMin: minimum values of the veleocities ascribed to different elements of x
    velMax: maximum values of the velocities ascribed to different elements of x
    fitnessFunction = f(x) 
    fitnessEvaluationFunction: func(optimType,currentState,localBestState,globalBestState=None)
    Function that takes in two states that define proposed solutions and outputs 
    the solution that most meets the objective. If passing in the global best state,
    and empty list is passed in for localBestState. For typical PSO the states are state = (fitness,x)

[1] Cura. "A particle swarm optimizatin approach to clustering" Expert Systems With Applications. 2012
[2] Chawla et al. "A unified approach to clustering and outlier detection" Proceedings of the 2013 SIAM International Conference on Data Mining
[3] Chou, C.H.  et al."A new cluster validity measure and its application to image compression." Pattern Anal. Appl. 7 (2), 205– 220.
[4] Behesti et al., "Non-parametric particle swarm optimization for global optimization",Applied Soft. Computing. 2015
"""
import numpy as np
import copy
import time
import sys,os
import logging as logger
import sys, os
import time

#Define outer particle function that takes the data and sigma (hyper parameter of gaussian kernel) as an input
def psoParticleClustersAndOutliersFunc(data,sigma):  
    #Particle class that finds optimal partitioning and number of outliers using standard PSO                 
    class psoParticleClusterAndOutliers():
        #Initialize parameters important for propagating particles and evaluating their fitness with respect
        def __init__(self,optimType,c1,c2,w,posMin,posMax,velMin,velMax,fitnessFunction,fitnessEvaluationFunction):
            #Initialize parameters relevant to updating the velocity and position of particles
            self.w=w;             
            self.localBestFit=float("inf");
            self.c1=c1; self.c2=c2;                
            self.KpMin=posMin[0]; self.KpMax=posMax[0];
            self.lMin=posMin[1];  self.lMax=posMax[1];
            self.velMinKp=velMin[0]; self.velMaxKp=velMax[0];
            self.velMinl=velMin[1]; self.velMaxl=velMax[1];
            self.velMinCentroids=velMin[2]; self.velMaxCentroids=velMin[2];
            self.featureMin=posMin[2]; self.featureMax=posMax[2];
            
            #Initialize parameters relevant to describing the data and the assignment of data to centroids
            self.data=copy.deepcopy(data);
            shpData=np.shape(data);
            self.nData=shpData[0];
            self.nFeatures=len(self.featureMin);
            self.numEmptyClusters=0;        
            
            #Initiaize strings and functions related to evaluating fitness
            self.optimType=optimType.lower(); 
            self.fitnessFunction=fitnessFunction;
            self.fitnessEvaluationFunction=fitnessEvaluationFunction;
    
            #Initialize function that calculates distance between data
            self.sigma=sigma;
            #self.distFunc=self.euclidDist;#self.gaussDist;  
            self.distFunc=self.gaussDist; 
               
            #Initialize number of centroies, number of outliers, centroid positions,
            #weight array that will indicate which points are outliers and all 
            #corresponding velocities                   
            r1=np.random.uniform();
            self.Kp = copy.deepcopy(round((self.KpMax-self.KpMin)*r1+self.KpMin));
            self.velKp = 0;            
            r1=np.random.uniform();
            self.l = copy.deepcopy(round((self.lMax-self.lMin)*r1+self.lMin));
            self.vell = 0;            
            self.centroids = []
            self.velCentroids = []
            for k in range(int(self.Kp)):
                self.centroids.append(np.array([0.0]*len(self.featureMin)));
                self.velCentroids.append(np.array([0.0]*len(self.featureMin)));
                for l in range(self.nFeatures):
                    r1=np.random.uniform();
                    self.centroids[k][l]=(self.featureMax[l]-self.featureMin[l])*r1+self.featureMin[l];              
            self.weightsOutliers=copy.deepcopy(np.array([0]*self.nData)) 
            
            #Assign data to initial centroids using the weights array, and remove the outliers, 
            #calculated according to some distance measure, from the data.
            self.weights = self.assignData(self.Kp,self.centroids,self.data);            
            self.weightsShape = np.shape(self.weights); 
            self.centroids=self.calcCentroids(self.Kp,self.centroids,self.weights,self.data);
            self.outliers=np.reshape(self.findOutliers(self.centroids,self.weights,self.data,self.l),(self.weightsShape[0]*self.weightsShape[1],1)); 
            self.outliers=np.flip(np.argsort(self.outliers,axis=0),axis=0); #Store index of sorted outliers          
            self.outlierData = [] 
            self.removeOutliersFromClusters();
            
            #Recalculate centroids with outliers removed, count the number of empy centroids,
            #and evaluate particle fitnes
            self.centroids=self.calcCentroids(self.Kp,self.centroids,self.weights,self.data);                   
            self.cntEmptyClusters();                
            self.currentFitness=self.fitnessFunction(self.distFunc,self.Kp,self.centroids,self.weights,self.data);#self.constKpFitFunc();

            #Store initial state as personal best local state
            self.localBestFitness=copy.deepcopy(self.currentFitness);
            self.localBestKp=copy.deepcopy(self.Kp);
            self.localBestCentroid=copy.deepcopy(self.centroids);
            self.localBestNumEmptyClusters=copy.deepcopy(self.numEmptyClusters);
            self.localBestl=copy.deepcopy(self.l);
            self.localBestOutliers=copy.deepcopy(self.outlierData);
            self.localBestWeights=copy.deepcopy(self.weights)
            self.localBestWeightsOutliers=copy.deepcopy(self.weightsOutliers)

        #Update velocity and position of particle
        def updateVelocityandPosition(self,globalBestState,t,constrict=None):  
            #Obtain global best components of particle solution
            globalBestKp=globalBestState[1]
            globalBestCentroid=globalBestState[2]
            globalBestl =  globalBestState[4]
            
            #Obtain coefficients used in particle propagation for particular time step
            if not np.isscalar(self.w): 
                w=self.w[t];
            else: w=self.w;          
            if not np.isscalar(self.c1): 
                c1=self.c1[t];
            else: c1=self.c1;       
            if not np.isscalar(self.c2): 
                c2=self.c2[t];
            else: c2=self.c2;
            if (constrict is not None):
                phi=c1+c2;
                constrFactor = 2.0/np.abs(2-phi-np.sqrt(phi**2-4*phi));
            else:
                constrFactor = 1.0
                
            #Update the velocity and position of number of clusters, and change number
            #of centroids to match updated number of clusters. Assign data to clusters.
            r1=np.random.uniform(0,1);   r2=np.random.uniform(0,1);
            cognitiveVel = r1*c1*(self.localBestKp-self.Kp); 
            socialVel = r2*c2*(globalBestKp-self.Kp)
            self.velKp = constrFactor*(w*self.velKp + cognitiveVel + socialVel)
            if (constrict is None):
                self.velKp = min(max(self.velKp,self.velMinKp),self.velMaxKp)
            self.Kp = min(max(np.round(self.Kp + self.velKp),self.KpMin),self.KpMax);
            if self.Kp > len(self.centroids):
                self.splitCentroids(); 
            elif self.Kp <len(self.centroids):
                self.deleteCentroids();
            else:                
                self.weights = self.assignData(self.Kp,self.centroids,self.data);

            #Remove outliers from the data assigned to centroid
            self.weightsShape=np.shape(self.weights);
            self.outliers=np.reshape(self.findOutliers(self.centroids,self.weights,self.data,self.l),(self.weightsShape[0]*self.weightsShape[1],1)); 
            self.outliers=np.flip(np.argsort(self.outliers,axis=0),axis=0); 
            self.outlierData = [] 
            self.removeOutliersFromClusters();
            
            #Update the velocity and position of number of outliers, and change number
            #of outliers to match updated number of outliers
            r1=np.random.uniform(0,1);   r2=np.random.uniform(0,1);
            cognitiveVel = r1*c1*(self.localBestl-self.l); 
            socialVel = r2*c2*(globalBestl-self.l)
            self.vell = constrFactor*(w*self.vell + cognitiveVel + socialVel)
            if (constrict is None):
                self.vell = min(max(self.vell,self.velMinl),self.velMaxl);
            self.l0=copy.deepcopy(self.l); 
            self.l = min(max(np.round(self.l + self.vell),self.lMin),self.lMax);
            if (self.l>self.l0):
                self.removeOutliersFromClusters();
            elif (self.l<self.l0):
                self.addOutliersToCluster();
            
            #Pick centroid that will be randomly updated
            k = np.random.randint(0,self.Kp);
            
            #Find centroid in personal best and swarm best solution that is closest to the centroid being updated
            globalBestCentroid = globalBestCentroid[np.argmin(self.distFunc(self.centroids[k],np.array(globalBestCentroid)))];
            localBestCentroid = self.localBestCentroid[np.argmin(self.distFunc(self.centroids[k],np.array(self.localBestCentroid)))];
            
            #Update the kth centroid velocity and position according to nearest global best 
            #and personal best centroids. Assign data to these centroids, and re-calcuate centroids
            for j in range(self.nFeatures):          
                r1=np.random.uniform(0,1);   r2=np.random.uniform(0,1);
                cognitiveVel = r1*c1*(localBestCentroid[j]-self.centroids[k][j]);
                socialVel = r2*c2*(globalBestCentroid[j]-self.centroids[k][j]);
                self.velCentroids[k][j] = constrFactor*(w*self.velCentroids[k][j] + cognitiveVel + socialVel);
                if (constrict is None):
                    self.velCentroids[k][j] = min(max(self.velCentroids[k][j],self.velMinCentroids[j]),self.velMaxCentroids[j]);
                self.centroids[k][j] = self.centroids[k][j] + self.velCentroids[k][j]
            self.weights = self.assignData(self.Kp,self.centroids,self.data);        
            self.calcCentroids(self.Kp,self.centroids,self.weights,self.data)
            
            #Remove outliers from the data assigned to centroids, calculate new centroids, and count number of clusters
            self.weightsShape=np.shape(self.weights); 
            self.outliers=np.reshape(self.findOutliers(self.centroids,self.weights,self.data,self.l),(self.weightsShape[0]*self.weightsShape[1],1)); 
            self.outliers=np.flip(np.argsort(self.outliers,axis=0),axis=0); 
            self.outlierData = [] 
            self.removeOutliersFromClusters();      
            self.centroids=self.calcCentroids(self.Kp,self.centroids,self.weights,self.data); 
            self.cntEmptyClusters(); 
        
        #Update personal best solution
        def updateLocalBest(self):
            #Calculate the fitness of the current solution, and store current state
            self.currentFitness=self.fitnessFunction(self.distFunc,self.Kp,self.centroids,self.weights,self.data);#self.constKpFitFunc();              
            currentState = (self.currentFitness,self.Kp,self.centroids,self.numEmptyClusters,self.l,self.outlierData,self.weights,self.weightsOutliers)
            
            #Store state of personal best solution and update personal best solution based on the 
            #fitness evaluation function
            localBestState = (self.localBestFitness,self.localBestKp,self.localBestCentroid,self.localBestNumEmptyClusters,self.localBestl,self.localBestOutliers,self.localBestWeights,self.localBestWeightsOutliers)
            localBestState = self.fitnessEvaluationFunction(self.optimType,currentState,localBestState);
            self.localBestFitness=localBestState[0]
            self.localBestKp=localBestState[1]
            self.localBestCentroid=localBestState[2]
            self.localBestNumEmptyClusters=localBestState[3]  
            self.localBestl=localBestState[4]
            self.localBestOutliers=localBestState[5]
            self.localBestWeights=localBestState[6]
            self.localBestWeightsOutliers=localBestState[7]; 
 
            return localBestState 
        
        #Assign data to centroids and update weights based on this        
        def assignData(self,K,centroids,data):  
            nData=np.shape(data)[0];
            #Store distances between data points and each centroids
            for k in range(int(K)):        
                d=self.distFunc(centroids[k],data);         
                if (k==0):
                    distArr = np.array(d);
                    w = np.array([0]*nData)
                else:
                    distArr = np.vstack([distArr,d]);
                    w = np.vstack([w,np.array([0]*nData)])
            
            #Assign data to centroids based minimal distance
            #and update weights
            if (self.Kp>1):
                centroidAssignments=np.argmin(distArr,axis=0); 
            else:
                centroidAssignments=np.array([0]*nData)
            if (self.Kp>1):
                for j in range(nData):
                    w[centroidAssignments[j]][j]=1;   
            else:
                w = np.ones((1,nData)).astype(int)
            return w;
        
        #Calculate centroids based on where data is assigned
        def calcCentroids(self,K,centroids,weights,data):
            for k in range(int(K)):
                if (np.sum(weights[k])>0):
                    centroids[k]=np.mean(data[np.where(weights[k]==1)[0]],axis=0)
            return centroids
        
        #Calculate euclidean distance
        def euclidDist(self,xi,xq):
            shape=np.shape(xi)
            shape2=np.shape(xq)
            if (xi.ndim>1 or xq.ndim>1):
                norm=np.linalg.norm(xi-xq,axis=1);
            else:
                norm=np.linalg.norm(xi-xq)  
            return norm
        
        #Calculate distance based on gaussian kernel
        def gaussDist(self,xi,xq):
            shape=np.shape(xi)
            shape2=np.shape(xq)
            if (xi.ndim>1 or xq.ndim>1):
                norm=np.linalg.norm(xi-xq,axis=1);
            else:
                norm=np.linalg.norm(xi-xq)  
                
            sigma=self.sigma
            return 2*(1-np.exp(-0.5*(norm**2)/sigma**2))
        
        #Count number of centroids with no assigned data
        def cntEmptyClusters(self):
            empCluster=np.where(np.sum(self.weights,axis=1)<=1)[0];
            self.numEmptyClusters=len(empCluster)           
        
        #Split largest centroid in particle until it matches the 
        #updated number of centroids
        def splitCentroids(self):
            nCurr=len(self.centroids);
            targetKp=copy.deepcopy(self.Kp)

            while(nCurr<targetKp):
                maxClust = np.argmax(np.sum(self.weights,axis=1)); #Find cluster with most data assigned to it
                datInMaxClust = self.data[np.where(self.weights[maxClust,0:]==1)[0],0:];
                datUpperBound = np.max(datInMaxClust,axis=0);
                datLowerBound = np.min(datInMaxClust,axis=0);
                newCent = np.array([0.0]*self.nFeatures);
                for j in range(self.nFeatures):
                    r1=np.random.uniform();
                    newCent[j] = (datUpperBound[j]-datLowerBound[j])*r1+datLowerBound[j];
                self.centroids.append(newCent);
                nCurr=nCurr+1;
                self.Kp=nCurr
                self.velCentroids.append(self.velCentroids[maxClust])
                self.weights = self.assignData(self.Kp,self.centroids,self.data)
                self.cntEmptyClusters(); 
    
        #Find the distance between data points and their assigned centroid
        #The outliers will be sorted based on this distance
        def findOutliers(self,centroids,weights,data,l):
            distOutlier=weights*0.0;
            for k in range(int(self.Kp)):
                datInClust=data[np.where(weights[k][0:]==1)[0],0:];
                if (len(datInClust)>0):
                    d=self.distFunc(centroids[k],datInClust); 
                    distOutlier[k][np.where(weights[k][0:]==1)[0]]=d;
            return distOutlier    
        
        #Delete smallest centroids until it matches the updated 
        #number of centroids
        def deleteCentroids(self):
            nCurr=len(self.centroids);
            targetKp=copy.deepcopy(self.Kp)

            while(nCurr>targetKp):
               minClust = np.argmin(np.sum(self.weights,axis=1)); 
               del self.centroids[minClust]
               del self.velCentroids[minClust]           
               nCurr=nCurr-1
               self.Kp=nCurr
               self.weights = self.assignData(self.Kp,self.centroids,self.data); 
               self.cntEmptyClusters(); 
         
        #Remove top l outliers from the data assigned to centroids
        #and store the data in an outliers weight array
        def removeOutliersFromClusters(self):  
            self.weightsOutliers=copy.deepcopy(self.weightsOutliers*0)
            for j in range(int(self.l)):
                #Store data that corresponds to outliers 
                if (j==0):
                    self.outlierData=copy.deepcopy(self.data[np.array(self.outliers[j]).astype(int)
                        -np.floor(self.outliers[j]/self.weightsShape[1]).astype(int)*self.weightsShape[1],0:]);
                else:
                    self.outlierData=np.vstack([copy.deepcopy(self.outlierData),copy.deepcopy(self.data[np.array(self.outliers[j]).astype(int)
                        -np.floor(self.outliers[j]/self.weightsShape[1]).astype(int)*self.weightsShape[1],0:])])

                #Remove outliers from weights that assign data to centroids, and add them
                #to outlier weights
                self.weights[np.floor(self.outliers[j]/self.weightsShape[1]).astype(int),np.array(self.outliers[j]).astype(int)
                    -np.floor(self.outliers[j]/self.weightsShape[1]).astype(int)*self.weightsShape[1]] = 0;
                self.weightsOutliers[np.array(self.outliers[j]).astype(int)
                    -np.floor(self.outliers[j]/self.weightsShape[1]).astype(int)*self.weightsShape[1]] = 1; 
        
        #Add outliers to cluster until it matches updated number of outliers       
        def addOutliersToCluster(self):
            for j in range(int(self.l0)-1,int(self.l)-1,-1):
                #Add outliers to weights that assign data to centroids, and remove 
                #them from outlier weights
                self.weights[np.floor(self.outliers[j]/self.weightsShape[1]).astype(int),np.array(self.outliers[j]).astype(int)-np.floor(self.outliers[j]/self.weightsShape[1]).astype(int)*self.weightsShape[1]] = 1;
                self.weightsOutliers[np.array(self.outliers[j]).astype(int)
                    -np.floor(self.outliers[j]/self.weightsShape[1]).astype(int)*self.weightsShape[1]] = 0;
            del list(self.outlierData)[int(self.l)-1:int(self.l0)-1]; 
       
        #Return the current state
        def getCurrState(self):
            currentState = (self.currentFitness,self.Kp,self.centroids,self.numEmptyClusters,self.l,self.outlierData,self.weights,self.weightsOutliers)
            return currentState
        
    return psoParticleClusterAndOutliers