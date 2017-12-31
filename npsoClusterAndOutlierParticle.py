#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:09:23 2017

@author: Ohi Dibua
This file contains a class that defines a non-parametric particle used in the problem of finding the optimal
partitioning (number of clusters) of a data set and the number of outliers that this data set uses.
It combines algorithms proposed from (SITE and SITE). The measure of optimal partitoning proposed 
is used in (SITE), and the method of choosing the best particle is inspired by (SITE). The details of
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
    posMin: minimum values of elements of x
    posMax: maximum values of elements of x
    velMin: minimum values of the veleocities ascribed to different elements of x
    velMax: maximum values of the velocities ascribed to different elements of x
    fitnessFunction = f(x) 
    fitnessEvaluationFunction: func(optimType,currentState,localBestState,globalBestState=None)
    Function that takes in two states that define proposed solutions and outputs 
    the solution that most meets the objective. If passing in the global best state,
    and empty list is passed in for localBestState. For typical PSO the states are state = (fitness,x)
""" 
import numpy as np
import copy
import time
import sys,os
import logging as logger
import sys, os
import time

def npsoParticleClustersAndOutliersFunc(data,sigma):                   
    class npsoParticleClusterAndOutliers():
        def __init__(self,optimType,w,posMin,posMax,velMin,velMax,fitnessFunction,fitnessEvaluationFunction):
            #Initialize parameters relevant to updating the velocity and position of particles
            self.w=w;              
            self.localBestFit=float("inf");                 
            self.KpMin=posMin[0]; self.KpMax=posMax[0];
            self.lMin=posMin[1];  self.lMax=posMax[1];
            self.velMinKp=velMin[0]; self.velMaxKp=velMax[0];
            self.velMinl=velMin[1]; self.velMaxl=velMax[1];
            self.velMinCentroids=velMin[2]; self.velMaxCentroids=velMin[2];
            self.featureMin=posMin[2]; self.featureMax=posMax[2];
            
            #Initialize parameters relevant to describing the data and the assignment of data to centroids
            self.data=copy.deepcopy(data);
            self.dataPermanent=copy.deepcopy(data);
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
            self.Kp = int(copy.deepcopy(round((self.KpMax-self.KpMin)*r1+self.KpMin)));
            self.velKp = 0;             
            r1=np.random.uniform();
            self.l = int(copy.deepcopy(round((self.lMax-self.lMin)*r1+self.lMin)));
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
            self.outliers=np.reshape(self.findOutliers(self.centroids,self.weights,self.distFunc,self.data),(self.weightsShape[0]*self.weightsShape[1],1)); 
            self.outliers=np.flip(np.argsort(self.outliers,axis=0),axis=0);             
            self.outlierData = [] 
            outputRmClusters=self.removeOutliersFromClusters(self.outliers,self.outlierData,self.weights,self.weightsOutliers,self.l);
            self.outlierData=outputRmClusters[0]; self.weights=outputRmClusters[1]; self.weightsOutliers=outputRmClusters[2];  
            
            #Recalculate centroids with outliers removed, count the number of empy centroids,
            #and evaluate particle fitnes
            self.centroids=self.calcCentroids(self.Kp,self.centroids,self.weights,self.data);                     
            self.numEmptyClusters=self.cntEmptyClusters(self.weights);                   
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

 
        def updateVelocityandPosition(self,bestNeighborState,globalBestState,t,constrict=None): 
             #Obtain global best and neighbor best components of particle solution
            globalBestKp=globalBestState[1]; globalBestCentroid=globalBestState[2]; globalBestl=globalBestState[4]
            bestNeighborKp=bestNeighborState[1]; bestNeighborCentroid=bestNeighborState[2]; bestNeighborl=bestNeighborState[4];
            
            #Obtain coefficients used in particle propagation for particular time step
            if (constrict is not None):
                pass;
            else:
                constrFactor = 1.0
            if not np.isscalar(self.w): 
                w=self.w[t];
            else: w=self.w;  
            
            #Update the velocity and position of number of clusters, and change number
            #of centroids to match updated number of clusters. Assign data to clusters.
            r1=np.random.uniform(0,1);   r2=np.random.uniform(0,1); r3=np.random.uniform(0,1);
            cognitiveVel = r1*(self.localBestKp-self.Kp); 
            socialVel = r2*(bestNeighborKp-self.Kp)
            self.velKp = constrFactor*(w*self.velKp + cognitiveVel + socialVel)
            if (constrict is None):
                self.velKp = min(max(self.velKp,self.velMinKp),self.velMaxKp)
            self.Kp = int(min(max(np.round(self.Kp + self.velKp + r3*(globalBestKp-self.Kp)),self.KpMin),self.KpMax)); 
            if self.Kp > np.shape(self.centroids)[0]: 
                outputSplitCentroids=self.splitCentroids(self.Kp,self.weights,self.centroids,self.velCentroids,self.data,self.assignData,self.cntEmptyClusters);
                self.Kp=outputSplitCentroids[0]; self.weights=outputSplitCentroids[1]; self.centroids=outputSplitCentroids[2];  
                self.numEmptyClusters=outputSplitCentroids[3];
            elif self.Kp <np.shape(self.centroids)[0]:
                outputDeleteCentroids=self.deleteCentroids(self.Kp,self.weights,self.centroids,self.velCentroids,self.assignData,self.cntEmptyClusters,self.data);
                self.Kp=outputDeleteCentroids[0]; self.weights=outputDeleteCentroids[1]; self.centroids=outputDeleteCentroids[2];
                self.numEmptyClusters=outputDeleteCentroids[3];
            else:                 
                self.weights = self.assignData(self.Kp,self.centroids,self.data);
  
            
            #Remove outliers from the data assigned to centroid
            self.weightsShape=np.shape(self.weights);
            self.outliers=np.reshape(self.findOutliers(self.centroids,self.weights,self.distFunc,self.data),(self.weightsShape[0]*self.weightsShape[1],1)); 
            self.outliers=np.flip(np.argsort(self.outliers,axis=0),axis=0); 
            self.outlierData = []  
            outputRmClusters=self.removeOutliersFromClusters(self.outliers,self.outlierData,self.weights,self.weightsOutliers,self.l);
            self.outlierData=outputRmClusters[0]; self.weights=outputRmClusters[1]; self.weightsOutliers=outputRmClusters[2];  
      
            #Update the velocity and position of number of outliers, and change number
            #of outliers to match updated number of outliers
            r1=np.random.uniform(0,2);   r2=np.random.uniform(0,2);
            cognitiveVel = r1*(self.localBestl-self.l); 
            socialVel = r2*(bestNeighborl-self.l)
            self.vell = constrFactor*(w*self.vell + cognitiveVel + socialVel)
            if (constrict is None):
                self.vell = min(max(self.vell,self.velMinl),self.velMaxl);
            self.l0=copy.deepcopy(self.l); 
            self.l = int(min(max(np.round(self.l + self.vell + r3*(globalBestl-self.l)),self.lMin),self.lMax)); 
            if (self.l>self.l0): 
                outputRmClusters=self.removeOutliersFromClusters(self.outliers,self.outlierData,self.weights,self.weightsOutliers,self.l);
                self.outlierData=outputRmClusters[0]; self.weights=outputRmClusters[1]; self.weightsOutliers=outputRmClusters[2]; 
            elif (self.l<self.l0):
                self.addOutliersToCluster();  
            
            #Pick centroid that will be randomly updated
            k = np.random.randint(0,self.Kp); 
            
            #Find centroid in personal best, neighborhood best and swarm best 
            #solution that is closest to the centroid being updated
            globalBestCentroid = globalBestCentroid[np.argmin(self.distFunc(self.centroids[k],np.array(globalBestCentroid)))];
            bestNeighborCentroid = bestNeighborCentroid[np.argmin(self.distFunc(self.centroids[k],np.array(bestNeighborCentroid)))];
            localBestCentroid = self.localBestCentroid[np.argmin(self.distFunc(self.centroids[k],np.array(self.localBestCentroid)))];
            
            #Update the kth centroid velocity and position according to nearest global best 
            #and personal best centroids. Assign data to these centroids, and re-calcuate centroids
            for j in range(self.nFeatures):           
                r1=np.random.uniform(0,2);   r2=np.random.uniform(0,2);
                cognitiveVel = r1*(localBestCentroid[j]-self.centroids[k][j]);
                socialVel = r2*(bestNeighborCentroid[j]-self.centroids[k][j]);
                self.velCentroids[k][j] = constrFactor*(w*self.velCentroids[k][j] + cognitiveVel + socialVel);
                if (constrict is None):
                    self.velCentroids[k][j] = min(max(self.velCentroids[k][j],self.velMinCentroids[j]),self.velMaxCentroids[j]);
                self.centroids[k][j] = self.centroids[k][j] + self.velCentroids[k][j] + r3*(globalBestCentroid[j]-self.centroids[k][j])
            self.weights = self.assignData(self.Kp,self.centroids,self.data);                        
            self.centroids=self.calcCentroids(self.Kp,self.centroids,self.weights,self.data); 
            
            #Remove outliers from the data assigned to centroids, calculate new centroids, and count number of clusters
            self.weightsShape=np.shape(self.weights); 
            self.outliers=np.reshape(self.findOutliers(self.centroids,self.weights,self.distFunc,self.data),(self.weightsShape[0]*self.weightsShape[1],1)); 
            self.outliers=np.flip(np.argsort(self.outliers,axis=0),axis=0); 
            self.outlierData = [] 
            outputRmClusters=self.removeOutliersFromClusters(self.outliers,self.outlierData,self.weights,self.weightsOutliers,self.l);
            self.outlierData=outputRmClusters[0]; self.weights=outputRmClusters[1]; self.weightsOutliers=outputRmClusters[2];                
            self.centroids=self.calcCentroids(self.Kp,self.centroids,self.weights,self.data); 
            self.numEmptyClusters=self.cntEmptyClusters(self.weights);  
          
        def updateLocalBest(self):
            #Calculate the fitness of the current solution, and store current state
            self.currentFitness=self.fitnessFunction(self.distFunc,self.Kp,self.centroids,self.weights,self.data);#self.constKpFitFunc();   
            currentState = (self.currentFitness,self.Kp,self.centroids,self.numEmptyClusters,self.l,self.outlierData,self.weights,self.weightsOutliers)
            localBestState = (self.localBestFitness,self.localBestKp,self.localBestCentroid,self.localBestNumEmptyClusters,self.localBestl,self.localBestOutliers,self.localBestWeights,self.localBestWeightsOutliers)
            
            #Store state of personal best solution and update personal best solution based on the 
            #fitness evaluation function
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
            nData=np.shape(data)[0]; K=int(K)
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
            if (K>1):
                centroidAssignments=np.argmin(distArr,axis=0); 
            else:
                centroidAssignments=np.array([0]*nData)
            if (K>1):
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
        def cntEmptyClusters(self,weights): 
            empCluster=np.where(np.sum(weights,axis=1)<=1)[0];
            numEmptyClusters=len(empCluster) 
            return numEmptyClusters 
        
        #Split largest centroid in particle until it matches the 
        #updated number of centroids
        def splitCentroids(self,Kp,weights,centroids,velCentroids,data,assignData,cntEmptyClusters): 
            nCurr=len(centroids); nFeatures=np.shape(data)[1]
            targetKp=copy.deepcopy(Kp)
            while(nCurr<targetKp):
                maxClust = np.argmax(np.sum(weights,axis=1));
                datInMaxClust = data[np.where(weights[maxClust,0:]==1)[0],0:];
                datUpperBound = np.max(datInMaxClust,axis=0);
                datLowerBound = np.min(datInMaxClust,axis=0);
                newCent = np.array([0.0]*nFeatures);
                for j in range(nFeatures):
                    r1=np.random.uniform();
                    newCent[j] = (datUpperBound[j]-datLowerBound[j])*r1+datLowerBound[j];
                centroids.append(newCent);
                nCurr=nCurr+1;
                Kp=nCurr
                if (len(velCentroids)>0):
                    velCentroids.append(velCentroids[maxClust])
                weights = assignData(Kp,centroids,data)
                numEmptyClusters=cntEmptyClusters(weights); 
            return (Kp,weights,centroids,numEmptyClusters) 
     
        #Find the distance between data points and their assigned centroid
        #The outliers will be sorted based on this distance
        def findOutliers(self,centroids,weights,distFunc,data):
            Kp=len(centroids); 
            distOutlier=weights*0.0;
            for k in range(int(Kp)):
                datInClust=data[np.where(weights[k][0:]==1)[0],0:];
                if (len(datInClust)>0):
                    d=distFunc(centroids[k],datInClust);  
                    distOutlier[k][np.where(weights[k][0:]==1)[0]]=d;
            return distOutlier        
        
        #Delete smallest centroids until it matches the updated 
        #number of centroids
        def deleteCentroids(self,Kp,weights,centroids,velCentroids,assignData,cntEmptyClusters,data):   
            nCurr=len(centroids);
            targetKp=copy.deepcopy(Kp)
            numEmptyClusters=cntEmptyClusters(weights); 

            while(nCurr>targetKp):
               minClust = np.argmin(np.sum(weights,axis=1));
               del centroids[minClust]
               if (len(velCentroids)>0):
                   del velCentroids[minClust]  
               nCurr=nCurr-1  
               Kp=nCurr
               weights = assignData(Kp,centroids,data); 
               numEmptyClusters=cntEmptyClusters(weights);  
            return (Kp,weights,centroids,numEmptyClusters)
        
        #Remove top l outliers from the data assigned to centroids
        #and store the data in an outliers weight array
        def removeOutliersFromClusters(self,outliers,outlierData,weights,weightsOutliers,l):   
            weightsShape=np.shape(weights);
            weightsOutliers=copy.deepcopy(weightsOutliers*0)
            #Store data that corresponds to outliers
            for j in range(int(l)):
                #Store data that corresponds to outliers 
                if (j==0):
                    outlierData=copy.deepcopy(data[np.array(outliers[j]).astype(int)
                        -np.floor(outliers[j]/weightsShape[1]).astype(int)*weightsShape[1],0:]);
                else:
                    outlierData=np.vstack([copy.deepcopy(outlierData),copy.deepcopy(data[np.array(outliers[j]).astype(int)
                        -np.floor(outliers[j]/weightsShape[1]).astype(int)*weightsShape[1],0:])])
 
                #Remove outliers from weights that assign data to centroids, and add them
                #to outlier weights
                weights[np.floor(outliers[j]/weightsShape[1]).astype(int),np.array(outliers[j]).astype(int)
                    -np.floor(outliers[j]/weightsShape[1]).astype(int)*weightsShape[1]] = 0;
                weightsOutliers[np.array(outliers[j]).astype(int)
                    -np.floor(outliers[j]/weightsShape[1]).astype(int)*weightsShape[1]] = 1;
            return (outlierData,weights,weightsOutliers)
        
        #Add outliers to cluster until it matches updated number of outliers           
        def addOutliersToCluster(self):
            #Add outliers to weights that assign data to centroids, and remove 
            #them from outlier weights
            for j in range(int(self.l0)-1,int(self.l)-1,-1):
                self.weights[np.floor(self.outliers[j]/self.weightsShape[1]).astype(int),np.array(self.outliers[j]).astype(int)-np.floor(self.outliers[j]/self.weightsShape[1]).astype(int)*self.weightsShape[1]] = 1;
                self.weightsOutliers[np.array(self.outliers[j]).astype(int)
                    -np.floor(self.outliers[j]/self.weightsShape[1]).astype(int)*self.weightsShape[1]] = 0;
            del list(self.outlierData)[int(self.l)-1:int(self.l0)-1]; 
        
        #Return the current state
        def getCurrState(self):
            currentState = (self.currentFitness,self.Kp,self.centroids,self.numEmptyClusters,self.l,self.outlierData,self.weights,self.weightsOutliers)
            return currentState
        
        #Return personal best state
        def getLocalBestState(self):
            localBestState = (self.localBestFitness,self.localBestKp,self.localBestCentroid,self.localBestNumEmptyClusters,self.localBestl,self.localBestOutliers,self.localBestWeights,self.localBestWeightsOutliers);
            return localBestState
    return npsoParticleClusterAndOutliers