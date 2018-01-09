# Automatic Partitioning And Outlier Detection

This file contains code that executes an algorithm that automatically determines the optimal paritioning of data, 
and that simulatneously determines the optimal number of outliers in the data. The outliers are detected 
based on a distance measurement. This algorithm is novel for two reasons:

     1) It attempts to simultaneously determine the number of clusters and number of outliers that 
        yields an optimally partitioned data set.
     2) It makes use of a non-parametric particle swarm optimization to combat pre-mature convergence 
        common to standard particle swarm optimization
    
The algorithm is essentially a combination of the algoritms proposed in [1] and [2]. Using this algorithm requires
optimizing the CS measure proposed in [3] which we optimize using the standard particle swarm optimization, the
constricted particle swarm optimization, and the non-parametric particle swarm optimization [4]. Combining [1] 
and [2] introduces a hyper-parameter epsilon that sets a threshold that determines how much the objective function
needs to change when an outlier is removed in order for the outlier to be considered real.

[1] Cura. "A particle swarm optimizatin approach to clustering" Expert Systems With Applications. 2012
http://www.sciencedirect.com/science/article/pii/S0957417411010852

[2] Chawla et al. "A unified approach to clustering and outlier detection" Proceedings of the 2013 SIAM International Conference on Data Mining
http://epubs.siam.org/doi/abs/10.1137/1.9781611972832.21

[3] Chou, C.H.  et al."A new cluster validity measure and its application to image compression." Pattern Anal. Appl. 7 (2), 205– 220. 2004
https://link.springer.com/article/10.1007/s10044-004-0218-1

[4] Behesti et al., "Non-parametric particle swarm optimization for global optimization",Applied Soft. Computing. 2015
http://www.sciencedirect.com/science/article/pii/S1568494614006474


# Running Tests

Tests can be run through the testClusteringCode file. At the top of the file you can choose the data set
to test the algorithm on. The algorithm works perfectly on the artificial data sets.

     testString =['shuttle','wdbc','wine','iris','ArtificialOutliers','ArtificialNoOutliers']
     #Choose data for testing algorithm
     testUse=testString[5]

The variables that characterize the particles to be used in the optimization are modified below:

     #Defines maximum and minimum number of clusters, number of outliers
     #and feature components
     nData=np.shape(X)[0]
     KpMin=1.0;
     KpMax=20.0;
     lMin=0.0;
     lMax=0.1*nData#20.0#0.01*nData;#20.0;
     numEvalState=8;
     
and the variables that characterize the PSO process are modified in the following block of code

     numParticles=40;
     neighborSize = 2#np.ceil(numParticles/5).astype(int);
     w=1.0;
     tol=1e-3;
     numIters=100
     kappa = 0.5;
     mult=1;
     c1=2.0
     c2 = c1*mult;
     constrict=1.0
     optimType='Min';
     ...
     nTrials=1;
     eps=0.05; #Hyper-parameter that determines extent to which objective must 
                 #change in order for solution that has different number of outliers 
                 #to be accepted.            
     wCS=1.0; wF=1.0; wKp=0.1 #Define weights to be used for each component of feature 
     ...
     #clusterOutlierParticles=psoParticleClustersAndOutliersFunc(X,sigma) 
     clusterOutlierParticles=npsoParticleClustersAndOutliersFunc(X,sigma); #Define npso particle class using the data and          sigma
     
Finally, the code is executed, and the relevant information stored, in the following chunk:

     pso=PSO(); #Define PSO class that contains functions that execute different versions of PSO
     for j in range(nTrials): 
         print('Trial',j+1)
         #Execute standard particle swarm optimization        
         output1=pso.executePSO(c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,clusterOutlierParticles,optimType,
         numEvalState,variableKpFitCSMeasFunc,evaluationFunc)
         
         #Execute constrict particle swarm optimization
         #c1=2.05; c2=c1;
         output1=pso.executeGCPSO(constrict,c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,clusterOutlierParticles
         ,optimType,numEvalState,variableKpFitCSMeasFunc,evaluationFunc)
         
         #Execute non-parameteric particle swarm optimization  
         output1=pso.executeNPSO(neighborSize,w,posMin,posMax,velMin,velMax,numIters,numParticles,clusterOutlierParticles,
         optimType,numEvalState,variableKpFitCSMeasFunc,evaluationFunc,npsoClustersAndOutliersInterpFunc)
 

         
    
         

