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

[2] Chawla et al. "A unified approach to clustering and outlier detection" Proceedings of the 2013 SIAM International Conference on Data Mining

[3] Chou, C.H.  et al."A new cluster validity measure and its application to image compression." Pattern Anal. Appl. 7 (2), 205– 220.

[4] Behesti et al., "Non-parametric particle swarm optimization for global optimization",Applied Soft. Computing. 2015
