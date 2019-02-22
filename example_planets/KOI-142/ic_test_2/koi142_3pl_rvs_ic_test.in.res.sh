#!/bin/bash 
 
mpirun -np 20 -output-filename outf/sbatch.o --tag-output demcmc koi142_rvs.txt empty.txt demcmc_.out.res bestchisq_.pldin.res gamma_.txt.res
