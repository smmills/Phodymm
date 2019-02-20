#!/bin/bash 
 
mpirun -np 20 -output-filename outf/sbatch.o --tag-output demcmc koi142_3pl_rvs.in empty.txt -rv0=koi142_rvs.txt demcmc_koi142_3pl_rvs_120w.out.res bestchisq_koi142_3pl_rvs_120w.pldin.res gamma_koi142_3pl_rvs_120w.txt.res
