#!/bin/bash 
 
mpirun -np 20 -output-filename outf/sbatch.o --tag-output demcmc koi142_3pl_rvs_ic_test.in empty.txt -rv0=koi142_rvs.txt demcmc_koi142_3pl_rvs_ic_test.out.res bestchisq_koi142_3pl_rvs_ic_test.pldin.res gamma_koi142_3pl_rvs_ic_test.txt.res
