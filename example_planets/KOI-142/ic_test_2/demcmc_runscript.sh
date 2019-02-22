#!/bin/bash 
 
mpirun -np 20 -output-filename outf/sbatch.o --tag-output ./demcmc koi142_3pl_rvs_ic_test.in koi142_3pl_rvs_ic_test.pldin -rv0=koi142_rvs.txt
