A basic model of a 2-planet system using only long-cadence data.


After phodymm.cpp has been compiled to produce an executable called lcout (see directions in README.md), 
copy the lcout executable to this directory.

Then run the `lightcurve_runscript.sh` script to generate
 (1) A lightcurve model
 (2) Transit times files for each planet
 (3) Some diagnostic output

Notes:





To run a DEMCMC, compile the demcmc executable (directions in README.md), and copy it to this directory.
Depending on your computing setup, you might then run runscript.sh (if you are already on the machine you wish to run the MCMC on)
or demcmc.sbatch (in order to submit your job to a slurm queue). 

5. OUTPUT of demcmc executable 
 - demcmc_XXX.out


To restart a DEMCMC, the helper script `restart.sh` is included. You may run XXXX

 
