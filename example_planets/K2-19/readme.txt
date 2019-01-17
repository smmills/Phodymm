A model of a 3-planet system with long-cadence photometry from K2, transit timing variation measurements from other sources, and mass priors from RVs. 
- Some TTV data taken from Narita et al. 2015 
- RV/TTV from CPS (Petigura et al. 2019)


# Single Forward Model

After phodymm.cpp has been compiled to produce an executable called lcout (see directions in README.md), 
copy the lcout executable to this directory.

Then run the `lightcurve_runscript.sh` script to generate
 (1) A lightcurve model
 (2) Transit times files for each planet
 (3) Ancillary and diagnostic output as described in the Phodymm README.md Appendix

The scripts in example_planets/analysis_tools can then be copied to this directory to produce some quick diagnostic plots 


# DEMCMC Model Fit

To run a DEMCMC, compile the demcmc executable (directions in README.md), and copy it to this directory.
Depending on your computing setup, you might then run runscript.sh (if you are already on the machine you wish to run the MCMC on)
or demcmc.sbatch (in order to submit your job to a slurm queue). 

This will generate the file with the state of the MCMC chain every 100 generations which can be used for generating posteriors, as well as several other output files as described in README.md MCMC Fit Output Files section.  

To restart a DEMCMC that has been stopped or crashed, the helper script `restart.sh` is included. After a DEMCMC is run for at least 100 generations (generating all necessary output files) running:
$ ./restart.sh demcmc_runscript.sh k2-19_massprior.in 
will generate several restart files ending in `.res` (see the Optional Input section in README.md) and a script called demcmc_runscript.sh.res.sh, which can be run to restart the MCMC from where it left off 


 