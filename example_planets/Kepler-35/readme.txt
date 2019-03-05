This is an example of a multiple-star system (2 stars with a circum-binary planet).
RV data is taken from Welsh et al. (2012) Nature, 481, 475-479. 

This code uses a Newtonian integrator, so it will not be exactly accurate since GR and tidal precession are not included. 
The detrending of the lightcurve data is also not particularly good, but it suffices for an example, and will produce both eclipses and transits in the proper location. 

To run a forward model, uncompress the photometry data:
$ tar -xvf kplr009837578_2880_detrend.txt.tar.gz

Compile phodymm.cpp to lcout as described in the main README.md, and place the executable in this directory. Then run:
$ ./lightcurve_runscript.sh
to model the photometry. This should produce a model of Kepler-35's lightcurve and all transit and eclipse times. 

To include the RV data in the model in addition to the photometry, instead run:
$ ./lightcurve_runscript_rvs.sh
There are RVs are for each star from 3 different telescopes. If we include RV jitter as a free parameter, that introduces 6 additional free model parameters (one jitter per telescope per star). Additionally, the optimal RV offset for each telescope and star is allowed to float in the model.  

One can set up and run a DEMCMC for either case just like any of the single star examples. 







