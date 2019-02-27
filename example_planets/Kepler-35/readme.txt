This is an example of a multiple-star system (2 stars with a circum-binary planet).
This example uses a Newtonian integrator, so it will not be accurate since GR and tidal precession are not included. 
The detrending of the lightcurve data is also not particularly good, but it suffices for an example

To run a forward model, uncompress the photometry data:
$ tar -xvf kplr009837578_2880_detrend.txt.tar.gz

Compile phodymm.cpp to lcout as described in the main README.md, and place the executable in this directory. Then run:
$ ./lightcurve_runscript.sh

This should produce a model of Kepler-35's lightcurve and all transit and eclipse times. 

One can set up and run a DEMCMC just like any of the single star examples. 


To Do: Demo RV fit with RVs from both stars


