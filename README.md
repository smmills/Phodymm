# Phodymm

## Requirements

This code requires: 
1. GSL (gnu scientific library - https://www.gnu.org/software/gsl/)
2. celerite (https://github.com/dfm/celerite)

If you are using the demcmc rather than just running the forward model, it also requires:
3. MPI (https://www.open-mpi.org)


## Running the model

After those are installed, you may compile Phodymm from source using:
```
$ g++ -w -O3 -o lcout -I/yourpathto/celerite/cpp/include -I/yourpathto/celerite/cpp/lib/eigen_3.3.3 -lm -lgsl -lgslcblas -fpermissive phodymm.cpp
```
where you would replace "yourpathto" with the path to your celerite install.
The `-lm` `-lgsl` and `-lgslcblas` flags should link the compiler to your math libraries (including gsl)

This generates an executable called `lcout` (short for light-curve output).
You may use it to run an N-body model given a data file, input file, and initial conditions file.
The output will be a theoretical lightcurve and list of transit times.  
For an example, see the readme.txt in `example_planets/Kepler-36`

## Fitting to Data to Generate Posteriors

To run a demcmc, you must change the first line in phodymm.c from
```#define demcmc_compile 0```
to 
```#define demcmc_compile 1```

Then recompile with
```
mpic++ -w -Ofast -o demcmc -I/yourpathto/celerite/celerite/cpp/include -I/yourpathto/celerite/celerite/cpp/lib/eigen_3.3.3 -lm -lgsl -lgslcblas -lmpi -fpermissive phodymm.cpp
```

This will create an executable called `demcmc` which can be used to run a differential evolution MCMC (DEMCMC). 
It is recommended you run the demcmc executable on a computing cluster. Some example scripts to run the DEMCMC are included in 
`example_planets/Kepler-36`
For more details look at the readme.txt in that folder.  

## Appendix


### Units

The units used internally are AU and days. Generally all quantities should be given in those units. 


### Input Files

1. DATA FILE  

   The file with the list of input times (e.g., example_planets/Kepler-36/kplr011401755_1440_1.txt) must have the following format:
   ```
   [Line Index] \t [Time (days)] \t [Ignored] \t [Ignored] \t [Flux] \t [Flux Error]
   ```
   In units:
   ```
   [none] \t [days] \t [none] \t [none] \t [Normalized to 1] \t [Relative to Flux Value]
   ```
   With type:
   ```
   [long int] \t [double] \t [numeric] \t [numeric] \t [double] \t [double]
   ```
   The 1st, 3rd, and 4th columns are currently ignored, but must be present. They represent the data point index from Kepler and the raw/un-normalized flux and uncertainty values. If short and long cadence data are used simultanously, the data file must also contain an additional column indicating the cadence of each point. The column should be of type int and is simply a 1 if the point is long cadence, and 0 if it is short cadence.  
   
   If only a forward model is being computed, then only the Time column (and cadence if present) are used.  
   
   To conveniently generate data files from the Kepler data, see https://github.com/smmills/kepler_detrend


2. INPUT FILE  

   This file specifies various parameters necessary for completing the fit (e.g., example_planets/Kepler-36/kepler36_longcadence.in). Examples of paramters to edit in this file include the name of the run, the location of the data filethe fitting basis, which parameters to let vary or keep fixed, and priors. 
This file is read in by the C code and must be in the exact format as the example. Commented lines must be retained. 

   The file is structured as the description of each variable in 1 or more commented lines beggining with //, followed by the variable name and declartion in the format: 
   ```
   type name= value
   ```
   Users should change the value, but leave the first two columns unchanged. It is important that spacing is retained.  

   All entries should be sufficiently described in the input file itself. 
 

3. INITIAL CONDITIONS  

   The starting state of the planetary system at the epoch specified in the input file must be provided by the user and is called the initial condition file (e.g., example_planets/Kepler-36/k36.pldin).  
   
   The format of this file is first a line which is ignored describing the column names. The first column is the planet name, the last the planetary radius in units of stellar radius, and the second to last the planetary mass in Jupiter masses. The other columns depend on the choice of input parameter basis selected in the input file (by the xyzflag variable).   
   ```
   [Planet Label] \t  [column name] \t [column name] \t [column name] \t [column name] \t [column name] \t [column name] \t [Mass] \t [Rp/Rstar] 
   [value] \t  [value] \t [value] \t [value] \t [value] \t [value] \t [value] \t [value] \t [value] 
   [[value] \t  [value] \t [value] \t [value] \t [value] \t [value] \t [value] \t [value] \t [value]] 
   ...
   [[value] \t  [value] \t [value] \t [value] \t [value] \t [value] \t [value] \t [value] \t [value]]
   ```
   This is followed by at least 5 rows of single value entries related to the overal system or stellar properties. The following must be included:
   ```
   [Stellar Mass (Solar)]
   [Stellar Radius (solar)]
   [c_1 quadratic limb darkening term]
   [c_2 quadratic limb darkening term]
   [dilution: the fraction of the flux in the system not from the star which the planets are transiting]
   ```
   Several optional rows follow, depending on what fitting is being done. For instance, celerite GP fits require the 4 celerite terms (one per row). RV fits can include an RV jitter term for each set of observations (1 per line), etc. For more details refer to the example_planets and input file. Extra lines at the end of this file are ignored.  
   
   Typically, the xyzflag in the input file is set to 0, and the basis for the initial condition is specified as:
   ```
   [Planet Label] \t  [period (d)] \t [T0 (d)] \t [e] \t [i (deg)] \t [Omega (deg)] \t [omega(deg)] \t [Mass] \t [Rp/Rstar]
   ```
   Where T0 is the time of conjunction of the planet and the star, i is the incliation, Omega is the nodal angle, and omega is the argument of periastron with the coordinate system such that the sky plane is 0. This initial condition is transformed into the basis selected in the input file before fitting (e.g. {e, omega} -> {sqrt(e)*sin(omega), sqrt(e)*cos(omega)).   
   
   Planets should be entered in increasing order of orbital period.  


### Forward Model (`lcout`) Output Files

1. Lightcurve file (lc_NAME.lcout), where NAME is the name specified in the input file.  
   
   This file lists the times of output and theoretical output as well as the measured flux and uncertainties. Useful for plotting the best fit.
   4 columns:
   ```
   [time (days)] \t [measured flux (normalized)] \t [model flux (normalized)] \t [measured uncertainty (normalized)]
   ```
2. Transit times files (tbv_XX_XX.out), where the first XX is the body being transited and the second XX is the body doing the transiting. Indexes are 00=star, 01=1st planet, 02= 2nd planet, etc.  

   4 columns:
   ```
   [transit index from epoch] \t [transit mid-time (days)] \t [closest approach of centers of both bodies (AU)] \t [transit velocity (AU/day)] 
   ```
 
3. Coordinate Conversion Files
   By default, the lcout command produces transformations of the inputted initial conditions to different coordinate systems. These are divided into two classes (those based on orbital elements, and those based on cartesian coordinates) and outputted into two files: 
   * xyz_NAME.xyzout
   *  aei_NAME.aeiout
   Descriptions within these files specify the coordinate system used for each entry within. 

4. [Optional] Celerite Continuum Fit  

   This file shows the Gaussian Process the celerite fit to the continuum if a celerite fit was chosen. It's format is 
   ```
   [time] [flux]
   ```
 

### MCMC Fit (`demcmc`) Output Files

1. Current best-fit solution.   

   Every time an MCMC walker encounters a parameter set with highest likelihood of any fit yet found by all of the chains, it outputs it to the file `mcmc_bestchisq_NAME.aei`, where NAME is the run name as specified in the input file. 

2. Chain state

   Every 100 generations, the current state of the chains is printed in the fitting basis to `demcmc_NAME.out`.  

3. Stepsize diagnostics

   Every 10 generations, the fraction of new parameter proposals that are accepted and the current scale factor on the differential evolution vector used to propose new steps is printed. A file is generated with name `gamma_NAME.txt` and format:
   ```
   [generation] \t [proposal acceptance fraction] \t [scale factor]
   ```

4. Diagnostic Output

   Some diagnostic output is saved in `demcmc.stdout` and/or the `outf` subdirectory, depending on how the run is executed. 
 



  
