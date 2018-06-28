#define demcmc_compile 3

// To compile lcout:
// make sure demcmc_compile is defined as 0
// Compile with:
// g++ -w -O3 -o lcout -I/home/smills/celerite/celerite/cpp/include -I/home/smills/celerite/celerite/cpp/lib/eigen_3.3.3 -lm -lgsl -lgslcblas -fpermissive demcmc.cpp 
// ./lcout demcmc.in kep35.pldin [[-rv0=rvs0.file] [-rv1=rvs1.file] ... ]

// To compile demcmc:
// make sure demcmc_compile is defined as 1
// mpic++ -w -Ofast -o demcmc -I/home/smills/celerite/celerite/cpp/include -I/home/smills/celerite/celerite/cpp/lib/eigen_3.3.3 -lm -lgsl -lgslcblas -lmpi -fpermissive demcmc.cpp
// Run with:
// mpirun ./demcmc demcmc.in kep.pldin

// To compile longterm stability
// make sure demcmc_compile is defined as 3
// Compile with:
// $ gcc -Ofast -o stability -lgsl -lgslcblas -fopenmp demcmc.c
// g++ -w -O3 -o lcout -I/home/smills/celerite/celerite/cpp/include -I/home/smills/celerite/celerite/cpp/lib/eigen_3.3.3 -lm -lgsl -lgslcblas -fpermissive demcmc.cpp


#if (demcmc_compile==1) 
#include <mpi.h>
#endif
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <memory.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <float.h>
#include <gsl/gsl_multimin.h>

// For CELERITE
#include <cmath>
#include <iostream>
#include <Eigen/Core> 
#include "celerite/celerite.h"
using Eigen::VectorXd;

int CELERITE;
int NCELERITE=4;
int RVCELERITE;
int NRVCELERITE=4;

// These variables are now all defined in the input file instead of here
// turn on N temperature annealing 
int NTEMPS;
// turn on one sided inclination distribution
int IGT90;
// turn on positive dilution
int INCPRIOR;
//
int DIGT0;
// sqrt(e) as parameter?
int SQRTE;
// restrict masses to > 0
int MGT0;
double MASSHIGH; 
// density cuts
int DENSITYCUTON;
double *MAXDENSITY;// g/cm^3
double MSUNGRAMS = 1.98855e33; // g
double RSUNCM = 6.95700e10; // cm
// eccentricity cuts. 
int ECUTON;
double *EMAX;
// turn on eccentricity prior.
int EPRIOR = 0;
int *EPRIORV;
// this sigma is the sigma parameter in a Rayleigh distribution
double ESIGMA;
// define rayleigh distribution
double rayleighpdf (double x) {
  double sigma = ESIGMA;
  if (x < 0.0) {
    printf(" rayleigh pdf requires positive x value\n");
    exit(0);
  }
  double f =  x / (sigma*sigma) * exp( -x*x / (2*sigma*sigma) );
  return f;
}
// define normal distribution
double normalpdf (double x) {
  double sigma = ESIGMA;
  double f = (1.0 / (sigma * sqrt(2.*M_PI))) *
        exp(-x*x / (2.0 * sigma * sigma));
  return f;
}


// Spectroscopy Constraints - Currently only works for sinngle star. Don't use it for multistar. 
int SPECTROSCOPY;
// Assumes assymetric Gaussian
double SPECRADIUS;
double SPECERRPOS;
double SPECERRNEG;

// Spectroscopy Constraints - Currently only works for sinngle star. Don't use it for multistar. 
int MASSSPECTROSCOPY;
// Assumes assymetric Gaussian
double SPECMASS;
double MASSSPECERRPOS;
double MASSSPECERRNEG;
int DILUTERANGE=1;

int RANK;
int SIZE;

double PRINTEPOCH = 800.0;
//// Global Variables
// Initialized in main
int RESTART;
int RVS;
// Initialized in getinput function call
int MULTISTAR;
int NBODIES;
int PPERPLAN;
int PSTAR = 5;  // Number of parameters for central star
char *OUTSTR;
int NPL;
double EPOCH;
long NWALKERS;
long NSTEPS;
int CADENCESWITCH;
char TFILE[1000];
int *PARFIX;
double T0;
double T1;
unsigned long SEED;
double DISPERSE;
double OPTIMAL;
double RELAX;
int NPERBIN;
double BINWIDTH;
double *PSTEP;
double *SSTEP;
double *STEP;
int STEPFLAG;
double *CSTEP;
int BIMODF;
int *BIMODLIST;
double OFFSETMULT;
double OFFSETMIN;
double OFFSETMINOUT;
double DIST2DIVISOR;
int LTE;
int SPLITINCO;
int XYZFLAG;
int *XYZLIST;
int *RVARR;
char **RVFARR;

int RVJITTERFLAG;
int NSTARSRV;
int NTELESCOPES;
int RVJITTERTOT = 0;

int TTVJITTERFLAG;
int TTVJITTERTOT = 0;
int NSTARSTTV;
int NTELESCOPESTTV;

int OOO = 2;
int CONVERT=1;

#if (demcmc_compile==3)
double TINTERVAL = 1000.0;
double AFRACSTABLE = 0.10;
#endif


// Variables for Fitting additional TTVs 
// as of now the ttv input files MUST be SORTED numerically and at same epoch
int TTVCHISQ;
//char **TTVFILENAMES;
//char TTVFILENAME1[] = "ttvsecondary.txt";
long **NTTV;  //number
double **TTTV; // time
double **ETTV; //error
double **MTTV; //modeled


// Often used system constants
const int SOFD = sizeof(double);
const int SOFDS = sizeof(double*);
const int SOFI = sizeof(int);
const int SOFIS = sizeof(int*);


///* Integration parameters */
#define DY 1e-14                   ///* Error allowed in parameter values per timestep. */
#define HStart 1e-5                ///* Timestep to start.  If you get NaNs, try reducing this. */

///* Some physical constants */
#define G 2.9591220363e-4         ///*  Newton's constant, AU^3*days^-2 */
#define RSUNAU  0.0046491            ///* solar radius in au */
#define REARTHORSUN 0.009171    ///* earth radius divided by solar radius */
#define MPSTOAUPD 5.77548327363993e-7    ///* meters per second to au per day conversion factor */
#define MSOMJ 1.04737701464237e3  ///* solar mass in terms of jupiter masses */ 
#define CAUPD 173.144632674240  ///* speed of light in AU per day */

//Note in dpintegrator and related code:
///* y vector is [x,y,z,v_x,v_y,v_z] */
///* f is d/dt y */


// Check if two doubles are equal (at the DBL_EPSILON level)
int dbleq (double a, double b) {

  return fabs(a-b) < DBL_EPSILON;

}


// Find the greater of two values
int compare (const void* a, const void* b) {

  double dbl_a = * ( (double*) a);
  double dbl_b = * ( (double*) b);

  if (dbl_a < dbl_b) return -1;
  else if (dbl_b < dbl_a) return 1;
  else return 0;

}


// set a 6*npl element vector equal to another 6*npl element vector
int seteq (int npl, double ynew[], const double y[]) {

  int i;
  for(i=0;i<6*npl;i++) ynew[i]=y[i];

}


// N-body interaction between planets with position vector y and masses in params
int func (double t, const double y[], double f[], void *params) {

  int i,j;
  int i1,i2;
  double * npl_masses = (double *)params;
  double * masses = &npl_masses[1];
  double dnpl = npl_masses[0];
  const int npl = dnpl;
  double gmc1, gmc2, gm1, gm2;
  double rc1m3, rc2m3, r12m3;

  for(i=0; i<npl; i++) {
    gmc1 = G*(masses[0]+masses[i+1]);
    rc1m3 = pow(pow(y[i*6+0],2)+pow(y[i*6+1],2)+pow(y[i*6+2],2),-3.0/2);
    for(j=0; j<3; j++) {
      f[i*6+j] = y[i*6+3+j];  /* x dot = v */
      f[i*6+3+j] = -gmc1*y[i*6+j]*rc1m3;   /* Pull of the star. */
    }   
  }
 
  /* Interaction between each pair of planets. */
  /* Eqn 6.8,6.9 in Murray and Dermott */
  /* Astrocentric coordinates are used (i.e. center of star is origin) */
  for(i1=0; i1<npl-1; i1++) {
    gm1 = G*masses[i1+1];
    rc1m3 = pow(pow(y[i1*6+0],2)+pow(y[i1*6+1],2)+pow(y[i1*6+2],2),-3.0/2);
    for(i2=i1+1; i2<npl; i2++) {
      gm2 = G*masses[i2+1];
      rc2m3 = pow(pow(y[i2*6+0],2)+pow(y[i2*6+1],2)+pow(y[i2*6+2],2),-3.0/2);
      r12m3 = pow(pow(y[i1*6+0]-y[i2*6+0],2)+pow(y[i1*6+1]-y[i2*6+1],2)+pow(y[i1*6+2]-y[i2*6+2],2),-3.0/2);
  
      for(j=0; j<3; j++) f[i1*6+3+j] += -gm2*( (y[i1*6+j]-y[i2*6+j])*r12m3 + y[i2*6+j]*rc2m3 );
      for(j=0; j<3; j++) f[i2*6+3+j] += -gm1*( (y[i2*6+j]-y[i1*6+j])*r12m3 + y[i1*6+j]*rc1m3 );
    }
  }

  return GSL_SUCCESS;
}


// We don't use the Jacobian, but still need to define a pointer for gsl
void *jac;


// Compute transit lightcurve for case of more than 1 luminous body 
double ***dpintegrator_multi(double ***int_in, double **tfe, double **tve, int *cadencelist) {

  int nbodies = NBODIES;
  int cadenceswitch = CADENCESWITCH;
  double t0 = T0;
  double t1 = T1;
  int nperbin = NPERBIN;
  double binwidth = BINWIDTH;
  const int posstrans = NBODIES;
  double offsetmult = OFFSETMULT;
  double offsetmin = OFFSETMIN;
  double offsetminout = OFFSETMINOUT;
  double dist2divisor = DIST2DIVISOR;
  const int sofd = SOFD;
  const int sofds = SOFDS;
  const int sofi = SOFI;
  const int sofis = SOFIS;

  int nplplus1 = int_in[0][0][0];
  const int npl = nplplus1-1;
  double tstart = int_in[1][0][0]; //epoch
  double rstar = int_in[3][0][0];
  double dilute = int_in[3][0][1];

  long kk = (long) tfe[0][0];
  double *timelist;
  long *order;
  double *timelist_orig = &tfe[0][1];

  long kkorig = kk;
  int nb;
  if (cadenceswitch == 1 ) {
    timelist = malloc(kk*nperbin*sofd);
    long q;
    for (q=0; q<kk; q++) {
      double tcenter = timelist_orig[q];
      for (nb=0; nb<nperbin; nb++) {
        timelist[q*nperbin+nb] = tcenter + (binwidth/nperbin)*(nb+0.5) - binwidth*0.5;
      }
    }
    kk *= nperbin;
  } else if (cadenceswitch == 0) {
    timelist = timelist_orig;
  } else if (cadenceswitch == 2) {
    timelist = malloc(kk*nperbin*sofd);
    order = malloc(kk*nperbin*sizeof(long));
    long lccount=0;
    long sccount=0;
    long q;
    for (q=0; q<kk; q++) {
      if (cadencelist[q] == 1) {
        double tcenter = timelist_orig[q];
        for (nb=0; nb<nperbin; nb++) {
          timelist[sccount+lccount*nperbin+nb] = tcenter + (binwidth/nperbin)*(nb+0.5) - binwidth*0.5;
          order[sccount+lccount*nperbin+nb] = q;
        }
        lccount++;
      } else {
        timelist[sccount+lccount*nperbin] = timelist_orig[q];
        order[sccount+lccount*nperbin] = q;
        sccount++;
      }
    }
    kk = sccount + lccount*nperbin; 


    if (OOO == 2) {
      OOO = 0;
      for (q=0; q<kk-1; q++) {
        if (timelist[q+1] < timelist[q]) {
          OOO = 1;
          break;
        }
      }
    }

    if (OOO) { 
      double *tclist = malloc((2*sofd)*kk); 
      for (q=0; q<kk; q++) {
        tclist[2*q] = timelist[q];
        tclist[2*q+1] = (double) order[q];
      }
      qsort(tclist, kk, sofd*2, compare);
      for (q=0; q<kk; q++) {
        timelist[q] = tclist[2*q];
        order[q] = (long) tclist[2*q+1];
      }
      free(tclist);
    }

  } else {
    printf("cadenceswitch err\n");
    exit(0);
  }
 

  // Set up integrator
  const gsl_odeiv_step_type * T 
  /*   = gsl_odeiv_step_bsimp;   14 s */
  = gsl_odeiv_step_rk8pd;      /* 3 s */
  /*  = gsl_odeiv_step_rkf45;    14 s */
  /*  = gsl_odeiv_step_rk4;      26 s */

  gsl_odeiv_step * s 
    = gsl_odeiv_step_alloc (T, 6*npl);
  gsl_odeiv_control * c 
    = gsl_odeiv_control_y_new (DY, 0.0);
  gsl_odeiv_evolve * e 
    = gsl_odeiv_evolve_alloc (6*npl);
  
  long i;
  double mu[npl+1];
  double npl_mu[npl+2];
  gsl_odeiv_system sys = {func, jac, 6*npl, npl_mu};


  double t; /* this is the given epoch. */
  double hhere;  
  double y[6*npl], yin[6*npl];
  //note:rad is in units of rp/rstar
  double *rad = malloc(npl*sofd);
  double brightness[npl+1];
  double c1list[npl+1];
  double c2list[npl+1];

  mu[0] = int_in[2][0][0]; 
  brightness[0] = int_in[2][0][8];
  c1list[0] = int_in[2][0][9];
  c2list[0] = int_in[2][0][10]; 

  for (i=0; i<npl; i++) {
    mu[i+1] = int_in[2][i+1][0];
    yin[i*6+0] = int_in[2][i+1][1];
    yin[i*6+1] = int_in[2][i+1][2];
    yin[i*6+2] = int_in[2][i+1][3];
    yin[i*6+3] = int_in[2][i+1][4];
    yin[i*6+4] = int_in[2][i+1][5];
    yin[i*6+5] = int_in[2][i+1][6];
    rad[i] = int_in[2][i+1][7];  
    brightness[i+1] = int_in[2][i+1][8];
    c1list[i+1] = int_in[2][i+1][9];
    c2list[i+1] = int_in[2][i+1][10];
  }

  memcpy(&npl_mu[1], mu, (npl+1)*sofd);
  npl_mu[0] = npl;

  double mtot=mu[0];
  for(i=0;i<npl;i++) mtot += mu[i+1];

  double yhone[6*npl], dydt[6*npl], yerrhone[6*npl];
  double yhone0[6*npl];
  double thone,tstep,dist,vtrans;
  double vstar,astar;

  int k;
  int brights=0;
  int *brightsindex = malloc((npl+1)*sofi);
  for (k=0; k<npl+1; k++) {
    if (brightness[k] > 0) {
      brightsindex[brights] = k;
      brights++;
    }
  }

  double **fluxlist = malloc(brights*sofds);
  for (k=0; k<brights; k++) {
    fluxlist[k] = malloc(kk*sofd);
  }
  double *netflux = malloc((kk+1)*sofd);
  double **tmte = malloc(4*sofds);

  tmte[0] = &tfe[0][0];
  tmte[1] = &tfe[1][0]; // measured flux list;
  tmte[3] = &tfe[2][0]; // error list;

  int **transitcount = malloc(brights*sofis);
  for (k=0; k<brights; k++) {
    transitcount[k] = malloc((npl+1)*sofi);
    for (i=0; i<(npl+1); i++) {
      transitcount[k][i] = -1;
    }
  }

  int pl;
  int **numplanets = malloc(brights*sofis);
  for (k=0; k<brights; k++) {
    numplanets[k] = calloc(kk, sofi);
  }
  int ***whereplanets = malloc(brights*sizeof(int**));
  for (k=0; k<brights; k++) {
    whereplanets[k] = malloc(kk*sofis);
    long kkc;
    for (kkc=0; kkc<kk; kkc++) {
      whereplanets[k][kkc] = calloc(nbodies, sofi);
    }
  }

  double ****tranarray = malloc(brights*sizeof(double***));
  for (k=0; k<brights; k++) {
    tranarray[k] = malloc(kk*sizeof(double**));
    long kkc;
    for (kkc=0; kkc<kk; kkc++) {
      tranarray[k][kkc] = malloc(posstrans*sofds);
      long kkcc;
      for (kkcc=0; kkcc<posstrans; kkcc++) {
        tranarray[k][kkc][kkcc] = malloc(3*sofd);
      }
    }
  }

 
  // set up directories for output if we are just doing a single run 
#if ( demcmc_compile == 0 ) 
  FILE ***directory = malloc(brights*sizeof(FILE**));
  for (k=0; k<brights; k++) {
    directory[k] = malloc((npl+1)*sizeof(FILE*));
  }
  for (k=0; k<brights; k++) {
    int k1 = brightsindex[k];
    for (i=0; i<(npl+1); i++) {
      if (i != k1) {
        char str[30];
        sprintf(str, "tbv%02i_%02i.out", k1, i);
        directory[k][i]=fopen(str,"w");
      }
    }
  }
#endif


  /* Setup forward integration */
  double ps2, dist2;  /* ps2 = projected separation squared. */
  t = tstart;
  double h = HStart;
  long maxtransits = 10000;  //total transits per planet
  int ntarrelem = 6;
  double **transitarr = malloc(brights*sofds);
  for (k=0; k<brights; k++) {
    transitarr[k] = malloc((maxtransits+1)*ntarrelem*sofd);
  }
  if (transitarr[brights-1] == NULL) {
      printf("Allocation Error\n");
      exit(0);
  }
  long *ntransits = calloc(brights, sizeof(long));

  seteq(npl, y, yin);
  double **dp = malloc(brights*sofds);
  double **dpold = malloc(brights*sofds);
  double ddpdt;
  for (i=0; i<brights; i++) {
      dp[i] = malloc(npl*sofd);
      dpold[i] = malloc(npl*sofd);
  }
  for(pl=0; pl<npl; pl++) dp[0][pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6];
  for (i=1; i<brights; i++) {
      for(pl=0; pl<npl; pl++) {
          dp[i][pl]=(y[0+pl*6]-y[0+(i-1)*6])*y[3+pl*6] + (y[1+pl*6]-y[1+(i-1)*6])*y[4+pl*6];
      }
  }

  long vv = 0;
  // This messes up reduced chi squared if your rv values are outside of the integration time!!!  
  // But we are only comparing relative chi squareds
  // Plus just don't include rv times that aren't within your integration bounds, because that's silly
  double *rvarr;
  double *rvtarr;
  int onrv = 0;
  int rvcount, startrvcount; 
  if (RVS) {
    vv = (long) tve[0][0];
    rvarr = calloc(vv+1, sofd);
    rvarr[0] = (double) vv;
    rvtarr = malloc((vv+2)*sofd);
    rvtarr[0] = -HUGE_VAL;
    rvtarr[vv+1] = HUGE_VAL;
    memcpy(&rvtarr[1], &tve[0][1], vv*sofd);
    startrvcount = 1;
    while (rvtarr[startrvcount] < tstart) startrvcount+=1;
    rvcount = startrvcount;
  }


  double eps=1e-10;
  double yhonedt[6*npl];
  double dydtdt[6*npl];
  double thonedt;
  double dt;
  double baryz, baryz2;
  double zb, zb2;
  double zp, zs;
  double dt1, dt2;
  double t2;
  double baryc[6*npl];
  double baryc2[6*npl];
  double dbarycdt[6*npl];
  double dbarycdt2[6*npl];
  double starimage[6*npl];
  double dstarimagedt[6*npl];
  int ii;
  int j;

  // Integrate forward from epoch
  while (t < t1 ) {      

    if (RVS) {
      onrv=0;
      if (h + t > rvtarr[rvcount]) {  
        onrv = 1;
        hhere = h;
        h = rvtarr[rvcount] - t;
      }
    }

    int status = gsl_odeiv_evolve_apply (e, c, s,
                                         &sys, 
                                         &t, t1,
                                         &h, y);

    if (status != GSL_SUCCESS)
        break;

    if (RVS) {
       if (onrv ==1) {
         h = hhere;
         func(t, y, dydt, mu);
         vstar = 0;
         for (i=0; i<npl; i++) {
           vstar += -y[i*6+5]*mu[i+1]/mtot;
         }
         if (tve[3][rvcount] == 0) {
           rvarr[rvcount] = vstar; 
         } else {
           rvarr[rvcount] = vstar + y[(((int) tve[3][rvcount])-1)*6+5];
         }
         rvcount += 1;


       }
    }

    //distance squared to first planet
    dist2 = pow(y[0],2)+pow(y[1],2)+pow(y[2],2);

    for(pl=0; pl<npl; pl++) {  /* Cycle through the planets, searching for a transit. */
        dpold[0][pl]=dp[0][pl];
        dp[0][pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6];
        ps2=y[0+pl*6]*y[0+pl*6]+y[1+pl*6]*y[1+pl*6];

      if( dp[0][pl]*dpold[0][pl] <= 0 && ps2 < dist2/dist2divisor && y[2+pl*6]<0 && brights>1) { 
                     // A minimum projected-separation occurred: "Tc".  
                //   ps2 constraint makes sure it's when y^2+z^2 is small-- its on the face of the star.  
                  //   y[2] constraint means a primary eclipse, presuming that the observer is on the positive x=y[2] axis.
        seteq(npl, yhone, y);
        thone = t;

        if (LTE) {
          ii=0;
          dt = -yhone[2+pl*6] / CAUPD;
          seteq(npl, yhonedt, yhone);
          gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
          thonedt = thone+dt;
 
          do {
            func(thone, yhone, dydt, npl_mu);
            func(thonedt, yhonedt, dydtdt, npl_mu);
 
            for (j=0; j<6; j++) {
              baryc[j] = 0.;
              baryc2[j] = 0.;
              dbarycdt[j] = 0.;
              dbarycdt2[j] = 0.;
              for (i=0; i<npl; i++) {
                baryc[j] += yhone[j+i*6]*mu[i+1];
                baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                dbarycdt[j] += dydt[j+i*6]*mu[i+1];
                dbarycdt2[j] += dydtdt[j+i*6]*mu[i+1];
              }
              baryc[j] /= mtot;
              baryc2[j] /= mtot;
              dbarycdt[j] /= mtot;
              dbarycdt2[j] /= mtot;
 
              starimage[j] = baryc2[j]-baryc[j];
              dstarimagedt[j] = dbarycdt2[j]-dbarycdt[j];
            }
 
            dp[0][pl] = (yhonedt[0+pl*6]-starimage[0])*(yhonedt[3+pl*6]-starimage[3]) + (yhonedt[1+pl*6]-starimage[1])*(yhonedt[4+pl*6]-starimage[4]);
            double ddpdt = (dydtdt[0+pl*6]-dstarimagedt[0])*(dydtdt[0+pl*6]-dstarimagedt[0]) + (dydtdt[1+pl*6]-dstarimagedt[1])*(dydtdt[1+pl*6]-dstarimagedt[1]) + (yhonedt[0+pl*6]-starimage[0])*(dydtdt[3+pl*6]-dstarimagedt[3]) + (yhonedt[1+pl*6]-starimage[1])*(dydtdt[4+pl*6]-dstarimagedt[4]);
 
 
            zs = 0.0;
            zp = yhonedt[2+pl*6]/CAUPD;
            zb = baryc[2]/CAUPD;
            zb2 = baryc2[2]/CAUPD;
            zs -= zb;
            zp -= zb2;
            dt1 = zs;
            dt2 = -zp;
            dt = dt1+dt2;
            
            tstep = - dp[0][pl]/ddpdt;
            gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, NULL, NULL, &sys);
            seteq(npl, yhonedt, yhone);
            gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
            thone += tstep;
            thonedt = thone+dt;
            ii++;
 
          } while (ii<10 && fabs(tstep)>eps);

          dist = sqrt( pow( (yhonedt[0+pl*6]-starimage[0]), 2) + pow( (yhonedt[1+pl*6]-starimage[1]), 2) ); 
 
        } else {

          ii=0;
          do {
            func (thone, yhone, dydt, npl_mu);
            ddpdt = dydt[0+pl*6]*dydt[0+pl*6] + dydt[1+pl*6]*dydt[1+pl*6] + yhone[0+pl*6]*dydt[3+pl*6] + yhone[1+pl*6]*dydt[4+pl*6];
            tstep = - dp[0][pl] / ddpdt;
            gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, dydt, NULL, &sys);
            thone += tstep;
            dp[0][pl]=yhone[0+pl*6]*yhone[3+pl*6]+yhone[1+pl*6]*yhone[4+pl*6];
            ii++;
          } while (ii<5 && fabs(tstep)>eps);
            
          dist = sqrt(pow(yhone[0+pl*6],2)+pow(yhone[1+pl*6],2));

        }

        dp[0][pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6];

        if (dist < 2.0*((rstar + rstar*rad[pl])*RSUNAU)) {

          transitcount[0][pl+1] += 1;  // Update the transit number. 

          if (LTE) {
            for (j=0; j<6; j++) {
              baryc[j] = 0.;
              baryc2[j] = 0.;
              for (i=0; i<npl; i++) {
                baryc[j] += yhone[j+i*6]*mu[i+1];
                baryc2[j] += yhonedt[j+i*6]*mu[i+1];
              }
              baryc[j] /= mtot;
              baryc2[j] /= mtot;
 
              starimage[j] = baryc2[j]-baryc[j];
            }
 
            zs = 0.0;
            zp = yhonedt[2+pl*6]/CAUPD;
            zb = baryc[2]/CAUPD;
            zb2 = baryc2[2]/CAUPD;
            zs -= zb;
            zp -= zb2;
            dt1 = zs;
            dt2 = -zp;
            dt = dt1+dt2;
 
            dist = sqrt( pow( yhonedt[0+pl*6] - starimage[0], 2) + pow( yhonedt[1+pl*6] - starimage[1], 2) ); 
            vtrans = sqrt( pow( yhonedt[3+pl*6] - starimage[3], 2) + pow( yhonedt[4+pl*6] - starimage[4], 2) ); 
          } else {
            dist = sqrt(pow(yhone[0+pl*6],2)+pow(yhone[1+pl*6],2));
            vtrans = sqrt(pow(yhone[3+pl*6],2)+pow(yhone[4+pl*6],2)); 
            zs=0;
            zb2=0;
          }

          t2 =  thone + zs;

#if ( demcmc_compile == 0 ) 
          fprintf(directory[0][pl+1], "%6d  %.18e  %.10e  %.10e\n", transitcount[0][pl+1], t2, dist, vtrans);
#endif

          double vplanet, toffset, tinit, tfin;
          long ic, nc, ac;
          double thone0, hold;

          vplanet = fmax(vtrans, 0.0001); // can't divide by 0, so set min value. 
          toffset = offsetmult*(rstar*(1+rad[pl]))*RSUNAU/vplanet;
          toffset = fmin(toffset, offsetmin);
          tinit = t2 - toffset;
          tfin = t2 + toffset;
          ic=0;
          while (ic<(kk-1) && timelist[ic]<tinit) {
            ic++;
          } 
          nc=0;
          while ((ic+nc)<(kk-1) && timelist[ic+nc]<tfin) {
            nc++;
          }
          seteq(npl, yhone0, yhone);
          thone0=thone;
          hold=h;

          for (ac=0; ac<nc; ac++) {

            if (whereplanets[0][ic+ac][pl+1] == 0) { // i.e. if this body's location not already computed
 
              int tnum;
              double tcur, ts1, ts2;

              tnum = numplanets[0][ic+ac];
              // stellar time 
              tcur = timelist[ic+ac] - zs;
              ts1 = tcur-thone;
              if (fabs(ts1) > fabs(h)) {
                h = fabs(h)*fabs(ts1)/ts1;
              } else {
                h = ts1;
              }
              while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur, thone)) ){
                status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thone, tcur, &h, yhone); 
              }
              if (LTE) {
                ts2 = (tcur+dt)-thonedt;
                if (fabs(ts2) > fabs(h)) {
                  h = fabs(h)*fabs(ts2)/ts2;
                } else {
                  h = ts2;
                }
                while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur+dt, thonedt)) ){
                  status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thonedt, tcur+dt, &h, yhonedt); 
                }
              }
              if (status != GSL_SUCCESS) {
                printf("Big problem\n");
                exit(0);
              }
  
              if (LTE) {
                for (j=0; j<6; j++) {
                  baryc[j] = 0.;
                  baryc2[j] = 0.;
                  for (i=0; i<npl; i++) {
                    baryc[j] += yhone[j+i*6]*mu[i+1];
                    baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                  }
                  baryc[j] /= mtot;
                  baryc2[j] /= mtot;
     
                  starimage[j] = baryc2[j]-baryc[j];
                }
     
                zs = 0.0;
                zp = yhonedt[2+pl*6]/CAUPD;
                zb = baryc[2]/CAUPD;
                zb2 = baryc2[2]/CAUPD;
                zs -= zb;
                zp -= zb2;
                dt1 = zs;
                dt2 = -zp;
                dt = dt1+dt2;
              }
 
              if (yhone[2+pl*6] < 0.0) {

                whereplanets[0][ic+ac][pl+1] = 1;
                tranarray[0][ic+ac][tnum][0] = rad[pl]*rstar*RSUNAU; 
                numplanets[0][ic+ac]+=1;

                if (LTE) {

                  tranarray[0][ic+ac][tnum][1] = yhonedt[0+pl*6]-starimage[0];
                  tranarray[0][ic+ac][tnum][2] = yhonedt[1+pl*6]-starimage[1];

                } else {

                  tranarray[0][ic+ac][tnum][1] = yhone[0+pl*6];
                  tranarray[0][ic+ac][tnum][2] = yhone[1+pl*6];

                }
              } 
            }
          }
          seteq(npl, yhone, yhone0);
          thone=thone0;
          h=hold;
        } 
      }
    }
    int st;
    int st1;
    for (st1=1; st1<brights; st1++) {
      st = brightsindex[st1];
      int stm1 = st-1;
      dpold[st1][stm1]=dp[st1][stm1];
      dp[st1][stm1]=y[0+stm1*6]*y[3+stm1*6]+y[1+stm1*6]*y[4+stm1*6];
      ps2=y[0+stm1*6]*y[0+stm1*6]+y[1+stm1*6]*y[1+stm1*6];
 
      if( dp[st1][stm1]*dpold[st1][stm1] <= 0 && ps2 < dist2/dist2divisor && y[2+stm1*6]>0 && brights > 1) { 

        seteq(npl, yhone, y);
        thone = t;

        if (LTE) {
          ii=0;
          dt = yhone[2+stm1*6] / CAUPD;
          seteq(npl, yhonedt, yhone);
          gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
          thonedt = thone+dt;
 
          do {
            func(thone, yhone, dydt, npl_mu);
            func(thonedt, yhonedt, dydtdt, npl_mu);
 
            for (j=0; j<6; j++) {
              baryc[j] = 0.;
              baryc2[j] = 0.;
              dbarycdt[j] = 0.;
              dbarycdt2[j] = 0.;
              for (i=0; i<npl; i++) {
                baryc[j] += yhone[j+i*6]*mu[i+1];
                baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                dbarycdt[j] += dydt[j+i*6]*mu[i+1];
                dbarycdt2[j] += dydtdt[j+i*6]*mu[i+1];
              }
              baryc[j] /= mtot;
              baryc2[j] /= mtot;
              dbarycdt[j] /= mtot;
              dbarycdt2[j] /= mtot;
 
              starimage[j] = -baryc2[j]+baryc[j];
              dstarimagedt[j] = -dbarycdt2[j]+dbarycdt[j];
            }
 
            dp[st1][stm1] = (yhone[0+stm1*6]-starimage[0])*(yhone[3+stm1*6]-starimage[3]) + (yhone[1+stm1*6]-starimage[1])*(yhone[4+stm1*6]-starimage[4]);
            double ddpdt = (dydt[0+stm1*6]-dstarimagedt[0])*(dydt[0+stm1*6]-dstarimagedt[0]) + (dydt[1+stm1*6]-dstarimagedt[1])*(dydt[1+stm1*6]-dstarimagedt[1]) + (yhone[0+stm1*6]-starimage[0])*(dydt[3+stm1*6]-dstarimagedt[3]) + (yhone[1+stm1*6]-starimage[1])*(dydt[4+stm1*6]-dstarimagedt[4]);
 
 
            zs = yhone[2+stm1*6]/CAUPD;
            zp = 0.0;
            zb = baryc[2]/CAUPD;
            zb2 = baryc2[2]/CAUPD;
            zs -= zb;
            zp -= zb2;
            dt1 = zs;
            dt2 = -zp;
            dt = dt1+dt2;
            
            tstep = - dp[st1][stm1]/ddpdt;
            gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, NULL, NULL, &sys);
            seteq(npl, yhonedt, yhone);
            gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
            thone += tstep;
            thonedt = thone+dt;
            ii++;
 
          } while (ii<10 && fabs(tstep)>eps);

          dist = sqrt( pow( (yhone[0+stm1*6]-starimage[0]), 2) + pow( (yhone[1+stm1*6]-starimage[1]), 2) ); 
 
        } else {

          ii=0;
          do {
            func (thone, yhone, dydt, npl_mu);
            ddpdt = dydt[0+stm1*6]*dydt[0+stm1*6] + dydt[1+stm1*6]*dydt[1+stm1*6] + yhone[0+stm1*6]*dydt[3+stm1*6] + yhone[1+stm1*6]*dydt[4+stm1*6];
            tstep = - dp[st1][stm1] / ddpdt;
            gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, dydt, NULL, &sys);
            thone += tstep;
            dp[st1][stm1]=yhone[0+stm1*6]*yhone[3+stm1*6]+yhone[1+stm1*6]*yhone[4+stm1*6];
            ii++;
          } while (ii<5 && fabs(tstep)>eps);
          
          dist = sqrt(pow(yhone[0+stm1*6],2)+pow(yhone[1+stm1*6],2));

        }

        dp[st1][stm1]=y[0+stm1*6]*y[3+stm1*6]+y[1+stm1*6]*y[4+stm1*6];

        if (dist < 2.0*((rstar + rstar*rad[stm1])*RSUNAU)) {

          transitcount[st1][0] += 1;  // Update the transit number. 

          if (LTE) {
            for (j=0; j<6; j++) {
              baryc[j] = 0.;
              baryc2[j] = 0.;
              for (i=0; i<npl; i++) {
                baryc[j] += yhone[j+i*6]*mu[i+1];
                baryc2[j] += yhonedt[j+i*6]*mu[i+1];
              }
              baryc[j] /= mtot;
              baryc2[j] /= mtot;
 
              starimage[j] = -baryc2[j]+baryc[j];
            }
            zs = yhone[2+stm1*6]/CAUPD;
            zp = 0.0;
            zb = baryc[2]/CAUPD;
            zb2 = baryc2[2]/CAUPD;
            zs -= zb;
            zp -= zb2;
            dt1 = zs;
            dt2 = -zp;
            dt = dt1+dt2;
            
            dist = sqrt( pow( yhone[0+stm1*6] - starimage[0], 2) + pow( yhone[1+stm1*6] - starimage[1], 2) ); 
            vtrans = sqrt( pow( yhone[3+stm1*6] - starimage[3], 2) + pow( yhone[4+stm1*6] - starimage[4], 2) ); 
          } else {
            dist = sqrt(pow(yhone[0+stm1*6],2)+pow(yhone[1+stm1*6],2));
            vtrans = sqrt(pow(yhone[3+stm1*6],2)+pow(yhone[4+stm1*6],2)); 
            zs=0;
            zb2=0;
          }

          t2 = thone + zs;

#if ( demcmc_compile == 0 )
          fprintf(directory[st1][0], "%6d  %.18e  %.10e  %.10e\n", transitcount[st1][0], t2, dist, vtrans);
#endif

          double vplanet, toffset, tinit, tfin;
          long ic, nc, ac;
          double thone0, hold;

          vplanet = fmax(vtrans, 0.0001);
          toffset = offsetmult*(rstar*(1+rad[stm1]))*RSUNAU/vplanet;
          toffset = fmin(toffset, offsetmin);
          tinit = t2 - toffset;
          tfin = t2 + toffset;
          ic=0;
          while (ic<(kk-1) && timelist[ic]<tinit) {
            ic++;
          } 
          nc=0;
          while ((ic+nc)<(kk-1) && timelist[ic+nc]<tfin) {
            nc++;
          }
          seteq(npl, yhone0, yhone);
          thone0=thone;
          hold=h;

          for (ac=0; ac<nc; ac++) {

            if (whereplanets[st1][ic+ac][0] == 0) { // i.e. if this body's location not already computed
 
              int tnum;
              double tcur, ts1, ts2;

              tnum = numplanets[st1][ic+ac];
              // stellar time 
              tcur = timelist[ic+ac] - zs;
              ts1 = (tcur-thone);
              if (fabs(ts1) > fabs(h)) {
                h = fabs(h)*fabs(ts1)/ts1;
              } else {
                h = ts1;
              }
              while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur, thone)) ){
                status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thone, tcur, &h, yhone); 
              }
              if (LTE) {
                ts2 = (tcur+dt)-thonedt;
                if (fabs(ts2) > fabs(h)) {
                  h = fabs(h)*fabs(ts2)/ts2;
                } else {
                  h = ts2;
                }
                while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur+dt, thonedt)) ){
                  status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thonedt, tcur+dt, &h, yhonedt); 
                }
              }
              if (status != GSL_SUCCESS) {
                printf("Big problem\n");
                exit(0);
              }
  
              if (LTE) {
                for (j=0; j<6; j++) {
                  baryc[j] = 0.;
                  baryc2[j] = 0.;
                  for (i=0; i<npl; i++) {
                    baryc[j] += yhone[j+i*6]*mu[i+1];
                    baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                  }
                  baryc[j] /= mtot;
                  baryc2[j] /= mtot;
     
                  starimage[j] = -baryc2[j]+baryc[j];
                }
                zs = yhone[2+stm1*6]/CAUPD;
                zp = 0.0;
                zb = baryc[2]/CAUPD;
                zb2 = baryc2[2]/CAUPD;
                zs -= zb;
                zp -= zb2;
                dt1 = zs;
                dt2 = -zp;
                dt = dt1+dt2;
              }
 
              if (yhone[2+stm1*6] > 0.0) {

                whereplanets[st1][ic+ac][0] = 1;
                tranarray[st1][ic+ac][tnum][0] = rstar*RSUNAU; 
                numplanets[st1][ic+ac]+=1;

                if (LTE) {

                  tranarray[st1][ic+ac][tnum][1] = -(yhone[0+stm1*6]-starimage[0]);
                  tranarray[st1][ic+ac][tnum][2] = -(yhone[1+stm1*6]-starimage[1]);

                } else {

                  tranarray[st1][ic+ac][tnum][1] = -yhone[0+stm1*6];
                  tranarray[st1][ic+ac][tnum][2] = -yhone[1+stm1*6];

                }
              } 
            }
          }
          seteq(npl, yhone, yhone0);
          thone=thone0;
          h=hold;
        } 
      }
      for (pl=0; pl<npl; pl++) {
        if (pl != stm1) {
             dpold[st1][pl]=dp[st1][pl];
          dp[st1][pl]=(y[0+pl*6]-y[0+st1*6])*(y[3+pl*6]-y[3+st1*6]) + (y[1+pl*6]-y[1+st1*6])*(y[4+pl*6]-y[4+st1*pl]);
          ps2=pow((y[0+pl*6]-y[0+st1*6]),2) + pow((y[1+pl*6]-y[1+st1*6]),2);

          if (dp[st1][pl]*dpold[st1][pl] <= 0 && ps2 < dist2/dist2divisor && y[2+st1*6]>y[2+pl*6] && brights > 1) { 
    
                seteq(npl, yhone, y);
                thone = t;
    
            if (LTE) {
              ii=0;
              dt = -(yhone[2+pl*6] - yhone[2+st1*6]) / CAUPD;
              seteq(npl, yhonedt, yhone);
              gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
              thonedt = thone+dt;
     
              do {
                func(thone, yhone, dydt, npl_mu);
                func(thonedt, yhonedt, dydtdt, npl_mu);
     
                for (j=0; j<6; j++) {
                  baryc[j] = 0.;
                  baryc2[j] = 0.;
                  dbarycdt[j] = 0.;
                  dbarycdt2[j] = 0.;
                  for (i=0; i<npl; i++) {
                    baryc[j] += yhone[j+i*6]*mu[i+1];
                    baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                    dbarycdt[j] += dydt[j+i*6]*mu[i+1];
                    dbarycdt2[j] += dydtdt[j+i*6]*mu[i+1];
                  }
                  baryc[j] /= mtot;
                  baryc2[j] /= mtot;
                  dbarycdt[j] /= mtot;
                  dbarycdt2[j] /= mtot;
    
                  //starimage should really be starshift here
                  starimage[j] = baryc2[j]-baryc[j];
                  dstarimagedt[j] = dbarycdt2[j]-dbarycdt[j];
                }
    
                dp[st1][pl] = (yhonedt[0+pl*6]-yhone[0+st1*6]-starimage[0])*(yhonedt[3+pl*6]-yhone[3+st1*6]-starimage[3]) + (yhonedt[1+pl*6]-yhone[1+st1*6]-starimage[1])*(yhonedt[4+pl*6]-yhone[4+st1*6]-starimage[4]);
                double ddpdt = (dydtdt[0+pl*6]-dydt[0+st1*6]-dstarimagedt[0])*(dydtdt[0+pl*6]-dydt[0+st1*6]-dstarimagedt[0]) + (dydtdt[1+pl*6]-dydt[1+st1*6]-dstarimagedt[1])*(dydtdt[1+pl*6]-dydt[1+st1*6]-dstarimagedt[1]) + (yhonedt[0+pl*6]-yhone[0+st1*6]-starimage[0])*(dydtdt[3+pl*6]-dydt[3+st1*6]-dstarimagedt[3]) + (yhonedt[1+pl*6]-yhone[1+st1*6]-starimage[1])*(dydt[4+pl*6]-dydt[4+st1*6]-dstarimagedt[4]);
     
     
                zs = yhone[2+st1*6]/CAUPD;
                zp = yhonedt[2+pl*6]/CAUPD;
                zb = baryc[2]/CAUPD;
                zb2 = baryc2[2]/CAUPD;
                zs -= zb;
                zp -= zb2;
                dt1 = zs;
                dt2 = -zp;
                dt = dt1+dt2;
                
                tstep = - dp[st1][pl]/ddpdt;
                gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, NULL, NULL, &sys);
                seteq(npl, yhonedt, yhone);
                gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
                thone += tstep;
                thonedt = thone+dt;
                ii++;
     
              } while (ii<10 && fabs(tstep)>eps);
                
              dist = sqrt( pow( (yhonedt[0+pl*6]-yhone[0+st1*6]-starimage[0]), 2) + pow( (yhonedt[1+pl*6]-yhone[1+st1*6]-starimage[1]), 2) ); 
     
            } else {
    
              ii=0;
              do {
                func (thone, yhone, dydt, npl_mu);
                ddpdt = (dydt[0+pl*6]-dydt[0+stm1*6])*(dydt[0+pl*6]-dydt[0+stm1*6]) + (dydt[1+pl*6]-dydt[1+stm1*6])*(dydt[1+pl*6]-dydt[1+stm1*6]) + (yhone[0+pl*6]-yhone[0+stm1*6])*(dydt[3+pl*6]-dydt[3+stm1*6]) + (yhone[1+pl*6]-yhone[1+stm1*6])*(dydt[4+pl*6]-dydt[4+stm1*6]);
                tstep = - dp[st1][pl] / ddpdt;
                gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, dydt, NULL, &sys);
                thone += tstep;
                dp[st1][pl]=(yhone[0+pl*6]-yhone[0+stm1*6])*(yhone[3+pl*6]-yhone[3+stm1*6]) + (yhone[1+pl*6]-yhone[1+stm1*6])*(yhone[4+pl*6]-yhone[4+stm1*6]);
                ii++;
              } while (ii<5 && fabs(tstep)>eps);
                
              dist = sqrt( pow( (yhone[0+pl*6]-yhone[0+stm1*6]), 2) + pow( (yhone[1+pl*6]-yhone[1+stm1*6]), 2) ); 
    
            }
    
            dp[st1][pl]=(y[0+pl*6]-y[0+stm1*6])*(y[3+pl*6]-y[3+stm1*6]) + (y[1+pl*6]-y[1+stm1*6])*(y[4+pl*6]-y[4+stm1*6]);
    
            if (dist < 2.0*((rstar*rad[st1] + rstar*rad[pl])*RSUNAU)) {
    
              transitcount[st1][pl+1] += 1;  // Update the transit number. 
    
              if (LTE) {
                for (j=0; j<6; j++) {
                  baryc[j] = 0.;
                  baryc2[j] = 0.;
                  for (i=0; i<npl; i++) {
                    baryc[j] += yhone[j+i*6]*mu[i+1];
                    baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                  }
                  baryc[j] /= mtot;
                  baryc2[j] /= mtot;
     
                  starimage[j] = baryc2[j]-baryc[j];
                }
                zs = yhone[2+st1*6]/CAUPD;
                zp = yhonedt[2+pl*6]/CAUPD;
                zb = baryc[2]/CAUPD;
                zb2 = baryc2[2]/CAUPD;
                zs -= zb;
                zp -= zb2;
                dt1 = zs;
                dt2 = -zp;
                dt = dt1+dt2;
                
                dist = sqrt( pow( (yhonedt[0+pl*6]-yhone[0+st1*6]-starimage[0]), 2) + pow( (yhonedt[1+pl*6]-yhone[1+st1*6]-starimage[1]), 2) ); 
                vtrans = sqrt(pow(yhonedt[3+pl*6]-yhone[3+st1*6] - starimage[3],2) + pow(yhonedt[4+pl*6]-yhone[4+st1*6] - starimage[4],2)); 
              } else {
                dist = sqrt( pow( (yhone[0+pl*6]-yhone[0+stm1*6]), 2) + pow( (yhone[1+pl*6]-yhone[1+stm1*6]), 2) ); 
                vtrans = sqrt(pow(yhone[3+pl*6]-yhone[3+stm1*6], 2) + pow(yhone[4+pl*6]-yhone[4+stm1*6], 2)); 
                //dist = sqrt( pow( (yhonedt[0+pl*6]-yhone[0+st1*6]), 2) + pow( (yhonedt[1+pl*6]-yhone[1+st1*6]), 2) ); 
                //vtrans = sqrt(pow(yhonedt[3+pl*6]-yhone[3+st1*6], 2) + pow(yhonedt[4+pl*6]-yhone[4+st1*6], 2)); 
                zs=0;
                zb2=0;
              }
    
              t2 = thone + zs;
    
#if ( demcmc_compile == 0 )//|| demcmc_compile == 3) 
              fprintf(directory[st1][pl+1], "%6d  %.18e  %.10e  %.10e\n", transitcount[st1][pl+1], t2, dist, vtrans);
#endif


              double vplanet, toffset, tinit, tfin;
              long ic, nc, ac;
              double thone0, hold;

              vplanet = fmax(vtrans, 0.0001);
              toffset = offsetmult*(rstar*(rad[st1]+rad[pl]))*RSUNAU/vplanet;
              toffset = fmin(toffset, offsetminout);
              tinit = t2 - toffset;
              tfin = t2 + toffset;
              ic=0;
              while (ic<(kk-1) && timelist[ic]<tinit) ic++; 
              nc=0;
              while ((ic+nc)<(kk-1) && timelist[ic+nc]<tfin) nc++;

              seteq(npl, yhone0, yhone);
              thone0=thone;
              hold=h;

              //This is fairly inexact for first time if yhone_Z is changing considerably Can try to fix this in future versions. 
              for (ac=0; ac<nc; ac++) {
                if (whereplanets[st1][ic+ac][pl+1] == 0) { // i.e. if this body's location not already computed

                  int tnum;
                  double tcur, ts1, ts2;

                  tnum = numplanets[st1][ic+ac];
                  tcur = timelist[ic+ac] - zs;
                  ts1 = (tcur-thone);
                  if (fabs(ts1) > fabs(h)) {
                    h = fabs(h)*fabs(ts1)/ts1;
                  } else {
                    h = ts1;
                  }
                  while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur, thone)) ){
                    status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thone, tcur, &h, yhone); 
                  }
                  if (LTE) {
                    ts2 = (tcur+dt)-thonedt;
                    if (fabs(ts2) > fabs(h)) {
                      h = fabs(h)*fabs(ts2)/ts2;
                    } else {
                      h = ts2;
                    }
                    while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur+dt, thonedt)) ){
                      status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thonedt, tcur+dt, &h, yhonedt); 
                    }
                  }
                  if (status != GSL_SUCCESS) {
                    printf("Big problem\n");
                    exit(0);
                  }
      
                  if (LTE) {
                    for (j=0; j<6; j++) {
                      baryc[j] = 0.;
                      baryc2[j] = 0.;
                      for (i=0; i<npl; i++) {
                        baryc[j] += yhone[j+i*6]*mu[i+1];
                        baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                      }
                      baryc[j] /= mtot;
                      baryc2[j] /= mtot;
         
                      starimage[j] = baryc2[j]-baryc[j];
                    }
         
                    zs = yhone[2+st1*6]/CAUPD;
                    zp = yhonedt[2+pl*6]/CAUPD;
                    zb = baryc[2]/CAUPD;
                    zb2 = baryc2[2]/CAUPD;
                    zs -= zb;
                    zp -= zb2;
                    dt1 = zs;
                    dt2 = -zp;
                    dt = dt1+dt2;
                  }
 
                  if (yhone[2+st1*6] > yhone[2+pl*6]) {

                    whereplanets[st1][ic+ac][pl+1] = 1;
                    numplanets[st1][ic+ac]+=1;
                    tranarray[st1][ic+ac][tnum][0] = rad[pl]*rstar*RSUNAU;

                    if (LTE) {

                      tranarray[st1][ic+ac][tnum][1] = yhonedt[0+pl*6] - yhone[0+st1*6] - starimage[0];
                      tranarray[st1][ic+ac][tnum][2] = yhonedt[1+pl*6] - yhone[1+st1*6] - starimage[1];
                    
                    } else {
                    
                      tranarray[st1][ic+ac][tnum][1] = yhone[0+pl*6] - yhone[0+st1*6];
                      tranarray[st1][ic+ac][tnum][2] = yhone[1+pl*6] - yhone[1+st1*6];

                    }
                  }
                } 
              }

              seteq(npl, yhone, yhone0);
              thone=thone0;
              h=hold;
            }
          }
        }
      }
    }
  }




  seteq(npl, y, yin);
  for(pl=0; pl<npl; pl++) dp[0][pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6];
  for (i=1; i<brights; i++) {
      for(pl=0; pl<npl; pl++) {
          dp[i][pl]=(y[0+pl*6]-y[0+(i-1)*6])*y[3+pl*6] + (y[1+pl*6]-y[1+(i-1)*6])*y[4+pl*6];
      }
  }
  t = tstart;
  h = -HStart;
  for (k=0; k<brights; k++) {
    for(pl=0; pl<(npl+1); pl++) transitcount[k][pl] = 0;
  }
    

  if (RVS) {
    rvcount = startrvcount-1;
  }



  while (t > t0+1) {

    if (RVS) {
      onrv=0;
      if ( (h + t) < rvtarr[rvcount]) {  
        onrv = 1;
        hhere = h;
        h = rvtarr[rvcount] - t;
      }
    }

    int status = gsl_odeiv_evolve_apply (e, c, s,
                                         &sys, 
                                         &t, t0,
                                         &h, y);

    if (status != GSL_SUCCESS)
        break;

    if (RVS) {
       if (onrv ==1) {
         h = hhere;
         func(t, y, dydt, mu);
         double vstar = 0;
         for (i=0; i<npl; i++) {
           vstar += -y[i*6+5]*mu[i+1]/mtot;
         }
         if (tve[3][rvcount] == 0) {
           rvarr[rvcount] = vstar; 
         } else {
           rvarr[rvcount] = vstar + y[(((int) tve[3][rvcount])-1)*6+5];
         }
         rvcount += 1;
       }
    }

    dist2 = pow(y[0],2)+pow(y[1],2)+pow(y[2],2);

    for(pl=0; pl<npl; pl++) {  /* Cycle through the planets, searching for a transit. */
        dpold[0][pl]=dp[0][pl];
        dp[0][pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6];
        ps2=y[0+pl*6]*y[0+pl*6]+y[1+pl*6]*y[1+pl*6];

      if( dp[0][pl]*dpold[0][pl] <= 0 && ps2 < dist2/dist2divisor && y[2+pl*6]<0 && brights>1) { 
                     // A minimum projected-separation occurred: "Tc".  
                //   ps2 constraint makes sure it's when y^2+z^2 is small-- its on the face of the star.  
                  //   y[2] constraint means a primary eclipse, presuming that the observer is on the positive x=y[2] axis.

        seteq(npl, yhone, y);
        thone = t;

        if (LTE) {
          ii=0;
          dt = -yhone[2+pl*6] / CAUPD;
          seteq(npl, yhonedt, yhone);
          gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
          thonedt = thone+dt;
 
          do {
            func(thone, yhone, dydt, npl_mu);
            func(thonedt, yhonedt, dydtdt, npl_mu);
 
            for (j=0; j<6; j++) {
              baryc[j] = 0.;
              baryc2[j] = 0.;
              dbarycdt[j] = 0.;
              dbarycdt2[j] = 0.;
              for (i=0; i<npl; i++) {
                baryc[j] += yhone[j+i*6]*mu[i+1];
                baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                dbarycdt[j] += dydt[j+i*6]*mu[i+1];
                dbarycdt2[j] += dydtdt[j+i*6]*mu[i+1];
              }
              baryc[j] /= mtot;
              baryc2[j] /= mtot;
              dbarycdt[j] /= mtot;
              dbarycdt2[j] /= mtot;
 
              starimage[j] = baryc2[j]-baryc[j];
              dstarimagedt[j] = dbarycdt2[j]-dbarycdt[j];
            }
 
            dp[0][pl] = (yhonedt[0+pl*6]-starimage[0])*(yhonedt[3+pl*6]-starimage[3]) + (yhonedt[1+pl*6]-starimage[1])*(yhonedt[4+pl*6]-starimage[4]);
            double ddpdt = (dydtdt[0+pl*6]-dstarimagedt[0])*(dydtdt[0+pl*6]-dstarimagedt[0]) + (dydtdt[1+pl*6]-dstarimagedt[1])*(dydtdt[1+pl*6]-dstarimagedt[1]) + (yhonedt[0+pl*6]-starimage[0])*(dydtdt[3+pl*6]-dstarimagedt[3]) + (yhonedt[1+pl*6]-starimage[1])*(dydtdt[4+pl*6]-dstarimagedt[4]);
 
 
            zs = 0.0;
            zp = yhonedt[2+pl*6]/CAUPD;
            zb = baryc[2]/CAUPD;
            zb2 = baryc2[2]/CAUPD;
            zs -= zb;
            zp -= zb2;
            dt1 = zs;
            dt2 = -zp;
            dt = dt1+dt2;
            
            tstep = - dp[0][pl]/ddpdt;
            gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, NULL, NULL, &sys);
            seteq(npl, yhonedt, yhone);
            gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
            thone += tstep;
            thonedt = thone+dt;
            ii++;
 
          } while (ii<10 && fabs(tstep)>eps);

          dist = sqrt( pow( (yhonedt[0+pl*6]-starimage[0]), 2) + pow( (yhonedt[1+pl*6]-starimage[1]), 2) ); 
 
        } else {

          ii=0;
          do {
            func (thone, yhone, dydt, npl_mu);
            ddpdt = dydt[0+pl*6]*dydt[0+pl*6] + dydt[1+pl*6]*dydt[1+pl*6] + yhone[0+pl*6]*dydt[3+pl*6] + yhone[1+pl*6]*dydt[4+pl*6];
            tstep = - dp[0][pl] / ddpdt;
            gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, dydt, NULL, &sys);
            thone += tstep;
            dp[0][pl]=yhone[0+pl*6]*yhone[3+pl*6]+yhone[1+pl*6]*yhone[4+pl*6];
            ii++;
          } while (ii<5 && fabs(tstep)>eps);
            
          dist = sqrt(pow(yhone[0+pl*6],2)+pow(yhone[1+pl*6],2));

        }

        dp[0][pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6];

        if (dist < 2.0*((rstar + rstar*rad[pl])*RSUNAU)) {

          transitcount[0][pl+1] -= 1;  // Update the transit number. 

          if (LTE) {
            for (j=0; j<6; j++) {
              baryc[j] = 0.;
              baryc2[j] = 0.;
              for (i=0; i<npl; i++) {
                baryc[j] += yhone[j+i*6]*mu[i+1];
                baryc2[j] += yhonedt[j+i*6]*mu[i+1];
              }
              baryc[j] /= mtot;
              baryc2[j] /= mtot;
 
              starimage[j] = baryc2[j]-baryc[j];
            }
 
            zs = 0.0;
            zp = yhonedt[2+pl*6]/CAUPD;
            zb = baryc[2]/CAUPD;
            zb2 = baryc2[2]/CAUPD;
            zs -= zb;
            zp -= zb2;
            dt1 = zs;
            dt2 = -zp;
            dt = dt1+dt2;
 
            dist = sqrt( pow( yhonedt[0+pl*6] - starimage[0], 2) + pow( yhonedt[1+pl*6] - starimage[1], 2) ); 
            vtrans = sqrt( pow( yhonedt[3+pl*6] - starimage[3], 2) + pow( yhonedt[4+pl*6] - starimage[4], 2) ); 
          } else {
            dist = sqrt(pow(yhone[0+pl*6],2)+pow(yhone[1+pl*6],2));
            vtrans = sqrt(pow(yhone[3+pl*6],2)+pow(yhone[4+pl*6],2)); 
            zs=0;
            zb2=0;
          }

          t2 =  thone + zs;

#if ( demcmc_compile == 0 )//|| demcmc_compile == 3) 
          fprintf(directory[0][pl+1], "%6d  %.18e  %.10e  %.10e\n", transitcount[0][pl+1], t2, dist, vtrans);
#endif

          double vplanet, toffset, tinit, tfin;
          long ic, nc, ac;
          double thone0, hold;

          vplanet = fmax(vtrans, 0.0001); // can't divide by 0, so set min value. 
          toffset = offsetmult*(rstar*(1+rad[pl]))*RSUNAU/vplanet;
          toffset = fmin(toffset, offsetmin);
          tinit = t2 - toffset;
          tfin = t2 + toffset;
          ic=0;
          while (ic<(kk-1) && timelist[ic]<tinit) {
            ic++;
          } 
          nc=0;
          while ((ic+nc)<(kk-1) && timelist[ic+nc]<tfin) {
            nc++;
          }
          seteq(npl, yhone0, yhone);
          thone0=thone;
          hold=h;

          for (ac=0; ac<nc; ac++) {

            if (whereplanets[0][ic+ac][pl+1] == 0) { // i.e. if this body's location not already computed
 
              int tnum;
              double tcur, ts1, ts2;

              tnum = numplanets[0][ic+ac];
              // stellar time 
              tcur = timelist[ic+ac] - zs;
              ts1 = (tcur-thone);
              if (fabs(ts1) > fabs(h)) {
                h = fabs(h)*fabs(ts1)/ts1;
              } else {
                h = ts1;
              }
              while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur, thone)) ){
                status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thone, tcur, &h, yhone); 
              }
              if (LTE) {
                ts2 = (tcur+dt)-thonedt;
                if (fabs(ts2) > fabs(h)) {
                  h = fabs(h)*fabs(ts2)/ts2;
                } else {
                  h = ts2;
                }
                while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur+dt, thonedt)) ){
                  status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thonedt, tcur+dt, &h, yhonedt); 
                }
              }
              if (status != GSL_SUCCESS) {
                printf("Big problem\n");
                exit(0);
              }
  
              if (LTE) {
                for (j=0; j<6; j++) {
                  baryc[j] = 0.;
                  baryc2[j] = 0.;
                  for (i=0; i<npl; i++) {
                    baryc[j] += yhone[j+i*6]*mu[i+1];
                    baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                  }
                  baryc[j] /= mtot;
                  baryc2[j] /= mtot;
     
                  starimage[j] = baryc2[j]-baryc[j];
                }
     
                zs = 0.0;
                zp = yhonedt[2+pl*6]/CAUPD;
                zb = baryc[2]/CAUPD;
                zb2 = baryc2[2]/CAUPD;
                zs -= zb;
                zp -= zb2;
                dt1 = zs;
                dt2 = -zp;
                dt = dt1+dt2;
              }
 
              if (yhone[2+pl*6] < 0.0) {

                whereplanets[0][ic+ac][pl+1] = 1;
                tranarray[0][ic+ac][tnum][0] = rad[pl]*rstar*RSUNAU; 
                numplanets[0][ic+ac]+=1;

                if (LTE) {

                  tranarray[0][ic+ac][tnum][1] = yhonedt[0+pl*6]-starimage[0];
                  tranarray[0][ic+ac][tnum][2] = yhonedt[1+pl*6]-starimage[1];

                } else {

                  tranarray[0][ic+ac][tnum][1] = yhone[0+pl*6];
                  tranarray[0][ic+ac][tnum][2] = yhone[1+pl*6];

                }
              } 
            }
          }
          seteq(npl, yhone, yhone0);
          thone=thone0;
          h=hold;
        } 
      }
    }
    int st;
    int st1;
    for (st1=1; st1<brights; st1++) {
      st = brightsindex[st1];
      int stm1 = st-1;
      dpold[st1][stm1]=dp[st1][stm1];
      dp[st1][stm1]=y[0+stm1*6]*y[3+stm1*6]+y[1+stm1*6]*y[4+stm1*6];
      ps2=y[0+stm1*6]*y[0+stm1*6]+y[1+stm1*6]*y[1+stm1*6];
 
      if( dp[st1][stm1]*dpold[st1][stm1] <= 0 && ps2 < dist2/dist2divisor && y[2+stm1*6]>0 && brights > 1) { 

        seteq(npl, yhone, y);
        thone = t;

        if (LTE) {
          ii=0;
          dt = yhone[2+stm1*6] / CAUPD;
          seteq(npl, yhonedt, yhone);
          gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
          thonedt = thone+dt;
 
          do {
            func(thone, yhone, dydt, npl_mu);
            func(thonedt, yhonedt, dydtdt, npl_mu);
 
            for (j=0; j<6; j++) {
              baryc[j] = 0.;
              baryc2[j] = 0.;
              dbarycdt[j] = 0.;
              dbarycdt2[j] = 0.;
              for (i=0; i<npl; i++) {
                baryc[j] += yhone[j+i*6]*mu[i+1];
                baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                dbarycdt[j] += dydt[j+i*6]*mu[i+1];
                dbarycdt2[j] += dydtdt[j+i*6]*mu[i+1];
              }
              baryc[j] /= mtot;
              baryc2[j] /= mtot;
              dbarycdt[j] /= mtot;
              dbarycdt2[j] /= mtot;
 
              starimage[j] = -baryc2[j]+baryc[j];
              dstarimagedt[j] = -dbarycdt2[j]+dbarycdt[j];
            }
 
            dp[st1][stm1] = (yhone[0+stm1*6]-starimage[0])*(yhone[3+stm1*6]-starimage[3]) + (yhone[1+stm1*6]-starimage[1])*(yhone[4+stm1*6]-starimage[4]);
            double ddpdt = (dydt[0+stm1*6]-dstarimagedt[0])*(dydt[0+stm1*6]-dstarimagedt[0]) + (dydt[1+stm1*6]-dstarimagedt[1])*(dydt[1+stm1*6]-dstarimagedt[1]) + (yhone[0+stm1*6]-starimage[0])*(dydt[3+stm1*6]-dstarimagedt[3]) + (yhone[1+stm1*6]-starimage[1])*(dydt[4+stm1*6]-dstarimagedt[4]);
 
 
            zs = yhone[2+stm1*6]/CAUPD;
            zp = 0.0;
            zb = baryc[2]/CAUPD;
            zb2 = baryc2[2]/CAUPD;
            zs -= zb;
            zp -= zb2;
            dt1 = zs;
            dt2 = -zp;
            dt = dt1+dt2;
            
            tstep = - dp[st1][stm1]/ddpdt;
            gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, NULL, NULL, &sys);
            seteq(npl, yhonedt, yhone);
            gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
            thone += tstep;
            thonedt = thone+dt;
            ii++;
 
          } while (ii<10 && fabs(tstep)>eps);

          dist = sqrt( pow( (yhone[0+stm1*6]-starimage[0]), 2) + pow( (yhone[1+stm1*6]-starimage[1]), 2) ); 
 
        } else {

          ii=0;
          do {
            func (thone, yhone, dydt, npl_mu);
            ddpdt = dydt[0+stm1*6]*dydt[0+stm1*6] + dydt[1+stm1*6]*dydt[1+stm1*6] + yhone[0+stm1*6]*dydt[3+stm1*6] + yhone[1+stm1*6]*dydt[4+stm1*6];
            tstep = - dp[st1][stm1] / ddpdt;
            gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, dydt, NULL, &sys);
            thone += tstep;
            dp[st1][stm1]=yhone[0+stm1*6]*yhone[3+stm1*6]+yhone[1+stm1*6]*yhone[4+stm1*6];
            ii++;
          } while (ii<5 && fabs(tstep)>eps);
          
          dist = sqrt(pow(yhone[0+stm1*6],2)+pow(yhone[1+stm1*6],2));

        }

        dp[st1][stm1]=y[0+stm1*6]*y[3+stm1*6]+y[1+stm1*6]*y[4+stm1*6];

        if (dist < 2.0*((rstar + rstar*rad[stm1])*RSUNAU)) {

          transitcount[st1][0] -= 1;  // Update the transit number. 

          if (LTE) {
            for (j=0; j<6; j++) {
              baryc[j] = 0.;
              baryc2[j] = 0.;
              for (i=0; i<npl; i++) {
                baryc[j] += yhone[j+i*6]*mu[i+1];
                baryc2[j] += yhonedt[j+i*6]*mu[i+1];
              }
              baryc[j] /= mtot;
              baryc2[j] /= mtot;
 
              starimage[j] = -baryc2[j]+baryc[j];
            }
            zs = yhone[2+stm1*6]/CAUPD;
            zp = 0.0;
            zb = baryc[2]/CAUPD;
            zb2 = baryc2[2]/CAUPD;
            zs -= zb;
            zp -= zb2;
            dt1 = zs;
            dt2 = -zp;
            dt = dt1+dt2;
            
            dist = sqrt( pow( yhone[0+stm1*6] - starimage[0], 2) + pow( yhone[1+stm1*6] - starimage[1], 2) ); 
            vtrans = sqrt( pow( yhone[3+stm1*6] - starimage[3], 2) + pow( yhone[4+stm1*6] - starimage[4], 2) ); 
          } else {
            dist = sqrt(pow(yhone[0+stm1*6],2)+pow(yhone[1+stm1*6],2));
            vtrans = sqrt(pow(yhone[3+stm1*6],2)+pow(yhone[4+stm1*6],2)); 
            zs=0;
            zb2=0;
          }

          t2 = thone + zs;

#if ( demcmc_compile == 0 )//|| demcmc_compile == 3) 
          fprintf(directory[st1][0], "%6d  %.18e  %.10e  %.10e\n", transitcount[st1][0], t2, dist, vtrans);
#endif

          double vplanet, toffset, tinit, tfin;
          long ic, nc, ac;
          double thone0, hold;

          vplanet = fmax(vtrans, 0.0001);
          toffset = offsetmult*(rstar*(1+rad[stm1]))*RSUNAU/vplanet;
          toffset = fmin(toffset, offsetmin);
          tinit = t2 - toffset;
          tfin = t2 + toffset;
          ic=0;
          while (ic<(kk-1) && timelist[ic]<tinit) {
            ic++;
          } 
          nc=0;
          while ((ic+nc)<(kk-1) && timelist[ic+nc]<tfin) {
            nc++;
          }
          seteq(npl, yhone0, yhone);
          thone0=thone;
          hold=h;

          for (ac=0; ac<nc; ac++) {

            if (whereplanets[st1][ic+ac][0] == 0) { // i.e. if this body's location not already computed
 
              int tnum;
              double tcur, ts1, ts2;

              tnum = numplanets[st1][ic+ac];
              // stellar time 
              tcur = timelist[ic+ac] - zs;
              ts1 = (tcur-thone);
              if (fabs(ts1) > fabs(h)) {
                h = fabs(h)*fabs(ts1)/ts1;
              } else {
                h = ts1;
              }
              while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur, thone)) ){
                status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thone, tcur, &h, yhone); 
              }
              if (LTE) {
                ts2 = (tcur+dt)-thonedt;
                if (fabs(ts2) > fabs(h)) {
                  h = fabs(h)*fabs(ts2)/ts2;
                } else {
                  h = ts2;
                }
                while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur+dt, thonedt)) ){
                  status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thonedt, tcur+dt, &h, yhonedt); 
                }
              }
              if (status != GSL_SUCCESS) {
                printf("Big problem\n");
                exit(0);
              }
  
              if (LTE) {
                for (j=0; j<6; j++) {
                  baryc[j] = 0.;
                  baryc2[j] = 0.;
                  for (i=0; i<npl; i++) {
                    baryc[j] += yhone[j+i*6]*mu[i+1];
                    baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                  }
                  baryc[j] /= mtot;
                  baryc2[j] /= mtot;
     
                  starimage[j] = -baryc2[j]+baryc[j];
                }
                zs = yhone[2+stm1*6]/CAUPD;
                zp = 0.0;
                zb = baryc[2]/CAUPD;
                zb2 = baryc2[2]/CAUPD;
                zs -= zb;
                zp -= zb2;
                dt1 = zs;
                dt2 = -zp;
                dt = dt1+dt2;
              }
 
              if (yhone[2+stm1*6] > 0.0) {

                whereplanets[st1][ic+ac][0] = 1;
                tranarray[st1][ic+ac][tnum][0] = rstar*RSUNAU; 
                numplanets[st1][ic+ac]+=1;

                if (LTE) {

                  tranarray[st1][ic+ac][tnum][1] = -(yhone[0+stm1*6]-starimage[0]);
                  tranarray[st1][ic+ac][tnum][2] = -(yhone[1+stm1*6]-starimage[1]);

                } else {

                  tranarray[st1][ic+ac][tnum][1] = -yhone[0+stm1*6];
                  tranarray[st1][ic+ac][tnum][2] = -yhone[1+stm1*6];

                }
              } 
            }
          }
          seteq(npl, yhone, yhone0);
          thone=thone0;
          h=hold;
        } 
      }
      for (pl=0; pl<npl; pl++) {
        if (pl != stm1) {
             dpold[st1][pl]=dp[st1][pl];
          dp[st1][pl]=(y[0+pl*6]-y[0+stm1*6])*(y[3+pl*6]-y[3+stm1*6]) + (y[1+pl*6]-y[1+stm1*6])*(y[4+pl*6]-y[4+stm1*6]);
          ps2=pow((y[0+pl*6]-y[0+stm1*6]),2) + pow((y[1+pl*6]-y[1+stm1*6]),2);

          if (dp[st1][pl]*dpold[st1][pl] <= 0 && ps2 < dist2/dist2divisor && y[2+stm1*6]>y[2+pl*6] && brights > 1) { 
    
                seteq(npl, yhone, y);
                thone = t;
    
            if (LTE) {
              ii=0;
              dt = -(yhone[2+pl*6] - yhone[2+stm1*6]) / CAUPD;
              seteq(npl, yhonedt, yhone);
              gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
              thonedt = thone+dt;
     
              do {
                func(thone, yhone, dydt, npl_mu);
                func(thonedt, yhonedt, dydtdt, npl_mu);
     
                for (j=0; j<6; j++) {
                  baryc[j] = 0.;
                  baryc2[j] = 0.;
                  dbarycdt[j] = 0.;
                  dbarycdt2[j] = 0.;
                  for (i=0; i<npl; i++) {
                    baryc[j] += yhone[j+i*6]*mu[i+1];
                    baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                    dbarycdt[j] += dydt[j+i*6]*mu[i+1];
                    dbarycdt2[j] += dydtdt[j+i*6]*mu[i+1];
                  }
                  baryc[j] /= mtot;
                  baryc2[j] /= mtot;
                  dbarycdt[j] /= mtot;
                  dbarycdt2[j] /= mtot;
    
                  //starimage should really be starshift here
                  starimage[j] = baryc2[j]-baryc[j];
                  dstarimagedt[j] = dbarycdt2[j]-dbarycdt[j];
                }
    
                dp[st1][pl] = (yhonedt[0+pl*6]-yhone[0+stm1*6]-starimage[0])*(yhonedt[3+pl*6]-yhone[3+stm1*6]-starimage[3]) + (yhonedt[1+pl*6]-yhone[1+stm1*6]-starimage[1])*(yhonedt[4+pl*6]-yhone[4+stm1*6]-starimage[4]);
                double ddpdt = (dydtdt[0+pl*6]-dydt[0+stm1*6]-dstarimagedt[0])*(dydtdt[0+pl*6]-dydt[0+stm1*6]-dstarimagedt[0]) + (dydtdt[1+pl*6]-dydt[1+stm1*6]-dstarimagedt[1])*(dydtdt[1+pl*6]-dydt[1+stm1*6]-dstarimagedt[1]) + (yhonedt[0+pl*6]-yhone[0+stm1*6]-starimage[0])*(dydtdt[3+pl*6]-dydt[3+stm1*6]-dstarimagedt[3]) + (yhonedt[1+pl*6]-yhone[1+stm1*6]-starimage[1])*(dydt[4+pl*6]-dydt[4+stm1*6]-dstarimagedt[4]);
     
     
                zs = yhone[2+stm1*6]/CAUPD;
                zp = yhonedt[2+pl*6]/CAUPD;
                zb = baryc[2]/CAUPD;
                zb2 = baryc2[2]/CAUPD;
                zs -= zb;
                zp -= zb2;
                dt1 = zs;
                dt2 = -zp;
                dt = dt1+dt2;
                
                tstep = - dp[st1][pl]/ddpdt;
                gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, NULL, NULL, &sys);
                seteq(npl, yhonedt, yhone);
                gsl_odeiv_step_apply (s, thone, dt, yhonedt, yerrhone, NULL, NULL, &sys);
                thone += tstep;
                thonedt = thone+dt;
                ii++;
     
              } while (ii<10 && fabs(tstep)>eps);
                
              dist = sqrt( pow( (yhonedt[0+pl*6]-yhone[0+stm1*6]-starimage[0]), 2) + pow( (yhonedt[1+pl*6]-yhone[1+stm1*6]-starimage[1]), 2) ); 
     
            } else {
    
              ii=0;
              do {
                func (thone, yhone, dydt, npl_mu);
                //ddpdt = dydt[0+pl*6]*dydt[0+pl*6] + dydt[1+pl*6]*dydt[1+pl*6] + yhone[0+pl*6]*dydt[3+pl*6] + yhone[1+pl*6]*dydt[4+pl*6];
                ddpdt = (dydt[0+pl*6]-dydt[0+stm1*6])*(dydt[0+pl*6]-dydt[0+stm1*6]) + (dydt[1+pl*6]-dydt[1+stm1*6])*(dydt[1+pl*6]-dydt[1+stm1*6]) + (yhone[0+pl*6]-yhone[0+stm1*6])*(dydt[3+pl*6]-dydt[3+stm1*6]) + (yhone[1+pl*6]-yhone[1+stm1*6])*(dydt[4+pl*6]-dydt[4+stm1*6]);
                tstep = - dp[st1][pl] / ddpdt;
                gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, dydt, NULL, &sys);
                thone += tstep;
                    //dp[st1][pl]=yhone[0+pl*6]*yhone[3+pl*6]+yhone[1+pl*6]*yhone[4+pl*6];
                dp[st1][pl]=(yhone[0+pl*6]-yhone[0+stm1*6])*(yhone[3+pl*6]-yhone[3+stm1*6]) + (yhone[1+pl*6]-yhone[1+stm1*6])*(yhone[4+pl*6]-yhone[4+stm1*6]);
                ii++;
              } while (ii<5 && fabs(tstep)>eps);
                
              //dist = sqrt( pow( (yhonedt[0+pl*6]-yhone[0+stm1*6]), 2) + pow( (yhonedt[1+pl*6]-yhone[1+stm1*6]), 2) ); 
              dist = sqrt( pow( (yhone[0+pl*6]-yhone[0+stm1*6]), 2) + pow( (yhone[1+pl*6]-yhone[1+stm1*6]), 2) ); 
    
            }
    
            dp[st1][pl]=(y[0+pl*6]-y[0+stm1*6])*(y[3+pl*6]-y[3+stm1*6]) + (y[1+pl*6]-y[1+stm1*6])*(y[4+pl*6]-y[4+stm1*6]);
    
            if (dist < 2.0*((rstar*rad[stm1] + rstar*rad[pl])*RSUNAU)) {
    
              transitcount[st1][pl+1] -= 1;  // Update the transit number. 
    
              if (LTE) {
                for (j=0; j<6; j++) {
                  baryc[j] = 0.;
                  baryc2[j] = 0.;
                  for (i=0; i<npl; i++) {
                    baryc[j] += yhone[j+i*6]*mu[i+1];
                    baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                  }
                  baryc[j] /= mtot;
                  baryc2[j] /= mtot;
     
                  starimage[j] = baryc2[j]-baryc[j];
                }
                zs = yhone[2+stm1*6]/CAUPD;
                zp = yhonedt[2+pl*6]/CAUPD;
                zb = baryc[2]/CAUPD;
                zb2 = baryc2[2]/CAUPD;
                zs -= zb;
                zp -= zb2;
                dt1 = zs;
                dt2 = -zp;
                dt = dt1+dt2;
                
                dist = sqrt( pow( (yhonedt[0+pl*6]-yhone[0+stm1*6]-starimage[0]), 2) + pow( (yhonedt[1+pl*6]-yhone[1+stm1*6]-starimage[1]), 2) ); 
                vtrans = sqrt(pow(yhonedt[3+pl*6]-yhone[3+stm1*6] - starimage[3],2) + pow(yhonedt[4+pl*6]-yhone[4+stm1*6] - starimage[4],2)); 
              } else {
                //dist = sqrt( pow( (yhonedt[0+pl*6]-yhone[0+stm1*6]), 2) + pow( (yhonedt[1+pl*6]-yhone[1+stm1*6]), 2) ); 
                //vtrans = sqrt(pow(yhonedt[3+pl*6]-yhone[3+stm1*6], 2) + pow(yhonedt[4+pl*6]-yhone[4+stm1*6], 2)); 
                dist = sqrt( pow( (yhone[0+pl*6]-yhone[0+stm1*6]), 2) + pow( (yhone[1+pl*6]-yhone[1+stm1*6]), 2) ); 
                vtrans = sqrt(pow(yhone[3+pl*6]-yhone[3+stm1*6], 2) + pow(yhone[4+pl*6]-yhone[4+stm1*6], 2)); 
                zs=0;
                zb2=0;
              }
    
              t2 = thone + zs;
    
#if ( demcmc_compile == 0 )//|| demcmc_compile == 3) 
              fprintf(directory[st1][pl+1], "%6d  %.18e  %.10e  %.10e\n", transitcount[st1][pl+1], t2, dist, vtrans);
#endif


              double vplanet, toffset, tinit, tfin;
              long ic, nc, ac;
              double thone0, hold;

              vplanet = fmax(vtrans, 0.0001);
              toffset = offsetmult*(rstar*(rad[stm1]+rad[pl]))*RSUNAU/vplanet;
              toffset = fmin(toffset, offsetminout);
              tinit = t2 - toffset;
              tfin = t2 + toffset;
              ic=0;
              while (ic<(kk-1) && timelist[ic]<tinit) ic++; 
              nc=0;
              while ((ic+nc)<(kk-1) && timelist[ic+nc]<tfin) nc++;

              seteq(npl, yhone0, yhone);
              thone0=thone;
              hold=h;

              //This is fairly inexact for first time if yhone_Z is changing considerably Can try to fix this in future versions. 
              for (ac=0; ac<nc; ac++) {
                if (whereplanets[st1][ic+ac][pl+1] == 0) { // i.e. if this body's location not already computed

                  int tnum;
                  double tcur, ts1, ts2;

                  tnum = numplanets[st1][ic+ac];
                  tcur = timelist[ic+ac] - zs;
                  ts1 = (tcur-thone);
                  if (fabs(ts1) > fabs(h)) {
                    h = fabs(h)*fabs(ts1)/ts1;
                  } else {
                    h = ts1;
                  }
                  while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur, thone)) ){
                    status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thone, tcur, &h, yhone); 
                  }
                  if (LTE) {
                    ts2 = (tcur+dt)-thonedt;
                    if (fabs(ts2) > fabs(h)) {
                      h = fabs(h)*fabs(ts2)/ts2;
                    } else {
                      h = ts2;
                    }
                    while ( !(status!=GSL_SUCCESS) && !(dbleq(tcur+dt, thonedt)) ){
                      status = gsl_odeiv_evolve_apply(e, c, s, &sys, &thonedt, tcur+dt, &h, yhonedt); 
                    }
                  }
                  if (status != GSL_SUCCESS) {
                    printf("Big problem\n");
                    exit(0);
                  }
      
                  if (LTE) {
                    for (j=0; j<6; j++) {
                      baryc[j] = 0.;
                      baryc2[j] = 0.;
                      for (i=0; i<npl; i++) {
                        baryc[j] += yhone[j+i*6]*mu[i+1];
                        baryc2[j] += yhonedt[j+i*6]*mu[i+1];
                      }
                      baryc[j] /= mtot;
                      baryc2[j] /= mtot;
         
                      starimage[j] = baryc2[j]-baryc[j];
                    }
         
                    zs = yhone[2+stm1*6]/CAUPD;
                    zp = yhonedt[2+pl*6]/CAUPD;
                    zb = baryc[2]/CAUPD;
                    zb2 = baryc2[2]/CAUPD;
                    zs -= zb;
                    zp -= zb2;
                    dt1 = zs;
                    dt2 = -zp;
                    dt = dt1+dt2;
                  }
 
                  if (yhone[2+stm1*6] > yhone[2+pl*6]) {

                    whereplanets[st1][ic+ac][pl+1] = 1;
                    numplanets[st1][ic+ac]+=1;
                    tranarray[st1][ic+ac][tnum][0] = rad[pl]*rstar*RSUNAU;

                    if (LTE) {

                      tranarray[st1][ic+ac][tnum][1] = yhonedt[0+pl*6] - yhone[0+stm1*6] - starimage[0];
                      tranarray[st1][ic+ac][tnum][2] = yhonedt[1+pl*6] - yhone[1+stm1*6] - starimage[1];
                    
                    } else {
                    
                      tranarray[st1][ic+ac][tnum][1] = yhone[0+pl*6] - yhone[0+stm1*6];
                      tranarray[st1][ic+ac][tnum][2] = yhone[1+pl*6] - yhone[1+stm1*6];

                    }
                  }
                } 
              }

              seteq(npl, yhone, yhone0);
              thone=thone0;
              h=hold;
            }
          }
        }
      }
    }
  }

      



  if (RVS) {
    free(rvtarr);
  }

  
#if ( demcmc_compile == 0 )//|| demcmc_compile == 3) 

  for (k=0; k<brights; k++) {
    int k1 = brightsindex[k];
    for (i=0; i<(npl+1); i++){
      if (i != k1) {
        fclose(directory[k][i]);
      }
    }
  }

  for (k=0; k<brights; k++) {
    free(directory[k]);
  }
  free(directory);

#endif

  

  for (k=0; k <brights; k++) {
    long kkc;
    for (kkc=0; kkc<kk; kkc++) {
      free(whereplanets[k][kkc]);
    }
    free(whereplanets[k]);
  }
  free(whereplanets);


  double onetlc (int nplanets, double **rxy, double rstar, double c1, double c2); // prototype 

  for (k=0; k<brights; k++) {
    int k1 = brightsindex[k];
    long kk1;
    for (kk1=0; kk1<kk; kk1++) {
      if (k==0) fluxlist[k][kk1] = onetlc(numplanets[k][kk1], tranarray[k][kk1], 1.0*rstar*RSUNAU, c1list[k1], c2list[k1]);
      else fluxlist[k][kk1] = onetlc(numplanets[k][kk1], tranarray[k][kk1], rad[k1-1]*rstar*RSUNAU, c1list[k1], c2list[k1]);
    }
  }

  double omdilute = 1.0-dilute;
  long kk1;
  for (kk1=0; kk1<kk; kk1++) {
    netflux[kk1+1] = omdilute*brightness[0]*fluxlist[0][kk1] + dilute;
    for (k=1; k<brights; k++) {
      int k1 = brightsindex[k];
      netflux[kk1+1] += omdilute*brightness[k1]*fluxlist[k][kk1];
    }
  }

  for (k=0; k<brights; k++) {
    long kkc; 
    for (kkc=0; kkc<kk; kkc++) {
      long kkcc;
      for (kkcc=0; kkcc<posstrans; kkcc++) {
        free(tranarray[k][kkc][kkcc]);
      }
      free(tranarray[k][kkc]);
    }
    free(tranarray[k]);    
  }
  free(tranarray);


  if (cadenceswitch == 2) {
    if (OOO) {
      long q;
      double *tclist = malloc((2*sofd)*kk); 
      for (q=0; q<kk; q++) {
        tclist[2*q] = (double) order[q];
        tclist[2*q+1] = netflux[q];
      }
      qsort(tclist, kk, sofd*2, compare);
      for (q=0; q<kk; q++) {
        netflux[q] = tclist[2*q+1];
      }
      free(tclist);
    }
    free(order);
  }

  if (cadenceswitch == 1) {
    long q;
    for (q=0; q<kkorig; q++) {
      double fsum=0;
      for (k=0; k<nperbin; k++) {
        fsum+=netflux[q*nperbin+k+1];
      }
      netflux[q+1] = fsum/nperbin;
    }
    kk /= nperbin;
    free(timelist);
  } else if (cadenceswitch == 2) {
    long q;
    long lccount = 0;
    long sccount = 0;

    for (q=0; q<kkorig; q++) {
      if (cadencelist[q] == 0) {
        netflux[q+1] = netflux[sccount+lccount*nperbin+1];
        sccount++;
      } else {
        double fsum=0;
        for (k=0; k<nperbin; k++) {
          fsum+=netflux[sccount+lccount*nperbin+k+1];
        }
        netflux[q+1] = fsum/nperbin;
        lccount++;
      }
    }
    kk = kkorig;
    free(timelist);
  }

  netflux[0]= (double) kk;
  tmte[2]=&netflux[0];

  
  for (k=0; k<brights; k++) {
    free(dp[k]); free(dpold[k]);
  }  
  free(dp); free(dpold);


  for (k=0; k<brights; k++) {
    free(fluxlist[k]);
  }
  free(fluxlist);
  //Do not free netflux as it is being returned!!


  for (k=0; k<brights; k++) {
    free(transitarr[k]);
    free(transitcount[k]);
    free(numplanets[k]);
  }
  free(transitarr);
  free(ntransits);
  free(transitcount);
  free(numplanets);
  free(brightsindex);

  free(rad); 

  gsl_odeiv_evolve_free(e);
  gsl_odeiv_control_free(c);
  gsl_odeiv_step_free(s);



  int tele;
  int maxteles=10;
  double *rvoffset;
  rvoffset = malloc(maxteles*sofd);
  if (RVS) {
    // Compute RV offset
    double rvdiff;
    long vvw;
    double weight;

    long vvt=1;
    tele=0;
    while (vvt<(vv+1)) {
      double numerator = 0;
      double denominator = 0;
      for (vvw=1; vvw<(vv+1); vvw++) {
        if ( (int) tve[4][vvw] == tele ) {
          rvdiff = rvarr[vvw] - tve[1][vvw];
          weight = 1.0 / pow(tve[2][vvw], 2);
          numerator += rvdiff*weight;
          denominator += weight;
        }
      }
      rvoffset[tele] = numerator/denominator;
      for (vvw=1; vvw<(vv+1); vvw++) {
        if ( (int) tve[4][vvw] == tele ) {
          rvarr[vvw] -= rvoffset[tele];
          vvt+=1;
        }
      }
      tele+=1;
    }
  }




  double **rvtmte = malloc(4*sofds);

  rvtmte[0] = &tve[0][0];
  rvtmte[1] = &tve[1][0];
  rvtmte[3] = &tve[2][0];
  rvtmte[2] = rvarr;


#if (demcmc_compile==0)

  char tmtefstr[1000];
  strcpy(tmtefstr, "lc_");
  strcat(tmtefstr, OUTSTR);
  strcat(tmtefstr, ".lcout");
  FILE *tmtef = fopen(tmtefstr, "a");
  long ijk;
  if (CADENCESWITCH==2) {
    for (ijk=1; ijk<=kk; ijk++) {
      fprintf(tmtef, "%.12lf %.12lf %.12lf %.12lf %i\n", tmte[0][ijk], tmte[1][ijk], tmte[2][ijk], tmte[3][ijk], cadencelist[ijk-1]);
    }
  } else {
    for (ijk=1; ijk<=kk; ijk++) {
      fprintf(tmtef, "%lf %lf %lf %lf\n", tmte[0][ijk], tmte[1][ijk], tmte[2][ijk], tmte[3][ijk]);
    }
  }
  fclose(tmtef);// = openf(tmtstr,"w");

  if (RVS) {
    for (i=0; i<nbodies; i++) {
      if (RVARR[i]) {
        char rvtmtefstr[1000];
        char num[10];
        strcpy(rvtmtefstr, "rv");
        sprintf(num, "%01i", i);
        strcat(rvtmtefstr, num);
        strcat(rvtmtefstr, "_");
        strcat(rvtmtefstr, OUTSTR);
        strcat(rvtmtefstr, ".rvout");
        FILE *rvtmtef = fopen(rvtmtefstr, "a");
        for (ijk=1; ijk<vv+1; ijk++) {
          if ( ((int) tve[3][ijk]) == i) {
            fprintf(rvtmtef, "%lf\t%e\t%e\t%e\t%i\n", rvtmte[0][ijk], rvtmte[1][ijk]/MPSTOAUPD, rvtmte[2][ijk]/MPSTOAUPD, rvtmte[3][ijk]/MPSTOAUPD, (int) tve[4][ijk]);
          }
        }
        int l;
        for (l=0; l<tele; l++) fprintf(rvtmtef, "RV offset %i = %lf\n", l, rvoffset[l]/MPSTOAUPD);
        fclose(rvtmtef);// = openf(tmtstr,"w");
      }
    }   
  }


#endif

  free(rvoffset);

  double ***fl_rv = malloc(2*sizeof(double**));
  fl_rv[0] = tmte;
  fl_rv[1] = rvtmte;

  return fl_rv;

}



// Compute lightcurve if only 1 star / luminous object
double ***dpintegrator_single (double ***int_in, double **tfe, double **tve, double **nte, int *cadencelist) {

  const int cadenceswitch = CADENCESWITCH;
  double t0 = T0;
  double t1 = T1;
  int nperbin = NPERBIN;
  double binwidth = BINWIDTH;
  const int lte = LTE;

  const int sofi = SOFI;
  const int sofd = SOFD;
  const int sofds = SOFDS;

  long kk = (long) tfe[0][0];
  int nplplus1 = int_in[0][0][0];
  const int npl = nplplus1-1;
  double tstart = int_in[1][0][0]; //epoch
  double rstar = int_in[3][0][0];
  double c1 = int_in[3][0][1];
  double c2 = int_in[3][0][2];
  double dilute = int_in[3][0][3];


  const gsl_odeiv_step_type * T 
  /*   = gsl_odeiv_step_bsimp;   14 s */
  = gsl_odeiv_step_rk8pd;   /* 3 s */
  /*  = gsl_odeiv_step_rkf45;    14 s */
  /*  = gsl_odeiv_step_rk4;  26 s */

  gsl_odeiv_step * s 
    = gsl_odeiv_step_alloc (T, 6*npl);
  gsl_odeiv_control * c 
    = gsl_odeiv_control_y_new (DY, 0.0);
  gsl_odeiv_evolve * e 
    = gsl_odeiv_evolve_alloc (6*npl);
  
  long i;
  double mu[npl+1];
  double npl_mu[npl+2];
  gsl_odeiv_system sys = {func, jac, 6*npl, npl_mu};

  double t; /* this is the given epoch. */

  double hhere;  
  double y[6*npl], yin[6*npl];
  //note:rad is in units of rp/rstar
  double *rad = malloc(npl*sofd);

  mu[0] = int_in[2][0][0]; 
 
  for (i=0; i<npl; i++) {
    mu[i+1] = int_in[2][i+1][0];
    int ih;
    for (ih=0; ih<6; ih++) {
      yin[i*6+ih] = int_in[2][i+1][ih+1];
    }
    rad[i] = int_in[2][i+1][7];  
  }

  memcpy(&npl_mu[1], mu, (npl+1)*sofd);
  npl_mu[0] = npl;


  double mtot=mu[0];
  for(i=0;i<npl;i++) mtot += mu[i+1];

  double yhone[6*npl], dydt[6*npl], yerrhone[6*npl];
  double thone,tstep,dist,vtrans;
  double vstar,astar;


  int transitcount[npl];
  int pl;
  for (pl=0;pl<npl;pl++) transitcount[pl]=-1;

#if ( demcmc_compile==3)
  double msys[npl];
  msys[0]=mu[0]+mu[1];
  for (i=1; i<npl; i++) msys[i] = msys[i-1] + mu[i+1];

  double *statetokep(double x, double y, double z, double vx, double vy, double vz, double m); //prototype
  double *keptoorb(double x, double y, double z, double vx, double vy, double vz, double m); //prototype

  double *amax = malloc(npl*sofd);
  double *amin = malloc(npl*sofd);
  for (i=0; i<npl; i++) {
    double *kepelementsin = statetokep(yin[0+i*6], yin[1+i*6], yin[2+i*6], yin[3+i*6], yin[4+i*6], yin[5+i*6], msys[i]);
    double aorig = kepelementsin[0];
    free(kepelementsin);
    amax[i] = aorig*(1.0+AFRACSTABLE);
    amin[i] = aorig*(1.0-AFRACSTABLE);
  }
#endif
 
 /* Setup forward integration */
  seteq(npl, y, yin);
  double dp[npl], ddpdt, dpold[npl] ;  /* dp = dot-product = x.v */ 
  for(pl=0; pl<npl; pl++) dp[pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6];
  double ps2, dist2;  /* ps2 = projected separation squared. */ 
  t = tstart;
  double h = HStart;


  int k;

  long maxtransits = 10000;
  int toomany=0;
  int ntarrelem = 6;
  double *transitarr = malloc((maxtransits+1)*ntarrelem*sofd);
  if (transitarr == NULL) {
    printf("Allocation Error\n");
    exit(0);
  } 
  long ntransits = 0;

  long vv = 0;

  // This messes up reduced chi squared if your rv values are outside of the integration time!!!  
  // But we are only comparing relative chi squareds
  // Plus just don't include rv times that aren't within your integration bounds, because that's silly
  double *rvarr;
  double *rvtarr;
  int onrv = 0;
  int rvcount, startrvcount; 
  if (RVS) {
    vv = (long) tve[0][0];
    rvarr = calloc(vv+1, sofd);
    rvarr[0] = (double) vv;
    rvtarr = malloc((vv+2)*sofd);
    rvtarr[0] = -HUGE_VAL;
    rvtarr[vv+1] = HUGE_VAL;
    memcpy(&rvtarr[1], &tve[0][1], vv*sofd);
    startrvcount = 1;
    while (rvtarr[startrvcount] < tstart) startrvcount+=1;
    rvcount = startrvcount;
  }
  long vvt;
  double *ttvarr;
  double **ttvarr2;
  if (TTVCHISQ) {
    ttvarr2 = malloc(npl*sofds);
    for (i=0; i<npl; i++) {
      ttvarr2[i] = malloc(NTTV[i][0]*sofd);
    }
    vvt = (long) nte[0][0];
    ttvarr = malloc((vvt+1)*sofd);
    ttvarr[0] = (double) vvt;

  }


#if ( demcmc_compile==3)

  int stabilitystatus=0;

  double **orbarray = malloc(npl*sofds);
  int onorbout = 0;
  double orbt = tstart+TINTERVAL/2.0;
  int densestart = 1;
  double denseinterval=0.1;
  double densemax=5000.0;
  if (densestart) {
    orbt = tstart;// + denseinterval;// /2.0;
  }
  long nposorbouts=0;
  long nnegorbouts=0;

  char stabstr[1000];
  strcpy(stabstr, "stability_");
  strcat(stabstr, OUTSTR);
  strcat(stabstr, ".out");
  FILE *stabout = fopen(stabstr, "w");
  fprintf(stabout, "t\t\t\tpnum\t\tP\t\ta\t\te\t\ti\t\t\tlo\t\t\tbo\t\t\tf\n");
  char stabstr2[1000];
  strcpy(stabstr2, "stability_xyz_");
  strcat(stabstr2, OUTSTR);
  strcat(stabstr2, ".out");
  FILE *stabout2 = fopen(stabstr2, "w");
  fprintf(stabout2, "t\t\t\tpnum\t\tx\t\ty\t\tz\t\tvx\t\tvy\t\tvz\t\t\n");

  double tfirst, tlast;

#endif

#if ( demcmc_compile == 0 || demcmc_compile == 3)
  FILE **directory = malloc(npl*sizeof(FILE*));
  for (i=0; i<npl; i++) {
    char str[30];
    sprintf(str, "tbv00_%02i.out", i+1);
    directory[i]=fopen(str,"w");
  }
  double tbvmax = 36500.0;
#endif



  double **tmte;
  double **rvtmte;



  double dist2divisor=DIST2DIVISOR;
  double baryz;
  double eps=1e-10;

  double printepoch=PRINTEPOCH;

#if (demcmc_compile != 3)
  for (i=0; i<npl; i++) {
    int ih;
    for (ih=0; ih<6; ih++) {
      if (isnan(yin[i*6+ih])) {
        goto exitgracefully;
      }
    }
  }
#endif
 
  int *ttvi;
  int *ttviinit;
  if (TTVCHISQ) {
    ttvi = calloc(npl, sofi);
    ttviinit = malloc(npl*sofi);
    for (i=0; i<npl; i++) {
      while(NTTV[i][ttvi[i]+1] < 0) {
        ttvi[i]+=1;
      }
      ttviinit[i] = ttvi[i];
    }
  }

#if ( demcmc_compile==0 )
  printf("starting integration.\n");
  printf("t=%lf tepoch=%lf t1=%lf t0=%lf h=%lf\n", t, tstart, t1, t0, h); 
#endif

  while (t < t1 ) {      


#if ( demcmc_compile==0 )
    int printtime=0;
    double htimehere;
    if (h+t > printepoch) {
      printtime=1;
      htimehere=h;
      h = printepoch-t;
      if (h<0) {
        printf("Error, bad printepoch\n");
      }
      printepoch=HUGE_VAL;
    }
#endif


    if (RVS) {
      onrv=0;
      if (h + t > rvtarr[rvcount]) {  
        onrv = 1;
        hhere = h;
        h = rvtarr[rvcount] - t;
      }
    }

#if ( demcmc_compile==3)
    onorbout = 0;
    if (h + t > orbt) {
      onorbout = 1;
      hhere = h;
      h = orbt - t;
    }
#endif

    int status = gsl_odeiv_evolve_apply (e, c, s,
                                         &sys, 
                                         &t, t1,
                                         &h, y);

    if (status != GSL_SUCCESS)
        break;

#if ( demcmc_compile==0 )
    if (printtime) {
      char outfile2str[80];
      strcpy(outfile2str, "xyz_adjusted_");
      strcat(outfile2str, OUTSTR);
      strcat(outfile2str, ".pldin");
      FILE *outfile2 = fopen(outfile2str, "a");
      fprintf(outfile2, "planet                 x                        y                      z                      v_x                   v_y                   v_z                       m                      rpors             ");
      if (MULTISTAR) {
        fprintf(outfile2, "brightness               c1                    c2\n");
      } else {
        fprintf(outfile2, "\n");
      }
      double pnum = 0.1;
      for (i=0; i<NPL; i++) {
        fprintf(outfile2, "%1.1lf", pnum);
        int j;
        for (j=0; j<6; j++) {
          fprintf(outfile2, "\t%.15lf", y[6*i+j]);
        }
        fprintf(outfile2, "\t%.15lf", mu[i+1]*MSOMJ);
        fprintf(outfile2, "\t%.15lf", rad[i]);
  
        fprintf(outfile2, "\n");
        pnum+=0.1;
      }
      fprintf(outfile2, "%.15lf ; mstar\n", mu[0]);
      fprintf(outfile2, "%.15lf ; rstar\n", rstar);
      fprintf(outfile2, "%.15lf ; c1\n", c1);
      fprintf(outfile2, "%.15lf ; c2\n", c2);
      fprintf(outfile2, "%.15lf ; dilute\n", dilute);
      fprintf(outfile2, " ; These coordinates are stellar centric\n");
      fprintf(outfile2, " ; Tepoch = %0.15lf\n", PRINTEPOCH);

      fclose(outfile2);

      printtime=0;
      h=htimehere;
    }
#endif

#if ( demcmc_compile==3)
    if (onorbout == 1) {
      h = hhere;
      for (i=0; i<npl; i++) {
        orbarray[i] = statetokep(y[0+i*6], y[1+i*6], y[2+i*6], y[3+i*6], y[4+i*6], y[5+i*6], msys[i]);
        double *aeielem = keptoorb(orbarray[i][0], orbarray[i][1], orbarray[i][2], orbarray[i][3], orbarray[i][4], orbarray[i][5], msys[i]);
        fprintf(stabout, "%.13lf\t%i\t%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\n", orbt, i, aeielem[0], orbarray[i][0], orbarray[i][1], orbarray[i][2]*180.0/M_PI, orbarray[i][3]*180.0/M_PI, orbarray[i][4]*180.0/M_PI, orbarray[i][5]*180.0/M_PI);
        fprintf(stabout2, "%.13lf\t%i\t%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\n", orbt, i, y[0+i*6], y[1+i*6], y[2+i*6], y[3+i*6], y[4+i*6], y[5+i*6]);
        if (orbarray[i][0] > amax[i] || orbarray[i][0] < amin[i]) {
          stabilitystatus = 1; 
          tlast = orbt;
        }
        free(orbarray[i]);
        free(aeielem);
      }
      if (densestart) {
        if (t<densemax) {
          orbt += denseinterval;
        } else {
          if (t<365.*100000.) orbt += TINTERVAL;
          else orbt += TINTERVAL*50.;
        }
      } else {
        if (t<365.*100000.) orbt += TINTERVAL;
        else orbt += TINTERVAL*50.;
      }
      nposorbouts++;
      if (nposorbouts % 1 == 0) fflush(stabout);
      if (nposorbouts % 1 == 0) fflush(stabout2);
    }
    if (stabilitystatus == 1) { 
      printf("Went Unstable -> Break\n");
      break;
    }
#endif


    if (RVS) {
      if (onrv ==1) {
        h = hhere;
        func(t, y, dydt, mu);
        double vstar = 0;
        for (i=0; i<npl; i++) {
          vstar += -y[i*6+5]*mu[i+1]/mtot;
        }
        rvarr[rvcount] = vstar;    
        rvcount += 1;
      }
    }


    dist2 = pow(y[0],2)+pow(y[1],2)+pow(y[2],2);
    for(pl=0; pl<npl; pl++) {  /* Cycle through the planets, searching for a transit. */
      dpold[pl]=dp[pl];
      dp[pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6];
      ps2=y[0+pl*6]*y[0+pl*6]+y[1+pl*6]*y[1+pl*6];

      if( /* 0 == 1 */ dp[pl]*dpold[pl] <= 0 && ps2 < dist2/dist2divisor && y[2+pl*6] < 0 ) { 
                     /* A minimum projected-separation occurred: "Tc".  
                   ps2 constraint makes sure it's when y^2+z^2 is small-- its on the face of the star.  
                   y[2] constraint means a primary eclipse, presuming that the observer is on the positive x=y[2] axis. */

        transitcount[pl] += 1;  /* Update the transit number. */

        seteq(npl, yhone, y);
        thone = t;
        i=0;
        do { 
          func (thone, yhone, dydt, npl_mu);
          ddpdt = dydt[0+pl*6]*dydt[0+pl*6] + dydt[1+pl*6]*dydt[1+pl*6] + yhone[0+pl*6]*dydt[3+pl*6] + yhone[1+pl*6]*dydt[4+pl*6];
          tstep = - dp[pl] / ddpdt;
          gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, dydt, NULL, &sys);
          thone += tstep;
          dp[pl]=yhone[0+pl*6]*yhone[3+pl*6]+yhone[1+pl*6]*yhone[4+pl*6];
          i++;
        } while (i<5 && fabs(tstep)>eps);
        
        dp[pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6];

        double zp=0;
        double zb=0;
        if (LTE) {
          baryz = 0;
          for(i=0; i<npl; i++) baryz += yhone[2+i*6]*mu[i+1];
          baryz /= mtot; 
          zp = yhone[2+pl*6]/CAUPD;
          zb = baryz/CAUPD;
          zp -= zb;
        }
        double t2 = thone - zb + zp;


#if ( (demcmc_compile == 0) )//|| (demcmc_compile == 3) )
        dist = sqrt(pow(yhone[0+pl*6],2)+pow(yhone[1+pl*6],2));
        vtrans = sqrt(pow(yhone[3+pl*6],2)+pow(yhone[4+pl*6],2)); 
        fprintf(directory[pl], "%6d  %.18e  %.10e  %.10e\n", transitcount[pl], t2, dist, vtrans);
#endif

        if (TTVCHISQ) {
          if (transitcount[pl] == NTTV[pl][ttvi[pl]+1]) {
#if (demcmc_compile==0)
            MTTV[pl][ttvi[pl]+1]=t2;
#endif
            ttvarr2[pl][ttvi[pl]]=t2;
            ttvi[pl]+=1;
          }
        }

#if ( (demcmc_compile == 3) )
        if (t2 < tbvmax) {
          double bsign=1.;
          if (yhone[1+pl*6] < 0.) bsign=-1;
          dist = sqrt(pow(yhone[0+pl*6],2)+pow(yhone[1+pl*6],2));
          vtrans = sqrt(pow(yhone[3+pl*6],2)+pow(yhone[4+pl*6],2)); 
          fprintf(directory[pl], "%6d  %.18e  %.10e  %.10e\n", transitcount[pl], t2, bsign*dist, vtrans);
        }
#endif

#if (demcmc_compile != 3)
        transitarr[ntransits*ntarrelem+0] = t2;
        transitarr[ntransits*ntarrelem+1] = yhone[0+pl*6]; 
        transitarr[ntransits*ntarrelem+2] = yhone[1+pl*6];
        transitarr[ntransits*ntarrelem+3] = yhone[3+pl*6];
        transitarr[ntransits*ntarrelem+4] = yhone[4+pl*6];
        transitarr[ntransits*ntarrelem+5] = rad[pl];
        ntransits++;
        if (ntransits > maxtransits) {
          printf("Too many transits - increase maxtransits\n");
          fflush(stdout);
          toomany=1;
          goto exitgracefully; 
        }
#endif
      }
    }
  }


  long postransits;
  postransits = ntransits;


  /* Setup backward integration */
  seteq(npl, y, yin);
  for(pl=0; pl<npl; pl++) dp[pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6]; /* dp = dot-product = x.v */
  t = tstart;
  h = -HStart;
  for(pl=0;pl<npl;pl++) transitcount[pl] = 0;

  if (RVS) {
    rvcount = startrvcount-1;
  }

#if ( demcmc_compile==3)
  int posstability = 0;
  if (stabilitystatus==1) posstability=1;
  stabilitystatus=0;
  if (densestart) {
    orbt = t-denseinterval;// /2.0;
  } else {
    orbt = t-TINTERVAL;// /2.0;
  }
#endif
  
  if (TTVCHISQ) {
    for (i=0; i<npl; i++) {
      ttvi[i] = ttviinit[i]-1;
    }
  }

#if ( demcmc_compile==0 )
  printf("RE starting integration.\n");
  printf("t=%lf tepoch=%lf t1=%lf t0=%lf h=%lf\n", t, tstart, t1, t0, h); 
#endif

  while (t > t0+1) {      
    if (RVS) {
      onrv=0;
      if ( (h + t) < rvtarr[rvcount]) {  
        onrv = 1;
        hhere = h;
        h = rvtarr[rvcount] - t;
      }
    }

#if ( demcmc_compile==3)
    onorbout = 0;
    if (h + t < orbt) {
      onorbout = 1;
      hhere = h;
      h = orbt - t;
    }
#endif

    int status = gsl_odeiv_evolve_apply (e, c, s,
                                         &sys, 
                                         &t, t0,
                                         &h, y);

    if (status != GSL_SUCCESS)
        break;

#if ( demcmc_compile==3)
    if (onorbout == 1) {
      h = hhere;
      for (i=0; i<npl; i++) {
        orbarray[i] = statetokep(y[0+i*6], y[1+i*6], y[2+i*6], y[3+i*6], y[4+i*6], y[5+i*6], msys[i]);
        double *aeielem = keptoorb(orbarray[i][0], orbarray[i][1], orbarray[i][2], orbarray[i][3], orbarray[i][4], orbarray[i][5], msys[i]);
        fprintf(stabout, "%.13lf\t%i\t%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\n", orbt, i, aeielem[0], orbarray[i][0], orbarray[i][1], orbarray[i][2]*180.0/M_PI, orbarray[i][3]*180.0/M_PI, orbarray[i][4]*180.0/M_PI, orbarray[i][5]*180.0/M_PI);
        fprintf(stabout2, "%.13lf\t%i\t%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\n", orbt, i, y[0+i*6], y[1+i*6], y[2+i*6], y[3+i*6], y[4+i*6], y[5+i*6]);
        if (orbarray[i][0] > amax[i] || orbarray[i][0] < amin[i]) {
          stabilitystatus = 1; 
          tlast = orbt;
        }
        free(orbarray[i]);
        free(aeielem);
      }
      if (densestart) {
        orbt -= denseinterval;
      } else {
        orbt -= TINTERVAL;
      }
    }
    if (stabilitystatus == 1) { 
      printf("Went Unstable -> Break\n");
      break;
    }
#endif

    if (RVS) {
      if (onrv ==1) {
        h = hhere;
        func(t, y, dydt, mu);
        double vstar = 0;
        for (i=0; i<npl; i++) {
          vstar += -y[i*6+5]*mu[i+1]/mtot;
        }
        rvarr[rvcount] = vstar;    
        rvcount -= 1;
      }
    }


    dist2 = pow(y[0],2)+pow(y[1],2)+pow(y[2],2);

    for(pl=0; pl<npl; pl++) {  /* Cycle through the planets, searching for a transit. */
      dpold[pl]=dp[pl];
      dp[pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6];
      ps2=y[0+pl*6]*y[0+pl*6]+y[1+pl*6]*y[1+pl*6];

      if( /* 0 == 1 */ dp[pl]*dpold[pl] <= 0 && ps2 < dist2/dist2divisor && y[2+pl*6] < 0 ) { 
                /* A minimum projected-separation occurred: "Tc".  
                   ps2 constraint makes sure it's when y^2+z^2 is small-- its on the face of the star.  
                   y[0] constraint means a primary eclipse, presuming that the observer is on the positive x=y[0] axis. */

        transitcount[pl] -= 1;  /* Update the transit number. */

        seteq(npl, yhone, y);
        thone = t;
        i=0;
        do {
          func (thone, yhone, dydt, npl_mu);
          ddpdt = dydt[0+pl*6]*dydt[0+pl*6] + dydt[1+pl*6]*dydt[1+pl*6] + yhone[0+pl*6]*dydt[3+pl*6] + yhone[1+pl*6]*dydt[4+pl*6];
          tstep = - dp[pl] / ddpdt;
          gsl_odeiv_step_apply (s, thone, tstep, yhone, yerrhone, dydt, NULL, &sys);
          thone += tstep;
          dp[pl]=yhone[0+pl*6]*yhone[3+pl*6]+yhone[1+pl*6]*yhone[4+pl*6];
          i++;
        } while (i<5 && fabs(tstep)>eps); 
          dp[pl]=y[0+pl*6]*y[3+pl*6]+y[1+pl*6]*y[4+pl*6];

        double zp=0;
        double zb=0;
        if (LTE) {
          baryz = 0;
          for(i=0; i<npl; i++) baryz += yhone[2+i*6]*mu[i+1];
          baryz /= mtot; 
          zp = yhone[2+pl*6]/CAUPD;
          zb = baryz/CAUPD;
          zp -= zb;
        }
        double t2 = thone - zb + zp;

#if ( demcmc_compile == 0 )//|| demcmc_compile == 3)
        dist = sqrt(pow(yhone[0+pl*6],2)+pow(yhone[1+pl*6],2));
        vtrans = sqrt(pow(yhone[3+pl*6],2)+pow(yhone[4+pl*6],2)); 
        fprintf(directory[pl], "%6d  %.18e  %.10e  %.10e\n", transitcount[pl], t2, dist, vtrans);
#endif

        if (TTVCHISQ) {
          if (ttvi[pl] >= 0 && transitcount[pl] == NTTV[pl][ttvi[pl]+1]) { 
#if (demcmc_compile==0)
            MTTV[pl][ttvi[pl]+1]=t2;
#endif
            ttvarr2[pl][ttvi[pl]]=t2;
            ttvi[pl]-=1;
          }
        }
#if ( (demcmc_compile == 3) )
        if (t2 < tbvmax) {
          double bsign=1.;
          if (yhone[1+pl*6] < 0.) bsign=-1;
          dist = sqrt(pow(yhone[0+pl*6],2)+pow(yhone[1+pl*6],2));
          vtrans = sqrt(pow(yhone[3+pl*6],2)+pow(yhone[4+pl*6],2)); 
          fprintf(directory[pl], "%6d  %.18e  %.10e  %.10e\n", transitcount[pl], t2, bsign*dist, vtrans);
        }
#endif

#if (demcmc_compile != 3)
        transitarr[ntransits*ntarrelem+0] = t2;
        transitarr[ntransits*ntarrelem+1] = yhone[0+pl*6]; 
        transitarr[ntransits*ntarrelem+2] = yhone[1+pl*6];
        transitarr[ntransits*ntarrelem+3] = yhone[3+pl*6];
        transitarr[ntransits*ntarrelem+4] = yhone[4+pl*6];
        transitarr[ntransits*ntarrelem+5] = rad[pl];
        ntransits++;
        if (ntransits > maxtransits) {
          printf("Too many transits - increase maxtransits\n");
          fflush(stdout);
          toomany=1;
          goto exitgracefully; 
        }
#endif

      }
    }
  }

  if (RVS) {
    free(rvtarr);
  }

  long negtransits;
  negtransits = ntransits-postransits;
  
#if ( demcmc_compile==3)
  int negstability = 0;
  if (stabilitystatus==1) negstability=1;
  if (negstability) fprintf(stabout, "System went unstable at t=%lf as 1 or more planets had a deviation from its original semi-major axis by %lf or more\n", tfirst, AFRACSTABLE); 
  if (posstability) fprintf(stabout, "%lf : System went unstable at t=%lf as 1 or more planets had a deviation from its original semi-major axis by %lf or more\n", tlast, tlast, AFRACSTABLE);
  fclose(stabout);
  fclose(stabout2);
  free(amin); free(amax);
  free(orbarray);
#endif


#if ( demcmc_compile == 0 || demcmc_compile == 3)

  for (i=0; i<npl; i++){
    fclose(directory[i]);
  }
  free(directory);

#endif


#if (demcmc_compile != 3)
  long timelistlen; 
  timelistlen = kk;

  double *timelist;
  double *temptimelist;
  double *fluxlist;
  double *errlist;
 
  double *rvtimelist;
  double *rvlist;
  double *rverrlist;

  //array of times, measured, and theory data, and error
  tmte = malloc(4*sofds);
  tmte[2] = malloc((kk+1)*sofd);

  temptimelist= &tfe[0][1];
  fluxlist= &tfe[1][1];
  errlist= &tfe[2][1];
  tmte[0] = &tfe[0][0];
  tmte[1] = &tfe[1][0];
  tmte[3] = &tfe[2][0];

  int tele;
  int maxteles;
  maxteles=10;
  double *rvoffset;
  rvoffset = malloc(maxteles*sofd);
  if (RVS) {
    // Compute RV offset
    double rvdiff;
    long vvw;
    double weight;

    long vvt=1;
    tele=0;
    while (vvt<(vv+1)) {
      double numerator = 0;
      double denominator = 0;
      for (vvw=1; vvw<(vv+1); vvw++) {
        if ( (int) tve[4][vvw] == tele ) {
          rvdiff = rvarr[vvw] - tve[1][vvw];
          weight = 1.0 / pow(tve[2][vvw], 2);
          numerator += rvdiff*weight;
          denominator += weight;
        }
      }
      rvoffset[tele] = numerator/denominator;
      for (vvw=1; vvw<(vv+1); vvw++) {
        if ( (int) tve[4][vvw] == tele ) {
          rvarr[vvw] -= rvoffset[tele];
          vvt+=1;
        }
      }
      tele+=1;
    }
  }

  rvtmte = malloc(4*sofds);

  rvtimelist=&tve[0][1];
  rvlist=&tve[1][1];
  rverrlist=&tve[2][1];
  rvtmte[0] = &tve[0][0];
  rvtmte[1] = &tve[1][0];
  rvtmte[3] = &tve[2][0];
  rvtmte[2] = rvarr;
  
  
  double **ttvtmte;
  if (TTVCHISQ) {
    int ki;
    int ksofar=0;
    for (i=0; i<npl; i++) {
      for (ki=0; ki<NTTV[i][0]; ki++) {
        ttvarr[ksofar+ki+1] = ttvarr2[i][ki];
      }
      ksofar += NTTV[i][0];
    }
    for (i=0; i<npl; i++) {
      free(ttvarr2[i]);
    }
    free(ttvarr2);
    ttvtmte = malloc(4*sofds);
    ttvtmte[0] = &nte[0][0];
    ttvtmte[1] = &nte[1][0];
    ttvtmte[3] = &nte[2][0];
    ttvtmte[2] = ttvarr;
  }

  double *temptlist; 
  double *timedlc ( double *times, int *cadences, long ntimes, double **transitarr, int nplanets, double rstar, double c1, double c2); //prototype
  double *binnedlc ( double *times, int *cadences, long ntimes, double binwidth, int nperbin,  double **transitarr, int nplanets, double rstar, double c1, double c2);

  long marker, l;
  int ntran;
  l=0;  
  marker = 0;
  ntran=1;
  double vel;
  double lhs;
  double rhs;


  qsort(transitarr, ntransits, (ntarrelem*sofd), compare);

  double *tclist;
  double *order;

  long q;
  if (cadenceswitch==2) {
    if (OOO == 2) {
      OOO = 0;
      for (q=0; q<kk-1; q++) {
        if (temptimelist[q+1] < temptimelist[q]) {
          OOO = 1;
          break;
        }
      }
    }
    if (OOO) {
      
      timelist = malloc(kk*sofd);
      long ti;
      for (ti=0; ti<kk; ti++) {
        timelist[ti] = tfe[0][ti+1];
      }

      int nsort=3;
      tclist = malloc((nsort*sofd)*kk); 
      order = malloc(sofd*kk);
      for (q=0; q<kk; q++) {
        tclist[nsort*q] = timelist[q];
        tclist[nsort*q+1] = (double) q;
        tclist[nsort*q+2] = (double) cadencelist[q];
      }
      qsort(tclist, kk, sofd*nsort, compare);
      for (q=0; q<kk; q++) {
        timelist[q] = tclist[nsort*q];
        cadencelist[q] = (int) tclist[nsort*q+2];
        order[q] = (long) tclist[nsort*q+1];
      }
      free(tclist);
    } else {
      timelist = temptimelist;
    }
  } else {
    timelist = temptimelist;
  }

  double omdilute;
  omdilute = 1.0-dilute;

  // cycle through each transit w/ index l
  while (l<ntransits-1) {
    // velocity of planet relative to star
    vel = sqrt( pow(transitarr[l*ntarrelem+3],2) + pow(transitarr[l*ntarrelem+4],2) );
    if (vel <= 0) {
      printf("Error, invalid planet velocity\n");
      exit(0);
    }
    // check if transits overlap or are close
    lhs = transitarr[l*ntarrelem+0] + 5.*rstar*RSUNAU/vel;
    rhs = transitarr[(l+1)*ntarrelem+0];
    // if they don't, compute the light curve (either w/ 1 planet or however many previous overlaps there were)
    if ( lhs<= rhs) {
      double **temparr = malloc(sofds*ntran);
      double minvel=vel;
      int ii;
      for (ii=0; ii<ntran; ii++) {
        temparr[ii] = malloc(ntarrelem*sofd);
        memcpy(temparr[ii], &transitarr[((l-ntran)+1+ii)*ntarrelem], ntarrelem*sofd);
        minvel = fmin(minvel, sqrt(pow(temparr[ii][3],2) + pow(temparr[ii][4],2)));
      }

      double startt, stopt;
      double *trantlist;
      int *tranclist;
      startt = temparr[0][0] - 3.*rstar*RSUNAU/minvel;
      stopt = temparr[ntran-1][0] + 3.*rstar*RSUNAU/minvel;
      while (timelist[marker]<startt && marker<kk-1) {
        tmte[2][marker+1]=1.0;       
        marker++;
      }
      long ntimes=0; 
      while (timelist[marker+ntimes]<stopt && marker+ntimes<kk-1) {
        ntimes++; 
      } 
      if (ntimes!=0) {
        trantlist = &timelist[marker];
        if (cadenceswitch==2)  tranclist = &cadencelist[marker];
        if (cadenceswitch == 1 || cadenceswitch == 2) {
          temptlist = binnedlc( trantlist, tranclist, ntimes, binwidth, nperbin, temparr, ntran, rstar, c1, c2); 
        }
        if (cadenceswitch == 0) {
          temptlist = timedlc( trantlist, tranclist, ntimes, temparr, ntran, rstar, c1, c2);
        }
        if (temptlist!=NULL) {
          memcpy(&tmte[2][marker+1],&temptlist[0], ntimes*sofd);
          free(temptlist);
        }
      }
      marker+=ntimes;
      for (ii=0; ii<ntran; ii++) {
        free(temparr[ii]);
      } 
      free(temparr);
      ntran=1;
    } else {
      ntran+=1;
    }
    l++;
  }


  //last case - compute last transit (w/ any overlaps)
  if (ntransits>0) {
    l=ntransits-1;
    double **temparr = malloc(sofds*ntran);
    vel = sqrt( pow(transitarr[l*ntarrelem+3],2) + pow(transitarr[l*ntarrelem+4],2) );
    double minvel=vel;
    int ii;
    for (ii=0; ii<ntran; ii++) {
      temparr[ii] = malloc(ntarrelem*sofd);
      memcpy(temparr[ii], &transitarr[((l-ntran)+1+ii)*ntarrelem], ntarrelem*sofd);
      minvel = fmin(minvel, sqrt(pow(temparr[ii][3],2) + pow(temparr[ii][4],2)));
    }
    double startt, stopt;
    double *trantlist;
    int *tranclist;
    startt = temparr[0][0] - 3.*rstar*RSUNAU/minvel;
    stopt = temparr[ntran-1][0] + 3.*rstar*RSUNAU/minvel;
    while (timelist[marker]<startt && marker<kk-1) {
      tmte[2][marker+1]=1.0;       
      marker++;
    }
    long ntimes=0;
    while (timelist[marker+ntimes]<stopt && marker+ntimes<kk-1) {
      ntimes++; 
    } 
    if (ntimes!=0) {
      trantlist = &timelist[marker];
      if (cadenceswitch==2)  tranclist = &cadencelist[marker];
      if (cadenceswitch == 1 || cadenceswitch == 2) {
        temptlist = binnedlc( trantlist, tranclist, ntimes, binwidth, nperbin, temparr, ntran, rstar, c1, c2); 
      }
      if (cadenceswitch == 0) {
        temptlist = timedlc( trantlist, tranclist, ntimes, temparr, ntran, rstar, c1, c2);
      }
      if (temptlist!=NULL) {
        memcpy(&tmte[2][marker+1],&temptlist[0], ntimes*sofd);
        free(temptlist);
      }
    }
    marker+=ntimes;
  
    for (ii=0; ii<ntran; ii++) {
      free(temparr[ii]);
    } 
    free(temparr);
  }

  while (marker<kk) {
    tmte[2][marker+1] = 1.0;
    marker++;
  }
 
  if ( !(dbleq(dilute, 0.0)) ) {
    for (l=0; l<kk; l++) {
      tmte[2][l] = tmte[2][l]*omdilute+dilute;
    }
  }

  if (cadenceswitch==2) {
    if (OOO) { 
      int nsort=4;
      tclist = malloc((nsort*sofd)*kk); 
      for (q=0; q<kk; q++) {
        tclist[nsort*q] = (double) order[q]; //timelist[q];
        tclist[nsort*q+1] = tmte[2][q+1]; //  (double) q;
        tclist[nsort*q+3] = (double) cadencelist[q]; //  (double) q;
      }
      qsort(tclist, kk, sofd*nsort, compare);
      for (q=0; q<kk; q++) {
        tmte[2][q+1] = tclist[nsort*q+1];
        cadencelist[q] = (int) tclist[nsort*q+3];
      }
      free(tclist);
      free(order);
      free(timelist);
    }
  }




  tmte[0][0] = kk;
  tmte[1][0] = kk;
  tmte[2][0] = kk;
  tmte[3][0] = kk;



#if (demcmc_compile==0)

  char tmtefstr[1000];
  strcpy(tmtefstr, "lc_");
  strcat(tmtefstr, OUTSTR);
  strcat(tmtefstr, ".lcout");
  FILE *tmtef;
  tmtef = fopen(tmtefstr, "a");
  long ijk;
  if (CADENCESWITCH==2) {
    for (ijk=1; ijk<=kk; ijk++) {
      fprintf(tmtef, "%lf %lf %lf %lf %i\n", tmte[0][ijk], tmte[1][ijk], tmte[2][ijk], tmte[3][ijk], cadencelist[ijk-1]);
    }
  } else {
    for (ijk=1; ijk<=kk; ijk++) {
      fprintf(tmtef, "%lf %lf %lf %lf\n", tmte[0][ijk], tmte[1][ijk], tmte[2][ijk], tmte[3][ijk]);
    }
  }
  fclose(tmtef);

  int nbodies;
  nbodies=npl+1;
  if (RVS) {
    char rvtmtefstr[1000];
    char num[10];
    strcpy(rvtmtefstr, "rv");
    strcat(rvtmtefstr, "00");
    strcat(rvtmtefstr, "_");
    strcat(rvtmtefstr, OUTSTR);
    strcat(rvtmtefstr, ".rvout");
    FILE *rvtmtef = fopen(rvtmtefstr, "a");
    for (ijk=1; ijk<vv+1; ijk++) {
      fprintf(rvtmtef, "%lf\t%e\t%e\t%e\t%i\n", rvtmte[0][ijk], rvtmte[1][ijk]/MPSTOAUPD, rvtmte[2][ijk]/MPSTOAUPD, rvtmte[3][ijk]/MPSTOAUPD, (int) tve[4][ijk]);
    }
    int l;
    for (l=0; l<tele; l++) fprintf(rvtmtef, "RV offset %i = %lf\n", l, rvoffset[l]/MPSTOAUPD);
    fclose(rvtmtef);
  }
#endif


  free(rvoffset);

  exitgracefully:;
  if (toomany) {
    tmte = malloc(4*sofds);
    tmte[2] = calloc((kk+1),sofd);
    tmte[2][1] = HUGE_VAL;
    tmte[2][0] = (double) kk;
    tmte[0] = &tfe[0][0];
    tmte[1] = &tfe[1][0];
    tmte[3] = &tfe[2][0];
    tmte[0][0] = (double) kk;
    tmte[1][0] = (double) kk;
    tmte[3][0] = (double) kk;
    rvtmte = malloc(4*sofds);
    rvtmte[0] = &tve[0][0];
    rvtmte[1] = &tve[1][0];
    rvtmte[3] = &tve[2][0];
    if (RVS) {
      rvtmte[2] = calloc(vv+1, sofd);
      rvtmte[2][1] = HUGE_VAL;
      rvtmte[2][0] = (double) vv;
      rvtmte[0][0] = (double) vv;
      rvtmte[1][0] = (double) vv;
      rvtmte[3][0] = (double) vv;
    }
  }

  if (TTVCHISQ) {
    free(ttvi);
    free(ttviinit);
  }

  free(rad); 
  free(transitarr);
  gsl_odeiv_evolve_free(e);
  gsl_odeiv_control_free(c);
  gsl_odeiv_step_free(s);

  double ***fl_rv = malloc(3*sizeof(double**));
  fl_rv[0] = tmte;
  fl_rv[1] = rvtmte;
  fl_rv[2] = ttvtmte;

  
  return fl_rv;

#else

  free(rad); 
  free(transitarr);
  gsl_odeiv_evolve_free(e);
  gsl_odeiv_control_free(c);
  gsl_odeiv_step_free(s);
  double ***empty;
  return empty;

#endif


}



// Read the formatted input file
int getinput(char fname[]) {

  const int sofd = SOFD;
  const int sofi = SOFI;

  FILE *inputf = fopen(fname, "r"); 
  if (inputf == NULL) {
    printf("Bad Input File Name");
    exit(0);
  }

  OUTSTR = malloc(1000*sizeof(char));

  char buffer[1000];
  char type[100];
  char varname[100];
  int i;
  int j;
  for (i=0; i<8; i++) fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %s", type, varname, &OUTSTR[0]); fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &NBODIES); fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);

  NPL = NBODIES-1;

  fscanf(inputf, "%s %s %lf", type, varname, &EPOCH); fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %li", type, varname, &NWALKERS); fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %li", type, varname, &NSTEPS); fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &CADENCESWITCH); fgets(buffer, 1000, inputf); 
  printf("cadenceswitch = %i\n", CADENCESWITCH);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %s", type, varname, TFILE); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &MULTISTAR); fgets(buffer, 1000, inputf); 

  if (MULTISTAR) PPERPLAN = 11;
  else PPERPLAN = 8; 
  printf("pperplan=%i\n", PPERPLAN);
  
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i %i %i", type, varname, &RVJITTERFLAG, &NSTARSRV, &NTELESCOPES); fgets(buffer, 1000, inputf);
  printf("rvjit %i %i %i\n", RVJITTERFLAG, NSTARSRV, NTELESCOPES);
  if (RVJITTERFLAG) {
    RVJITTERTOT = NSTARSRV*NTELESCOPES;
    PSTAR += RVJITTERTOT;//*2
  }
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &TTVCHISQ); fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i %i %i", type, varname, &TTVJITTERFLAG, &NSTARSTTV, &NTELESCOPESTTV); fgets(buffer, 1000, inputf);
  if (TTVJITTERFLAG) {
    TTVJITTERTOT = NSTARSTTV*NTELESCOPESTTV;
    PSTAR += TTVJITTERTOT;//*2
  }

  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &CELERITE); fgets(buffer, 1000, inputf);
  if (CELERITE) {
    PSTAR += NCELERITE;//*2
  }
  printf("celerite=%i\n", CELERITE);
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &RVCELERITE); fgets(buffer, 1000, inputf);
  if (RVCELERITE) {
    PSTAR += NRVCELERITE;//*2
  }
  printf("RVcelerite=%i\n", RVCELERITE);

  printf("pstar=%i\n", PSTAR);

  PARFIX = malloc((PPERPLAN*NPL+PSTAR)*sofi);
  PSTEP = malloc(PPERPLAN*sofd);
  SSTEP = malloc(PSTAR*sofd);
  CSTEP = malloc((PPERPLAN*NPL+PSTAR)*sofd);
  BIMODLIST = malloc((PPERPLAN*NPL+PSTAR)*sofd);
  
  MAXDENSITY = malloc(NPL*sofd);
  EMAX = malloc(NPL*sofd); 
  EPRIORV = malloc(NPL*sofd);

  const int npl = NPL;

  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &SPECTROSCOPY); fgets(buffer, 1000, inputf); 
  fscanf(inputf, "%lf", &SPECRADIUS); fscanf(inputf, "%lf", &SPECERRPOS); fscanf(inputf, "%lf", &SPECERRNEG); fgets(buffer, 1000, inputf);
  printf("spectroscopy = %i, %lf, %lf, %lf\n", SPECTROSCOPY, SPECRADIUS, SPECERRPOS, SPECERRNEG);
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &MASSSPECTROSCOPY); fgets(buffer, 1000, inputf); 
  fscanf(inputf, "%lf", &SPECMASS); fscanf(inputf, "%lf", &MASSSPECERRPOS); fscanf(inputf, "%lf", &MASSSPECERRNEG); fgets(buffer, 1000, inputf);
  printf("MASS spectroscopy = %i, %lf, %lf, %lf\n", MASSSPECTROSCOPY, SPECMASS, MASSSPECERRPOS, MASSSPECERRNEG);
  fgets(buffer, 1000, inputf); //l. 29
  fscanf(inputf, "%s %s %i", type, varname, &DIGT0); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &IGT90); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &INCPRIOR); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &MGT0); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &DENSITYCUTON); fgets(buffer, 1000, inputf); 
  for (i=0; i<npl; i++) fscanf(inputf, "%lf", &MAXDENSITY[i]); fgets(buffer, 1000, inputf); //l.38
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &SQRTE); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &ECUTON); fgets(buffer, 1000, inputf); 
  for (i=0; i<npl; i++) fscanf(inputf, "%lf", &EMAX[i]); fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf); //l.45
  fgets(buffer, 1000, inputf); //l.45
  fscanf(inputf, "%s %s %i", type, varname, &EPRIOR); fgets(buffer, 1000, inputf); 
  for (i=0; i<npl; i++) fscanf(inputf, "%i", &EPRIORV[i]); fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %lf", type, varname, &ESIGMA); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &NTEMPS); fgets(buffer, 1000, inputf); 
  printf("sqrte, ecuton, eprior[0], esigma = %i, %i, %i, %lf\n", SQRTE, ECUTON, EPRIORV[0], ESIGMA);


  for (i=0; i<9; i++) fgets(buffer, 1000, inputf);
  for (i=0; i<npl; i++) {
    for (j=0; j<PPERPLAN; j++) fscanf(inputf, "%i", &PARFIX[PPERPLAN*i+j]); fgets(buffer, 1000, inputf);
  }
  for (i=0; i<PSTAR; i++) fscanf(inputf, "%i", &PARFIX[NPL*PPERPLAN+i]); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &SPLITINCO); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &BIMODF); fgets(buffer, 1000, inputf); 
  printf("bimodf = %i\n", BIMODF);
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  for (i=0; i<npl; i++) {
    for (j=0; j<PPERPLAN; j++) fscanf(inputf, "%i", &BIMODLIST[PPERPLAN*i+j]); fgets(buffer, 1000, inputf);
  }
  for (i=0; i<PSTAR; i++) fscanf(inputf, "%i", &BIMODLIST[NPL*PPERPLAN+i]); fgets(buffer, 1000, inputf);
  // Infrequently Edited Parameters
  for (i=0; i<4; i++) fgets(buffer, 1000, inputf); 
  fscanf(inputf, "%s %s %lf", type, varname, &T0); fgets(buffer, 1000, inputf); 
  printf("t0 = %lf\n", T0);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %lf", type, varname, &T1); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  printf("t1 = %lf\n", T1);
  fscanf(inputf, "%s %s %lu", type, varname, &SEED); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %lf", type, varname, &DISPERSE); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %lf", type, varname, &OPTIMAL); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %lf", type, varname, &RELAX); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &NPERBIN); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %lf", type, varname, &BINWIDTH); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &LTE); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %lf", type, varname, &OFFSETMULT); fgets(buffer, 1000, inputf); 
  fscanf(inputf, "%s %s %lf", type, varname, &OFFSETMIN); fgets(buffer, 1000, inputf); 
  fscanf(inputf, "%s %s %lf", type, varname, &OFFSETMINOUT); fgets(buffer, 1000, inputf); 
  fscanf(inputf, "%s %s %lf", type, varname, &DIST2DIVISOR); fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &XYZFLAG); fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  if (XYZFLAG==1 || XYZFLAG==2 || XYZFLAG==3) {
    XYZLIST=malloc(npl*sofi);
    fscanf(inputf, "%s %s", type, varname);
    for (j=0; j<npl; j++) fscanf(inputf, "%i", &XYZLIST[j]);
    fgets(buffer, 1000, inputf);
  } else {
    XYZLIST=calloc(npl, sofi);
    fgets(buffer, 1000, inputf);
  }
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s", type, varname); for (j=0; j<PPERPLAN; j++) fscanf(inputf, "%lf", &PSTEP[j]); fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s", type, varname); for (i=0; i<PSTAR; i++) fscanf(inputf, "%lf", &SSTEP[i]); fgets(buffer, 1000, inputf); 
  fgets(buffer, 1000, inputf);
  fscanf(inputf, "%s %s %i", type, varname, &STEPFLAG); fgets(buffer, 1000, inputf); 

  if (STEPFLAG) {
    for (i=0; i<9; i++) fgets(buffer, 1000, inputf);
    for (i=0; i<npl; i++) {
      for (j=0; j<PPERPLAN; j++) fscanf(inputf, "%lf", &CSTEP[PPERPLAN*i+j]); fgets(buffer, 1000, inputf);
    }
    for (i=0; i<PSTAR; i++) fscanf(inputf, "%lf", &CSTEP[NPL*PPERPLAN+i]);
    STEP = CSTEP;
  } else {
    STEP = malloc((PPERPLAN*NPL+PSTAR)*sofd);
    free(CSTEP);
    for (i=0; i<npl; i++) memcpy(&STEP[i*PPERPLAN], &PSTEP[0], PPERPLAN*sofd);
    memcpy(&STEP[NPL*PPERPLAN], &SSTEP[0], PSTAR*sofd);
  }

  fclose(inputf);
  return 0;
}


// Compute mean longitude from f, e, pomega
double getlambda(double f, double e, double pomega) {
  double bige = 2*atan(sqrt((1-e)/(1+e)) * tan(f/2));
  double lambda = pomega + bige - e*sin(bige);
  return lambda;
}


// Compute true anomaly from lambda, e, pomega
double getf(double lambda, double e, double pomega) {
  double bigm = lambda-pomega;
  double bige = bigm;
  int i;
  for (i=0; i<=25; i++) {
    bige = bigm + e*sin(bige);
  }
  double sendback = 2.0*atan( sqrt(1.0+e)/sqrt(1.0-e) * tan(bige/2.0) );
  return sendback;
}


// Modulo pi function
double *pushpi(double x[], long int lenx) {

  long int cyclecount = 0;
  long int i;
  for (i=0; i<lenx;i++){
    x[i] += 2.0*M_PI*cyclecount;
    while (x[i] > M_PI) {
      x[i] -= 2.0*M_PI;
      cyclecount -= 1;
    }
    while (x[i] < -M_PI) {
      x[i] += 2.0*M_PI;
      cyclecount += 1;
    }
  }
 
  return x;
}


// Get planetary coordinates in the i, omega, Omega reference frame
double *rotatekep(double x, double y, double i, double lo, double bo) {

  const int sofd = SOFD;

  double z=0.;
  double x1 = cos(lo)*x-sin(lo)*y;
  double y1 = sin(lo)*x+cos(lo)*y;
  double z1 = z;
  double x2 = x1;
  double y2 = cos(i)*y1-sin(i)*z1;
  double z2 = sin(i)*y1+cos(i)*z1;
  double x3 = cos(bo)*x2-sin(bo)*y2;
  double y3 = sin(bo)*x2+cos(bo)*y2;
  double z3 = z2;

  double *arr3 = malloc(3*sofd);
  arr3[0]=x3;arr3[1]=y3;arr3[2]=z3;
  return arr3;
}


// compute x, y, z, vx, vy, vz array from a, e, i, omega, Omega, f, mass array
double *keptostate(double a, double e, double i, double lo, double bo, double f, double m) {

  const int sofd = SOFD;

  double mass = m;
  if (isnan(m)) {
    printf("mass error\n");
    exit(0);
  }
  double r = a*(1.0-pow(e,2))/(1.0+e*cos(f));
  double x0 = r*cos(f);
  double y0 = r*sin(f);
  double vx0 = -sqrt(mass/(a*(1.0-pow(e,2)))) * sin(f);
  double vy0 = sqrt(mass/(a*(1.0-pow(e,2)))) * (e+cos(f));

  double *statearr = malloc(6*sofd);  
  double *state1arr, *state2arr;
  state1arr = rotatekep(x0, y0, i, lo, bo);
  state2arr = rotatekep(vx0, vy0, i, lo, bo);
  memcpy(statearr,state1arr,3*sofd);
  memcpy(statearr+3,state2arr,3*sofd);
  free(state1arr);free(state2arr);

  return statearr;

}


// Compute  a, e, i, omega, Omega, f array from x, y, zy, vx, vy, vz, mass array
double *statetokep (double x, double y, double z, double vx, double vy, double vz, double m) {

  double mass = G*m;

  double rsq = x*x + y*y + z*z;
  double r = sqrt(rsq);
  double vsq = vx*vx + vy*vy + vz*vz;
  double hx = y*vz - z*vy;
  double hy = z*vx - x*vz;
  double hz = x*vy - y*vx;
  double hsq = hx*hx + hy*hy + hz*hz;
  double h = sqrt(hsq);

  double rdot;
  if (vsq <= hsq/rsq) rdot=0.0;
  else rdot = sqrt( vsq - hsq/rsq );
  double rrdot = x*vx + y*vy + z*vz;
  if (rrdot < 0.0) rdot *= -1.0;

  double a = 1.0/(2.0/r - vsq/mass);
  double e, e1;
  e1 = 1.0 - hsq/(a*mass); // This little check is due to rounding errors when e=0.0 exactly causing negative square roots. 
  if (e1 <= 0) e=0.0;
  else e=sqrt(e1);
  double i = acos(hz/h);

  double sini = sin(i);
  double bo, lof;
  if (dbleq(sini, 0.0)) {
    bo = 0.0;
    lof = atan2(y/r, x/r);
  } else {
    double sinbo = hx / (h*sini);
    double cosbo = -hy / (h*sini);
    bo = atan2(sinbo, cosbo);

    double sinlof = z/(r*sini);
    double coslof = (1.0/cos(bo)) * (x/r + sin(bo)*sinlof*hz/h);
    lof = atan2(sinlof, coslof);
  }

  double lo, f;
  if (dbleq(e, 0.0)) {
    lo = 0.0;
    f = lof;
  } else {
    double sinf = a*(1.0-e*e)/(h*e)*rdot;
    double cosf = (1.0/e)*(a*(1.0-e*e)/r - 1.0);
    f = atan2(sinf, cosf);
    double lofmf = lof-f;
    double *lotemp = pushpi(&lofmf, 1);
    lo = lotemp[0];

  }


  double sofd = SOFD;
  double *kepelements = malloc(6*sofd); 
  kepelements[0] = a;
  kepelements[1] = e;
  kepelements[2] = i;
  kepelements[3] = lo;
  kepelements[4] = bo;
  kepelements[5] = f;

  return kepelements;

}


// Compute P, T0, e, i, Omega, omega array from a, e, i, omega, Omega, f, mass array
double *keptoorb(double a, double e, double i, double lo, double bo, double f, double mass) {

  double period = sqrt( (pow(a, 3)*4.0*M_PI*M_PI) / (G * mass) );
  double Tepoch = EPOCH;
  double E0 = 2.0*atan(sqrt( (1.0-e)/(1.0+e) ) * tan((M_PI/2.0-lo)/2.0) ); 
  double Ef = 2.0*atan(sqrt( (1.0-e)/(1.0+e) ) * tan(f/2.0) );
  double t0 = period/(2.0*M_PI) * ( 2.0*M_PI/period*Tepoch - Ef + E0 + e*sin(Ef) - e*sin(E0) );
  if (t0 < Tepoch) {
    do t0 += period;
    while (t0 < Tepoch);
  } else {
    while (t0 >= (Tepoch+period)) t0 -= period;
  }

  int sofd = SOFD;
  double *orbelements = malloc(6*sofd);
  orbelements[0] = period;
  orbelements[1] = t0;
  orbelements[2] = e;
  orbelements[3] = i;
  orbelements[4] = bo;
  orbelements[5] = lo;

  return orbelements;

}


// Compute Eccentric Anomaly from M and e
double getE(double M, double e) {
  int i;
  double bige=M;
  for (i=0; i<=25; i++) {
    bige = M + e*sin(bige);
  }
  return bige;
}


// Compute P, T0, e, i, Omega, omega vector from a, e, i, omega, Omega, MeanAnomoly, mass vector
double *keptoorbmean(double a, double e, double i, double lo, double bo, double M, double mass) {

  double period = sqrt( (pow(a, 3)*4.0*M_PI*M_PI) / (G * mass) );
  double Tepoch = EPOCH;
  double E0 = 2.0*atan(sqrt( (1.0-e)/(1.0+e) ) * tan((M_PI/2.0-lo)/2.0) ); 
  double Ef = getE(M, e); 
  double t0 = period/(2.0*M_PI) * ( 2.0*M_PI/period*Tepoch - Ef + E0 + e*sin(Ef) - e*sin(E0) );
  if (t0 < Tepoch) {
    do t0 += period;
    while (t0 < Tepoch);
  } else {
    while (t0 >= (Tepoch+period)) t0 -= period;
  }

  int sofd = SOFD;
  double *orbelements = malloc(6*sofd);
  orbelements[0] = period;
  orbelements[1] = t0;
  orbelements[2] = e;
  orbelements[3] = i;
  orbelements[4] = bo;
  orbelements[5] = lo;

  return orbelements;

}


// Compute Orbital xyzvxvyvz array from inputted orbital elements vector
double ***dsetup2 (double *p, const int npl){
  const int sofd = SOFD;
  const int sofds = SOFDS; 
 
  int pperplan = PPERPLAN;
  int pstar = PSTAR;

  double epoch = EPOCH;

  double brightstar=0;
  double bsum=0;
  int i;
  if (MULTISTAR) {
    for (i=0; i<npl; i++) bsum+=p[i*pperplan+8];
    brightstar=1.0-bsum;
    if (brightstar <= 0.) {
      printf("Central star has zero or negative brightness. Input must be incorrect (ds2)\n");
      exit(0);
    }
  }

  double ms = p[npl*pperplan+0];
  double rstar = p[npl*pperplan+1];
  double c1 = p[npl*pperplan+2];
  double c2 = p[npl*pperplan+3];
  double dilute = p[npl*pperplan+4];

  double bigg = 1.0e0; //Newton's constant
  double ghere = G; //2.9591220363e-4; 
  double jos = 1.0/MSOMJ;  //9.545e-4; //M_jup/M_sol

  double *mp = malloc(npl*sofd);
  double *mpjup = malloc(npl*sofd);
  double *msys = malloc((npl+1)*sofd);
  msys[0] = ms;  

  double *a = malloc(npl*sofd);
  double *e = malloc(npl*sofd);
  double *inc = malloc(npl*sofd);
  double *bo = malloc(npl*sofd); 
  double *lo = malloc(npl*sofd);
  double *lambda = malloc(npl*sofd);
  double *f = malloc(npl*sofd);   

  for (i=0;i<npl; i++) {
    if (SQRTE) {
      e[i] = pow( sqrt(pow(p[i*pperplan+2],2)+pow(p[i*pperplan+3],2)), 2);
    } else {
      e[i] = sqrt(pow(p[i*pperplan+2],2)+pow(p[i*pperplan+3],2));
    }
    inc[i] = p[i*pperplan+4]*M_PI/180;
    bo[i] = p[i*pperplan+5]*M_PI/180;
    lo[i] = atan2(p[i*pperplan+3],p[i*pperplan+2]);
    mp[i]= p[i*pperplan+6];
 
    mpjup[i] = mp[i]*jos;       //          ; M_Jup
    msys[i+1] = msys[i]+mpjup[i];
    a[i] = cbrt(ghere*(msys[i+1])) * pow(cbrt(p[i*pperplan+0]),2) * pow(cbrt(2*M_PI),-2);

    double pomega = bo[i]+lo[i];
    double lambda0 = getlambda( (M_PI/2-lo[i]), e[i], pomega);
    double m0 = lambda0-pomega;
    double me = m0 + 2*M_PI*(epoch - p[i*pperplan+1])/p[i*pperplan+0]; 
    double mepomega = me+pomega;
    double *lambdaepoint = pushpi(&mepomega,1);
    double lambdae = lambdaepoint[0];
    f[i] = getf(lambdae, e[i], pomega);

  }

  double **state = malloc(npl*sofds);
  for (i=0; i<npl;i++) {
    state[i] = keptostate(a[i],e[i],inc[i],lo[i],bo[i],f[i],(ghere*msys[i+1]));
    int j;
    for (j=0; j<6; j++) {
      state[i][j] = -state[i][j];
    }
  }


  // jacobian
  double *sum;
  if (XYZFLAG==1) {
    sum=calloc(6,sofd);
    if(XYZLIST[0]) {
      int j;
      for (j=0; j<6; j++) state[0][j] = p[j];
    }
    free(sum);
  }

#if (demcmc_compile == 0) 
  if (CONVERT) {
    char outfile2str[80];
    strcpy(outfile2str, "xyz_out_");
    strcat(outfile2str, OUTSTR);
    strcat(outfile2str, ".pldin");
    FILE *outfile2 = fopen(outfile2str, "a");
    fprintf(outfile2, "planet                 x                        y                      z                      v_x                   v_y                   v_z                       m                      rpors             ");
    if (MULTISTAR) {
      fprintf(outfile2, "brightness               c1                    c2\n");
    } else {
      fprintf(outfile2, "\n");
    }
    double pnum = 0.1;
    for (i=0; i<NPL; i++) {
      fprintf(outfile2, "%1.1lf", pnum);
      int j;
      for (j=0; j<6; j++) {
        fprintf(outfile2, "\t%.15lf", state[i][j]);
      }
      fprintf(outfile2, "\t%.15lf", mpjup[i]/jos);
      fprintf(outfile2, "\t%.15lf", p[i*pperplan+7]);
      if (MULTISTAR) {
        for (j=8; j<11; j++) {
          fprintf(outfile2, "\t%.15lf", p[i*pperplan+j]);
        }
      }

      fprintf(outfile2, "\n");
      pnum+=0.1;
    }
    fprintf(outfile2, "%.15lf ; mstar\n", ms);
    fprintf(outfile2, "%.15lf ; rstar\n", rstar);
    fprintf(outfile2, "%.15lf ; c1\n", c1);
    fprintf(outfile2, "%.15lf ; c2\n", c2);
    fprintf(outfile2, "%.15lf ; dilute\n", dilute);
            if (RVJITTERFLAG) {
              for (i=0; i<RVJITTERTOT; i++) {
                fprintf(outfile2, "%.15lf ; rvjitter \n", p[npl*pperplan+5+i]);
              }
            }
            if (TTVJITTERFLAG) {
              for (i=0; i<TTVJITTERTOT; i++) {
                fprintf(outfile2, "%.15lf ; rvjitter \n", p[npl*pperplan+5+RVJITTERTOT+i]);
              }
            }
            if (CELERITE) {
              for (i=0; i<NCELERITE; i++) {
                fprintf(outfile2, "%.15lf ; celerite \n", p[npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+i]);
              }
            }
            if (RVCELERITE) {
              for (i=0; i<NRVCELERITE; i++) {
                fprintf(outfile2, "%.15lf ; RV celerite \n", p[npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+i]);
              }
            }
    fprintf(outfile2, " ; These coordinates are Jacobian \n");

    fclose(outfile2);
  }
#endif


  
  double **statenew = malloc(npl*sofds);
  for (i=0; i<npl; i++) {
    statenew[i] = malloc(6*sofd);
  }

  memcpy(statenew[0], state[0], 6*sofd);
  int j;
  sum = calloc(6,sofd);
  for (i=1; i<npl; i++){
    int j;
    for (j=0; j<6; j++) {
      sum[j] += state[i-1][j]*mpjup[i-1]/msys[i];
      statenew[i][j] = state[i][j]+sum[j];
    }
  }
  free(sum);


  //barycentric
  if (XYZFLAG==3) {
    //printf("barycentric coordinate input is broken at the moment. try using stellar centric instead.\n");
    //exit(0);
    double *starpos = calloc(6,sofd);
    int j;
    for (j=0; j<6; j++) {
      for (i=0; i<npl; i++) {
        if (XYZLIST[i]) {
          starpos[j] -= p[i*pperplan+j]*mpjup[i];
        } else {
          starpos[j] -= state[i][j]*mpjup[i];
        }
      }
      starpos[j] /= ms;
    }
    for (i=0; i<npl; i++) {
      if (XYZLIST[i]) {
        int j;
        for (j=0; j<6; j++) {
          statenew[i][j] = p[i*pperplan+j]-starpos[j];
        }
      }
    }
    free(starpos);
  }

  //stellarcentric
  if (XYZFLAG==2) {
    for (i=0; i<npl; i++) {
      if (XYZLIST[i]) {
        int j;
        for (j=0; j<6; j++) {
          statenew[i][j] = p[i*pperplan+j];
        }
      }
    }
  }

#if (demcmc_compile == 0) 
  if (CONVERT) {
    char outfile2str[80];
    strcpy(outfile2str, "xyz_out_");
    strcat(outfile2str, OUTSTR);
    strcat(outfile2str, ".pldin");
    FILE *outfile2 = fopen(outfile2str, "a");
    fprintf(outfile2, "planet                 x                        y                      z                      v_x                   v_y                   v_z                       m                      rpors             ");
    if (MULTISTAR) {
      fprintf(outfile2, "brightness               c1                    c2\n");
    } else {
      fprintf(outfile2, "\n");
    }
    double pnum = 0.1;
    for (i=0; i<NPL; i++) {
      fprintf(outfile2, "%1.1lf", pnum);
      int j;
      for (j=0; j<6; j++) {
        fprintf(outfile2, "\t%.15lf", statenew[i][j]);
      }
      fprintf(outfile2, "\t%.15lf", mpjup[i]/jos);
      fprintf(outfile2, "\t%.15lf", p[i*pperplan+7]);
      if (MULTISTAR) {
        for (j=8; j<11; j++) {
          fprintf(outfile2, "\t%.15lf", p[i*pperplan+j]);
        }
      }

      fprintf(outfile2, "\n");
      pnum+=0.1;
    }
    fprintf(outfile2, "%.15lf ; mstar\n", ms);
    fprintf(outfile2, "%.15lf ; rstar\n", rstar);
    fprintf(outfile2, "%.15lf ; c1\n", c1);
    fprintf(outfile2, "%.15lf ; c2\n", c2);
    fprintf(outfile2, "%.15lf ; dilute\n", dilute);
    fprintf(outfile2, " ; These coordinates are stellar centric\n");

    fclose(outfile2);
  }
  if (CONVERT) {

    double *bary = calloc(6,sofd);
    int j;
    for (j=0; j<6; j++) {
      double mtot=ms;
      for (i=0; i<npl; i++) {
        if (XYZLIST[i]) {
          bary[j] += p[i*pperplan+j]*mpjup[i];
        } else {
          bary[j] += statenew[i][j]*mpjup[i];
        }
        mtot+=mpjup[i];
      }
      bary[j] /= mtot;
    }

    char outfile2str[80];
    strcpy(outfile2str, "xyz_out_");
    strcat(outfile2str, OUTSTR);
    strcat(outfile2str, ".pldin");
    FILE *outfile2 = fopen(outfile2str, "a");
    fprintf(outfile2, "planet                 x                        y                      z                      v_x                   v_y                   v_z                       m                      rpors             ");
    if (MULTISTAR) {
      fprintf(outfile2, "brightness               c1                    c2\n");
    } else {
      fprintf(outfile2, "\n");
    }
    double pnum = 0.0;

    fprintf(outfile2, "%1.1lf", pnum);
    for (j=0; j<6; j++) {
      fprintf(outfile2, "\t%.15lf", -bary[j]);
    }
    fprintf(outfile2, "\t%.15lf", ms/jos);
    fprintf(outfile2, "\t%.15lf", 1.0);
    fprintf(outfile2, "\t%.15lf", brightstar);
    fprintf(outfile2, "\t%.15lf", c1);
    fprintf(outfile2, "\t%.15lf\n", c2);

    pnum+=0.1;
    for (i=0; i<NPL; i++) {
      fprintf(outfile2, "%1.1lf", pnum);
      int j;
      for (j=0; j<6; j++) {
        fprintf(outfile2, "\t%.15lf", statenew[i][j]-bary[j]);
      }
      fprintf(outfile2, "\t%.15lf", mpjup[i]/jos);
      fprintf(outfile2, "\t%.15lf", p[i*pperplan+7]);
      if (MULTISTAR) {
        for (j=8; j<11; j++) {
          fprintf(outfile2, "\t%.15lf", p[i*pperplan+j]);
        }
      }

      fprintf(outfile2, "\n");
      pnum+=0.1;
    }
    fprintf(outfile2, "%.15lf ; mstar\n", ms);
    fprintf(outfile2, "%.15lf ; rstar\n", rstar);
    fprintf(outfile2, "%.15lf ; c1\n", c1);
    fprintf(outfile2, "%.15lf ; c2\n", c2);
    fprintf(outfile2, "%.15lf ; dilute\n", dilute);
    fprintf(outfile2, " ; These coordinates are barycentric\n");

    free(bary);
    fclose(outfile2);
  }
#endif

  double ***integration_in = malloc(4*sizeof(double**));
  integration_in[0] = malloc(sofds);
  integration_in[0][0] = malloc(sofd);
  integration_in[0][0][0] = npl+1;
  integration_in[1] = malloc(sofds);
  integration_in[1][0] = malloc(sofd);
  integration_in[1][0][0] = epoch;
  integration_in[2] = malloc((npl+1)*sofds);
  for (i=0; i<npl+1; i++) {
    integration_in[2][i] = malloc(pperplan*sofd);
  }
  integration_in[2][0][0] = bigg*ms;
  integration_in[2][0][1] = 0.;
  integration_in[2][0][2] = 0.;
  integration_in[2][0][3] = 0.;
  integration_in[2][0][4] = 0.;
  integration_in[2][0][5] = 0.;
  integration_in[2][0][6] = 0.;
  integration_in[2][0][7] = 1.;
  if (MULTISTAR) {
    integration_in[2][0][8] = brightstar;
    integration_in[2][0][9] = c1;
    integration_in[2][0][10] = c2;
  }
  for (i=0; i<npl; i++) {
    integration_in[2][i+1][0] = bigg*mpjup[i];
    int j;
    for (j=0; j<6; j++) {
      integration_in[2][i+1][j+1] = statenew[i][j];
    }
    integration_in[2][i+1][7] = p[i*pperplan+7]; 
    if (MULTISTAR) {
      for (j=8; j<11; j++) {
        integration_in[2][i+1][j] = p[i*pperplan+j];
      }
    }
  }
  integration_in[3] = malloc(sofds);
  if (!MULTISTAR) {
    integration_in[3][0] = malloc(4*sofd);
  } else {
    integration_in[3][0] = malloc(2*sofd);
  }
  integration_in[3][0][0] = rstar;
  if (!MULTISTAR) {
    integration_in[3][0][1] = c1;
    integration_in[3][0][2] = c2;
    integration_in[3][0][3] = dilute;
  } else {
    integration_in[3][0][1] = dilute;
  }




  free(mp);free(mpjup);free(msys);free(a);free(e);free(inc);free(bo);free(lo);free(lambda);free(f);
  for (i=0; i<npl; i++) {
    free(state[i]);free(statenew[i]);
  }
  free(state);free(statenew);

  return integration_in;

}


// Compute Orbital xyzvxvyvz array from inputted orbital elements array
double ***dsetup (double **p, const int npl){
  const int sofd = SOFD;
  const int sofds = SOFDS; 
 
  int pperplan = PPERPLAN;
  int pstar = PSTAR;

  //double ms = MS;
  double epoch = EPOCH;

  double brightstar=0;
  double bsum=0;
  int i;
  if (MULTISTAR) {
    for (i=0; i<npl; i++) bsum+=p[i][8];
    brightstar=1.0-bsum;
    if (brightstar <= 0.) {
      printf("Central star has zero or negative brightness. Input must be incorrect\n");
      exit(0);
    }
  }

  double ms = p[npl][0];
  double rstar = p[npl][1];
  double c1 = p[npl][2];
  double c2 = p[npl][3];
  double dilute = p[npl][4];

  double bigg = 1.0e0; //Newton's constant
  double ghere = G; //2.9591220363e-4; 
  double jos = 1.0/MSOMJ;  //9.545e-4; //M_jup/M_sol

  double *mp = malloc(npl*sofd);
  double *mpjup = malloc(npl*sofd);
  double *msys = malloc((npl+1)*sofd);
  msys[0] = ms;  

  double *a = malloc(npl*sofd);
  double *e = malloc(npl*sofd);
  double *inc = malloc(npl*sofd);
  double *bo = malloc(npl*sofd); 
  double *lo = malloc(npl*sofd);
  double *lambda = malloc(npl*sofd);
  double *f = malloc(npl*sofd);   

 
  for (i=0;i<npl; i++) {
    if (SQRTE) {
      e[i] = pow( sqrt(pow(p[i][2],2)+pow(p[i][3],2)), 2 );
    } else {
      e[i] = sqrt(pow(p[i][2],2)+pow(p[i][3],2));
    }
    inc[i] = p[i][4]*M_PI/180;
    bo[i] = p[i][5]*M_PI/180;
    lo[i] = atan2(p[i][3],p[i][2]);
    mp[i]= p[i][6];
 
    mpjup[i] = mp[i]*jos;       //          ; M_Jup
    msys[i+1] = msys[i]+mpjup[i];
    a[i] = cbrt(ghere*(msys[i+1])) * pow(cbrt(p[i][0]),2) * pow(cbrt(2*M_PI),-2);

    double pomega = bo[i]+lo[i];
    double lambda0 = getlambda( (M_PI/2-lo[i]), e[i], pomega);
    double m0 = lambda0-pomega;
    double me = m0 + 2*M_PI*(epoch - p[i][1])/p[i][0]; 
    double mepomega = me+pomega;
    double *lambdaepoint = pushpi(&mepomega,1);
    double lambdae = lambdaepoint[0];
    f[i] = getf(lambdae, e[i], pomega);

  }

  double **state = malloc(npl*sofds);
  for (i=0; i<npl;i++) {
    state[i] = keptostate(a[i],e[i],inc[i],lo[i],bo[i],f[i],(ghere*msys[i+1]));
    int j;
    for (j=0; j<6; j++) {
      state[i][j] = -state[i][j];
    }
  }

  // jacobian
  double *sum;
  if (XYZFLAG==1) {
    sum=calloc(6,sofd);
    if(XYZLIST[0]) {
      int j;
      for (j=0; j<6; j++) state[0][j] = p[0][j];
    }
    free(sum);
  }

#if (demcmc_compile == 0) 

  if (CONVERT) {
    char outfile2str[80];
    strcpy(outfile2str, "xyz_out_");
    strcat(outfile2str, OUTSTR);
    strcat(outfile2str, ".pldin");
    FILE *outfile2 = fopen(outfile2str, "a");
    fprintf(outfile2, "planet                 x                        y                      z                      v_x                   v_y                   v_z                       m                      rpors             ");
    if (MULTISTAR) {
      fprintf(outfile2, "brightness               c1                    c2\n");
    } else {
      fprintf(outfile2, "\n");
    }
    double pnum = 0.1;
    for (i=0; i<NPL; i++) {
      fprintf(outfile2, "%1.1lf", pnum);
      int j;
      for (j=0; j<6; j++) {
        fprintf(outfile2, "\t%.15lf", state[i][j]);
      }
      fprintf(outfile2, "\t%.15lf", mpjup[i]/jos);
      fprintf(outfile2, "\t%.15lf", p[i][7]);
      if (MULTISTAR) {
        for (j=8; j<11; j++) {
          fprintf(outfile2, "\t%.15lf", p[i][j]);
        }
      }

      fprintf(outfile2, "\n");
      pnum+=0.1;
    }
    fprintf(outfile2, "%.15lf ; mstar\n", ms);
    fprintf(outfile2, "%.15lf ; rstar\n", rstar);
    fprintf(outfile2, "%.15lf ; c1\n", c1);
    fprintf(outfile2, "%.15lf ; c2\n", c2);
    fprintf(outfile2, "%.15lf ; dilute\n", dilute);
    fprintf(outfile2, " ; These coordinates are Jacobian \n");

    fclose(outfile2);
  }

#endif


  double **statenew = malloc(npl*sofds);
  for (i=0; i<npl; i++) {
    statenew[i] = malloc(6*sofd);
  }

  memcpy(statenew[0], state[0], 6*sofd);
  int j;
  sum = calloc(6,sofd);
  for (i=1; i<npl; i++){
    int j;
    for (j=0; j<6; j++) {
      sum[j] += state[i-1][j]*mpjup[i-1]/msys[i];
      statenew[i][j] = state[i][j]+sum[j];
    }
  }
  free(sum);


  //barycentric
  if (XYZFLAG==3) {
    //printf("barycentric coordinate input is broken at the moment. try using stellar centric instead.\n");
    //exit(0);
    double *starpos = calloc(6,sofd);
    int j;
    for (j=0; j<6; j++) {
      for (i=0; i<npl; i++) {
        if (XYZLIST[i]) {
          starpos[j] -= p[i][j]*mpjup[i];
        } else {
          starpos[j] -= state[i][j]*mpjup[i];
        }
      }
      starpos[j] /= ms;
    }
    for (i=0; i<npl; i++) {
      if (XYZLIST[i]) {
        int j;
        for (j=0; j<6; j++) {
          statenew[i][j] = p[i][j]-starpos[j];
        }
      }
    }
    free(starpos);
  }

  //stellarcentric
  if (XYZFLAG==2) {
    for (i=0; i<npl; i++) {
      if (XYZLIST[i]) {
        int j;
        for (j=0; j<6; j++) {
          statenew[i][j] = p[i][j];
        }
      }
    }
  }


#if (demcmc_compile == 0) 
  if (CONVERT) {
    char outfile2str[80];
    strcpy(outfile2str, "xyz_out_");
    strcat(outfile2str, OUTSTR);
    strcat(outfile2str, ".pldin");
    FILE *outfile2 = fopen(outfile2str, "a");
    fprintf(outfile2, "planet                 x                        y                      z                      v_x                   v_y                   v_z                       m                      rpors             ");
    if (MULTISTAR) {
      fprintf(outfile2, "brightness               c1                    c2\n");
    } else {
      fprintf(outfile2, "\n");
    }
    double pnum = 0.1;
    for (i=0; i<NPL; i++) {
      fprintf(outfile2, "%1.1lf", pnum);
      int j;
      for (j=0; j<6; j++) {
        fprintf(outfile2, "\t%.15lf", statenew[i][j]);
      }
      fprintf(outfile2, "\t%.15lf", mpjup[i]/jos);
      fprintf(outfile2, "\t%.15lf", p[i][7]);
      if (MULTISTAR) {
        for (j=8; j<11; j++) {
          fprintf(outfile2, "\t%.15lf", p[i][j]);
        }
      }

      fprintf(outfile2, "\n");
      pnum+=0.1;
    }
    fprintf(outfile2, "%.15lf ; mstar\n", ms);
    fprintf(outfile2, "%.15lf ; rstar\n", rstar);
    fprintf(outfile2, "%.15lf ; c1\n", c1);
    fprintf(outfile2, "%.15lf ; c2\n", c2);
    fprintf(outfile2, "%.15lf ; dilute\n", dilute);
    fprintf(outfile2, " ; These coordinates are stellar centric\n");

    fclose(outfile2);
  }
  if (CONVERT) {

    double *bary = calloc(6,sofd);
    int j;
    for (j=0; j<6; j++) {
      double mtot=ms;
      for (i=0; i<npl; i++) {
        if (XYZLIST[i]) {
          bary[j] += p[i][j]*mpjup[i];
        } else {
          bary[j] += statenew[i][j]*mpjup[i];
        }
        mtot+=mpjup[i];
      }
      bary[j] /= mtot;
    }


    char outfile2str[80];
    strcpy(outfile2str, "xyz_out_");
    strcat(outfile2str, OUTSTR);
    strcat(outfile2str, ".pldin");
    FILE *outfile2 = fopen(outfile2str, "a");
    fprintf(outfile2, "planet                 x                        y                      z                      v_x                   v_y                   v_z                       m                      rpors             ");
    if (MULTISTAR) {
      fprintf(outfile2, "brightness               c1                    c2\n");
    } else {
      fprintf(outfile2, "\n");
    }
    double pnum = 0.0;

    fprintf(outfile2, "%1.1lf", pnum);
    for (j=0; j<6; j++) {
      fprintf(outfile2, "\t%.15lf", -bary[j]);
    }
    fprintf(outfile2, "\t%.15lf", ms/jos);
    fprintf(outfile2, "\t%.15lf", 1.0);
    fprintf(outfile2, "\t%.15lf", brightstar);
    fprintf(outfile2, "\t%.15lf", c1);
    fprintf(outfile2, "\t%.15lf\n", c2);

    pnum+=0.1;
    for (i=0; i<NPL; i++) {
      fprintf(outfile2, "%1.1lf", pnum);
      int j;
      for (j=0; j<6; j++) {
        fprintf(outfile2, "\t%.15lf", statenew[i][j]-bary[j]);
      }
      fprintf(outfile2, "\t%.15lf", mpjup[i]/jos);
      fprintf(outfile2, "\t%.15lf", p[i][7]);
      if (MULTISTAR) {
        for (j=8; j<11; j++) {
          fprintf(outfile2, "\t%.15lf", p[i][j]);
        }
      }

      fprintf(outfile2, "\n");
      pnum+=0.1;
    }
    fprintf(outfile2, "%.15lf ; mstar\n", ms);
    fprintf(outfile2, "%.15lf ; rstar\n", rstar);
    fprintf(outfile2, "%.15lf ; c1\n", c1);
    fprintf(outfile2, "%.15lf ; c2\n", c2);
    fprintf(outfile2, "%.15lf ; dilute\n", dilute);
    fprintf(outfile2, " ; These coordinates are barycentric\n");

    free(bary);
    fclose(outfile2);
  }
#endif

  double ***integration_in = malloc(4*sizeof(double**));
  integration_in[0] = malloc(sofds);
  integration_in[0][0] = malloc(sofd);
  integration_in[0][0][0] = npl+1;
  integration_in[1] = malloc(sofds);
  integration_in[1][0] = malloc(sofd);
  integration_in[1][0][0] = epoch;
  integration_in[2] = malloc((npl+1)*sofds);
  for (i=0; i<npl+1; i++) {
    integration_in[2][i] = malloc(pperplan*sofd);
  }
  integration_in[2][0][0] = bigg*ms;
  integration_in[2][0][1] = 0.;
  integration_in[2][0][2] = 0.;
  integration_in[2][0][3] = 0.;
  integration_in[2][0][4] = 0.;
  integration_in[2][0][5] = 0.;
  integration_in[2][0][6] = 0.;
  integration_in[2][0][7] = 1.;
  if (MULTISTAR) {
    integration_in[2][0][8] = brightstar;
    integration_in[2][0][9] = c1;
    integration_in[2][0][10] = c2;
  }
  for (i=0; i<npl; i++) {
    integration_in[2][i+1][0] = bigg*mpjup[i];
    int j;
    for (j=0; j<6; j++) {
      integration_in[2][i+1][j+1] = statenew[i][j];
    }
    integration_in[2][i+1][7] = p[i][7]; 
    if (MULTISTAR) {
      for (j=8; j<11; j++) {
        integration_in[2][i+1][j] = p[i][j];
      }
    }
  }
  integration_in[3] = malloc(sofds);
  if (!MULTISTAR) {
    integration_in[3][0] = malloc(4*sofd);
  } else {
    integration_in[3][0] = malloc(2*sofd);
  }
  integration_in[3][0][0] = rstar;
  if (!MULTISTAR) {
    integration_in[3][0][1] = c1;
    integration_in[3][0][2] = c2;
    integration_in[3][0][3] = dilute;
  } else {
    integration_in[3][0][1] = dilute;
  }

  free(mp);free(mpjup);free(msys);free(a);free(e);free(inc);free(bo);free(lo);free(lambda);free(f);
  for (i=0; i<npl; i++) {
    free(state[i]);free(statenew[i]);
  }
  free(state);free(statenew);

  return integration_in;

}



// Computes (observation - theory)/error for a [time, obs, theory, err] vector
double *devoerr (double **tmte) {

  const int sofd = SOFD;

  long ntimes = (long) tmte[0][0];
  double *resid = malloc((ntimes+1)*sofd);

  resid[0] = (double) ntimes;
  long i;
  for (i=0; i<ntimes; i++) {
    resid[i+1] = (tmte[1][i+1]-tmte[2][i+1])/tmte[3][i+1];
  }

  return resid;

}



//// These routines are from Pal 2008
// from icirc.c
//
typedef struct
{	double	x0,y0;
	double	r;
} circle;

typedef struct
{	int	cidx;
	double	phi0,dphi;
	int	noidx;
	int	*oidxs;
} arc;

int icirc_arclist_intersections(circle *circles,int ncircle,arc **routs,int *rnout)
{
    int	i,j;
    
    arc	*arcs,*aouts;
    int	*acnt;
    int	naout;
    
    arcs=(arc *)malloc(sizeof(arc)*ncircle*ncircle*2);
    acnt=(int *)malloc(sizeof(int)*ncircle);
    
    for ( i=0 ; i<ncircle ; i++ )
    {	acnt[i]=0;		}
    
    for ( i=0 ; i<ncircle ; i++ )
    { for ( j=0 ; j<ncircle ; j++ )
    {	double	xa,ya,xb,yb,ra,rb;
        double	dx,dy,d;
        double	w,phia,phi0;
        
        if ( i==j )
            continue;
        
        xa=circles[i].x0;
        ya=circles[i].y0;
        ra=circles[i].r;
        xb=circles[j].x0;
        yb=circles[j].y0;
        rb=circles[j].r;
        dx=xb-xa;
        dy=yb-ya;
        d=sqrt(dx*dx+dy*dy);
        if ( ra+rb <= d )
            continue;
        else if ( d+ra <= rb )
            continue;
        else if ( d+rb <= ra )
            continue;
        w=(ra*ra+d*d-rb*rb)/(2*ra*d);
        if ( ! ( -1.0 <= w && w <= 1.0 ) )
            continue;
        phia=acos(w);
        
        phi0=atan2(dy,dx);
        if ( phi0 < 0.0 )	phi0+=2*M_PI;
		
        if ( acnt[i] <= 0 )
        {	arc	*a;
            a=&arcs[2*ncircle*i];
            a[0].phi0=phi0-phia;
            a[0].dphi=2*phia;
            a[1].phi0=phi0+phia;
            a[1].dphi=2*(M_PI-phia);
            acnt[i]=2;
        }
        else
        {	arc	*a;
            double	wp[2],w,dw;
            int	k,n,l;
            wp[0]=phi0-phia;
            wp[1]=phi0+phia;
            a=&arcs[2*ncircle*i];
            n=acnt[i];
            for ( k=0 ; k<2 ; k++ )
            {	w=wp[k];
                for ( l=0 ; l<n ; l++ )
                {	dw=w-a[l].phi0;
                    while ( dw<0.0 )	dw+=2*M_PI;
                    while ( 2*M_PI<=dw )	dw-=2*M_PI;
                    if ( dw<a[l].dphi )
                        break;
                }
                if ( l<n )
                {	memmove(a+l+1,a+l,sizeof(arc)*(n-l));
                    a[l+1].phi0=a[l].phi0+dw;
                    a[l+1].dphi=a[l].dphi-dw;
                    a[l].dphi=dw;
                    n++;
                }
            }
            acnt[i]=n;
        }
        
    }
    }
    
    naout=0;
    for ( i=0 ; i<ncircle ; i++ )
    {	if ( acnt[i] <= 0 )
		naout++;
	else
		naout+=acnt[i];
    }
    aouts=(arc *)malloc(sizeof(arc)*naout);
    j=0;
    for ( i=0 ; i<ncircle ; i++ )
    {	int	k;
        if ( acnt[i] <= 0 )
        {	aouts[j].cidx=i;
            aouts[j].phi0=0.0;
            aouts[j].dphi=2*M_PI;
            j++;
        }
        else
        {	for ( k=0 ; k<acnt[i] ; k++ )
        {	aouts[j].cidx=i;
			aouts[j].phi0=arcs[2*ncircle*i+k].phi0;
			aouts[j].dphi=arcs[2*ncircle*i+k].dphi;
			j++;
        }
        }
    }
    for ( j=0 ; j<naout ; j++ )
    {	double	x,y,dx,dy;
        int	k;
        
        i=aouts[j].cidx;
        if ( acnt[i] <= 0 )
        {	x=circles[i].x0+circles[i].r;
            y=circles[i].y0;
        }
        else
        {	double	phi;
            phi=aouts[j].phi0+0.5*aouts[j].dphi;
            x=circles[i].x0+circles[i].r*cos(phi);
            y=circles[i].y0+circles[i].r*sin(phi);
        }
        aouts[j].noidx=0;
        aouts[j].oidxs=NULL;
        for ( k=0 ; k<ncircle ; k++ )
        {	if ( i==k )	continue;
            dx=x-circles[k].x0;
            dy=y-circles[k].y0;
            if ( dx*dx+dy*dy < circles[k].r*circles[k].r )
            {	aouts[j].oidxs=(int *)realloc(aouts[j].oidxs,sizeof(int)*(aouts[j].noidx+1));
                *(aouts[j].oidxs+aouts[j].noidx)=k;
                aouts[j].noidx++;
            }
        }
    }
    
    if ( routs != NULL )	*routs=aouts;
    if ( rnout != NULL )	*rnout=naout;
    
    free(acnt);
    free(arcs);
    
    return(0);
}

int icirc_arclist_free(arc *arcs,int narc)
{
    int	i;
    for ( i=0 ; i<narc ; i++ )
    {	if ( arcs[i].oidxs != NULL )
		free(arcs[i].oidxs);
    }
    free(arcs);
    return(0);
}


// From elliptic.c
//
//

#define		FMIN(a,b)	((a)<(b)?(a):(b))
#define		FMAX(a,b)	((a)<(b)?(a):(b))
#define		SQR(a)		((a)*(a))

#define C1 0.3
#define C2 (1.0/7.0)
#define C3 0.375
#define C4 (9.0/22.0)

double carlson_elliptic_rc(double x,double y)
{
    double alamb,ave,s,w,xt,yt,ans;
    
    if ( y > 0.0 )
    {	xt=x;
        yt=y;
        w=1.0;
    }
    else
    {	xt=x-y;
        yt = -y;
        w=sqrt(x)/sqrt(xt);
    }
    do
    {	alamb=2.0*sqrt(xt)*sqrt(yt)+yt;
        xt=0.25*(xt+alamb);
        yt=0.25*(yt+alamb);
        ave=(1.0/3.0)*(xt+yt+yt);
        s=(yt-ave)/ave;
    } while ( fabs(s) > 0.0012 );
    
    ans=w*(1.0+s*s*(C1+s*(C2+s*(C3+s*C4))))/sqrt(ave);
    
    return(ans);
}

#undef	C4
#undef	C3
#undef	C2
#undef	C1

double carlson_elliptic_rf(double x,double y,double z)
{
    double	alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt;
    xt=x;
    yt=y;
    zt=z;
    do
    {	sqrtx=sqrt(xt);
        sqrty=sqrt(yt);
        sqrtz=sqrt(zt);
        alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
        xt=0.25*(xt+alamb);
        yt=0.25*(yt+alamb);
        zt=0.25*(zt+alamb);
        ave=(1.0/3.0)*(xt+yt+zt);
        delx=(ave-xt)/ave;
        dely=(ave-yt)/ave;
        delz=(ave-zt)/ave;
    } while ( fabs(delx) > 0.0025 || fabs(dely) > 0.0025 || fabs(delz) > 0.0025 );
    e2=delx*dely-delz*delz;
    e3=delx*dely*delz;
    return((1.0+((1.0/24.0)*e2-(0.1)-(3.0/44.0)*e3)*e2+(1.0/14.0)*e3)/sqrt(ave));
}

#define C1 (3.0/14.0)
#define C2 (1.0/6.0)
#define C3 (9.0/22.0)
#define C4 (3.0/26.0)
#define C5 (0.25*C3)
#define C6 (1.5*C4)

double carlson_elliptic_rd(double x,double y,double z)
{
    double alamb,ave,delx,dely,delz,ea,eb,ec,ed,ee,fac,
	sqrtx,sqrty,sqrtz,sum,xt,yt,zt,ans;
    
    xt=x;
    yt=y;
    zt=z;
    sum=0.0;
    fac=1.0;
    do
    {	sqrtx=sqrt(xt);
        sqrty=sqrt(yt);
        sqrtz=sqrt(zt);
        alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
        sum+=fac/(sqrtz*(zt+alamb));
        fac=0.25*fac;
        xt=0.25*(xt+alamb);
        yt=0.25*(yt+alamb);
        zt=0.25*(zt+alamb);
        ave=0.2*(xt+yt+3.0*zt);
        delx=(ave-xt)/ave;
        dely=(ave-yt)/ave;
        delz=(ave-zt)/ave;
    } while ( fabs(delx) > 0.0015 || fabs(dely) > 0.0015 || fabs(delz) > 0.0015 );
    ea=delx*dely;
    eb=delz*delz;
    ec=ea-eb;
    ed=ea-6.0*eb;
    ee=ed+ec+ec;
    ans=3.0*sum+fac*(1.0+ed*(-C1+C5*ed-C6*delz*ee)
                     +delz*(C2*ee+delz*(-C3*ec+delz*C4*ea)))/(ave*sqrt(ave));
    return(ans);
}

#undef	C6
#undef	C5
#undef	C4
#undef	C3
#undef	C2
#undef	C1

#define C1 (3.0/14.0)
#define C2 (1.0/3.0)
#define C3 (3.0/22.0)
#define C4 (3.0/26.0)
#define C5 (0.75*C3)
#define C6 (1.5*C4)
#define C7 (0.5*C2)
#define C8 (C3+C3)

double carlson_elliptic_rj(double x,double y,double z,double p)
{
    double	a,alamb,alpha,ans,ave,b,beta,delp,delx,dely,delz,ea,eb,ec,
	ed,ee,fac,pt,rcx,rho,sqrtx,sqrty,sqrtz,sum,tau,xt,yt,zt;
    
    sum=0.0;
    fac=1.0;
    if ( p > 0.0 )
    {	xt=x;
        yt=y;
        zt=z;
        pt=p;
        a=b=rcx=0.0;
    }
    else
    {	xt=FMIN(FMIN(x,y),z);
        zt=FMAX(FMAX(x,y),z);
        yt=x+y+z-xt-zt;
        a=1.0/(yt-p);
        b=a*(zt-yt)*(yt-xt);
        pt=yt+b;
        rho=xt*zt/yt;
        tau=p*pt/yt;
        rcx=carlson_elliptic_rc(rho,tau);
    }
    do
    {	sqrtx=sqrt(xt);
        sqrty=sqrt(yt);
        sqrtz=sqrt(zt);
        alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
        alpha=SQR(pt*(sqrtx+sqrty+sqrtz)+sqrtx*sqrty*sqrtz);
        beta=pt*SQR(pt+alamb);
        sum += fac*carlson_elliptic_rc(alpha,beta);
        fac=0.25*fac;
        xt=0.25*(xt+alamb);
        yt=0.25*(yt+alamb);
        zt=0.25*(zt+alamb);
        pt=0.25*(pt+alamb);
        ave=0.2*(xt+yt+zt+pt+pt);
        delx=(ave-xt)/ave;
        dely=(ave-yt)/ave;
        delz=(ave-zt)/ave;
        delp=(ave-pt)/ave;
    } while ( fabs(delx)>0.0015 || fabs(dely)>0.0015 || fabs(delz)>0.0015 || fabs(delp)>0.0015 );
    ea=delx*(dely+delz)+dely*delz;
    eb=delx*dely*delz;
    ec=delp*delp;
    ed=ea-3.0*ec;
    ee=eb+2.0*delp*(ea-ec);
    
    ans=3.0*sum+fac*(1.0+ed*(-C1+C5*ed-C6*ee)+eb*(C7+delp*(-C8+delp*C4))
                     +delp*ea*(C2-delp*C3)-C2*delp*ec)/(ave*sqrt(ave));
    
    if ( p <= 0.0 ) ans=a*(b*ans+3.0*(rcx-carlson_elliptic_rf(xt,yt,zt)));
    
    return(ans);
}

#undef	C6
#undef	C5
#undef	C4
#undef	C3
#undef	C2
#undef	C1
#undef	C8
#undef	C7

#undef			SQR
#undef			FMAX
#undef			FMIN


// From mttr.c
//
//

// This function returns F[phi'] (Eq 34) from r=r, rho=c, x=phi'
double mttr_integral_primitive(double r,double c,double x)
{
    double	q2,s2,d2,sx,cx,w;
    double	rf,rd,rj;
    double	beta;
    double	iret;
    int         d2neq0, cneqr;
    
    // Eq 24-26 with r=r, rho=c, x=phi'
    q2=r*r+c*c+2*r*c*cos(x);
    d2=r*r+c*c-2*r*c;
    s2=r*r+c*c+2*r*c;


    double epsilon=3.0*DBL_EPSILON;
    d2neq0 = !((d2 < epsilon) && ( -d2 < epsilon));
    cneqr = !(((c-r) < epsilon) && (-(c-r) < epsilon));

    sx=sin(x/2);
    cx=cos(x/2);
    
    if ( 1.0<q2 )	q2=1.0;
    
    w=(1-q2)/(1-d2);
    if ( w<0.0 )	w=0.0;
    // Eq 31-33
    rf=carlson_elliptic_rf(w,sx*sx,1);
    rd=carlson_elliptic_rd(w,sx*sx,1);
    if ( d2neq0 && cneqr)	rj=carlson_elliptic_rj(w,sx*sx,1,q2/d2);
    else		        rj=0.0;
    
    // Eq 34 line 1
    beta=atan2((c-r)*sx,(c+r)*cx);
    iret=-beta/3;
    // Eq 34 line 2 first term
    iret+=x/6;
    
    w=cx/sqrt(1-d2);
    
    // Eq (34) lines 2-6
    iret+=
    +2.0/ 9.0*c*r*sin(x)*sqrt(1-q2)
    +1.0/ 3.0*(1+2*r*r*r*r-4*r*r)*w*rf
    +2.0/ 9.0*r*c*(4-7*r*r-c*c+5*r*c)*w*rf
    -4.0/27.0*r*c*(4-7*r*r-c*c)*w*cx*cx*rd;
    if ( d2neq0 && cneqr )
    //if ( d2neq0 )
        iret += 1.0/3.0*w*(r+c)/(r-c)*(rf-(q2-d2)/(3*d2)*rj);
    else
        iret -= 1.0/3.0*w*(r+c)*(q2-d2)*M_PI/(2*q2*sqrt(q2));

    return(iret);
}

double mttr_integral_definite(double r,double c,double x0,double dx)
{
    double	dc,nx;
    double	ret;
    
    
    // Eq (22) or (21) ?
    if ( c<=0.0 )
    {	if ( r<1.0 )
		ret=(1-(1-r*r)*sqrt(1-r*r))*dx/3.0;
	else /* this case implies r==1: */
		ret=dx/3.0;
        
        return(ret);
    }
    
    if ( dx<0.0 )
    {	x0+=dx;
        dx=-dx;
    }
    while ( x0<0.0 )	x0+=2*M_PI;
    while ( 2*M_PI<=x0 )	x0-=2*M_PI;
    
    ret=0.0;
    while ( 0.0<dx )
    {	dc=2*M_PI-x0;
        if ( dx<dc )	dc=dx,nx=x0+dx;
        else		nx=0.0;
        
        ret+=mttr_integral_primitive(r,c,x0+dc)-mttr_integral_primitive(r,c,x0);
        
        x0=nx;
        dx-=dc;
    }
   
    return(ret);
}

/*****************************************************************************/

// This function takes n circles, computes their overlap, and returns
// The c's are coefficients of the stellar flux terms
//c1 is constant term 
//c2 is polynomial term
double mttr_flux_general(circle *circles,int ncircle,double c0,double c1,double c2)
{
    arc	*arcs,*a;
    int	i,narc;
    double	fc,f0;
    
    // Get circle intersections into *arcs
    icirc_arclist_intersections(circles,ncircle,&arcs,&narc);
    
    fc=0.0;
    
    for ( i=0 ; i<narc ; i++ )
    {	double	sign,x0,y0,r,p0,dp,p1,df0,df1,df2;
        double	x2,y2,r2;
        
        a=&arcs[i];
        if ( a->cidx==0 && a->noidx<=0 )
            sign=+1;
        else if ( a->cidx != 0 && a->noidx==1 && a->oidxs[0]==0 )
            sign=-1;
        else
            continue;
        
        x0=circles[a->cidx].x0;
        y0=circles[a->cidx].y0;
        r =circles[a->cidx].r;
        p0=a->phi0;
        dp=a->dphi;
        p1=p0+dp;
        
        x2=x0*x0;
        y2=y0*y0;
        r2=r*r;
        
        // Eq 8 last line
        df0=0.5*r*(x0*(sin(p1)-sin(p0))+y0*(-cos(p1)+cos(p0)))+0.5*r*r*dp;
        
        // If c1, then constant term
        if ( c1 != 0.0 )
        {	double	delta,rho;
            delta=atan2(y0,x0);
            rho=sqrt(x2+y2);
            // this avoids edge cases where rho ~= r and the integral doesn't work properly
            if ( fabs(rho) > 1e-7 && fabs(r) > 1e-7  && fabs(rho-r) < 1e-7)  rho += 2e-7;
            df1=mttr_integral_definite(r,rho,p0-delta,dp);
        }
        else
            df1=0.0;
        
        // If c2 then polynomial term
        if ( c2 != 0.0 )
            // Eq (18)
        {	df2=(r/48.0)*(	+(24*(x2+y2)+12*r2)*r*dp
                          -4*y0*(6*x2+2*y2+9*r2)*(cos(p1)-cos(p0))
                          -24*x0*y0*r*(cos(2*p1)-cos(2*p0))
                          -4*y0*r2*(cos(3*p1)-cos(3*p0))
                          +4*x0*(2*x2+6*y2+9*r2)*(sin(p1)-sin(p0))
                          -4*x0*r2*(sin(3*p1)-sin(3*p0))
                          -r2*r*(sin(4*p1)-sin(4*p0)) );
        }
        else
            df2=0.0;
        
        fc += sign*(c0*df0+c1*df1+c2*df2);
        
        
    }
    
    // normalize
    f0=2.0*M_PI*(c0/2.0+c1/3.0+c2/4.0);
    
    icirc_arclist_free(arcs,narc);
    
    if ( 0.0<f0 )
        return(fc/f0);
    else
        return(0.0);
}


// Lightcurve output functions
//
// This function computes the lightcurve for a single time given the positions of the planets and properties of the star
double onetlc (int nplanets, double **rxy, double rstar, double c1, double c2) {

    if (nplanets==0) return 1.0;

    circle sun = {0.,0., rstar/rstar};
    circle system[nplanets+1];
    system[0] = sun;
    int i;
    for (i=0; i<nplanets; i++) {
      system[i+1].x0 = rxy[i][1]/rstar;
      system[i+1].y0 = rxy[i][2]/rstar;
      system[i+1].r = rxy[i][0]/rstar;///rstar;
   }

    double c0 = 1.0;
    double g0, g1, g2;
    g0 = c0-c1-2.0*c2;
    g1 = c1+2.0*c2;
    g2 = c2;
 
    double flux = mttr_flux_general(system, nplanets+1, g0, g1, g2);// /nflux; 

    return flux;

}


// Computes the lightcurve for a list of times (and cadences) for an array of planet positions as a function of time and the stellar properties
double *timedlc ( double *times, int *cadences, long ntimes, double **transitarr, int nplanets, double rstar, double c1, double c2) {

    const int sofd = SOFD;
    const int cadenceswitch = CADENCESWITCH;

    if (ntimes==0) {
      return NULL;
    }

    double *fluxlist = malloc(ntimes*sofd);
    double rsinau = RSUNAU; //0.0046491; // solar radius in au

    double xstart[nplanets];
    double ystart[nplanets];
    int i;
    for (i=0; i<nplanets; i++) {
       xstart[i] = (transitarr[i][1] - transitarr[i][3]*(transitarr[i][0]-times[0]))/(rstar*rsinau);
       ystart[i] = (transitarr[i][2] - transitarr[i][4]*(transitarr[i][0]-times[0]))/(rstar*rsinau);
    }

    circle sun = {0.,0., 1.};
    circle system[nplanets+1];
    system[0] = sun;
    for (i=0; i< nplanets; i++) {
        system[i+1].x0 = xstart[i];
        system[i+1].y0 = ystart[i];
        system[i+1].r = transitarr[i][5];
    }

    double c0 = 1.0;
    double g0, g1, g2;
    g0 = c0-c1-2.0*c2;
    g1 = c1+2.0*c2;
    g2 = c2;
 
    double flux;   
    double t_cur;
    double t_next;
    long n=0;
    if (cadenceswitch==2) {
      long j=0;
      long jj=0;
      for (n=0; n<ntimes-1; n++) {
        while (cadences[j]==0) j++;
        t_cur = times[j];
        jj=j+1;
        while (cadences[j]==0) jj++;
        t_next = times[n+1];
        flux = mttr_flux_general(system, nplanets+1, g0, g1, g2);// /nflux; 
        fluxlist[n] = flux;
        for (i=0; i<nplanets; i++) {
            system[i+1].x0 += transitarr[i][3]*(t_next-t_cur)/(rstar*rsinau);
            system[i+1].y0 += transitarr[i][4]*(t_next-t_cur)/(rstar*rsinau);
        }
        j=jj;
      }
      fluxlist[ntimes-1] = mttr_flux_general(system, nplanets+1, g0, g1, g2);


    } else {
      for (n=0; n<ntimes-1; n++) {
        t_cur = times[n];
        t_next = times[n+1];
        flux = mttr_flux_general(system, nplanets+1, g0, g1, g2);// /nflux; 
        fluxlist[n] = flux;
        for (i=0; i<nplanets; i++) {
            system[i+1].x0 += transitarr[i][3]*(t_next-t_cur)/(rstar*rsinau);
            system[i+1].y0 += transitarr[i][4]*(t_next-t_cur)/(rstar*rsinau);
        }
      }
      fluxlist[ntimes-1] = mttr_flux_general(system, nplanets+1, g0, g1, g2);
    }

    return fluxlist;

}


// Computes and bins the lightcurve for a list of times (and cadences) for an array of planet positions as a function of time and the stellar properties
double *binnedlc ( double *times, int *cadences, long ntimes, double binwidth, int nperbin,  double **transitarr, int nplanets, double rstar, double c1, double c2) {

    //need to take into account extra 1's on either side of transit if bin is wider than buffer.  
    const int sofd = SOFD;
    const int cadenceswitch = CADENCESWITCH;

    if (ntimes==0) {
      return NULL;
    }

    double *fluxlist = malloc(ntimes*sofd);
    double rsinau = RSUNAU; //0.0046491; // solar radius in au
    long maxcalls = ntimes*nperbin;// + 1;
    double *fulltimelist=malloc(maxcalls*sofd);
    long j = 0;
    long jj = 0;
    if (cadenceswitch==2) {
      for (j=0; j<ntimes; j++) {
        if (cadences[j]==1) {
          int k;
          for (k=0; k<nperbin; k++) {
            fulltimelist[jj]=times[j]-binwidth/2+binwidth/nperbin*k+binwidth/(2*nperbin);
            jj++;
          }
        } else {
          fulltimelist[jj] = times[j];
          jj++;
        }
      }
    } else {
      for(j=0; j<ntimes; j++) {
        int k;
        for (k=0; k<nperbin; k++) {
          fulltimelist[nperbin*j+k]=times[j]-binwidth/2+binwidth/nperbin*k+binwidth/(2*nperbin);
        }
      }
    }

    double xstart[nplanets];
    double ystart[nplanets];
    int i;
    for (i=0; i<nplanets; i++) {
       xstart[i] = (transitarr[i][1] - transitarr[i][3]*(transitarr[i][0]-fulltimelist[0]))/(rsinau*rstar);
       ystart[i] = (transitarr[i][2] - transitarr[i][4]*(transitarr[i][0]-fulltimelist[0]))/(rsinau*rstar);
    }

    circle sun = {0.,0., 1.};
    circle system[nplanets+1];
    system[0] = sun;
    for (i=0; i< nplanets; i++) {
        system[i+1].x0 = xstart[i];
        system[i+1].y0 = ystart[i];
        system[i+1].r = transitarr[i][5];
    }

    double c0 = 1.0;
    double g0, g1, g2;
    g0 = c0-c1-2.0*c2;
    g1 = c1+2.0*c2;
    g2 = c2;
 
    double flux;   
    long n=0;
    if (cadenceswitch==2) {
      int nlong=0;
      for (n=0; n<ntimes; n++) {
        int nn;
        double binnedflux=0;
        if (cadences[n] == 1) {
          for (nn=0; nn<nperbin; nn++) {
            double t_cur = fulltimelist[(n-nlong) + nlong*nperbin + nn];
            double t_next = fulltimelist[(n-nlong) + nlong*nperbin + nn + 1];
            binnedflux += mttr_flux_general(system, nplanets+1, g0, g1, g2);// /nflux; 
            for (i=0; i<nplanets; i++) {
              system[i+1].x0 += transitarr[i][3]*(t_next-t_cur)/(rstar*rsinau);
              system[i+1].y0 += transitarr[i][4]*(t_next-t_cur)/(rstar*rsinau);
            }
          }
          binnedflux = binnedflux/nperbin;
          nlong++;
        } else {
          double t_cur = fulltimelist[(n-nlong) + nlong*nperbin];
          double t_next = fulltimelist[(n-nlong) + nlong*nperbin + 1];
          binnedflux += mttr_flux_general(system, nplanets+1, g0, g1, g2);// /nflux; 
          for (i=0; i<nplanets; i++) {
            system[i+1].x0 += transitarr[i][3]*(t_next-t_cur)/(rstar*rsinau);
            system[i+1].y0 += transitarr[i][4]*(t_next-t_cur)/(rstar*rsinau);
          }
        }
        fluxlist[n] = binnedflux;
      }
    } else {
    for (n=0; n<ntimes; n++) {
        int nn;
        double binnedflux=0;
        for (nn=0; nn<nperbin; nn++) {
            double t_cur = fulltimelist[n*nperbin+nn];
            double t_next = fulltimelist[n*nperbin+nn+1];
            binnedflux += mttr_flux_general(system, nplanets+1, g0, g1, g2);// /nflux; 
            for (i=0; i<nplanets; i++) {
                system[i+1].x0 += transitarr[i][3]*(t_next-t_cur)/(rstar*rsinau);
                system[i+1].y0 += transitarr[i][4]*(t_next-t_cur)/(rstar*rsinau);
            }
        }
        binnedflux = binnedflux/nperbin;
        fluxlist[n] = binnedflux;
      }
    }

    free(fulltimelist);
    return fluxlist;

}




// This runs the DEMCMC 
int demcmc(char aei[], char chainres[], char bsqres[], char gres[]) {

  // Load in global vars (This isn't really necessary but is convenient to ensure not changing global vars)
  const long nwalkers = NWALKERS;
  const int nsteps = NSTEPS;
  const int cadenceswitch = CADENCESWITCH;
  const char *tfilename = TFILE;
  const int *parfix = PARFIX;
  const unsigned long seed = SEED;
  const double disperse = DISPERSE;
  const double optimal = OPTIMAL;
  const double relax = RELAX;
  const double *step = STEP;
  const int bimodf = BIMODF;
  const int *bimodlist = BIMODLIST;
  const int pperplan = PPERPLAN;
  const int pstar = PSTAR;
  const int npl = NPL;
  const int nbodies = NBODIES;
  const int splitincO = SPLITINCO;

  const int sofd = SOFD;
  const int sofds = SOFDS;
  const int sofi = SOFI;
  const int sofis = SOFIS;

  const gsl_rng_type *T;
  gsl_rng *r;
  gsl_rng *rnw;
  gsl_rng_env_setup();
  //T=gsl_rng_ranlxd2;
  T=gsl_rng_taus2;

  r=gsl_rng_alloc(T);
  gsl_rng_set(r, seed);
  rnw=gsl_rng_alloc(T);
  
  if (TTVCHISQ) {

    int i;
    FILE **ttvfiles=malloc(npl*sizeof(FILE*));
    for (i=0; i<npl; i++) {
      char tempchar[1000];
      sprintf(tempchar, "ttv_%02i.in", i+1);
      printf("ttvfilei = %s\n", tempchar);
      ttvfiles[i]=fopen(tempchar, "r");
    }

    long maxttvs = 1000;
    // NTTV[i] is the list of the transit index for each transit with a measured time for planet i
    // Like TTTV, ETTV, and MTTV, the first entry is the total number of data points
    NTTV=malloc(npl*sizeof(long*));
    for (i=0; i<npl; i++) NTTV[i] = malloc(maxttvs*sizeof(long));
    TTTV=malloc(npl*sofds);
    for (i=0; i<npl; i++) TTTV[i] = malloc(maxttvs*sofd);
    ETTV=malloc(npl*sofds);
    for (i=0; i<npl; i++) ETTV[i] = malloc(maxttvs*sofd);
#if (demcmc_compile==0)
    MTTV=malloc(npl*sofds);
    for (i=0; i<npl; i++) MTTV[i] = malloc(maxttvs*sofd);
#endif

    for (i=0; i<npl; i++) {
      if (ttvfiles[i] == NULL) {
        printf("Bad ttv file name");
        exit(0);
      }
      long tt=0;
      while (fscanf(ttvfiles[i], "%li %lf %lf", &NTTV[i][tt+1], &TTTV[i][tt+1], &ETTV[i][tt+1]) == 3) { 
        if (tt>=maxttvs-1) {
          FILE *sout = fopen("demcmc.stdout", "a");
          fprintf(sout, "Too many TTVs, adjust maxttvs or use correct file\n");
          fclose(sout);
          exit(0);
        }
        printf("ttvread in planet %i num %li index %i time %lf err %lf\n", i, tt, NTTV[i][tt+1], TTTV[i][tt+1], ETTV[i][tt+1]);
        tt++;
      }
      NTTV[i][0]=tt;
      TTTV[i][0]=tt;
      ETTV[i][0]=tt;
      NTTV[i][tt+1]=LONG_MAX;
      TTTV[i][tt+1]=HUGE_VAL;
      ETTV[i][tt+1]=HUGE_VAL;
    }
 
    for (i=0; i<npl; i++) {
      fclose(ttvfiles[i]);
    }
    free(ttvfiles);
  }

  FILE *sout;
  double **p = malloc((nbodies)*sofds);  

  if (!RESTART) {
 
    int i;
    for (i=0; i<npl; i++) {
      p[i] = malloc(pperplan*sofd); 
    }
    p[npl] = malloc(pstar*sofd);
    double *planet1 = malloc(npl*sofd);
    double *period1 = malloc(npl*sofd);
    double *t01 = malloc(npl*sofd);
    double *e1 = malloc(npl*sofd);
    double *inc1 = malloc(npl*sofd);
    double *bo1 = malloc(npl*sofd);
    double *lo1 = malloc(npl*sofd);
    double *mp1 = malloc(npl*sofd);
    double *rpors1 = malloc(npl*sofd);
    double *brightness1;
    double *c1bin1;
    double *c2bin1;
    double *celeriteps = malloc(4*sofd);
    double *rvceleriteps = malloc(4*sofd);
    if (MULTISTAR) {
      brightness1 = malloc(npl*sofd);
      c1bin1 = malloc(npl*sofd);
      c2bin1 = malloc(npl*sofd);
    }
    double ms;
    double c0;
    double c1;
    double rstar; 
    double dilute;
    double *jittersize;
    if (RVJITTERFLAG) {
      jittersize=malloc(RVJITTERTOT*sofd);
    }
    double *jittersizettv;
    if (TTVJITTERFLAG) {
      jittersizettv=malloc(TTVJITTERTOT*sofd);
    }
 
    char buffer[1000];
  
    FILE *aeifile = fopen(aei, "r");
    if (aeifile == NULL) {
      printf("Bad pldin Input File Name");
      exit(0);
    }
  
    fgets(buffer, 1000, aeifile);
    for (i=0; i<npl; i++) {
      fscanf(aeifile,"%lf %lf %lf %lf %lf %lf %lf %lf %lf", &planet1[i], &period1[i], &t01[i], &e1[i], &inc1[i], &bo1[i], &lo1[i], &mp1[i], &rpors1[i]);
      printf("%lf\n", period1[i]);
      p[i][0] = period1[i];
      p[i][1] = t01[i];
      if (XYZFLAG==4 || XYZFLAG==5 || XYZFLAG==6) {
        p[i][2] = e1[i];
        p[i][3] = inc1[i];
        p[i][4] = bo1[i];
        p[i][5] = lo1[i];
      } else {
        if (XYZLIST[i]) {
          p[i][2] = e1[i];
          p[i][3] = inc1[i];
          p[i][4] = bo1[i];
          p[i][5] = lo1[i];
        } else {
          if (SQRTE) {
            p[i][2] = sqrt(e1[i]) * cos(lo1[i]*M_PI/180);
            p[i][3] = sqrt(e1[i]) * sin(lo1[i]*M_PI/180);
          } else {
            p[i][2] = e1[i] * cos(lo1[i]*M_PI/180);
            p[i][3] = e1[i] * sin(lo1[i]*M_PI/180);
          }
          p[i][4] = inc1[i];
          p[i][5] = bo1[i];
        }
      }
      p[i][6] = mp1[i];
      p[i][7] = rpors1[i];
      if (MULTISTAR) {
        fscanf(aeifile, "%lf %lf %lf", &brightness1[i], &c1bin1[i], &c2bin1[i]);
        p[i][8] = brightness1[i];
        p[i][9] = c1bin1[i];
        p[i][10] = c2bin1[i]; 
      }
      fgets(buffer, 1000, aeifile); // This line usually unnecessarily advances the file pointer to a new line. 
                                    // Unless  you are not multistar and but have too many inputs. It then  saves you from bad read-ins 
    }
    fscanf(aeifile, "%lf", &ms);
    fgets(buffer, 1000, aeifile);
    fscanf(aeifile, "%lf", &rstar);
    fgets(buffer, 1000, aeifile);
    fscanf(aeifile, "%lf", &c0);
    fgets(buffer, 1000, aeifile);
    fscanf(aeifile, "%lf", &c1);
    fgets(buffer, 1000, aeifile);
    fscanf(aeifile, "%lf", &dilute);
printf("rvjf = %i\n", RVJITTERFLAG);
    if (RVJITTERFLAG) {
      int ki;
      for (ki=0; ki<RVJITTERTOT; ki++) {
printf("kih = %i\n", ki);
        fgets(buffer, 1000, aeifile);
        fscanf(aeifile, "%lf", &jittersize[ki]);
      }
    }
    if (TTVJITTERFLAG) {
      int ki;
      for (ki=0; ki<TTVJITTERTOT; ki++) {
        fgets(buffer, 1000, aeifile);
        fscanf(aeifile, "%lf", &jittersizettv[ki]);
      }
    }
            if (CELERITE) {
              int ki;
              for (ki=0; ki<NCELERITE; ki++) {
        fgets(buffer, 1000, aeifile);
        fscanf(aeifile, "%lf", &celeriteps[ki]);
              }
            }
            if (RVCELERITE) {
              int ki;
              for (ki=0; ki<NRVCELERITE; ki++) {
        fgets(buffer, 1000, aeifile);
        fscanf(aeifile, "%lf", &rvceleriteps[ki]);
              }
            }
 
    p[npl][0] = ms;
    p[npl][1] = rstar;
    p[npl][2] = c0;
    p[npl][3] = c1;
    p[npl][4] = dilute; 
    if (RVJITTERFLAG) {
      int ki;
      for (ki=0; ki<RVJITTERTOT; ki++) {
        p[npl][5+ki] = jittersize[ki];
      }
    }
    if (TTVJITTERFLAG) {
      int ki;
      for (ki=0; ki<TTVJITTERTOT; ki++) {
        p[npl][5+RVJITTERTOT+ki] = jittersizettv[ki];
      }
    }
    if (CELERITE) {
      int ki;
      for (ki=0; ki<NCELERITE; ki++) {
        p[npl][5+RVJITTERTOT+TTVJITTERTOT+ki] = celeriteps[ki];
      }
    }
    if (RVCELERITE) {
      int ki;
      for (ki=0; ki<NRVCELERITE; ki++) {
        p[npl][5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+ki] = rvceleriteps[ki];
      }
    }

    // if you gave me input in aei instead of PT0e, change it:
    if (XYZFLAG==5) {
      double masstot[npl+1]; 
      masstot[0] = ms;
      for (i=0; i<npl; i++) {
        masstot[i+1] = masstot[i];
        masstot[i+1] += p[i][6]/MSOMJ;
      }
      for (i=0; i<npl; i++) {
        double *orbelements = keptoorb(p[i][0], p[i][1], p[i][2]*M_PI/180., p[i][3]*M_PI/180., p[i][4]*M_PI/180., p[i][5]*M_PI/180., masstot[i+1]);
        p[i][0] = orbelements[0];
        p[i][1] = orbelements[1];
        if (SQRTE) {
          p[i][2] = sqrt(orbelements[2]) * cos(orbelements[5]);
          p[i][3] = sqrt(orbelements[2]) * sin(orbelements[5]);
        } else {
          p[i][2] = orbelements[2] * cos(orbelements[5]);
          p[i][3] = orbelements[2] * sin(orbelements[5]);
        }
        p[i][4] = orbelements[3]*180./M_PI;
        p[i][5] = orbelements[4]*180./M_PI;

        free(orbelements);
      }
    }
    // aei but with mean anomaly not true 
    if (XYZFLAG==6) {
      double masstot[npl+1]; 
      masstot[0] = ms;
      for (i=0; i<npl; i++) {
        masstot[i+1] = masstot[i];
        masstot[i+1] += p[i][6]/MSOMJ;
      }
      for (i=0; i<npl; i++) {
        double *orbelements = keptoorbmean(p[i][0], p[i][1], p[i][2]*M_PI/180., p[i][3]*M_PI/180., p[i][4]*M_PI/180., p[i][5]*M_PI/180., masstot[i+1]);
        p[i][0] = orbelements[0];
        p[i][1] = orbelements[1];
        if (SQRTE) {
          p[i][2] = sqrt(orbelements[2]) * cos(orbelements[5]);
          p[i][3] = sqrt(orbelements[2]) * sin(orbelements[5]);
        } else {
          p[i][2] = orbelements[2] * cos(orbelements[5]);
          p[i][3] = orbelements[2] * sin(orbelements[5]);
        }
        p[i][4] = orbelements[3]*180./M_PI;
        p[i][5] = orbelements[4]*180./M_PI;
        free(orbelements);
      }
    }

    fclose(aeifile);

    free(planet1); free(period1); free(t01); free (e1); free(inc1); free(bo1); free(lo1); free(mp1); free(rpors1); free(celeriteps); free(rvceleriteps); 
    if (MULTISTAR) {
      free(brightness1); free(c1bin1); free(c2bin1);
    }
    if (RVJITTERFLAG) {
      free(jittersize);
    }
    if (TTVJITTERFLAG) {
      free(jittersizettv);
    }

  }

  int nparam = pperplan*npl+pstar;

  // read in list of times
  double *timelist;
  double *fluxlist;
  double *errlist;
  int *cadencelist;
  long timelistlen = 2000000;
  timelist = malloc(timelistlen*sofd);
  errlist = malloc(timelistlen*sofd);
  fluxlist = malloc(timelistlen*sofd);
  if (fluxlist==NULL) {
    sout = fopen("demcmc.stdout", "a");
    fprintf(sout, "malloc error\n");
    fclose(sout);
    exit(0);
  }

  FILE *tfile = fopen(tfilename, "r");

  printf ("%s\n", tfilename);

  if (tfile == NULL) {
    sout = fopen("demcmc.stdout", "a");
    printf("Error opening tfile\n");
    fprintf(sout,"Error opening tfile\n");
    fclose(sout);
    exit(0);
  }
  long num;
  double flux1, err1;
  long kk=0;

  if (CADENCESWITCH==0 || CADENCESWITCH==1) {
    while (fscanf(tfile, "%ld %lf %lf %lf %lf %lf", &num, &timelist[kk+1], &flux1, &err1, &fluxlist[kk+1], &errlist[kk+1]) == 6) { 
      if (kk>=timelistlen-1) {
        timelistlen+=1000000;
        timelist = realloc(timelist, timelistlen*sofd);
        fluxlist = realloc(fluxlist, timelistlen*sofd);
        errlist = realloc(errlist, timelistlen*sofd);
        if (timelist==NULL) {
          sout = fopen("demcmc.stdout", "a");
          fprintf(sout, "timelist allocation failure\n");
          fclose(sout);
          exit(0);
        }
      }
      kk++;
    }
  } else {
    cadencelist = malloc(timelistlen*sofi);
    while (fscanf(tfile, "%ld %lf %lf %lf %lf %lf %i", &num, &timelist[kk+1], &flux1, &err1, &fluxlist[kk+1], &errlist[kk+1], &cadencelist[kk]) == 7) { 
      if (kk>=timelistlen-1) {
        timelistlen+=1000000;
        timelist = realloc(timelist, timelistlen*sofd);
        fluxlist = realloc(fluxlist, timelistlen*sofd);
        cadencelist = realloc(cadencelist, timelistlen*sofi);
        errlist = realloc(errlist, timelistlen*sofd);
        
        if (timelist==NULL) {
          sout = fopen("demcmc.stdout", "a");
          fprintf(sout, "timelist allocation failure\n");
          fclose(sout);
          exit(0);
        }
      }
      kk++;
    }
    printf ("\n%s read\n", tfilename);
  }
  fclose(tfile);
  
  timelist[0]=kk;
  fluxlist[0]=kk;
  errlist[0]=kk;

  double **tfe = malloc(3*sofds);
  tfe[0]=timelist;
  tfe[1]=fluxlist;
  tfe[2]=errlist;

  double *rvtimelist;
  double *rvlist;
  double *rverrlist;
  double *rvbodylist;
  int *rvtelelist;
  double *rvtelelistd;
  long rvlistlen = 5000; // Maximum number of RVs by default
  long vv=0;
  double **tve = malloc(5*sofds);

  printf("pre RV\n");
  if (RVS) {
    rvtimelist = malloc(rvlistlen*sofd);
    rvlist = malloc(rvlistlen*sofd);
    rverrlist = malloc(rvlistlen*sofd);
    rvbodylist = malloc(rvlistlen*sofd);
    rvtelelist = malloc(rvlistlen*sofi);
    rvtelelistd = malloc(rvlistlen*sofd);
    if (rverrlist==NULL) {
      sout = fopen("demcmc.stdout", "a");
      fprintf(sout, "malloc error\n");
      fclose(sout);
      exit(0);
    }

    int i;
    for(i=0; i<nbodies; i++) {
      printf("%i, %i\n", i, RVARR[i]);
      if(RVARR[i]) { 
        FILE *rvfile = fopen(RVFARR[i], "r");
        if (rvfile == NULL) {
          sout = fopen("demcmc.stdout", "a");
          fprintf(sout,"Error opening rvfile\n");
          fclose(sout);
          exit(0);
        } 
        while (fscanf(rvfile, "%lf %lf %lf %i", &rvtimelist[vv+1], &rvlist[vv+1], &rverrlist[vv+1], &rvtelelist[vv+1]) == 4) { 
          rvbodylist[vv+1] = (double) i;
          if (vv>=rvlistlen-1) {
            timelistlen+=1000;
            rvtimelist = realloc(rvtimelist, rvlistlen*sofd);
            rvlist = realloc(rvlist, rvlistlen*sofd);
            rverrlist = realloc(rverrlist, rvlistlen*sofd);
            if (rverrlist==NULL) {
              sout = fopen("demcmc.stdout", "a");
              fprintf(sout, "rvlist allocation failure\n");
              fclose(sout);
              exit(0);
            }
          }
          vv++;
        }
        fclose(rvfile);
      }
    }
  
    printf("vv=%li\n",vv);
  
    rvtimelist[0] = (double) vv;
    rvlist[0] = (double) vv;
    rverrlist[0] = (double) vv;
    rvbodylist[0] = (double) vv;
    rvtelelist[0] = (int) vv;
    rvtelelistd[0] = (double) vv;

    long w;
    for (w=1; w<(vv+1); w++) {
      rvlist[w] = rvlist[w] * MPSTOAUPD;
      rverrlist[w] = rverrlist[w] * MPSTOAUPD;
    }
 
    double bigrvlist[vv*5];
    long z;
    for (z=1; z<(vv+1); z++) {
      bigrvlist[(z-1)*5+0] = rvtimelist[z];
      bigrvlist[(z-1)*5+1] = rvlist[z];
      bigrvlist[(z-1)*5+2] = rverrlist[z];
      bigrvlist[(z-1)*5+3] = rvbodylist[z];
      bigrvlist[(z-1)*5+4] = (double) rvtelelist[z];
    }
    qsort(bigrvlist, vv, 5*sofd, compare);
    for (z=1; z<(vv+1); z++) {
      rvtimelist[z] = bigrvlist[(z-1)*5+0];
      rvlist[z] = bigrvlist[(z-1)*5+1];
      rverrlist[z] = bigrvlist[(z-1)*5+2];
      rvbodylist[z] = bigrvlist[(z-1)*5+3]; 
      rvtelelistd[z] = bigrvlist[(z-1)*5+4];
    }
    printf("here0.8\n");

    tve[0]=rvtimelist;
    tve[1]=rvlist;
    tve[2]=rverrlist;
    tve[3]=rvbodylist;
    tve[4]=rvtelelistd;
    free(rvtelelist);    
  }

  long ttvlistlen = 5000; // Max number TTVs by default
  long vvt=0;
  double **nte;
  if (TTVCHISQ) {
    nte = malloc(3*sofds);
    int kij;
    for (kij=0; kij<3; kij++) {
      nte[kij] = malloc(ttvlistlen*sofd);
    }
    int i;
    int ki;
    int ksofar=0;
    for (i=0; i<npl; i++) {
      for (ki=0; ki<NTTV[i][0]; ki++) {
        nte[0][1+ki+ksofar] = (double) NTTV[i][1+ki]; 
        nte[1][1+ki+ksofar] = TTTV[i][1+ki];
        nte[2][1+ki+ksofar] = ETTV[i][1+ki];
      }
      ksofar += NTTV[i][0];
    }
    nte[0][0] = (double) ksofar;
    nte[1][0] = (double) ksofar;
    nte[2][0] = (double) ksofar;
  }

#if (demcmc_compile==1)
  int i;
  double ***p0 = malloc(nwalkers*sizeof(double**));
  for (i=0; i<nwalkers; i++){
    p0[i] = malloc((npl+1)*sofds);
    int j;
    for (j=0; j<npl; j++) {
      p0[i][j] = malloc(pperplan*sofd); 
    }
    p0[i][npl] = malloc(pstar*sofd);
  }

  int w;

  double gamma;
  double xisqmin; 
  double *gammaN;

  int fixedpars = 0;
  for (i=0; i<nparam; i++) fixedpars += parfix[i];
  int ndim = nparam - fixedpars;
  if (ndim == 0) {
    printf("Warning! No free parameters!\n");
    ndim=1;
  }

  long jj = 0;


  if (RESTART) {
  
    if (! NTEMPS) {
      FILE *restartf = fopen(chainres, "r");
      if (restartf == NULL) {
        printf("Error: %s\n", chainres); 
        printf("nofile\n");
        exit(0);
      }
      double ignore; 
      char ignorec[1000];
      for (i=0; i<nwalkers; i++) {
        int j; 
        for (j=0; j<npl; j++) {
          fscanf(restartf, " %lf %lf %lf %lf %lf %lf %lf %lf %lf", &ignore, &p0[i][j][0], &p0[i][j][1], &p0[i][j][2], &p0[i][j][3], &p0[i][j][4], &p0[i][j][5], &p0[i][j][6], &p0[i][j][7]);
          if (MULTISTAR) {
            fscanf(restartf, "%lf %lf %lf", &p0[i][j][8], &p0[i][j][9], &p0[i][j][10]);
          }
        }
        fscanf(restartf, "%lf", &p0[i][npl][0]);
        fgets(ignorec, 1000, restartf);
        fscanf(restartf, "%lf", &p0[i][npl][1]);
        fgets(ignorec, 1000, restartf);
        fscanf(restartf, "%lf", &p0[i][npl][2]);
        fgets(ignorec, 1000, restartf);
        fscanf(restartf, "%lf", &p0[i][npl][3]);
        fgets(ignorec, 1000, restartf);
        fscanf(restartf, "%lf", &p0[i][npl][4]);
        fgets(ignorec, 1000, restartf);
        if (RVJITTERFLAG) {
          int ki;
          for (ki=0; ki<(RVJITTERTOT); ki++) {
            fscanf(restartf, "%lf", &p0[i][npl][5+ki]);
            fgets(ignorec, 1000, restartf);
          }
        }
        if (TTVJITTERFLAG) {
          int ki;
          for (ki=0; ki<(TTVJITTERTOT); ki++) {
            fscanf(restartf, "%lf", &p0[i][npl][5+RVJITTERTOT+ki]);
            fgets(ignorec, 1000, restartf);
          }
        }
        if (CELERITE) {
          int ki;
          for (ki=0; ki<NCELERITE; ki++) {
            fscanf(restartf, "%lf", &p0[i][npl][5+RVJITTERTOT+TTVJITTERTOT+ki]);
            fgets(ignorec, 1000, restartf);
          }
        }
        if (RVCELERITE) {
          int ki;
          for (ki=0; ki<NRVCELERITE; ki++) {
            fscanf(restartf, "%lf", &p0[i][npl][5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+ki]);
            fgets(ignorec, 1000, restartf);
          }
        }
        fgets(ignorec, 1000, restartf);
      }
      fclose(restartf);
      printf("Read in Restart pldin\n");
     
      FILE *regammaf = fopen(gres,"r");
      long genres;
      double grac, gammares;
      fscanf(regammaf, "%li %lf %lf", &genres, &grac, &gammares); 
      fclose(regammaf);
     
      printf("Read in Restart gamma\n");
      // rounding generation
      genres = genres/10;
      genres = genres*10;
      jj = genres+1;
      // optimal multiplier
      gamma = gammares;
     
      FILE *rebsqf = fopen(bsqres, "r");
      char garbage[10000];
      for (i=0; i<(1+npl+pstar+1); i++) {
        fgets(garbage, 10000, rebsqf);
      }
      char cgarbage;
      for (i=0; i<11; i++) {
        cgarbage = fgetc(rebsqf);
      }
      double rexisqmin;
      fscanf(rebsqf, "%lf", &rexisqmin);
      fclose(rebsqf);
      xisqmin = rexisqmin;
     
      printf("Read in Restart best chi sq\n");
      printf("xisqmin=%lf\n", xisqmin);
      printf("gamma=%lf\n", gamma); 
    
    } else { // if NTEMPS:
  
      for (w=0; w<NTEMPS; w++) {
 
        char chainresw[200]; 
        sprintf(chainresw, "%s_%i", chainres, w);
        FILE *restartf = fopen(chainresw, "r");
        if (restartf == NULL) {
          printf("nofile\n");
          exit(0);
        }
        double ignore; 
        char ignorec[1000];
        for (i=0; i<nwalkers; i++) {
          int j; 
          for (j=0; j<npl; j++) {
            fscanf(restartf, " %lf %lf %lf %lf %lf %lf %lf %lf %lf", &ignore, &p0N[w][i][j][0], &p0N[w][i][j][1], &p0N[w][i][j][2], &p0N[w][i][j][3], &p0N[w][i][j][4], &p0N[w][i][j][5], &p0N[w][i][j][6], &p0N[w][i][j][7]);
            if (MULTISTAR) {
              fscanf(restartf, "%lf %lf %lf", &p0N[w][i][j][8], &p0N[w][i][j][9], &p0N[w][i][j][10]);
            }
          }
          fscanf(restartf, "%lf", &p0N[w][i][npl][0]);
          fgets(ignorec, 1000, restartf);
          fscanf(restartf, "%lf", &p0N[w][i][npl][1]);
          fgets(ignorec, 1000, restartf);
          fscanf(restartf, "%lf", &p0N[w][i][npl][2]);
          fgets(ignorec, 1000, restartf);
          fscanf(restartf, "%lf", &p0N[w][i][npl][3]);
          fgets(ignorec, 1000, restartf);
          fscanf(restartf, "%lf", &p0N[w][i][npl][4]);
          fgets(ignorec, 1000, restartf);
          if (RVJITTERFLAG) {
            int ki;
            for (ki=0; ki<(RVJITTERTOT); ki++) {
              fscanf(restartf, "%lf", &p0N[w][i][npl][5+ki]);
              fgets(ignorec, 1000, restartf);
            }
          }
          if (TTVJITTERFLAG) {
            int ki;
            for (ki=0; ki<(TTVJITTERTOT); ki++) {
              fscanf(restartf, "%lf", &p0N[w][i][npl][5+RVJITTERTOT+ki]);
              fgets(ignorec, 1000, restartf);
            }
          }
          if (CELERITE) {
            int ki;
            for (ki=0; ki<NCELERITE; ki++) {
              fscanf(restartf, "%lf", &p0N[w][i][npl][5+RVJITTERTOT+TTVJITTERTOT+ki]);
              fgets(ignorec, 1000, restartf);
            }
          }
          if (RVCELERITE) {
            int ki;
            for (ki=0; ki<NRVCELERITE; ki++) {
              fscanf(restartf, "%lf", &p0N[w][i][npl][5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+ki]);
              fgets(ignorec, 1000, restartf);
            }
          }
          fgets(ignorec, 1000, restartf);
        }
        fclose(restartf);
    
        char gresw[200];
        sprintf(gresw, "%s_%i", gres, w);
        FILE *regammaf = fopen(gresw,"r");
        long genres;
        double grac, gammares;
        fscanf(regammaf, "%li %lf %lf", &genres, &grac, &gammares); 
        fclose(regammaf);
       
        // rounding generation
        genres = genres/10;
        genres = genres*10;    
        jj = genres+1;
   
        // optimal multiplier
        gammaN[w] = gammares;
     
        if (w ==0) {
          FILE *rebsqf = fopen(bsqres, "r");
          char garbage[10000];
          for (i=0; i<(1+npl+pstar+1); i++) {
            fgets(garbage, 10000, rebsqf);
          }
          char cgarbage;
          for (i=0; i<11; i++) {
            cgarbage = fgetc(rebsqf);
          }
          double rexisqmin;
          fscanf(rebsqf, "%lf", &rexisqmin);
          fclose(rebsqf);
          xisqmin = rexisqmin;
         
          printf("xisqmin=%lf\n", xisqmin);
          printf("gamma=%lf\n", gammaN[w]); 
        }
      }
    }
    
  
  } else {  //if not RESTART

    printf("Initializing Walkers\n");
    if (!NTEMPS) {
    // take small random steps to initialize walkers
    for (i=0; i<nwalkers; i++) {
      int j;
      for (j=0; j<npl; j++) {
        int k;
        for (k=0; k<pperplan; k++) {
          // make sure all inclinations still > 90.0
          do {
            double epsilon = (1-parfix[j*pperplan+k])*step[j*pperplan+k]*gsl_ran_gaussian(r, 1.0);
            // don't split inclinations....
            //if (splitincO && k==4 && ((i+nwalkers/2)/nwalkers)) p0[i][j][k] = 180. - p[j][k] + epsilon; 
            //else if (splitincO && k==5 && ((i+nwalkers/2)/nwalkers)) p0[i][j][k] = -p[j][k] + epsilon;
            //else p0[i][j][k] = p[j][k] + epsilon;
            p0[i][j][k] = p[j][k] + epsilon;
            if ( (int) ceil(disperse) ) p0[i][j][k] += disperse*(1-parfix[j*pperplan+k])*step[j*pperplan+k]*gsl_ran_gaussian(r, 1.0);
          } while ( (IGT90 && (k==4 && p0[i][j][k] < 90.0)) || (MGT0 && (k==6 && p0[i][j][k] < 0.0)) );
        }
      }
      for (j=0; j<pstar; j++) {
        do {
        p0[i][npl][j] = p[npl][j] + (1-parfix[npl*pperplan+j])*step[npl*pperplan+j]*gsl_ran_gaussian(r, 1.0);
        if ( (int) ceil(disperse) ) p0[i][npl][j] += disperse*(1-parfix[npl*pperplan+j])*step[npl*pperplan+j]*gsl_ran_gaussian(r, 1.0);
        } while ( (TTVJITTERFLAG || RVJITTERFLAG) && (j > 4 && p0[i][npl][j] < 0.0) ); 
      }
    } 
 
    // optimal multiplier
    gamma = 2.38 / sqrt(2.*ndim);

    } else { // if NTEMPS
      for (w=0; w<NTEMPS; w++) {
        // take small random steps to initialize walkers
        for (i=0; i<nwalkers; i++) {
          int j;
          for (j=0; j<npl; j++) {
            int k;
            for (k=0; k<pperplan; k++) {
              // make sure all inclinations still > 90.0
              do {
                double epsilon = (1-parfix[j*pperplan+k])*step[j*pperplan+k]*gsl_ran_gaussian(r, 1.0);
                // don't split inclinations....
                //if (splitincO && k==4 && ((i+nwalkers/2)/nwalkers)) p0N[w][i][j][k] = 180. - p[j][k] + epsilon; 
                //else if (splitincO && k==5 && ((i+nwalkers/2)/nwalkers)) p0N[w][i][j][k] = -p[j][k] + epsilon;
                //else p0N[w][i][j][k] = p[j][k] + epsilon;
                p0N[w][i][j][k] = p[j][k] + epsilon;
                if ( (int) ceil(disperse) ) p0N[w][i][j][k] += disperse*(1-parfix[j*pperplan+k])*step[j*pperplan+k]*gsl_ran_gaussian(r, 1.0);
              } while ( (IGT90 && (k==4 && p0N[w][i][j][k] < 90.0)) ||  (MGT0 && (k==6 && p0N[w][i][j][k] < 0.0)) );
            }
          }
          for (j=0; j<pstar; j++) {
            p0N[w][i][npl][j] = p[npl][j] + (1-parfix[npl*pperplan+j])*step[npl*pperplan+j]*gsl_ran_gaussian(r, 1.0);
            if ( (int) ceil(disperse) ) p0N[w][i][npl][j] += disperse*(1-parfix[npl*pperplan+j])*step[npl*pperplan+j]*gsl_ran_gaussian(r, 1.0);
          }
        } 
     
        gammaN[w] = 2.38 / sqrt(2.*ndim);
      }
    }
  }

  printf("Initialized Walkers\n");

  int pperwalker = npl*pperplan+pstar;
  int totalparams = nwalkers*pperwalker;
  double *p0local = malloc(totalparams*sofd);
  double **p0localN; 

  if (NTEMPS) {
    p0localN = malloc(NTEMPS*sofds);
    for (w=0; w<NTEMPS; w++) {
      p0localN[w] = malloc(totalparams*sofd);
    }
    for (w=0; w<NTEMPS; w++) {
      for (i=0; i<nwalkers; i++) {
        int j;
        for (j=0; j<npl; j++) {
          int k;
          for (k=0; k<pperplan; k++) {
            p0localN[w][i*pperwalker + j*pperplan + k] = p0N[w][i][j][k];
          }
        }
        for (j=0; j<pstar; j++) {
          p0localN[w][i*pperwalker + npl*pperplan + j] = p0N[w][i][npl][j];
        }
      }
    }
  } else {
    for (i=0; i<nwalkers; i++) {
      int j;
      for (j=0; j<npl; j++) {
        int k;
        for (k=0; k<pperplan; k++) {
          p0local[i*pperwalker + j*pperplan + k] = p0[i][j][k];
        }
      }
      for (j=0; j<pstar; j++) {
        p0local[i*pperwalker + npl*pperplan + j] = p0[i][npl][j];
      }
    }
  }

#endif


  double ***dsetup (double **p, const int npl); //prototype 
  double ***dsetup2 (double *p, const int npl); //prototype 
  double ***dpintegrator_single (double ***int_in, double **tfe, double **tve, double **nte, int *cadencelist); //prototype 
  int dpintegrator_single_megno (double ***int_in); //prototype 
  double ***dpintegrator_multi (double ***int_in, double **tfe, double **tve, int *cadencelist); //prototype 
  double *devoerr (double **tmte); //prototype

#if (demcmc_compile==0)
  printf("Sanity check:\n");
  printf("p[][2]=%lf, orbelemets[2]=%lf\n", p[0][0], p[0][0]);

  double ***int_in = dsetup(p, npl);
  printf("int_in %lf, %lf, %lf, %lf\n", int_in[0][0][0], int_in[1][0][0], int_in[2][0][0], int_in[2][1][0]);

  double ***flux_rvs; 
  if (MULTISTAR) {
    flux_rvs = dpintegrator_multi(int_in, tfe, tve, cadencelist);
  } else {
    flux_rvs = dpintegrator_single(int_in, tfe, tve, nte, cadencelist);
  }
  printf("Integration Complete\n");
  
  double **ttvts = flux_rvs[2];
  double **flux = flux_rvs[0];
  double **radvs = flux_rvs[1];
  double *dev = devoerr(flux);
  double xisq = 0;
  long il;
  long maxil = (long) dev[0];
  printf("kk=%li\n", maxil);
 
  if (! CELERITE) { 
    for (il=0; il<maxil; il++) xisq += dev[il+1]*dev[il+1];
  } else { // if celerite
    double *xs = flux_rvs[0][0];
    long maxil = (long) xs[0];
    double *trueys = flux_rvs[0][1];
    double *modelys = flux_rvs[0][2];
    double *es = flux_rvs[0][3];
    double *diffys = malloc(sofd*maxil);
    for (il=0; il<maxil; il++) { 
      diffys[il] = trueys[il+1]-modelys[il+1];
    }
    double *yvarp = malloc(sofd*maxil);
    for (il=0; il<maxil; il++) { 
      yvarp[il] = es[il+1]*es[il+1]; 
    }
    double *xp = &xs[1]; 
    
    int j_real = 0;
    int j_complex;
    double jitter, k1, k2, k3, S0, w0, Q;
    jitter = p[npl][5+RVJITTERTOT+TTVJITTERTOT+0];
    S0 = p[npl][5+RVJITTERTOT+TTVJITTERTOT+1];
    w0 = p[npl][5+RVJITTERTOT+TTVJITTERTOT+2];
    Q = p[npl][5+RVJITTERTOT+TTVJITTERTOT+3];
    if (Q >= 0.5) {
      j_complex = 1;
    } else {
      j_complex = 2;
    }
    VectorXd a_real(j_real),
            c_real(j_real),
            a_comp(j_complex),
            b_comp(j_complex),
            c_comp(j_complex),
            d_comp(j_complex);
    if (Q >= 0.5) {
      k1 = S0*w0*Q;
      k2 = sqrt(4.*Q*Q - 1.);
      k3 = w0/(2.*Q);
      a_comp << k1;
      b_comp << k1/k2;
      c_comp << k3;
      d_comp << k3*k2;
    } else {
      j_complex = 2;
      k1 = 0.5*S0*w0*Q;
      k2 = sqrt(1. - 4.*Q*Q);
      k3 = w0/(2.*Q);
      a_comp << k1*(1. + 1./k2), k1*(1. - 1./k2);
      b_comp << 0., 0.;
      c_comp << k3*(1. - k2), k3*(1. + k2);
      d_comp << 0., 0.;
    }

    VectorXd x = VectorXd::Map(xp, maxil);
    VectorXd yvar = VectorXd::Map(yvarp, maxil);
    VectorXd dy = VectorXd::Map(diffys, maxil);
  
    celerite::solver::CholeskySolver<double> solver;
    solver.compute(
          jitter,
          a_real, c_real,
          a_comp, b_comp, c_comp, d_comp,
          x, yvar  // Note: this is the measurement _variance_
      );
  
    // see l.186-192 in celerite.py
    double logdet, diffs, llike;
    logdet = solver.log_determinant();
    diffs = solver.dot_solve(dy); 
    llike = -0.5 * (diffs + logdet); 
  
    xisq = diffs+logdet;
   
#if (demcmc_compile==0)
    printf("Celerite params:\n");
    printf("a+comp=%25.17lf\n", a_comp[0]);
    printf("b+comp=%25.17lf\n", b_comp[0]);
    printf("c+comp=%25.17lf\n", c_comp[0]);
    printf("d+comp=%25.17lf\n", d_comp[0]);
    printf("diffs=%lf\n", diffs);
    printf("logdet=%lf\n", logdet);
    printf("llike=%lf\n", llike);

    VectorXd prediction = solver.predict(dy, x);
    char tmtefstr[1000];
    strcpy(tmtefstr, "lc_");
    strcat(tmtefstr, OUTSTR);
    strcat(tmtefstr, ".gp_lcout");
    FILE *tmtef;
    tmtef = fopen(tmtefstr, "a");
    long ijk;
    for (ijk=0; ijk<maxil; ijk++) {
      fprintf(tmtef, "%.12lf \n", prediction[ijk]);
    }
    fclose(tmtef);// = openf(tmtstr,"w");
 
#endif
  
    free(yvarp);
    free(diffys);
  }

  printf("xisq=%lf\n", xisq);
  printf("prervjitter\n");

  if (RVS) {
    if (! RVCELERITE) { 
      double *newelist;
      if (RVJITTERFLAG) {
        long kj;
        long maxkj = (long) tve[0][0]; 
        newelist=malloc((maxkj+1)*sofd);
        newelist[0] = (double) maxkj;
        for (kj=0; kj<maxkj; kj++) {
          int jitterindex = (int) tve[3][1+kj]*NTELESCOPES + tve[4][1+kj];
          double sigmajitter = p[npl][5+jitterindex]*MPSTOAUPD;
          double quadsum = sigmajitter*sigmajitter + radvs[3][1+kj]*radvs[3][1+kj];
          // double check this... factor of 1/2
          xisq += log(quadsum / (radvs[3][1+kj]*radvs[3][1+kj]) );
          newelist[1+kj] = sqrt( quadsum );
        }
        radvs[3] = newelist;
      }
      double *rvdev = devoerr(radvs);
      long maxil = (long) rvdev[0];
      for (il=0; il<maxil; il++) xisq += rvdev[il+1]*rvdev[il+1];
      free(rvdev);
    } else { // if rvcelerite
      double *xs = flux_rvs[1][0];
      long maxil = (long) xs[0];
      double *trueys = flux_rvs[1][1];
      double *modelys = flux_rvs[1][2];
      double *es = flux_rvs[1][3];
      double *diffys = malloc(sofd*maxil);
      for (il=0; il<maxil; il++) { 
        diffys[il] = trueys[il+1]-modelys[il+1];
      }
      double *yvarp = malloc(sofd*maxil);
      for (il=0; il<maxil; il++) { 
        yvarp[il] = es[il+1]*es[il+1]; 
      }
      double *xp = &xs[1]; 
    
      int j_real = 0;
      int j_complex;
      double jitter, k1, k2, k3, S0, w0, Q;
      jitter = p[npl][5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+0];
      S0 = p[npl][5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+1];
      w0 = p[npl][5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+2];
      Q = p[npl][5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+3];
      if (Q >= 0.5) {
        j_complex = 1;
      } else {
        j_complex = 2;
      }
      VectorXd a_real(j_real),
              c_real(j_real),
              a_comp(j_complex),
              b_comp(j_complex),
              c_comp(j_complex),
              d_comp(j_complex);
      if (Q >= 0.5) {
        k1 = S0*w0*Q;
        k2 = sqrt(4.*Q*Q - 1.);
        k3 = w0/(2.*Q);
        a_comp << k1;
        b_comp << k1/k2;
        c_comp << k3;
        d_comp << k3*k2;
      } else {
        j_complex = 2;
        k1 = 0.5*S0*w0*Q;
        k2 = sqrt(1. - 4.*Q*Q);
        k3 = w0/(2.*Q);
        a_comp << k1*(1. + 1./k2), k1*(1. - 1./k2);
        b_comp << 0., 0.;
        c_comp << k3*(1. - k2), k3*(1. + k2);
        d_comp << 0., 0.;
      }
   
      VectorXd x = VectorXd::Map(xp, maxil);
      VectorXd yvar = VectorXd::Map(yvarp, maxil);
      VectorXd dy = VectorXd::Map(diffys, maxil);
    
      celerite::solver::CholeskySolver<double> solver;
      solver.compute(
            jitter,
            a_real, c_real,
            a_comp, b_comp, c_comp, d_comp,
            x, yvar  // Note: this is the measurement _variance_
        );
    
      // see l.186-192 in celerite.py
      double logdet, diffs, llike;
      logdet = solver.log_determinant();
      diffs = solver.dot_solve(dy); 
      llike = -0.5 * (diffs + logdet); 
    
      xisq += diffs+logdet;
     
#if (demcmc_compile==0)
      printf("a+comp=%25.17lf\n", a_comp[0]);
      printf("b+comp=%25.17lf\n", b_comp[0]);
      printf("c+comp=%25.17lf\n", c_comp[0]);
      printf("d+comp=%25.17lf\n", d_comp[0]);
      printf("diffs=%lf\n", diffs);
      printf("logdet=%lf\n", logdet);
      printf("llike=%lf\n", llike);
  
      VectorXd prediction = solver.predict(dy, x);
      char tmtefstr[1000];
      strcpy(tmtefstr, "rv_");
      strcat(tmtefstr, OUTSTR);
      strcat(tmtefstr, ".gp_rvout");
      FILE *tmtef;
      tmtef = fopen(tmtefstr, "a");
      long ijk;
      for (ijk=0; ijk<maxil; ijk++) {
        fprintf(tmtef, "%.12lf \n", prediction[ijk]/MPSTOAUPD);
      }
      fclose(tmtef);// = openf(tmtstr,"w");
   
#endif
  
    
      free(yvarp);
      free(diffys);
  
    }
  }

  if (TTVCHISQ) {
    double *newelistttv;
    if (TTVJITTERFLAG) {
      long kj;
      long maxkj = (long) ttvts[0][0];
      newelistttv = malloc((maxkj+1)*sofd);
      newelistttv[0] = (double) maxkj;
      for (kj=0; kj<maxkj; kj++) {
        int jitterindex = (kj < NTTV[0][0]) ? 0 : 1 ; 
        double sigmajitter = p[npl][5+RVJITTERTOT+jitterindex];
        double quadsum = sigmajitter*sigmajitter + ttvts[3][1+kj]*ttvts[3][1+kj];
        // check that the index on ttvts should be 2
        xisq += log(quadsum / (ttvts[3][1+kj]*ttvts[3][1+kj]) );
        newelistttv[1+kj] = sqrt(quadsum);
      }
      ttvts[3] = &newelistttv[0];
    }
    printf("%lf %lf %lf %lf\n", ttvts[0][0], ttvts[1][0], ttvts[2][0], ttvts[3][0]);
    double *ttvdev = devoerr(ttvts);
    printf("ttvts\n");
    long maxil = (long) ttvdev[0];
    for (il=0; il<maxil; il++) xisq += ttvdev[il+1]*ttvdev[il+1];
    free (ttvdev);
    printf("4xisq=%lf\n", xisq);
    if (TTVJITTERFLAG) {
      free(newelistttv);
    }
  }
  printf("postttvjitter\n");
  printf("xisq=%lf\n", xisq);
  
  double photoradius = int_in[3][0][0]; 
  if (SPECTROSCOPY) {
    if (photoradius > SPECRADIUS) xisq += pow( (photoradius - SPECRADIUS) / SPECERRPOS, 2 );
    else xisq += pow( (photoradius - SPECRADIUS) / SPECERRNEG, 2 );
  }
  double photomass = int_in[2][0][0]; 
  if (MASSSPECTROSCOPY) {
    if (photomass > SPECMASS) xisq += pow( (photomass - SPECMASS) / MASSSPECERRPOS, 2 );
    else xisq += pow( (photomass - SPECMASS) / MASSSPECERRNEG, 2 );
  }
  printf("xisqnoinc=%lf\n", xisq);
  if (INCPRIOR) {
    int i0;
    for (i0=0; i0<npl; i0++) {
      printf("%lf\n", xisq);
      xisq += -2.0*log( sin(p[i0][4] *M_PI/180.) ); 
    }
  }
  printf("xisqnoe=%lf\n", xisq);
  double* evector = malloc(npl*sofd);
  if (ECUTON || EPRIOR) {
    if (SQRTE) {
      int i0;
      for (i0=0; i0<npl; i0++) {
        evector[i0] = pow(sqrt( pow(p[i0][2], 2) + pow(p[i0][3], 2) ), 2);
      }
    } else {
      int i0;
      for (i0=0; i0<npl; i0++) {
        evector[i0] = sqrt( pow(p[i0][2], 2) + pow(p[i0][3], 2) );
      }
    }
  }
  if (EPRIOR) {
    int i0;
    for (i0=0; i0<NPL; i0++) {
      if (EPRIORV[i0]) {
        double priorprob;
        if (EPRIOR==1) {
          priorprob = rayleighpdf(evector[i0]);
        } else if (EPRIOR==2) {
          priorprob = normalpdf(evector[i0]);
        }
        xisq += -2.0*log( priorprob );
      }
    }
  }
  printf("xisq=%lf\n", xisq);
  free(dev);
  free(evector);

  if (CONVERT) {  
    int i, j;
    double **aeiparam  = malloc(npl*sofds);
    double **orbparam  = malloc(npl*sofds);
    double masstot[npl+1]; 
    double stateorig[npl][6];
    masstot[0] = int_in[2][0][0];
    for (i=0; i<npl; i++) {
      masstot[i+1] = masstot[i];
      masstot[i+1] += int_in[2][i+1][0];
    }

    for (j=0; j<6; j++) {
      stateorig[0][j] = -int_in[2][1][j+1];
    }
    double *sum = calloc(6,sofd);
    for (i=1; i<npl; i++){
      for (j=0; j<6; j++) {
        sum[j] += int_in[2][i][j+1]*int_in[2][i][0]/masstot[i];
        stateorig[i][j] = -(int_in[2][i+1][j+1] - sum[j]);
      }
    }
    free(sum);
    
    for (i=0; i<npl; i++) {
      aeiparam[i] = statetokep(stateorig[i][0], stateorig[i][1], stateorig[i][2], stateorig[i][3], stateorig[i][4], stateorig[i][5], masstot[i+1]); 
      orbparam[i] = keptoorb(aeiparam[i][0], aeiparam[i][1], aeiparam[i][2], aeiparam[i][3], aeiparam[i][4], aeiparam[i][5], masstot[i+1]);
    }

    char outfile2str[80];
    strcpy(outfile2str, "aei_out_");
    strcat(outfile2str, OUTSTR);
    strcat(outfile2str, ".pldin");
    FILE *outfile2 = fopen(outfile2str, "a");
    fprintf(outfile2, "planet         period (d)               T0 (d)                  e                   i (deg)                 Omega (deg)               omega(deg)               mp (mjup)              rpors           ");
    if (MULTISTAR) {
      fprintf(outfile2, "brightness               c1                    c2\n");
    } else {
      fprintf(outfile2, "\n");
    }
    char ch = 'a';
    double pnum = 0.1;
    int ip;
    for (ip=0; ip<npl; ip++) {
      fprintf(outfile2, "%1.1lf", pnum);
      ch++; 
      pnum+=0.1;
      int ii;
      fprintf(outfile2, "\t%.15lf", orbparam[ip][0]); 
      fprintf(outfile2, "\t%.15lf", orbparam[ip][1]);
      fprintf(outfile2, "\t%.15lf", orbparam[ip][2]);
      fprintf(outfile2, "\t%.15lf", orbparam[ip][3]*180.0/M_PI);
      fprintf(outfile2, "\t%.15lf", orbparam[ip][4]*180.0/M_PI);
      fprintf(outfile2, "\t%.15lf", orbparam[ip][5]*180.0/M_PI);
      for (ii=6; ii<8; ii++) {
        fprintf(outfile2, "\t%.15lf", p[ip][ii]); 
      }
      if (MULTISTAR) {
        for (ii=8; ii<11; ii++) {
          fprintf(outfile2, "\t%.15lf", p[ip][ii]);
        }
      }
      fprintf(outfile2, "\n");
    }
    fprintf(outfile2, "%.15lf ; Mstar (M_sol)\n", p[npl][0]);
    fprintf(outfile2, "%.15lf ; Rstar (R_sol)\n", p[npl][1]);
    fprintf(outfile2, "%.15lf ; c1 (linear limb darkening) \n", p[npl][2]);
    fprintf(outfile2, "%.15lf ; c2 (quadratic limb darkening) \n", p[npl][3]);
    fprintf(outfile2, "%.15lf ; dilution (frac light not from stars in system)\n", p[npl][4]);
    if (RVJITTERFLAG) {
      int ki;
      for (ki=0; ki<(RVJITTERTOT); ki++) {
        fprintf(outfile2, "%.15lf ; rv jitter %i\n", p[npl][5+ki], ki);
      }
    }
    if (TTVJITTERFLAG) {
      int ki;
      for (ki=0; ki<(TTVJITTERTOT); ki++) {
        fprintf(outfile2, "%.15lf ; ttv jitter %i\n", p[npl][5+RVJITTERTOT+ki], ki);
      }
    }
    if (CELERITE) {
      int ki;
      for (ki=0; ki<NCELERITE; ki++) {
        fprintf(outfile2, "%.15lf ; celerite \n", p[npl][5+RVJITTERTOT+TTVJITTERTOT+ki]);
      }
    }
    if (RVCELERITE) {
      int ki;
      for (ki=0; ki<NRVCELERITE; ki++) {
        fprintf(outfile2, "%.15lf ; rvcelerite \n", p[npl][5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+ki]);
      }
    }
    fprintf(outfile2, " ; Comments: These coordinates are jacobian (see Lee & Peale 2003).\n");
    fprintf(outfile2, " ; chisq = %.15lf\n", xisq);

    if (SQRTE) {
      fprintf(outfile2, "planet         period (d)               T0 (d)              sqrt[e] cos(omega)        sqrt[e] sin(omega)        i (deg)                 Omega (deg)            mp (mjup)              rpors           ");
    } else {
      fprintf(outfile2, "planet         period (d)               T0 (d)              e cos(omega)             e sin(omega)             i (deg)                 Omega (deg)            mp (mjup)              rpors           ");
    }
    if (MULTISTAR) {
      fprintf(outfile2, "      brightness               c1                    c2\n");
    } else {
      fprintf(outfile2, "\n");
    }
    ch = 'a';
    pnum = 0.1;
    for (ip=0; ip<npl; ip++) {
      fprintf(outfile2, "%1.1lf", pnum);
      ch++; 
      pnum+=0.1;
      int ii;
      fprintf(outfile2, "\t%.15lf", orbparam[ip][0]); 
      fprintf(outfile2, "\t%.15lf", orbparam[ip][1]);
      if (SQRTE) {
        fprintf(outfile2, "\t%.15lf", sqrt(orbparam[ip][2])*cos(orbparam[ip][5]));
        fprintf(outfile2, "\t%.15lf", sqrt(orbparam[ip][2])*sin(orbparam[ip][5]));
      } else {
        fprintf(outfile2, "\t%.15lf", orbparam[ip][2]*cos(orbparam[ip][5]));
        fprintf(outfile2, "\t%.15lf", orbparam[ip][2]*sin(orbparam[ip][5]));
      }
      fprintf(outfile2, "\t%.15lf", orbparam[ip][3]*180.0/M_PI);
      fprintf(outfile2, "\t%.15lf", orbparam[ip][4]*180.0/M_PI);
      for (ii=6; ii<8; ii++) {
        fprintf(outfile2, "\t%.15lf", p[ip][ii]); 
      }
      if (MULTISTAR) {
        for (ii=8; ii<11; ii++) {
          fprintf(outfile2, "\t%.15lf", p[ip][ii]);
        }
      }
      fprintf(outfile2, "\n");
    }
    fprintf(outfile2, "%.15lf ; Mstar (M_sol)\n", p[npl][0]);
    fprintf(outfile2, "%.15lf ; Rstar (R_sol)\n", p[npl][1]);
    fprintf(outfile2, "%.15lf ; c1 (linear limb darkening) \n", p[npl][2]);
    fprintf(outfile2, "%.15lf ; c2 (quadratic limb darkening) \n", p[npl][3]);
    fprintf(outfile2, "%.15lf ; dilution (frac light not from stars in system)\n", p[npl][4]);
    if (RVJITTERFLAG) {
      int ki;
      for (ki=0; ki<(RVJITTERTOT); ki++) {
        fprintf(outfile2, "%.15lf ; rv jitter %i\n", p[npl][5+ki], ki);
      }
    }
    if (TTVJITTERFLAG) {
      int ki;
      for (ki=0; ki<(TTVJITTERTOT); ki++) {
        fprintf(outfile2, "%.15lf ; ttv jitter %i\n", p[npl][5+RVJITTERTOT+ki], ki);
      }
    }
    if (CELERITE) {
      int ki;
      for (ki=0; ki<NCELERITE; ki++) {
        fprintf(outfile2, "%.15lf ; celerite \n", p[npl][5+RVJITTERTOT+TTVJITTERTOT+ki]);
      }
    }
    if (RVCELERITE) {
      int ki;
      for (ki=0; ki<NRVCELERITE; ki++) {
        fprintf(outfile2, "%.15lf ; rvcelerite \n", p[npl][5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+ki]);
      }
    }
    fprintf(outfile2, " ; Comments: These coordinates are jacobian (see Lee & Peale 2003).\n");
    fprintf(outfile2, " ; chisq = %.15lf\n", xisq);


    fprintf(outfile2, "planet         a (AU)                   e                       i (deg)                omega (deg)          Omega (deg)               f (deg)               mp (mjup)              rpors           ");
    if (MULTISTAR) {
      fprintf(outfile2, "       brightness               c1                    c2\n");
    } else {
      fprintf(outfile2, "\n");
    }
    ch = 'a';
    pnum = 0.1;
    for (ip=0; ip<npl; ip++) {
      fprintf(outfile2, "%1.1lf", pnum);
      ch++; 
      pnum+=0.1;
      int ii;
      fprintf(outfile2, "\t%.15lf", aeiparam[ip][0]); 
      fprintf(outfile2, "\t%.15lf", aeiparam[ip][1]);
      fprintf(outfile2, "\t%.15lf", aeiparam[ip][2]*180.0/M_PI);
      fprintf(outfile2, "\t%.15lf", aeiparam[ip][3]*180.0/M_PI);
      fprintf(outfile2, "\t%.15lf", aeiparam[ip][4]*180.0/M_PI);
      fprintf(outfile2, "\t%.15lf", aeiparam[ip][5]*180.0/M_PI);
      for (ii=6; ii<8; ii++) {
        fprintf(outfile2, "\t%.15lf", p[ip][ii]); 
      }
      if (MULTISTAR) {
        for (ii=8; ii<11; ii++) {
          fprintf(outfile2, "\t%.15lf", p[ip][ii]);
        }
      }
      fprintf(outfile2, "\n");
    }
    fprintf(outfile2, "%.15lf ; Mstar (M_sol)\n", p[npl][0]);
    fprintf(outfile2, "%.15lf ; Rstar (R_sol)\n", p[npl][1]);
    fprintf(outfile2, "%.15lf ; c1 (linear limb darkening) \n", p[npl][2]);
    fprintf(outfile2, "%.15lf ; c2 (quadratic limb darkening) \n", p[npl][3]);
    fprintf(outfile2, "%.15lf ; dilution (frac light not from stars in system)\n", p[npl][4]);
    if (RVJITTERFLAG) {
      int ki;
      for (ki=0; ki<(RVJITTERTOT); ki++) {
        fprintf(outfile2, "%.15lf ; rv jitter %i\n", p[npl][5+ki], ki);
      }
    }
    if (TTVJITTERFLAG) {
      int ki;
      for (ki=0; ki<(TTVJITTERTOT); ki++) {
        fprintf(outfile2, "%.15lf ; ttv jitter %i\n", p[npl][5+RVJITTERTOT+ki], ki);
      }
    }
    if (CELERITE) {
      int ki;
      for (ki=0; ki<NCELERITE; ki++) {
        fprintf(outfile2, "%.15lf ; celerite \n", p[npl][5+RVJITTERTOT+TTVJITTERTOT+ki]);
      }
    }
    if (RVCELERITE) {
      int ki;
      for (ki=0; ki<NRVCELERITE; ki++) {
        fprintf(outfile2, "%.15lf ; rvcelerite \n", p[npl][5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+ki]);
      }
    }
    fprintf(outfile2, " ; Comments: These coordinates are jacobian (see Lee & Peale 2003).\n");
    fprintf(outfile2, " ; chisq = %.15lf\n", xisq);

    fclose(outfile2);

    for (i=0; i<npl; i++) free(aeiparam[i]);
    free(aeiparam);
    for (i=0; i<npl; i++) free(orbparam[i]);
    free(orbparam);

  }

  int i;
  free(int_in[0][0]);
  free(int_in[0]);
  free(int_in[1][0]);
  free(int_in[1]);
  int i1;
  for(i1=0; i1<npl+1; i1++) free(int_in[2][i1]);
  free(int_in[2]);
  free(int_in[3][0]);
  free(int_in[3]);
  free(int_in);

  //Only 1 array is malloced!
  free(flux_rvs[0][2]);
  if (RVS) {
    free(flux_rvs[1][2]);
  }
  free(flux_rvs[0]);
  free(flux_rvs[1]);
  free(flux_rvs);

#elif (demcmc_compile==1)
  printf("made it3\n");
  // xi squared array (one value per walker)
  double *xisq0 = malloc(nwalkers*sofd);
  double **xisq0N;
  if (NTEMPS) {
    xisq0N = malloc(NTEMPS*sofds);
    for (w=0; w<NTEMPS; w++) {
      xisq0N[w] = malloc(nwalkers*sofd);
    }
  }

  if (! NTEMPS) {
    for (i=0; i<nwalkers; i++) {
      double ***int_in = dsetup2(&p0local[pperwalker*i], npl);
      printf("Converted to XYZ\n");
      double ***flux_rvs; 
      if (MULTISTAR) flux_rvs = dpintegrator_multi(int_in, tfe, tve, cadencelist);
      else flux_rvs = dpintegrator_single(int_in, tfe, tve, nte, cadencelist);
      printf("Computed Flux\n");
      double **ttvts = flux_rvs[2];
      double **flux = flux_rvs[0];
      double **radvs = flux_rvs[1];
      double *dev = devoerr(flux);
      double xisqtemp = 0;
      long il;
      long maxil = (long) dev[0];
      if (! CELERITE) { 
        for (il=0; il<maxil; il++) xisqtemp += dev[il+1]*dev[il+1];
      } else { // if celerite
        double *xs = flux_rvs[0][0];
            long maxil = (long) xs[0];
        double *trueys = flux_rvs[0][1];
        double *modelys = flux_rvs[0][2];
        double *es = flux_rvs[0][3];
        double *diffys = malloc(sofd*maxil);
        for (il=0; il<maxil; il++) { 
           diffys[il] = trueys[il+1]-modelys[il+1];
        }
        double *yvarp = malloc(sofd*maxil);
        for (il=0; il<maxil; il++) { 
           yvarp[il] = es[il+1]*es[il+1]; 
        }
        double *xp = &xs[1]; 
        
        int j_real = 0;
        int j_complex;
        double jitter, k1, k2, k3, S0, w0, Q;
        jitter = p0local[pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+0];// p[npl][5+RVJITTERTOT+TTVJITTERTOT+0];
        S0 = p0local[pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+1]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+1];
        w0 = p0local[pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+2]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+2];
        Q = p0local[pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+3];//p[npl][5+RVJITTERTOT+TTVJITTERTOT+3];
        if (Q >= 0.5) {
          j_complex = 1;
        } else {
          j_complex = 2;
        }
        VectorXd a_real(j_real),
                c_real(j_real),
                a_comp(j_complex),
                b_comp(j_complex),
                c_comp(j_complex),
                d_comp(j_complex);
        if (Q >= 0.5) {
          k1 = S0*w0*Q;
          k2 = sqrt(4.*Q*Q - 1.);
          k3 = w0/(2.*Q);
          a_comp << k1;
          b_comp << k1/k2;
          c_comp << k3;
          d_comp << k3*k2;
        } else {
          j_complex = 2;
          k1 = 0.5*S0*w0*Q;
          k2 = sqrt(1. - 4.*Q*Q);
          k3 = w0/(2.*Q);
          a_comp << k1*(1. + 1./k2), k1*(1. - 1./k2);
          b_comp << 0., 0.;
          c_comp << k3*(1. - k2), k3*(1. + k2);
          d_comp << 0., 0.;
        }
      
     
        printf("%lf %lf %lf %lf\n", a_comp[0], b_comp[0], c_comp[0], d_comp[0]);
    
      
        VectorXd x = VectorXd::Map(xp, maxil);
        VectorXd yvar = VectorXd::Map(yvarp, maxil);
        VectorXd dy = VectorXd::Map(diffys, maxil);
     
        celerite::solver::CholeskySolver<double> solver;
        try {
        solver.compute(
              jitter,
              a_real, c_real,
              a_comp, b_comp, c_comp, d_comp,
              x, yvar  // Note: this is the measurement _variance_
          );
        } catch ( std::exception e1 ) {
          xisqtemp = HUGE_VAL;
          goto celeritefail;
        }
      
        // see l.186-192 in celerite.py
        double logdet, diffs, llike;
        logdet = solver.log_determinant();
        diffs = solver.dot_solve(dy); 
        llike = -0.5 * (diffs + logdet); // hm, this is wrong 
    
      
      
        xisqtemp = diffs+logdet;
        printf("xsiqtemp=%lf\n", xisqtemp);
    
        free(yvarp);
        free(diffys);
    
      }
      double *newelist;
   
      if (RVS) {
        if (! RVCELERITE) { 
          double *newelist;
          if (RVJITTERFLAG) {
            long kj;
            long maxkj = (long) tve[0][0]; 
            newelist=malloc((maxkj+1)*sofd);
            newelist[0] = (double) maxkj;
            for (kj=0; kj<maxkj; kj++) {
              int jitterindex = (int) tve[3][1+kj]*NTELESCOPES + tve[4][1+kj];
              //double sigmajitter = p[npl][5+jitterindex]*MPSTOAUPD;
              double sigmajitter = p0local[pperwalker*i+npl*pperplan+5+jitterindex]*MPSTOAUPD;
              double quadsum = sigmajitter*sigmajitter + radvs[3][1+kj]*radvs[3][1+kj];
              // double check this... factor of 1/2
              xisqtemp += log(quadsum / (radvs[3][1+kj]*radvs[3][1+kj]) );
              newelist[1+kj] = sqrt( quadsum );
            }
            radvs[3] = newelist;
          }
  
          double *rvdev = devoerr(radvs);
          long maxil = (long) rvdev[0];
          for (il=0; il<maxil; il++) xisqtemp += rvdev[il+1]*rvdev[il+1];
          free(rvdev);
    
    
        } else { // if rvcelerite
          double *xs = flux_rvs[1][0];
          long maxil = (long) xs[0];
          double *trueys = flux_rvs[1][1];
          double *modelys = flux_rvs[1][2];
          double *es = flux_rvs[1][3];
          double *diffys = malloc(sofd*maxil);
          for (il=0; il<maxil; il++) { 
             diffys[il] = trueys[il+1]-modelys[il+1];
          }
          double *yvarp = malloc(sofd*maxil);
          for (il=0; il<maxil; il++) { 
             yvarp[il] = es[il+1]*es[il+1]; 
          }
          double *xp = &xs[1]; 
          
          int j_real = 0;
          int j_complex;
          double jitter, k1, k2, k3, S0, w0, Q;
          jitter = p0local[pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+0];// p[npl][5+RVJITTERTOT+TTVJITTERTOT+0];
          S0 = p0local[pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+1]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+1];
          w0 = p0local[pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+2]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+2];
          Q = p0local[pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+3];//p[npl][5+RVJITTERTOT+TTVJITTERTOT+3];
          if (Q >= 0.5) {
            j_complex = 1;
          } else {
            j_complex = 2;
          }
          VectorXd a_real(j_real),
                  c_real(j_real),
                  a_comp(j_complex),
                  b_comp(j_complex),
                  c_comp(j_complex),
                  d_comp(j_complex);
          if (Q >= 0.5) {
            k1 = S0*w0*Q;
            k2 = sqrt(4.*Q*Q - 1.);
            k3 = w0/(2.*Q);
            a_comp << k1;
            b_comp << k1/k2;
            c_comp << k3;
            d_comp << k3*k2;
          } else {
            j_complex = 2;
            k1 = 0.5*S0*w0*Q;
            k2 = sqrt(1. - 4.*Q*Q);
            k3 = w0/(2.*Q);
            a_comp << k1*(1. + 1./k2), k1*(1. - 1./k2);
            b_comp << 0., 0.;
            c_comp << k3*(1. - k2), k3*(1. + k2);
            d_comp << 0., 0.;
          }
       
        
          VectorXd x = VectorXd::Map(xp, maxil);
          VectorXd yvar = VectorXd::Map(yvarp, maxil);
          VectorXd dy = VectorXd::Map(diffys, maxil);
        
          celerite::solver::CholeskySolver<double> solver;
          solver.compute(
                jitter,
                a_real, c_real,
                a_comp, b_comp, c_comp, d_comp,
                x, yvar  // Note: this is the measurement _variance_
            );
        
          // see l.186-192 in celerite.py
          double logdet, diffs, llike;
          logdet = solver.log_determinant();
          diffs = solver.dot_solve(dy); 
          llike = -0.5 * (diffs + logdet); 
        
          xisqtemp += diffs+logdet;
         
#if (demcmc_compile==0)
          printf("diffs=%lf\n", diffs);
          printf("logdet=%lf\n", logdet);
          printf("llike=%lf\n", llike);
      
          VectorXd prediction = solver.predict(dy, x);
          char tmtefstr[1000];
          strcpy(tmtefstr, "rv_");
          strcat(tmtefstr, OUTSTR);
          strcat(tmtefstr, ".gp_rvout");
          FILE *tmtef;
          tmtef = fopen(tmtefstr, "a");
          long ijk;
          for (ijk=0; ijk<maxil; ijk++) {
            fprintf(tmtef, "%.12lf \n", prediction[ijk]);
          }
          fclose(tmtef);// = openf(tmtstr,"w");
       
#endif        
          free(yvarp);
          free(diffys);
        }
      }

      if (TTVCHISQ) {
        double *newelistttv;
        if (TTVJITTERFLAG) {
          long kj;
          long maxkj = (long) ttvts[0][0];
          newelistttv = malloc((maxkj+1)*sofd);
          newelistttv[0] = (double) maxkj;
          for (kj=0; kj<maxkj; kj++) {
            int jitterindex = (kj < NTTV[0][0]) ? 0 : 1 ; 
            double sigmajitter = p0local[pperwalker*i+npl*pperplan+5+RVJITTERTOT+jitterindex];
            double quadsum = sigmajitter*sigmajitter + ttvts[3][1+kj]*ttvts[3][1+kj];
            // check that the index on ttvts should be 2
            xisqtemp += log(quadsum / (ttvts[3][1+kj]*ttvts[3][1+kj]) );
            newelistttv[1+kj] = sqrt(quadsum);
          }
          ttvts[3] = newelistttv;
        }
        double *ttvdev = devoerr(ttvts);
        long maxil = (long) ttvdev[0];
        for (il=0; il<maxil; il++) xisqtemp += ttvdev[il+1]*ttvdev[il+1];
        free (ttvdev);
        if (TTVJITTERFLAG) {
          free(newelistttv);
        }
      }


      double photoradius;
      photoradius = p0local[i*pperwalker+npl*pperplan+1]; 
      double photomass;
      photomass = p0local[i*pperwalker+npl*pperplan+0]; 
      double* evector; 
      evector = malloc(npl*sofd);
      if (ECUTON || EPRIOR) {
        if (SQRTE) {
          int i0;
          for (i0=0; i0<npl; i0++) {
            evector[i0] = pow(sqrt( pow(p0local[i*pperwalker+i0*pperplan+2], 2) + pow(p0local[i*pperwalker+i0*pperplan+3], 2) ), 2);
          }
        } else {
          int i0;
          for (i0=0; i0<npl; i0++) {
            evector[i0] = sqrt( pow(p0local[i*pperwalker+i0*pperplan+2], 2) + pow(p0local[i*pperwalker+i0*pperplan+3], 2) );
          }
        }
      }
      if (SPECTROSCOPY) {
        if (photoradius > SPECRADIUS) xisqtemp += pow( (photoradius - SPECRADIUS) / SPECERRPOS, 2 );
        else xisqtemp += pow( (photoradius - SPECRADIUS) / SPECERRNEG, 2 );
      }
      if (MASSSPECTROSCOPY) {
        if (photomass > SPECMASS) xisqtemp += pow( (photomass - SPECMASS) / MASSSPECERRPOS, 2 );
        else xisqtemp += pow( (photomass - SPECMASS) / MASSSPECERRNEG, 2 );
      }
      if (INCPRIOR) {
        int i0;
        for (i0=0; i0<npl; i0++) {
          xisqtemp += -2.0*log( sin(p0local[i*pperwalker+i0*pperplan+4] *M_PI/180.) ); 
        }
      }
      if (EPRIOR) {
        int i0;
        for (i0=0; i0<NPL; i0++) {
          if (EPRIORV[i0]) {
            double priorprob;
            if (EPRIOR==1) {
              priorprob = rayleighpdf(evector[i0]);
            } else if (EPRIOR==2) {
              priorprob = normalpdf(evector[i0]);
            }
            xisqtemp += -2.0*log( priorprob );
          }
        }
      }

      celeritefail:

      xisq0[i] = xisqtemp;
 
      free(evector);

      free(int_in[0][0]);
      free(int_in[0]);
      free(int_in[1][0]);
      free(int_in[1]);
      int i1;
      for(i1=0; i1<npl+1; i1++) free(int_in[2][i1]);
      free(int_in[2]);
      free(int_in[3][0]);
      free(int_in[3]);
      free(int_in);
  
      //Only 1 array is malloced!
      free(flux_rvs[0][2]);
      if (RVS) {
        free(flux_rvs[1][2]);
        if (RVJITTERFLAG) {
          free(newelist);
        }
      }
  
      free(flux_rvs[0]);
      free(flux_rvs[1]);
      free(flux_rvs);
  
      free(dev);
  
  
    }
  } else { //if NTEMPS:
    for (i=0; i<nwalkers; i++) {
      for (w=0; w<NTEMPS; w++) {
        double ***int_in = dsetup2(&p0localN[w][pperwalker*i], npl);
        double ***flux_rvs; 
        if (MULTISTAR) flux_rvs = dpintegrator_multi(int_in, tfe, tve, cadencelist);
        else flux_rvs = dpintegrator_single(int_in, tfe, tve, nte, cadencelist);
        double **ttvts = flux_rvs[2];
        double **flux = flux_rvs[0];
        double **radvs = flux_rvs[1];
        double *dev = devoerr(flux);
        double xisqtemp = 0;
        long il;
        long maxil = (long) dev[0];
        if (! CELERITE) { 
          for (il=0; il<maxil; il++) xisqtemp += dev[il+1]*dev[il+1];
        } else { // if celerite
          double *xs = flux_rvs[0][0];
              long maxil = (long) xs[0];
          double *trueys = flux_rvs[0][1];
          double *modelys = flux_rvs[0][2];
          double *es = flux_rvs[0][3];
          double *diffys = malloc(sofd*maxil);
          for (il=0; il<maxil; il++) { 
             diffys[il] = trueys[il+1]-modelys[il+1];
          }
          double *yvarp = malloc(sofd*maxil);
          for (il=0; il<maxil; il++) { 
             yvarp[il] = es[il+1]*es[il+1]; 
          }
          double *xp = &xs[1]; 
          
          int j_real = 0;
          int j_complex;
          double jitter, k1, k2, k3, S0, w0, Q;
          jitter = p0localN[w][pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+0];// p[npl][5+RVJITTERTOT+TTVJITTERTOT+0];
          S0 = p0localN[w][pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+1]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+1];
          w0 = p0localN[w][pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+2]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+2];
          Q = p0localN[w][pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+3];//p[npl][5+RVJITTERTOT+TTVJITTERTOT+3];
          if (Q >= 0.5) {
            j_complex = 1;
          } else {
            j_complex = 2;
          }
          VectorXd a_real(j_real),
                  c_real(j_real),
                  a_comp(j_complex),
                  b_comp(j_complex),
                  c_comp(j_complex),
                  d_comp(j_complex);
          if (Q >= 0.5) {
            k1 = S0*w0*Q;
            k2 = sqrt(4.*Q*Q - 1.);
            k3 = w0/(2.*Q);
            a_comp << k1;
            b_comp << k1/k2;
            c_comp << k3;
            d_comp << k3*k2;
          } else {
            j_complex = 2;
            k1 = 0.5*S0*w0*Q;
            k2 = sqrt(1. - 4.*Q*Q);
            k3 = w0/(2.*Q);
            a_comp << k1*(1. + 1./k2), k1*(1. - 1./k2);
            b_comp << 0., 0.;
            c_comp << k3*(1. - k2), k3*(1. + k2);
            d_comp << 0., 0.;
          }
        
        
          VectorXd x = VectorXd::Map(xp, maxil);
          VectorXd yvar = VectorXd::Map(yvarp, maxil);
          VectorXd dy = VectorXd::Map(diffys, maxil);
        
          celerite::solver::CholeskySolver<double> solver;
          solver.compute(
                jitter,
                a_real, c_real,
                a_comp, b_comp, c_comp, d_comp,
                x, yvar  // Note: this is the measurement _variance_
            );
        
          // see l.186-192 in celerite.py
          double logdet, diffs, llike;
          logdet = solver.log_determinant();
          diffs = solver.dot_solve(dy); 
          llike = -0.5 * (diffs + logdet); 
        
          xisqtemp = diffs+logdet;
         
          free(yvarp);
          free(diffys);
        }
        double *newelist;
  
        if (RVS) {
          if (! RVCELERITE) { 
            double *newelist;
            if (RVJITTERFLAG) {
              long kj;
              long maxkj = (long) tve[0][0]; 
              newelist=malloc((maxkj+1)*sofd);
              newelist[0] = (double) maxkj;
              for (kj=0; kj<maxkj; kj++) {
                int jitterindex = (int) tve[3][1+kj]*NTELESCOPES + tve[4][1+kj];
                double sigmajitter = p0localN[w][pperwalker*i+npl*pperplan+5+jitterindex]*MPSTOAUPD;
                double quadsum = sigmajitter*sigmajitter + radvs[3][1+kj]*radvs[3][1+kj];
                // double check this... factor of 1/2
                xisqtemp += log(quadsum / (radvs[3][1+kj]*radvs[3][1+kj]) );
                newelist[1+kj] = sqrt( quadsum );
              }
              radvs[3] = newelist;
            }
    
            double *rvdev = devoerr(radvs);
            long maxil = (long) rvdev[0];
            for (il=0; il<maxil; il++) xisqtemp += rvdev[il+1]*rvdev[il+1];
            free(rvdev);
  
          } else { // if rvcelerite
            double *xs = flux_rvs[1][0];
            long maxil = (long) xs[0];
            double *trueys = flux_rvs[1][1];
            double *modelys = flux_rvs[1][2];
            double *es = flux_rvs[1][3];
            double *diffys = malloc(sofd*maxil);
            for (il=0; il<maxil; il++) { 
               diffys[il] = trueys[il+1]-modelys[il+1];
            }
            double *yvarp = malloc(sofd*maxil);
            for (il=0; il<maxil; il++) { 
               yvarp[il] = es[il+1]*es[il+1]; 
            }
            double *xp = &xs[1]; 
            
            int j_real = 0;
            int j_complex;
            double jitter, k1, k2, k3, S0, w0, Q;
            jitter = p0localN[w][pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+0];// p[npl][5+RVJITTERTOT+TTVJITTERTOT+0];
            S0 = p0localN[w][pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+1]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+1];
            w0 = p0localN[w][pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+2]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+2];
            Q = p0localN[w][pperwalker*i+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+3];//p[npl][5+RVJITTERTOT+TTVJITTERTOT+3];
            if (Q >= 0.5) {
              j_complex = 1;
            } else {
              j_complex = 2;
            }
            VectorXd a_real(j_real),
                    c_real(j_real),
                    a_comp(j_complex),
                    b_comp(j_complex),
                    c_comp(j_complex),
                    d_comp(j_complex);
            if (Q >= 0.5) {
              k1 = S0*w0*Q;
              k2 = sqrt(4.*Q*Q - 1.);
              k3 = w0/(2.*Q);
              a_comp << k1;
              b_comp << k1/k2;
              c_comp << k3;
              d_comp << k3*k2;
            } else {
              j_complex = 2;
              k1 = 0.5*S0*w0*Q;
              k2 = sqrt(1. - 4.*Q*Q);
              k3 = w0/(2.*Q);
              a_comp << k1*(1. + 1./k2), k1*(1. - 1./k2);
              b_comp << 0., 0.;
              c_comp << k3*(1. - k2), k3*(1. + k2);
              d_comp << 0., 0.;
            }
         
          
            VectorXd x = VectorXd::Map(xp, maxil);
            VectorXd yvar = VectorXd::Map(yvarp, maxil);
            VectorXd dy = VectorXd::Map(diffys, maxil);
          
            celerite::solver::CholeskySolver<double> solver;
            solver.compute(
                  jitter,
                  a_real, c_real,
                  a_comp, b_comp, c_comp, d_comp,
                  x, yvar  // Note: this is the measurement _variance_
              );
          
            // see l.186-192 in celerite.py
            double logdet, diffs, llike;
            logdet = solver.log_determinant();
            diffs = solver.dot_solve(dy); 
            llike = -0.5 * (diffs + logdet); 
          
            xisqtemp += diffs+logdet;
           
  #if (demcmc_compile==0)
            printf("a+comp=%lf\n", a_comp[0]);
            printf("b+comp=%lf\n", b_comp[0]);
            printf("c+comp=%lf\n", c_comp[0]);
            printf("d+comp=%lf\n", d_comp[0]);
            printf("diffs=%lf\n", diffs);
            printf("logdet=%lf\n", logdet);
            printf("llike=%lf\n", llike);
        
            VectorXd prediction = solver.predict(dy, x);
            char tmtefstr[1000];
            strcpy(tmtefstr, "rv_");
            strcat(tmtefstr, OUTSTR);
            strcat(tmtefstr, ".gp_rvout");
            FILE *tmtef;
            tmtef = fopen(tmtefstr, "a");
            long ijk;
            for (ijk=0; ijk<maxil; ijk++) {
              fprintf(tmtef, "%.12lf \n", prediction[ijk]);
            }
            fclose(tmtef);// = openf(tmtstr,"w"); 
  #endif      
          
            free(yvarp);
            free(diffys);
        
          }
        }
        if (TTVCHISQ) {
          double *newelistttv;
          if (TTVJITTERFLAG) {
            long kj;
            long maxkj = (long) ttvts[0][0];
            newelistttv = malloc((maxkj+1)*sofd);
            newelistttv[0] = (double) maxkj;
            for (kj=0; kj<maxkj; kj++) {
              int jitterindex = (kj < NTTV[0][0]) ? 0 : 1 ; 
              //double sigmajitter = p[npl][5+RVJITTERTOT+jitterindex];
              double sigmajitter = p0localN[w][pperwalker*i+npl*pperplan+5+RVJITTERTOT+jitterindex];
              double quadsum = sigmajitter*sigmajitter + ttvts[3][1+kj]*ttvts[3][1+kj];
              // check that the index on ttvts should be 2
              xisqtemp += log(quadsum / (ttvts[3][1+kj]*ttvts[3][1+kj]) );
              newelistttv[1+kj] = sqrt(quadsum);
            }
            ttvts[3] = newelistttv;
          }
          
          double *ttvdev = devoerr(ttvts);
          long maxil = (long) ttvdev[0];
          for (il=0; il<maxil; il++) xisqtemp += ttvdev[il+1]*ttvdev[il+1];
          free (ttvdev);
          if (TTVJITTERFLAG) {
            free(newelistttv);
          }
  
        }
        double photoradius = p0localN[w][i*pperwalker+npl*pperplan+1];
        double photomass = p0localN[w][i*pperwalker+npl*pperplan+0];
        double* evector = malloc(npl*sofd);
        if (ECUTON || EPRIOR) {
          if (SQRTE) {
            int i0;
            for (i0=0; i0<npl; i0++) {
              evector[i0] = pow(sqrt( pow(p0localN[w][i*pperwalker+i0*pperplan+2], 2) + pow(p0localN[w][i*pperwalker+i0*pperplan+3], 2) ), 2);
            }
          } else {
            int i0;
            for (i0=0; i0<npl; i0++) {
              evector[i0] = sqrt( pow(p0localN[w][i*pperwalker+i0*pperplan+2], 2) + pow(p0localN[w][i*pperwalker+i0*pperplan+3], 2) );
            }
          }
        }
        if (SPECTROSCOPY) {
          if (photoradius > SPECRADIUS) xisqtemp += pow( (photoradius - SPECRADIUS) / SPECERRPOS, 2 );
          else xisqtemp += pow( (photoradius - SPECRADIUS) / SPECERRNEG, 2 );
        }
        if (MASSSPECTROSCOPY) {
          if (photomass > SPECMASS) xisqtemp += pow( (photomass - SPECMASS) / MASSSPECERRPOS, 2 );
          else xisqtemp += pow( (photomass - SPECMASS) / MASSSPECERRNEG, 2 );
        }
        if (INCPRIOR) {
                int i0;
          for (i0=0; i0<npl; i0++) {
            xisqtemp += -2.0*log( sin(p0localN[w][i*pperwalker+i0*pperplan+4] * M_PI/180.)); 
          }
        }
        if (EPRIOR) {
          int i0;
          for (i0=0; i0<NPL; i0++) {
            if (EPRIORV[i0]) {
              double priorprob;
              if (EPRIOR==1) {
                priorprob = rayleighpdf(evector[i0]);
              } else if (EPRIOR==2) {
                priorprob = normalpdf(evector[i0]);
              }
              xisqtemp += -2.0*log( priorprob );
            }
          }
        }
        xisq0N[w][i] = xisqtemp;
        free(evector);
    
        free(int_in[0][0]);
        free(int_in[0]);
        free(int_in[1][0]);
        free(int_in[1]);
        int i1;
        for(i1=0; i1<npl+1; i1++) free(int_in[2][i1]);
        free(int_in[2]);
        free(int_in[3][0]);
        free(int_in[3]);
        free(int_in);
    
        //Only 1 array is malloced!
        free(flux_rvs[0][2]);
        if (RVS) {
          free(flux_rvs[1][2]);
          if (RVJITTERFLAG) {
            free(newelist);
          }
        }
    
        free(flux_rvs[0]);
        free(flux_rvs[1]);
        free(flux_rvs);
    
        free(dev);
      }
    }
  }

  

  if (! RESTART ) {
    // set lowest xisq
    xisqmin = HUGE_VAL;
  }


  int k = 0;
  int *acceptance = malloc(nwalkers*sofi);
  int **acceptanceN;
  FILE *outfile;

  double *p0localcopy = malloc(pperwalker*sofd);

  int *acceptanceglobal = malloc(nwalkers*sofi);
  double *p0global = malloc(totalparams*sofd);
  double *xisq0global = malloc(nwalkers*sofd);
  int **acceptanceglobalN;
  double **p0globalN;
  double **xisq0globalN;

  if (NTEMPS) {
    acceptanceN = malloc(NTEMPS*sofis);
    acceptanceglobalN = malloc(NTEMPS*sofis);
    p0globalN = malloc(NTEMPS*sofds);
    xisq0globalN = malloc(NTEMPS*sofds);
    for (w=0; w<NTEMPS; w++) {
      acceptanceN[w] = malloc(nwalkers*sofi);
      acceptanceglobalN[w] = malloc(nwalkers*sofi);
      p0globalN[w] = malloc(totalparams*sofd);
      xisq0globalN[w] = malloc(nwalkers*sofd);
    }
  }

  double *beta;
  if (NTEMPS) {
    beta = malloc(NTEMPS*sofd);
    beta[0] = 1.0;
    for (w=1; w<NTEMPS; w++) {
      //beta[w] = beta[w-1] / sqrt(2.0);
      beta[w] = beta[w-1] / 2.;
    }
  }

  // loop over generations
  unsigned long nwcore = (unsigned long) RANK;
  unsigned long nwalkersul = (unsigned long) nwalkers;
  gsl_rng_set(rnw, seed*(nwcore+2));

  while (jj<nsteps) {
 
    if (RANK==0 && jj % 10 == 0) {
      sout = fopen("demcmc.stdout", "a");
      fprintf(sout, "begin gen %li\n", jj);
      fclose(sout);
    }

    //time testing
    struct timespec start, finish;
    double elapsed;
    if (RANK==0 && jj % 10 ==0) { 
      clock_gettime(CLOCK_MONOTONIC, &start);
    }

    int nwi;
    int ncores;
    MPI_Comm_size(MPI_COMM_WORLD, &ncores);
    int npercore = nwalkers / ncores;
    //nwcore = (unsigned long) RANK;
    printf("ncores %i nwalkers %i\n", ncores, nwalkers);
    
    unsigned long nw;
    long nwinit=nwcore*npercore;
    long nwfin=(nwcore+1)*npercore;
    // This loops allows you to have more walkers than cores
    for (nw=nwinit; nw < nwfin; nw++) {
  
      if (! NTEMPS) {
        memcpy(p0localcopy, &p0local[nw*pperwalker], pperwalker*sofd);
  
        acceptance[nw] = 1;
    
        unsigned long nw1;
        do nw1 = gsl_rng_uniform_int(rnw, nwalkersul); while (nw1 == nw); 
        unsigned long nw2;
        do nw2 = gsl_rng_uniform_int(rnw, nwalkersul); while (nw2 == nw || nw2 == nw1); 
    
        int notallowed=0;
    
        int ip;
        if (bimodf) {
          if ( jj % bimodf == 0) {
            for (ip=0; ip<pstar; ip++) {
              p0local[nw*pperwalker+npl*pperplan+ip] += (gamma+(1-gamma)*bimodlist[npl*pperplan+ip])*(p0local[nw1*pperwalker+npl*pperplan+ip]-p0local[nw2*pperwalker+npl*pperplan+ip])*(1-parfix[npl*pperplan+ip])*(1+(1+(gamma-1)*bimodlist[npl*pperplan+ip])*gsl_ran_gaussian(rnw, 0.1));
            }
            if ( DIGT0 && (p0local[nw*pperwalker+npl*pperplan+4] < 0.0) ) notallowed=1;
            for (ip=0; ip<npl; ip++) {
              int ii;
              for (ii=0; ii<pperplan; ii++) {
                p0local[nw*pperwalker+ip*pperplan+ii] += (gamma+(1-gamma)*bimodlist[ip*pperplan+ii])*(p0local[nw1*pperwalker+ip*pperplan+ii]-p0local[nw2*pperwalker+ip*pperplan+ii])*(1-parfix[ip*pperplan+ii])*(1+(1+(gamma-1)*bimodlist[ip*pperplan+ii])*gsl_ran_gaussian(rnw, 0.1));
              }
              // make sure i and Omega angles are not cycling through:
              if ( p0local[nw*pperwalker+ip*pperplan+4] < 0.0 || p0local[nw*pperwalker+ip*pperplan+4] > 180.0 ) notallowed=1;
              if ( p0local[nw*pperwalker+ip*pperplan+5] < -180.0 || p0local[nw*pperwalker+ip*pperplan+5] > 180.0 ) notallowed=1;
              // make sure i>=90
              if ( IGT90 && (p0local[nw*pperwalker+ip*pperplan+4] < 90.0) ) notallowed=1;
              // make sure m>=0
              if ( MGT0 && (p0local[nw*pperwalker+ip*pperplan+6] < 0.0) ) notallowed=1;
              //make sure density is allowed
              if (DENSITYCUTON) {
                double massg = p0local[nw*pperwalker+ip*pperplan+6] / MSOMJ * MSUNGRAMS; 
                double radcm = p0local[nw*pperwalker+npl*pperplan+1] * p0local[nw*pperwalker+ip*pperplan+7] * RSUNCM;
                double rhogcc = massg / (4./3.*M_PI*radcm*radcm*radcm);
                if (rhogcc > MAXDENSITY[ip]) notallowed=1;
              }
            } 
          } else {
            for (ip=0; ip<pstar; ip++) {
              p0local[nw*pperwalker+npl*pperplan+ip] += (gamma+(1-gamma)*bimodlist[npl*pperplan+ip])*(p0local[nw1*pperwalker+npl*pperplan+ip]-p0local[nw2*pperwalker+npl*pperplan+ip])*(1-parfix[npl*pperplan+ip])*(1+(1+(gamma-1)*bimodlist[npl*pperplan+ip])*gsl_ran_gaussian(rnw, 0.1));
            }
            if ( DIGT0 && (p0local[nw*pperwalker+npl*pperplan+4] < 0.0) ) notallowed=1;
            for (ip=0; ip<npl; ip++) {
              int ii;
              for (ii=0; ii<pperplan; ii++) {
                p0local[nw*pperwalker+ip*pperplan+ii] += gamma*(p0local[nw1*pperwalker+ip*pperplan+ii]-p0local[nw2*pperwalker+ip*pperplan+ii])*(1-parfix[ip*pperplan+ii])*(1+gsl_ran_gaussian(rnw, 0.1));
              }
              // make sure i and Omega angles are not cycling through:
              if ( p0local[nw*pperwalker+ip*pperplan+4] < 0.0 || p0local[nw*pperwalker+ip*pperplan+4] > 180.0 ) notallowed=1;
              if ( p0local[nw*pperwalker+ip*pperplan+5] < -180.0 || p0local[nw*pperwalker+ip*pperplan+5] > 180.0 ) notallowed=1;
              // make sure i>=90
              if ( IGT90 && (p0local[nw*pperwalker+ip*pperplan+4] < 90.0) ) notallowed=1;
              // make sure m>=0
              if ( MGT0 && (p0local[nw*pperwalker+ip*pperplan+6] < 0.0) ) notallowed=1;
              //make sure density is allowed
              if (DENSITYCUTON) {
                double massg = p0local[nw*pperwalker+ip*pperplan+6] / MSOMJ * MSUNGRAMS; 
                double radcm = p0local[nw*pperwalker+npl*pperplan+1] * p0local[nw*pperwalker+ip*pperplan+7] * RSUNCM;
                double rhogcc = massg / (4./3.*M_PI*radcm*radcm*radcm);
                if (rhogcc > MAXDENSITY[ip]) notallowed=1;
              }
            } 
          }
        } else {
          for (ip=0; ip<pstar; ip++) {
            p0local[nw*pperwalker+npl*pperplan+ip] += gamma*(p0local[nw1*pperwalker+npl*pperplan+ip]-p0local[nw2*pperwalker+npl*pperplan+ip])*(1-parfix[npl*pperplan+ip])*(1+gsl_ran_gaussian(rnw, 0.1));
          }
          if ( DIGT0 && (p0local[nw*pperwalker+npl*pperplan+4] < 0.0) ) notallowed=1;
          for (ip=0; ip<npl; ip++) {
            int ii;
            for (ii=0; ii<pperplan; ii++) {
              p0local[nw*pperwalker+ip*pperplan+ii] += gamma*(p0local[nw1*pperwalker+ip*pperplan+ii]-p0local[nw2*pperwalker+ip*pperplan+ii])*(1-parfix[ip*pperplan+ii])*(1+gsl_ran_gaussian(rnw, 0.1));
            }
            // make sure i and Omega angles are not cycling through:
            if ( p0local[nw*pperwalker+ip*pperplan+4] < 0.0 || p0local[nw*pperwalker+ip*pperplan+4] > 180.0 ) notallowed=1;
            if ( p0local[nw*pperwalker+ip*pperplan+5] < -180.0 || p0local[nw*pperwalker+ip*pperplan+5] > 180.0 ) notallowed=1;
            // make sure i>=90
            if ( IGT90 && (p0local[nw*pperwalker+ip*pperplan+4] < 90.0) ) notallowed=1;
            // make sure m>=0
            if ( MGT0 && (p0local[nw*pperwalker+ip*pperplan+6] < 0.0) ) notallowed=1;
            //make sure density is allowed
            if (DENSITYCUTON) {
              double massg = p0local[nw*pperwalker+ip*pperplan+6] / MSOMJ * MSUNGRAMS; 
              double radcm = p0local[nw*pperwalker+npl*pperplan+1] * p0local[nw*pperwalker+ip*pperplan+7] * RSUNCM;
              double rhogcc = massg / (4./3.*M_PI*radcm*radcm*radcm);
              if (rhogcc > MAXDENSITY[ip]) notallowed=1;
            }
          } 
        }
        double photoradius = p0local[nw*pperwalker+npl*pperplan+1]; 
        double photomass = p0local[nw*pperwalker+npl*pperplan+0]; 
        
        // check that RVJITTER >= 0
        if (RVS) {
          if (RVJITTERFLAG) {
            int ki;
            for (ki=0; ki<RVJITTERTOT; ki++) {
              if (p0local[nw*pperwalker+npl*pperplan+(5+ki)] < 0.0) {
                notallowed=1;
              }
            }
          }
        }
        
        // check that celerite terms >= 0
        if (CELERITE) {
            int ki;
            for (ki=0; ki<NCELERITE; ki++) {
              if (p0local[nw*pperwalker+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+ki] < 0.0) {
                notallowed=1;
              }
            }
        }
        if (RVCELERITE) {
            int ki;
            for (ki=0; ki<NRVCELERITE; ki++) {
              if (p0local[nw*pperwalker+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+CELERITE*4+ki] < 0.0) {
                notallowed=1;
              }
            }
        }
    
        double* evector = malloc(npl*sofd);
        if (ECUTON || EPRIOR) {
          if (SQRTE) {
            int i0;
            for (i0=0; i0<npl; i0++) {
              evector[i0] = pow(sqrt( pow(p0local[nw*pperwalker+i0*pperplan+2], 2) + pow(p0local[nw*pperwalker+i0*pperplan+3], 2) ), 2);
            }
          } else {
            int i0;
            for (i0=0; i0<npl; i0++) {
              evector[i0] = sqrt( pow(p0local[nw*pperwalker+i0*pperplan+2], 2) + pow(p0local[nw*pperwalker+i0*pperplan+3], 2) );
            }
          }
        }
        // make sure e_d cos  w_d is within its range
        if (ECUTON) {
          double* emax = EMAX;
          int i0;
          for (i0=0; i0<npl; i0++) {
            if (evector[i0] > emax[i0] ) notallowed=1;
          }
        }
        //make sure ld constants in range
        if ( DILUTERANGE && (p0local[nw*pperwalker+npl*pperplan+2] < 0.0 || p0local[nw*pperwalker+npl*pperplan+2] > 1.0 || p0local[nw*pperwalker+npl*pperplan+3] < 0.0 || p0local[nw*pperwalker+npl*pperplan+3] > 1.0) ) notallowed=1;
  
        double ***nint_in = dsetup2(&p0local[nw*pperwalker], npl);
    
        if (notallowed) { 
            
          acceptance[nw] = 0;
    
        } else {
    
          double ***nflux_rvs; 
          if (MULTISTAR) nflux_rvs = dpintegrator_multi(nint_in, tfe, tve, cadencelist);
          else nflux_rvs = dpintegrator_single(nint_in, tfe, tve, nte, cadencelist);
          double **nttvts = nflux_rvs[2];
          double **nflux = nflux_rvs[0];
          double **nradvs = nflux_rvs[1];
          double *ndev = devoerr(nflux);
          double nxisqtemp = 0;
          long il;
          long maxil = (long) ndev[0];
          if (! CELERITE) { 
            for (il=0; il<maxil; il++) nxisqtemp += ndev[il+1]*ndev[il+1];
          } else { // if celerite
            double *xs = nflux_rvs[0][0];
                long maxil = (long) xs[0];
            double *trueys = nflux_rvs[0][1];
            double *modelys = nflux_rvs[0][2];
            double *es = nflux_rvs[0][3];
            double *diffys = malloc(sofd*maxil);
            for (il=0; il<maxil; il++) { 
               diffys[il] = trueys[il+1]-modelys[il+1];
            }
            double *yvarp = malloc(sofd*maxil);
            for (il=0; il<maxil; il++) { 
               yvarp[il] = es[il+1]*es[il+1]; 
            }
            double *xp = &xs[1]; 
            
            int j_real = 0;
            int j_complex;
            double jitter, k1, k2, k3, S0, w0, Q;
            jitter = p0local[pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+0];// p[npl][5+RVJITTERTOT+TTVJITTERTOT+0];
            S0 = p0local[pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+1]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+1];
            w0 = p0local[pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+2]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+2];
            Q = p0local[pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+3];//p[npl][5+RVJITTERTOT+TTVJITTERTOT+3];
            if (Q >= 0.5) {
              j_complex = 1;
            } else {
              j_complex = 2;
            }
            VectorXd a_real(j_real),
                    c_real(j_real),
                    a_comp(j_complex),
                    b_comp(j_complex),
                    c_comp(j_complex),
                    d_comp(j_complex);
            if (Q >= 0.5) {
              k1 = S0*w0*Q;
              k2 = sqrt(4.*Q*Q - 1.);
              k3 = w0/(2.*Q);
              a_comp << k1;
              b_comp << k1/k2;
              c_comp << k3;
              d_comp << k3*k2;
            } else {
              j_complex = 2;
              k1 = 0.5*S0*w0*Q;
              k2 = sqrt(1. - 4.*Q*Q);
              k3 = w0/(2.*Q);
              a_comp << k1*(1. + 1./k2), k1*(1. - 1./k2);
              b_comp << 0., 0.;
              c_comp << k3*(1. - k2), k3*(1. + k2);
              d_comp << 0., 0.;
            }
          
          
          
            VectorXd x = VectorXd::Map(xp, maxil);
            VectorXd yvar = VectorXd::Map(yvarp, maxil);
            VectorXd dy = VectorXd::Map(diffys, maxil);
          
            celerite::solver::CholeskySolver<double> solver;
            solver.compute(
                  jitter,
                  a_real, c_real,
                  a_comp, b_comp, c_comp, d_comp,
                  x, yvar  // Note: this is the measurement _variance_
              );
          
            // see l.186-192 in celerite.py
            double logdet, diffs, llike;
            logdet = solver.log_determinant();
            diffs = solver.dot_solve(dy); 
            llike = -0.5 * (diffs + logdet); 
          
            nxisqtemp = diffs+logdet;
          
            free(yvarp);
            free(diffys);
        
          }
          double *newelist;
          if (RVS) {
            if (! RVCELERITE) { 
              double *newelist;
              if (RVJITTERFLAG) {
                long kj;
                long maxkj = (long) tve[0][0]; 
                newelist=malloc((maxkj+1)*sofd);
                newelist[0] = (double) maxkj;
                for (kj=0; kj<maxkj; kj++) {
                  int jitterindex = (int) tve[3][1+kj]*NTELESCOPES + tve[4][1+kj];
                  double sigmajitter = p0local[pperwalker*nw+npl*pperplan+5+jitterindex]*MPSTOAUPD;
                  double quadsum = sigmajitter*sigmajitter + nradvs[3][1+kj]*nradvs[3][1+kj];
                  // double check this... factor of 1/2
                  nxisqtemp += log(quadsum / (nradvs[3][1+kj]*nradvs[3][1+kj]) );
                  newelist[1+kj] = sqrt( quadsum );
                }
                nradvs[3] = newelist;
              }
      
              double *rvdev = devoerr(nradvs);
              long maxil = (long) rvdev[0];
              for (il=0; il<maxil; il++) nxisqtemp += rvdev[il+1]*rvdev[il+1];
              free(rvdev);
    
            } else { // if rvcelerite
              double *xs = nflux_rvs[1][0];
              long maxil = (long) xs[0];
              double *trueys = nflux_rvs[1][1];
              double *modelys = nflux_rvs[1][2];
              double *es = nflux_rvs[1][3];
              double *diffys = malloc(sofd*maxil);
              for (il=0; il<maxil; il++) { 
                 diffys[il] = trueys[il+1]-modelys[il+1];
              }
              double *yvarp = malloc(sofd*maxil);
              for (il=0; il<maxil; il++) { 
                 yvarp[il] = es[il+1]*es[il+1]; 
              }
              double *xp = &xs[1]; 
              
              int j_real = 0;
              int j_complex;
              double jitter, k1, k2, k3, S0, w0, Q;
              jitter = p0local[pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+0];// p[npl][5+RVJITTERTOT+TTVJITTERTOT+0];
              S0 = p0local[pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+1]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+1];
              w0 = p0local[pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+2]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+2];
              Q = p0local[pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+3];//p[npl][5+RVJITTERTOT+TTVJITTERTOT+3];
              if (Q >= 0.5) {
                j_complex = 1;
              } else {
                j_complex = 2;
              }
              VectorXd a_real(j_real),
                      c_real(j_real),
                      a_comp(j_complex),
                      b_comp(j_complex),
                      c_comp(j_complex),
                      d_comp(j_complex);
              if (Q >= 0.5) {
                k1 = S0*w0*Q;
                k2 = sqrt(4.*Q*Q - 1.);
                k3 = w0/(2.*Q);
                a_comp << k1;
                b_comp << k1/k2;
                c_comp << k3;
                d_comp << k3*k2;
              } else {
                j_complex = 2;
                k1 = 0.5*S0*w0*Q;
                k2 = sqrt(1. - 4.*Q*Q);
                k3 = w0/(2.*Q);
                a_comp << k1*(1. + 1./k2), k1*(1. - 1./k2);
                b_comp << 0., 0.;
                c_comp << k3*(1. - k2), k3*(1. + k2);
                d_comp << 0., 0.;
              }
            
              VectorXd x = VectorXd::Map(xp, maxil);
              VectorXd yvar = VectorXd::Map(yvarp, maxil);
              VectorXd dy = VectorXd::Map(diffys, maxil);
            
              celerite::solver::CholeskySolver<double> solver;
              solver.compute(
                    jitter,
                    a_real, c_real,
                    a_comp, b_comp, c_comp, d_comp,
                    x, yvar  // Note: this is the measurement _variance_
                );
            
              // see l.186-192 in celerite.py
              double logdet, diffs, llike;
              logdet = solver.log_determinant();
              diffs = solver.dot_solve(dy); 
              llike = -0.5 * (diffs + logdet); 
            
              nxisqtemp += diffs+logdet;
             
#if (demcmc_compile==0)
          
              VectorXd prediction = solver.predict(dy, x);
              char tmtefstr[1000];
              strcpy(tmtefstr, "rv_");
              strcat(tmtefstr, OUTSTR);
              strcat(tmtefstr, ".gp_rvout");
              FILE *tmtef;
              tmtef = fopen(tmtefstr, "a");
              long ijk;
              for (ijk=0; ijk<maxil; ijk++) {
                fprintf(tmtef, "%.12lf \n", prediction[ijk]);
              }
              fclose(tmtef);// = openf(tmtstr,"w");
           
#endif
            
              free(yvarp);
              free(diffys);
          
            }
          }
          if (TTVCHISQ) {
            double *newelistttv;
            if (TTVJITTERFLAG) {
              long kj;
              long maxkj = (long) nttvts[0][0];
              newelistttv = malloc((maxkj+1)*sofd);
              newelistttv[0] = (double) maxkj;
              for (kj=0; kj<maxkj; kj++) {
                int jitterindex = (kj < NTTV[0][0]) ? 0 : 1 ; 
                //double sigmajitter = p[npl][5+RVJITTERTOT+jitterindex];
                double sigmajitter = p0local[pperwalker*nw+npl*pperplan+5+RVJITTERTOT+jitterindex];
                double quadsum = sigmajitter*sigmajitter + nttvts[3][1+kj]*nttvts[3][1+kj];
                // check that the index on ttvts should be 2
                nxisqtemp += log(quadsum / (nttvts[3][1+kj]*nttvts[3][1+kj]) );
                newelistttv[1+kj] = sqrt(quadsum);
              }
              nttvts[3] = newelistttv;
            }
          
            double *ttvdev = devoerr(nttvts);
            long maxil = (long) ttvdev[0];
            for (il=0; il<maxil; il++) nxisqtemp += ttvdev[il+1]*ttvdev[il+1];
            free (ttvdev);
            if (TTVJITTERFLAG) {
              free(newelistttv);
            }
    
          }
          
          //double photoradius = p0local[nw*pperwalker+npl*pperplan+1]; 
          if (SPECTROSCOPY) {
            if (photoradius > SPECRADIUS) nxisqtemp += pow( (photoradius - SPECRADIUS) / SPECERRPOS, 2 );
            else nxisqtemp += pow( (photoradius - SPECRADIUS) / SPECERRNEG, 2 );
          }
          if (MASSSPECTROSCOPY) {
            if (photomass > SPECMASS) nxisqtemp += pow( (photomass - SPECMASS) / MASSSPECERRPOS, 2 );
            else nxisqtemp += pow( (photomass - SPECMASS) / MASSSPECERRNEG, 2 );
          }
          if (INCPRIOR) {
                  int i0;
            for (i0=0; i0<npl; i0++) {
              nxisqtemp += -2.0*log( sin(p0local[nw*pperwalker+i0*pperplan+4] * M_PI/180.)       ); 
            }
          }
          if (EPRIOR) {
            int i0;
            for (i0=0; i0<NPL; i0++) {
              if (EPRIORV[i0]) {
                double priorprob;
                if (EPRIOR==1) {
                  priorprob = rayleighpdf(evector[i0]);
                } else if (EPRIOR==2) {
                  priorprob = normalpdf(evector[i0]);
                }
                nxisqtemp += -2.0*log( priorprob );
              }
            }
          }
          double xisq = nxisqtemp;
          
  
          free(nint_in[0][0]);
          free(nint_in[0]);
          free(nint_in[1][0]);
          free(nint_in[1]);
          int i1;
          for(i1=0; i1<npl+1; i1++) free(nint_in[2][i1]);
          free(nint_in[2]);
          free(nint_in[3][0]);
          free(nint_in[3]);
          free(nint_in);
      
          free(nflux_rvs[0][2]);
          if (RVS) {
            free(nflux_rvs[1][2]);
            if (RVJITTERFLAG) {
              free(newelist);
            }
          }
      
          free(nflux_rvs[0]);
          free(nflux_rvs[1]);
          free(nflux_rvs);
      
          free(ndev);
      
          // prob that you should take new state
          double prob;
          prob = exp((xisq0[nw]-xisq)/2.);
   
          double bar = gsl_rng_uniform(rnw);
      
          // accept new state?
          if (prob <= bar || isnan(prob)) {
            acceptance[nw] = 0;
          } else {
            xisq0[nw] = xisq;
          } 
    
        }
        free(evector);
    
        // switch back to old ones if not accepted
        if (acceptance[nw] == 0) {
          memcpy(&p0local[nw*pperwalker], p0localcopy, pperwalker*sofd);
        }
  
      } else {  // if NTEMPS:
        for (w=0; w<NTEMPS; w++) {
          memcpy(p0localcopy, &p0localN[w][nw*pperwalker], pperwalker*sofd);
          acceptanceN[w][nw] = 1;
          unsigned long nw1;
          do nw1 = gsl_rng_uniform_int(rnw, nwalkersul); while (nw1 == nw); 
          unsigned long nw2;
          do nw2 = gsl_rng_uniform_int(rnw, nwalkersul); while (nw2 == nw || nw2 == nw1); 
      
          int notallowed=0;
      
          int ip;
          if (bimodf) {
            if ( jj % bimodf == 0) {
              for (ip=0; ip<npl; ip++) {
                int ii;
                for (ii=0; ii<pperplan; ii++) {
                  p0localN[w][nw*pperwalker+ip*pperplan+ii] += (gammaN[w]+(1-gammaN[w])*bimodlist[ip*pperplan+ii])*(p0localN[w][nw1*pperwalker+ip*pperplan+ii]-p0localN[w][nw2*pperwalker+ip*pperplan+ii])*(1-parfix[ip*pperplan+ii])*(1+(1+(gammaN[w]-1)*bimodlist[ip*pperplan+ii])*gsl_ran_gaussian(rnw, 0.1));
                }
                // make sure i and Omega angles are not cycling through:
                if ( p0localN[w][nw*pperwalker+ip*pperplan+4] < 0.0 || p0localN[w][nw*pperwalker+ip*pperplan+4] > 180.0 ) notallowed=1;
                if ( p0localN[w][nw*pperwalker+ip*pperplan+5] < -180.0 || p0localN[w][nw*pperwalker+ip*pperplan+5] > 180.0 ) notallowed=1;
                // make sure i>=90
                if ( IGT90 && (p0localN[w][nw*pperwalker+ip*pperplan+4] < 90.0) ) notallowed=1;
                // make sure m>=0
                if ( MGT0 && (p0localN[w][nw*pperwalker+ip*pperplan+6] < 0.0) ) notallowed=1;
                //make sure density is allowed
                if (DENSITYCUTON) {
                  double massg = p0localN[w][nw*pperwalker+ip*pperplan+6] / MSOMJ * MSUNGRAMS; 
                  double radcm = p0localN[w][nw*pperwalker+npl*pperplan+1] * p0local[nw*pperwalker+ip*pperplan+7] * RSUNCM;
                  double rhogcc = massg / (4./3.*M_PI*radcm*radcm*radcm);
                  if (rhogcc > MAXDENSITY[ip]) notallowed=1;
                }
              } 
              for (ip=0; ip<pstar; ip++) {
                p0localN[w][nw*pperwalker+npl*pperplan+ip] += (gammaN[w]+(1-gammaN[w])*bimodlist[npl*pperplan+ip])*(p0localN[w][nw1*pperwalker+npl*pperplan+ip]-p0localN[w][nw2*pperwalker+npl*pperplan+ip])*(1-parfix[npl*pperplan+ip])*(1+(1+(gammaN[w]-1)*bimodlist[npl*pperplan+ip])*gsl_ran_gaussian(rnw, 0.1));
              }
              if ( DIGT0 && (p0localN[w][nw*pperwalker+npl*pperplan+4] < 0.0) ) notallowed=1;
            } else {
              for (ip=0; ip<npl; ip++) {
                int ii;
                for (ii=0; ii<pperplan; ii++) {
                  p0localN[w][nw*pperwalker+ip*pperplan+ii] += gammaN[w]*(p0localN[w][nw1*pperwalker+ip*pperplan+ii]-p0localN[w][nw2*pperwalker+ip*pperplan+ii])*(1-parfix[ip*pperplan+ii])*(1+gsl_ran_gaussian(rnw, 0.1));
                }
                // make sure i and Omega angles are not cycling through:
                if ( p0localN[w][nw*pperwalker+ip*pperplan+4] < 0.0 || p0localN[w][nw*pperwalker+ip*pperplan+4] > 180.0 ) notallowed=1;
                if ( p0localN[w][nw*pperwalker+ip*pperplan+5] < -180.0 || p0localN[w][nw*pperwalker+ip*pperplan+5] > 180.0 ) notallowed=1;
                // make sure i>=90
                if ( IGT90 && (p0localN[w][nw*pperwalker+ip*pperplan+4] < 90.0) ) notallowed=1;
                // make sure m>=0
                if ( MGT0 && (p0localN[w][nw*pperwalker+ip*pperplan+6] < 0.0) ) notallowed=1;
                //make sure density is allowed
                if (DENSITYCUTON) {
                  double massg = p0localN[w][nw*pperwalker+ip*pperplan+6] / MSOMJ * MSUNGRAMS; 
                  double radcm = p0localN[w][nw*pperwalker+npl*pperplan+1] * p0local[nw*pperwalker+ip*pperplan+7] * RSUNCM;
                  double rhogcc = massg / (4./3.*M_PI*radcm*radcm*radcm);
                  if (rhogcc > MAXDENSITY[ip]) notallowed=1;
                }
              } 
              for (ip=0; ip<pstar; ip++) {
                p0localN[w][nw*pperwalker+npl*pperplan+ip] += gammaN[w]*(p0localN[w][nw1*pperwalker+npl*pperplan+ip]-p0localN[w][nw2*pperwalker+npl*pperplan+ip])*(1-parfix[npl*pperplan+ip])*(1+gsl_ran_gaussian(rnw, 0.1));
              }
              if ( DIGT0 && (p0localN[w][nw*pperwalker+npl*pperplan+4] < 0.0) ) notallowed=1;
            }
          } else {
            for (ip=0; ip<npl; ip++) {
              int ii;
              for (ii=0; ii<pperplan; ii++) {
                p0localN[w][nw*pperwalker+ip*pperplan+ii] += gammaN[w]*(p0localN[w][nw1*pperwalker+ip*pperplan+ii]-p0localN[w][nw2*pperwalker+ip*pperplan+ii])*(1-parfix[ip*pperplan+ii])*(1+gsl_ran_gaussian(rnw, 0.1));
              }
              // make sure i and Omega angles are not cycling through:
              if ( p0localN[w][nw*pperwalker+ip*pperplan+4] < 0.0 || p0localN[w][nw*pperwalker+ip*pperplan+4] > 180.0 ) notallowed=1;
              if ( p0localN[w][nw*pperwalker+ip*pperplan+5] < -180.0 || p0localN[w][nw*pperwalker+ip*pperplan+5] > 180.0 ) notallowed=1;
              // make sure i>=90
              if ( IGT90 && (p0localN[w][nw*pperwalker+ip*pperplan+4] < 90.0) ) notallowed=1;
              // make sure m>=0
              if ( MGT0 && (p0localN[w][nw*pperwalker+ip*pperplan+6] < 0.0) ) notallowed=1;
              //make sure density is allowed
              if (DENSITYCUTON) {
                double massg = p0localN[w][nw*pperwalker+ip*pperplan+6] / MSOMJ * MSUNGRAMS; 
                double radcm = p0localN[w][nw*pperwalker+npl*pperplan+1] * p0local[nw*pperwalker+ip*pperplan+7] * RSUNCM;
                double rhogcc = massg / (4./3.*M_PI*radcm*radcm*radcm);
                if (rhogcc > MAXDENSITY[ip]) notallowed=1;
              }
            } 
            for (ip=0; ip<pstar; ip++) {
              p0localN[w][nw*pperwalker+npl*pperplan+ip] += gammaN[w]*(p0localN[w][nw1*pperwalker+npl*pperplan+ip]-p0localN[w][nw2*pperwalker+npl*pperplan+ip])*(1-parfix[npl*pperplan+ip])*(1+gsl_ran_gaussian(rnw, 0.1));
            }
            if ( DIGT0 && (p0localN[w][nw*pperwalker+npl*pperplan+4] < 0.0) ) notallowed=1;
          }
          double photoradius = p0localN[w][nw*pperwalker+npl*pperplan+1]; 
          double photomass = p0localN[w][nw*pperwalker+npl*pperplan+0]; 
        
          // check that RVJITTER >= 0
          if (RVS) {
            if (RVJITTERFLAG) {
              int ki;
              for (ki=0; ki<RVJITTERTOT; ki++) {
                if (p0local[nw*pperwalker+npl*pperplan+(5+ki)] < 0.0) {
                  notallowed=1;
                }
              }
            }
          }
          // check that TTVJITTER >= 0
          if (TTVJITTERFLAG) {
              int ki;
              for (ki=0; ki<TTVJITTERTOT; ki++) {
                if (p0local[nw*pperwalker+npl*pperplan+(5+ki)+RVJITTERTOT] < 0.0) {
                  notallowed=1;
                }
              }
          }
          // check that celerite terms >= 0
          if (CELERITE) {
              int ki;
              for (ki=0; ki<NCELERITE; ki++) {
                if (p0local[nw*pperwalker+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+ki] < 0.0) {
                  notallowed=1;
                }
              }
          }
          if (RVCELERITE) {
              int ki;
              for (ki=0; ki<NRVCELERITE; ki++) {
                if (p0local[nw*pperwalker+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+ki] < 0.0) {
                  notallowed=1;
                }
              }
          }
      
          // make sure e_d cos  w_d is within its range
          double* evector = malloc(npl*sofd);
          if (ECUTON || EPRIOR) {
            if (SQRTE) {
              int i0;
              for (i0=0; i0<npl; i0++) {
                evector[i0] = pow(sqrt( pow(p0localN[w][nw*pperwalker+i0*pperplan+2], 2) + pow(p0localN[w][nw*pperwalker+i0*pperplan+3], 2) ), 2);
              }
            } else {
              int i0;
              for (i0=0; i0<npl; i0++) {
                evector[i0] = sqrt( pow(p0localN[w][nw*pperwalker+i0*pperplan+2], 2) + pow(p0localN[w][nw*pperwalker+i0*pperplan+3], 2) );
              }
            }
          }
          // make sure e_d cos  w_d is within its range
          if (ECUTON) {
            double* emax = EMAX;
            int i0;
            for (i0=0; i0<npl; i0++) {
              if (evector[i0] > emax[i0] ) notallowed=1;
            }
          }
          //make sure ld constants in range
          if ( DILUTERANGE && (p0localN[w][nw*pperwalker+npl*pperplan+2] < 0.0 || p0localN[w][nw*pperwalker+npl*pperplan+2] > 1.0 || p0localN[w][nw*pperwalker+npl*pperplan+3] < 0.0 || p0localN[w][nw*pperwalker+npl*pperplan+3] > 1.0) ) notallowed=1;
      
          double ***nint_in = dsetup2(&p0localN[w][nw*pperwalker], npl);
          if (notallowed) { 
              
            acceptanceN[w][nw] = 0;
      
          } else {
      
            double ***nflux_rvs; 
            if (MULTISTAR) nflux_rvs = dpintegrator_multi(nint_in, tfe, tve, cadencelist);
            else nflux_rvs = dpintegrator_single(nint_in, tfe, tve, nte, cadencelist);
            double **nttvts = nflux_rvs[2];
            double **nflux = nflux_rvs[0];
            double **nradvs = nflux_rvs[1];
            double *ndev = devoerr(nflux);
            double nxisqtemp = 0;
            long il;
            long maxil = (long) ndev[0];
            if (! CELERITE) { 
              for (il=0; il<maxil; il++) nxisqtemp += ndev[il+1]*ndev[il+1];
            } else { // if celerite
              double *xs = nflux_rvs[0][0];
                  long maxil = (long) xs[0];
              double *trueys = nflux_rvs[0][1];
              double *modelys = nflux_rvs[0][2];
              double *es = nflux_rvs[0][3];
              double *diffys = malloc(sofd*maxil);
              for (il=0; il<maxil; il++) { 
                 diffys[il] = trueys[il+1]-modelys[il+1];
              }
              double *yvarp = malloc(sofd*maxil);
              for (il=0; il<maxil; il++) { 
                 yvarp[il] = es[il+1]*es[il+1]; 
              }
              double *xp = &xs[1]; 
              
              int j_real = 0;
              int j_complex;
              double jitter, k1, k2, k3, S0, w0, Q;
              jitter = p0localN[w][pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+0];// p[npl][5+RVJITTERTOT+TTVJITTERTOT+0];
              S0 = p0localN[w][pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+1]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+1];
              w0 = p0localN[w][pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+2]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+2];
              Q = p0localN[w][pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+3];//p[npl][5+RVJITTERTOT+TTVJITTERTOT+3];
              if (Q >= 0.5) {
                j_complex = 1;
              } else {
                j_complex = 2;
              }
              VectorXd a_real(j_real),
                      c_real(j_real),
                      a_comp(j_complex),
                      b_comp(j_complex),
                      c_comp(j_complex),
                      d_comp(j_complex);
              if (Q >= 0.5) {
                k1 = S0*w0*Q;
                k2 = sqrt(4.*Q*Q - 1.);
                k3 = w0/(2.*Q);
                a_comp << k1;
                b_comp << k1/k2;
                c_comp << k3;
                d_comp << k3*k2;
              } else {
                j_complex = 2;
                k1 = 0.5*S0*w0*Q;
                k2 = sqrt(1. - 4.*Q*Q);
                k3 = w0/(2.*Q);
                a_comp << k1*(1. + 1./k2), k1*(1. - 1./k2);
                b_comp << 0., 0.;
                c_comp << k3*(1. - k2), k3*(1. + k2);
                d_comp << 0., 0.;
              }
            
            
            
              VectorXd x = VectorXd::Map(xp, maxil);
              VectorXd yvar = VectorXd::Map(yvarp, maxil);
              VectorXd dy = VectorXd::Map(diffys, maxil);
            
              celerite::solver::CholeskySolver<double> solver;
              solver.compute(
                    jitter,
                    a_real, c_real,
                    a_comp, b_comp, c_comp, d_comp,
                    x, yvar  // Note: this is the measurement _variance_
                );
            
              // see l.186-192 in celerite.py
              double logdet, diffs, llike;
              logdet = solver.log_determinant();
              diffs = solver.dot_solve(dy); 
              llike = -0.5 * (diffs + logdet); 
            
              nxisqtemp = diffs+logdet;
            
              free(yvarp);
              free(diffys);
          
            }
            double *newelist;
            
            if (RVS) {
              if (! RVCELERITE) { 
        
                double *newelist;
                if (RVJITTERFLAG) {
                  long kj;
                  long maxkj = (long) tve[0][0]; 
                  newelist=malloc((maxkj+1)*sofd);
                  newelist[0] = (double) maxkj;
                  for (kj=0; kj<maxkj; kj++) {
                    int jitterindex = (int) tve[3][1+kj]*NTELESCOPES + tve[4][1+kj];
                    //double sigmajitter = p[npl][5+jitterindex]*MPSTOAUPD;
                    double sigmajitter = p0localN[w][pperwalker*nw+npl*pperplan+5+jitterindex]*MPSTOAUPD;
                    double quadsum = sigmajitter*sigmajitter + nradvs[3][1+kj]*nradvs[3][1+kj];
                    // double check this... factor of 1/2
                    nxisqtemp += log(quadsum / (nradvs[3][1+kj]*nradvs[3][1+kj]) );
                    newelist[1+kj] = sqrt( quadsum );
                  }
                  nradvs[3] = newelist;
                }
        
                double *rvdev = devoerr(nradvs);
                long maxil = (long) rvdev[0];
                for (il=0; il<maxil; il++) nxisqtemp += rvdev[il+1]*rvdev[il+1];
                free(rvdev);
        
              } else { // if rvcelerite
                double *xs = nflux_rvs[1][0];
                    long maxil = (long) xs[0];
                double *trueys = nflux_rvs[1][1];
                double *modelys = nflux_rvs[1][2];
                double *es = nflux_rvs[1][3];
                double *diffys = malloc(sofd*maxil);
                for (il=0; il<maxil; il++) { 
                   diffys[il] = trueys[il+1]-modelys[il+1];
                }
                double *yvarp = malloc(sofd*maxil);
                for (il=0; il<maxil; il++) { 
                   yvarp[il] = es[il+1]*es[il+1]; 
                }
                double *xp = &xs[1]; 
                
                int j_real = 0;
                int j_complex;
                double jitter, k1, k2, k3, S0, w0, Q;
                jitter = p0localN[w][pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+0];// p[npl][5+RVJITTERTOT+TTVJITTERTOT+0];
                S0 = p0localN[w][pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+1]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+1];
                w0 = p0localN[w][pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+2]; //p[npl][5+RVJITTERTOT+TTVJITTERTOT+2];
                Q = p0localN[w][pperwalker*nw+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+3];//p[npl][5+RVJITTERTOT+TTVJITTERTOT+3];
                if (Q >= 0.5) {
                  j_complex = 1;
                } else {
                  j_complex = 2;
                }
                VectorXd a_real(j_real),
                        c_real(j_real),
                        a_comp(j_complex),
                        b_comp(j_complex),
                        c_comp(j_complex),
                        d_comp(j_complex);
                if (Q >= 0.5) {
                  k1 = S0*w0*Q;
                  k2 = sqrt(4.*Q*Q - 1.);
                  k3 = w0/(2.*Q);
                  a_comp << k1;
                  b_comp << k1/k2;
                  c_comp << k3;
                  d_comp << k3*k2;
                } else {
                  j_complex = 2;
                  k1 = 0.5*S0*w0*Q;
                  k2 = sqrt(1. - 4.*Q*Q);
                  k3 = w0/(2.*Q);
                  a_comp << k1*(1. + 1./k2), k1*(1. - 1./k2);
                  b_comp << 0., 0.;
                  c_comp << k3*(1. - k2), k3*(1. + k2);
                  d_comp << 0., 0.;
                }
              
                VectorXd x = VectorXd::Map(xp, maxil);
                VectorXd yvar = VectorXd::Map(yvarp, maxil);
                VectorXd dy = VectorXd::Map(diffys, maxil);
              
                celerite::solver::CholeskySolver<double> solver;
                solver.compute(
                      jitter,
                      a_real, c_real,
                      a_comp, b_comp, c_comp, d_comp,
                      x, yvar  // Note: this is the measurement _variance_
                  );
              
                // see l.186-192 in celerite.py
                double logdet, diffs, llike;
                logdet = solver.log_determinant();
                diffs = solver.dot_solve(dy); 
                llike = -0.5 * (diffs + logdet); 
              
                nxisqtemp += diffs+logdet;
               
    #if (demcmc_compile==0)
            
                VectorXd prediction = solver.predict(dy, x);
                char tmtefstr[1000];
                strcpy(tmtefstr, "rv_");
                strcat(tmtefstr, OUTSTR);
                strcat(tmtefstr, ".gp_rvout");
                FILE *tmtef;
                tmtef = fopen(tmtefstr, "a");
                long ijk;
                for (ijk=0; ijk<maxil; ijk++) {
                  fprintf(tmtef, "%.12lf \n", prediction[ijk]);
                }
                fclose(tmtef);// = openf(tmtstr,"w");
             
    #endif
              
                free(yvarp);
                free(diffys);
            
              }
            }
            if (TTVCHISQ) {
              double *newelistttv;
              if (TTVJITTERFLAG) {
                long kj;
                long maxkj = (long) nttvts[0][0];
                newelistttv = malloc((maxkj+1)*sofd);
                newelistttv[0] = (double) maxkj;
                for (kj=0; kj<maxkj; kj++) {
                  int jitterindex = (kj < NTTV[0][0]) ? 0 : 1 ; 
                  //double sigmajitter = p[npl][5+RVJITTERTOT+jitterindex];
                  double sigmajitter = p0localN[w][pperwalker*nw+npl*pperplan+5+RVJITTERTOT+jitterindex];
                  double quadsum = sigmajitter*sigmajitter + nttvts[3][1+kj]*nttvts[3][1+kj];
                  // check that the index on ttvts should be 2
                  nxisqtemp += log(quadsum / (nttvts[3][1+kj]*nttvts[3][1+kj]) );
                  newelistttv[1+kj] = sqrt(quadsum);
                }
                nttvts[3] = newelistttv;
              }
              
              double *ttvdev = devoerr(nttvts);
              long maxil = (long) ttvdev[0];
              for (il=0; il<maxil; il++) nxisqtemp += ttvdev[il+1]*ttvdev[il+1];
              free (ttvdev);
              if (TTVJITTERFLAG) {
                free(newelistttv);
              }
      
            }
                
            if (SPECTROSCOPY) {
              if (photoradius > SPECRADIUS) nxisqtemp += pow( (photoradius - SPECRADIUS) / SPECERRPOS, 2 );
              else nxisqtemp += pow( (photoradius - SPECRADIUS) / SPECERRNEG, 2 );
            }
            if (MASSSPECTROSCOPY) {
              if (photomass > SPECMASS) nxisqtemp += pow( (photomass - SPECMASS) / MASSSPECERRPOS, 2 );
              else nxisqtemp += pow( (photomass - SPECMASS) / MASSSPECERRNEG, 2 );
            }
            if (INCPRIOR) {
              int i0;
              for (i0=0; i0<npl; i0++) {
                nxisqtemp += -2.0*log( sin(p0localN[w][nw*pperwalker+i0*pperplan+4] * M_PI/180.)); 
              }
            }
            if (EPRIOR) {
              int i0;
              for (i0=0; i0<NPL; i0++) {
                if (EPRIORV[i0]) {
                  double priorprob;
                  if (EPRIOR==1) {
                    priorprob = rayleighpdf(evector[i0]);
                  } else if (EPRIOR==2) {
                    priorprob = normalpdf(evector[i0]);
                  }
                  nxisqtemp += -2.0*log( priorprob );
                }
              }
            }
            double xisq = nxisqtemp;
        
            free(nint_in[0][0]);
            free(nint_in[0]);
            free(nint_in[1][0]);
            free(nint_in[1]);
            int i1;
            for(i1=0; i1<npl+1; i1++) free(nint_in[2][i1]);
            free(nint_in[2]);
            free(nint_in[3][0]);
            free(nint_in[3]);
            free(nint_in);
        
            free(nflux_rvs[0][2]);
            if (RVS) {
              free(nflux_rvs[1][2]);
              if (RVJITTERFLAG) {
                free(newelist);
              }
            }
        
            free(nflux_rvs[0]);
            free(nflux_rvs[1]);
            free(nflux_rvs);
        
            free(ndev);
        
            // prob that you should take new state
            double prob;
            prob = exp(beta[w]*(xisq0N[w][nw]-xisq)/2.);
      
      
            double bar = gsl_rng_uniform(rnw);
        
            // accept new state?
            if (prob <= bar || isnan(prob)) {
              acceptanceN[w][nw] = 0;
            } else {
              xisq0N[w][nw] = xisq;
            } 
      
          }
          free(evector);
  
          // switch back to old ones if not accepted
          if (acceptanceN[w][nw] == 0) {
            memcpy(&p0localN[w][nw*pperwalker], p0localcopy, pperwalker*sofd);
          }
        }
        for (w=0; w<NTEMPS-1; w++) {
          // prob that we should swap temperature chains
          double pswap;
          double barswap;
          if (jj % 10 == 0) {
            pswap=0.0;
            pswap = exp( (beta[w]-beta[w+1]) * (xisq0N[w][nw] - xisq0N[w+1][nw]) ); 
            barswap = gsl_rng_uniform(rnw);
            if (pswap >= barswap) {
              memcpy(p0localcopy, &p0localN[w][nw*pperwalker], pperwalker*sofd);
              memcpy(&p0localN[w][nw*pperwalker], &p0localN[w+1][nw*pperwalker], pperwalker*sofd);
              memcpy(&p0localN[w+1][nw*pperwalker], p0localcopy, pperwalker*sofd);
              sout = fopen("demcmc.stdout", "a");
              fprintf(sout, "Tswap chain %i between %i and %i \n", nw, w, w+1);
              fclose(sout);
            }
          }
        }
      }
    ///must free all thes *N[w] s that I have alloced
    }

    if (RANK==0 && jj % 10 ==0) {
      //time testing
      clock_gettime(CLOCK_MONOTONIC, &finish);

      elapsed = (finish.tv_sec - start.tv_sec);
      elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
      sout = fopen("demcmc.stdout", "a");
      fprintf(sout, "Gen %li  run time = %3.12lf secs\n", jj, elapsed);
      fclose(sout);
    }

    if (! NTEMPS) {
  
      MPI_Allgather(&p0local[nwinit*pperwalker], pperwalker*npercore, MPI_DOUBLE, p0global, pperwalker*npercore, MPI_DOUBLE, MPI_COMM_WORLD);
      MPI_Gather(&acceptance[nwinit], 1*npercore, MPI_INT, acceptanceglobal, 1*npercore, MPI_INT, 0, MPI_COMM_WORLD); 
      MPI_Gather(&xisq0[nwinit], 1*npercore, MPI_DOUBLE, xisq0global, 1*npercore, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  
      memcpy(p0local, p0global, totalparams*sofd);
  
      int nwdex;
      if (RANK == 0) {
        memcpy(acceptance, acceptanceglobal, nwalkers*sofi);
        memcpy(xisq0, xisq0global, nwalkers*sofd);
  
        long naccept = 0;
        for (nwdex=0; nwdex<nwalkers; nwdex++) naccept += acceptance[nwdex];
  
        double fracaccept = 1.0*naccept/nwalkers;
        if (fracaccept >= optimal) {
          gamma *= (1+relax);
        } else {
          gamma *= (1-relax);
        }
  
        char gammafstr[80];
        strcpy(gammafstr, "gamma_");
        strcat(gammafstr, OUTSTR);
        strcat(gammafstr, ".txt");
        if (jj % 10 == 0) {
          FILE *gammaf = fopen(gammafstr, "a");
          fprintf(gammaf, "%li\t%lf\t%lf\n", jj, fracaccept, gamma);
          fclose(gammaf);
        }
  
        // print out occasionally
        if (jj % 100 == 0) { 
          char outfilestr[80];
          strcpy(outfilestr, "demcmc_");
          strcat(outfilestr, OUTSTR);
          strcat(outfilestr, ".out");
          outfile = fopen(outfilestr, "a");
          
          int nwdex;
          for (nwdex=0; nwdex<nwalkers; nwdex++) {
            char ch = 'a';
            double pnum = 0.1;
            for (i=0; i<npl; i++) {
              fprintf(outfile, "%1.1lf", pnum);
              ch++; 
              pnum+=0.1;
              int ii;
              for (ii=0; ii<pperplan; ii++) {
                fprintf(outfile, "\t%.15lf", p0local[nwdex*pperwalker+i*pperplan+ii]);  
              }
              fprintf(outfile, "\n");
            }
            fprintf(outfile, "%.15lf ; Mstar (R_sol)\n", p0local[nwdex*pperwalker+npl*pperplan+0]);
            fprintf(outfile, "%.15lf ; Rstar (R_sol)\n", p0local[nwdex*pperwalker+npl*pperplan+1]);
            fprintf(outfile, "%.15lf ; c1 (linear limb darkening) \n", p0local[nwdex*pperwalker+npl*pperplan+2]);
            fprintf(outfile, "%.15lf ; c2 (quadratic limb darkening) \n", p0local[nwdex*pperwalker+npl*pperplan+3]);
            fprintf(outfile, "%.15lf ; dilution (frac light not from stars in system)\n", p0local[nwdex*pperwalker+npl*pperplan+4]);
            if (RVJITTERFLAG) {
              for (i=0; i<RVJITTERTOT; i++) {
                fprintf(outfile, "%.15lf ; rvjitter \n", p0local[nwdex*pperwalker+npl*pperplan+5+i]);
              }
            }
            if (TTVJITTERFLAG) {
              for (i=0; i<TTVJITTERTOT; i++) {
                fprintf(outfile, "%.15lf ; rvjitter \n", p0local[nwdex*pperwalker+npl*pperplan+5+RVJITTERTOT+i]);
              }
            }
            if (CELERITE) {
              for (i=0; i<NCELERITE; i++) {
                fprintf(outfile, "%.15lf ; celerite \n", p0local[nwdex*pperwalker+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+i]);
              }
            }
            if (RVCELERITE) {
              for (i=0; i<NRVCELERITE; i++) {
                fprintf(outfile, "%.15lf ; rvcelerite \n", p0local[nwdex*pperwalker+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+i]);
              }
            }
            fprintf(outfile, "; chisq = %18.11lf, %i, %li\n", xisq0[nwdex], nwdex, jj);  
          }
          fclose(outfile);
        }
  
        FILE *outfile2;
        for (nwdex=0; nwdex<nwalkers; nwdex++) {
          if (xisq0[nwdex] < xisqmin) {
            xisqmin = xisq0[nwdex];
            sout = fopen("demcmc.stdout", "a");
            fprintf(sout, "Chain %lu has Best xisq so far: %lf\n", nwdex, xisq0[nwdex]);
            fclose(sout);
            char outfile2str[80];
            strcpy(outfile2str, "mcmc_bestchisq_");
            strcat(outfile2str, OUTSTR);
            strcat(outfile2str, ".aei");
            outfile2 = fopen(outfile2str, "a");
            fprintf(outfile2, "planet         period (d)               T0 (d)                  e                   i (deg)                 Omega (deg)               omega(deg)               mp (mjup)              rpors           ");
            if (MULTISTAR) {
              fprintf(outfile2, "brightness               c1                    c2\n");
            } else {
              fprintf(outfile2, "\n");
            }
            char ch = 'a';
            double pnum = 0.1;
            int ip;
            for (ip=0; ip<npl; ip++) {
              //fprintf(outfile2, "%c", ch);
              fprintf(outfile2, "%1.1lf", pnum);
              ch++; 
              pnum+=0.1;
              int ii;
              for (ii=0; ii<2; ii++) {
                fprintf(outfile2, "\t%.15lf", p0local[nwdex*pperwalker+ip*pperplan+ii]); 
              }
              if (SQRTE) {
                fprintf(outfile2, "\t%18.15lf", pow( sqrt( pow(p0local[nwdex*pperwalker+ip*pperplan+2],2) + pow(p0local[nwdex*pperwalker+ip*pperplan+3],2) ), 2) );
              } else {
                fprintf(outfile2, "\t%18.15lf", sqrt( pow(p0local[nwdex*pperwalker+ip*pperplan+2],2) + pow(p0local[nwdex*pperwalker+ip*pperplan+3],2) ) );
              }
              for (ii=4; ii<6; ii++) {
                fprintf(outfile2, "\t%.15lf", p0local[nwdex*pperwalker+ip*pperplan+ii]); 
              }
              fprintf(outfile2, "\t%18.15lf", atan2( p0local[nwdex*pperwalker+ip*pperplan+3] , p0local[nwdex*pperwalker+ip*pperplan+2] ) * 180./M_PI);
              for (ii=6; ii<8; ii++) {
                fprintf(outfile2, "\t%.15lf", p0local[nwdex*pperwalker+ip*pperplan+ii]); 
              }
              if (MULTISTAR) {
                for (ii=8; ii<11; ii++) {
                  fprintf(outfile2, "\t%.15lf", p0local[nwdex*pperwalker+ip*pperplan+ii]);
                }
              }
              fprintf(outfile2, "\n");
            }
            fprintf(outfile2, "%.15lf ; Mstar (M_sol)\n", p0local[nwdex*pperwalker+npl*pperplan+0]);
            fprintf(outfile2, "%.15lf ; Rstar (R_sol)\n", p0local[nwdex*pperwalker+npl*pperplan+1]);
            fprintf(outfile2, "%.15lf ; c1 (linear limb darkening) \n", p0local[nwdex*pperwalker+npl*pperplan+2]);
            fprintf(outfile2, "%.15lf ; c2 (quadratic limb darkening) \n", p0local[nwdex*pperwalker+npl*pperplan+3]);
            fprintf(outfile2, "%.15lf ; dilution (frac light not from stars in system)\n", p0local[nwdex*pperwalker+npl*pperplan+4]);
            if (RVJITTERFLAG) {
              for (i=0; i<RVJITTERTOT; i++) {
                fprintf(outfile2, "%.15lf ; rvjitter \n", p0local[nwdex*pperwalker+npl*pperplan+5+i]);
              }
            }
            if (TTVJITTERFLAG) {
              for (i=0; i<TTVJITTERTOT; i++) {
                fprintf(outfile2, "%.15lf ; rvjitter \n", p0local[nwdex*pperwalker+npl*pperplan+5+RVJITTERTOT+i]);
              }
            }
            if (CELERITE) {
              for (i=0; i<NCELERITE; i++) {
                fprintf(outfile2, "%.15lf ; celerite \n", p0local[nwdex*pperwalker+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+i]);
              }
            }
            if (RVCELERITE) {
              for (i=0; i<NRVCELERITE; i++) {
                fprintf(outfile2, "%.15lf ; rvcelerite \n", p0local[nwdex*pperwalker+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+i]);
              }
            }
            fprintf(outfile2, " ; Comments: These coordinates are jacobian (see Lee & Peale 2003).   gen = %li   ch=%li\n", jj, nwdex);
            fprintf(outfile2, " ; chisq = %.15lf\n", xisq0[nwdex]);
            fclose(outfile2);
          }
        }
  
  
      }
  
      MPI_Bcast(&gamma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
      if ((RANK==0) && (jj % 10 == 0 )) {
        //time testing
        clock_gettime(CLOCK_MONOTONIC, &finish);
  
        elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        sout = fopen("demcmc.stdout", "a");
        fprintf(sout, "Gen %li  run time + message passing = %3.12lf secs\n", jj, elapsed);
        fclose(sout);
      }
  
    } else {

      // correct how to copy these variables
      for (w=0; w<NTEMPS; w++) {

        MPI_Allgather(&p0localN[w][nw*pperwalker], pperwalker, MPI_DOUBLE, p0globalN[w], pperwalker, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Gather(&acceptanceN[w][nw], 1, MPI_INT, acceptanceglobalN[w], 1, MPI_INT, 0, MPI_COMM_WORLD); 
        MPI_Gather(&xisq0N[w][nw], 1, MPI_DOUBLE, xisq0globalN[w], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
        memcpy(p0localN[w], p0globalN[w], totalparams*sofd);
    
        int nwdex;
        if (RANK == 0) {
          memcpy(acceptanceN[w], acceptanceglobalN[w], nwalkers*sofi);
          memcpy(xisq0N[w], xisq0globalN[w], nwalkers*sofd);
    
          long naccept = 0;
          for (nwdex=0; nwdex<nwalkers; nwdex++) naccept += acceptanceN[w][nwdex];
    
          double fracaccept = 1.0*naccept/nwalkers;
          if (fracaccept >= optimal) {
            gammaN[w] *= (1+relax);
          } else {
            gammaN[w] *= (1-relax);
          }
    
          char gammafstr[80];
          strcpy(gammafstr, "gamma_");
          strcat(gammafstr, OUTSTR);
          strcat(gammafstr, ".txt");
          sprintf(gammafstr, "%s_%i", gammafstr, w);
          if (jj % 10 == 0) {
            FILE *gammaf = fopen(gammafstr, "a");
            fprintf(gammaf, "%li\t%lf\t%lf\n", jj, fracaccept, gammaN[w]);
            fclose(gammaf);
          }
    
          // print out occasionally
          if (jj % 100 == 0) { 
            char outfilestr[80];
            strcpy(outfilestr, "demcmc_");
            strcat(outfilestr, OUTSTR);
            strcat(outfilestr, ".out");
            sprintf(outfilestr, "%s_%i", outfilestr, w);
            outfile = fopen(outfilestr, "a");
            
            int nwdex;
            for (nwdex=0; nwdex<nwalkers; nwdex++) {
              //printf("demcmc herey\n");
              char ch = 'a';
              double pnum = 0.1;
              for (i=0; i<npl; i++) {
                //fprintf(outfile, "%c", ch);
                fprintf(outfile, "%1.1lf", pnum);
                ch++; 
                pnum+=0.1;
                int ii;
                for (ii=0; ii<pperplan; ii++) {
                  fprintf(outfile, "\t%.15lf", p0localN[w][nwdex*pperwalker+i*pperplan+ii]);  
                }
                fprintf(outfile, "\n");
              }
              fprintf(outfile, "%.15lf ; Mstar (R_sol)\n", p0localN[w][nwdex*pperwalker+npl*pperplan+0]);
              fprintf(outfile, "%.15lf ; Rstar (R_sol)\n", p0localN[w][nwdex*pperwalker+npl*pperplan+1]);
              fprintf(outfile, "%.15lf ; c1 (linear limb darkening) \n", p0localN[w][nwdex*pperwalker+npl*pperplan+2]);
              fprintf(outfile, "%.15lf ; c2 (quadratic limb darkening) \n", p0localN[w][nwdex*pperwalker+npl*pperplan+3]);
              fprintf(outfile, "%.15lf ; dilution (frac light not from stars in system)\n", p0localN[w][nwdex*pperwalker+npl*pperplan+4]);
              fprintf(outfile, "; chisq = %18.11lf, %i, %li\n", xisq0N[w][nwdex], nwdex, jj);  
            }
            fclose(outfile);
          }
    
          FILE *outfile2;
          for (nwdex=0; nwdex<nwalkers; nwdex++) {
            if (w==0 && xisq0N[w][nwdex] < xisqmin) {
              xisqmin = xisq0N[w][nwdex];
              sout = fopen("demcmc.stdout", "a");
              fprintf(sout, "Chain %lu has Best xisq so far: %lf\n", nwdex, xisq0N[w][nwdex]);
              fclose(sout);
              char outfile2str[80];
              strcpy(outfile2str, "mcmc_bestchisq_");
              strcat(outfile2str, OUTSTR);
              strcat(outfile2str, ".aei");
              outfile2 = fopen(outfile2str, "a");
              fprintf(outfile2, "planet         period (d)               T0 (d)                  e                   i (deg)                 Omega (deg)               omega(deg)               mp (mjup)              rpors           ");
              if (MULTISTAR) {
                fprintf(outfile2, "brightness               c1                    c2\n");
              } else {
                fprintf(outfile2, "\n");
              }
              char ch = 'a';
              double pnum = 0.1;
              int ip;
              for (ip=0; ip<npl; ip++) {
                fprintf(outfile2, "%1.1lf", pnum);
                ch++; 
                pnum+=0.1;
                int ii;
                for (ii=0; ii<2; ii++) {
                  fprintf(outfile2, "\t%.15lf", p0localN[w][nwdex*pperwalker+ip*pperplan+ii]); 
                }
                if (SQRTE) {
                  fprintf(outfile2, "\t%18.15lf", pow( sqrt( pow(p0localN[w][nwdex*pperwalker+ip*pperplan+2],2) + pow(p0localN[w][nwdex*pperwalker+ip*pperplan+3],2) ), 2) );
                } else {
                  fprintf(outfile2, "\t%18.15lf", sqrt( pow(p0localN[w][nwdex*pperwalker+ip*pperplan+2],2) + pow(p0localN[w][nwdex*pperwalker+ip*pperplan+3],2) ) );
                }
                for (ii=4; ii<6; ii++) {
                  fprintf(outfile2, "\t%.15lf", p0localN[w][nwdex*pperwalker+ip*pperplan+ii]); 
                }
                fprintf(outfile2, "\t%18.15lf", atan2( p0localN[w][nwdex*pperwalker+ip*pperplan+3] , p0localN[w][nwdex*pperwalker+ip*pperplan+2] ) * 180./M_PI);
                for (ii=6; ii<8; ii++) {
                  fprintf(outfile2, "\t%.15lf", p0localN[w][nwdex*pperwalker+ip*pperplan+ii]); 
                }
                if (MULTISTAR) {
                  for (ii=8; ii<11; ii++) {
                    fprintf(outfile2, "\t%.15lf", p0localN[w][nwdex*pperwalker+ip*pperplan+ii]);
                  }
                }
                fprintf(outfile2, "\n");
              }
              fprintf(outfile2, "%.15lf ; Mstar (M_sol)\n", p0localN[w][nwdex*pperwalker+npl*pperplan+0]);
              fprintf(outfile2, "%.15lf ; Rstar (R_sol)\n", p0localN[w][nwdex*pperwalker+npl*pperplan+1]);
              fprintf(outfile2, "%.15lf ; c1 (linear limb darkening) \n", p0localN[w][nwdex*pperwalker+npl*pperplan+2]);
              fprintf(outfile2, "%.15lf ; c2 (quadratic limb darkening) \n", p0localN[w][nwdex*pperwalker+npl*pperplan+3]);
              fprintf(outfile2, "%.15lf ; dilution (frac light not from stars in system)\n", p0localN[w][nwdex*pperwalker+npl*pperplan+4]);
            if (RVJITTERFLAG) {
              for (i=0; i<RVJITTERTOT; i++) {
                fprintf(outfile2, "%.15lf ; rvjitter \n", p0localN[w][nwdex*pperwalker+npl*pperplan+5+i]);
              }
            }
            if (TTVJITTERFLAG) {
              for (i=0; i<TTVJITTERTOT; i++) {
                fprintf(outfile, "%.15lf ; rvjitter \n", p0localN[w][nwdex*pperwalker+npl*pperplan+5+RVJITTERTOT+i]);
              }
            }
            if (CELERITE) {
              for (i=0; i<NCELERITE; i++) {
                fprintf(outfile, "%.15lf ; celerite \n", p0localN[w][nwdex*pperwalker+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+i]);
              }
            }
            if (RVCELERITE) {
              for (i=0; i<NRVCELERITE; i++) {
                fprintf(outfile, "%.15lf ; rvcelerite \n", p0localN[w][nwdex*pperwalker+npl*pperplan+5+RVJITTERTOT+TTVJITTERTOT+NCELERITE*CELERITE+i]);
              }
            }
              fprintf(outfile2, " ; Comments: These coordinates are jacobian (see Lee & Peale 2003).   gen = %li \n", jj);
              fprintf(outfile2, " ; chisq = %.15lf\n", xisq0N[w][nwdex]);
              fclose(outfile2);
            }
          }
        }
    
        MPI_Bcast(&gammaN[w], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    
        if ((RANK==0) && (jj % 10 == 0 )) {
          //time testing
          clock_gettime(CLOCK_MONOTONIC, &finish);
    
          elapsed = (finish.tv_sec - start.tv_sec);
          elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
          sout = fopen("demcmc.stdout", "a");
          fprintf(sout, "Gen %li  run time + message passing = %3.12lf secs\n", jj, elapsed);
          fclose(sout);
        }
      }
    }

    jj+=1;

  }


  free(acceptance);
  free(acceptanceglobal);
  free(p0local);
  free(p0global);
  free(p0localcopy);
  free(xisq0global);


  for(i=0; i<nwalkers; i++) {
    int i1;
    for (i1=0; i1<npl; i1++) {
      free(p0[i][i1]);
    }
    free(p0[i][npl]);
    free(p0[i]);
  }
  free(p0);

  free(xisq0);

#elif (demcmc_compile == 3)

  int i;
  double ***int_in = dsetup(p, npl);
  double ***flux_rvs; 
  if (MULTISTAR) flux_rvs = dpintegrator_multi(int_in, tfe, tve, cadencelist);
  else flux_rvs = dpintegrator_single(int_in, tfe, tve, nte, cadencelist);
  free(int_in[0][0]);
  free(int_in[0]);
  free(int_in[1][0]);
  free(int_in[1]);
  int i1;
  for(i1=0; i1<npl+1; i1++) free(int_in[2][i1]);
  free(int_in[2]);
  free(int_in[3][0]);
  free(int_in[3]);
  free(int_in);

#endif
  if (TTVCHISQ) {

#if (demcmc_compile==0) 

    for (i=0; i<npl; i++) {
      char tempchar[1000];
      sprintf(tempchar, "ttvmcmc00_%02i.out", i+1);
      FILE* ttvouti = fopen(tempchar, "w");
      int ii;
      for (ii=0; ii<NTTV[i][0]; ii++) {
        fprintf(ttvouti, "%8li \t %.15lf \t %.15lf \t %.15lf \n", NTTV[i][ii+1], TTTV[i][ii+1], MTTV[i][ii+1], ETTV[i][ii+1]);
      }
      fclose(ttvouti);
    }

    for (i=0; i<npl; i++) free(MTTV[i]);
    free(MTTV);
#endif

    for (i=0; i<npl; i++) free(NTTV[i]);
    free(NTTV);
    for (i=0; i<npl; i++) free(TTTV[i]);
    free(TTTV);
    for (i=0; i<npl; i++) free(ETTV[i]);
    free(ETTV);
  }

  if (!RESTART) {
    for (i=0; i<nbodies; i++) free(p[i]);
  }
  free(p);

  for (i=0; i<3; i++) free(tfe[i]);
  free(tfe);
  if (RVS) {
    for (i=0; i<5; i++) free(tve[i]);
  }
  if (TTVCHISQ) {
    int kij;
    for (kij=0; kij<3; kij++) {
      free(nte[kij]);
    }
    free(nte);    
  }
  free(tve);
  if (CADENCESWITCH==2) {
    free(cadencelist); 
  }

  gsl_rng_free(r);
  gsl_rng_free(rnw);

  return 0;
}





int main (int argc, char *argv[]) {

#if (demcmc_compile==1)
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
  MPI_Comm_rank(MPI_COMM_WORLD, &SIZE);
#endif

  struct timespec start, finish;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &start);

  getinput(argv[1]);
  
  void * nullptr;

  int sofi = SOFI;
  RVARR = calloc(NBODIES, sofi);
  RVFARR = malloc(NBODIES*sizeof(char*));
  int i;
  for (i=0; i<NBODIES; i++) {
    RVFARR[i] = malloc(100*sizeof(char));
  }
  int rvscount=0;
  for (i=1; i<argc; i++) {
    if (argv[i][0] == '-' && argv[i][1]=='r' && argv[i][2]=='v') {
      int body = argv[i][3]-'0';
      RVARR[body] = 1;
      strcpy(RVFARR[body], &argv[i][5]); 
      RVS = 1;
      rvscount++;
    }
  }
   
  argc-=rvscount;

  if (argc == 3) {
    RESTART = 0;
    if (RVS) {
      demcmc(argv[2], nullptr, nullptr, nullptr); 
    } else {
      demcmc(argv[2], nullptr, nullptr, nullptr); 
    }
  } else if (argc == 6) {
    RESTART = 1;
    if (RVS) {
      demcmc(argv[2], argv[3+rvscount], argv[4+rvscount], argv[5+rvscount]);
    } else {
      demcmc(argv[2], argv[3], argv[4], argv[5]);
    }
  } else {
    printf("usage: $ ./demcmc demcmc.in kep11.aei [[-rv0=rvs0.file] [-rv1=rvs1.file] ... ] [demcmc.res mcmcbsq.res gamma.res]\n");
    exit(0);
  }

  
  for(i=0; i<NBODIES; i++) {
    free(RVFARR[i]);
  }
  free(RVFARR);
  free(PARFIX); free(PSTEP); free(SSTEP); free(STEP); free(BIMODLIST); free(OUTSTR); free(RVARR);
  free(MAXDENSITY); free(EMAX); free(EPRIORV);

  free(XYZLIST);

  clock_gettime(CLOCK_MONOTONIC, &finish);

  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
  printf("Total run time = %3.12lf secs\n", elapsed);

#if (demcmc_compile==1)
  MPI_Finalize();
#endif

  return 0;

}



