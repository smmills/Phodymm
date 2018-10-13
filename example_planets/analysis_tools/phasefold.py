import numpy as np
import matplotlib.pyplot as plt
import glob

colorlist = ['b', 'r', 'g', 'y', 'c', 'm', 'midnightblue', 'yellow'] 

lcdatafile = glob.glob("./lc_*.lcout") 
if len(lcdatafile) == 0:
  print("This script must be run in the directory containing the lc_RUNNAME.lcout")
  print("    file produced with the 'lcout' command. No such file was not found here.")
  print("    Aborting")
  exit()
if len(lcdatafile) > 1:
  print("Warning: Multiple lc_RUNNAME.lcout files found in this directory")
  print("    The default behavior is to plot the first one alphabetically")
lcdata = np.loadtxt(lcdatafile[0])
time = lcdata[:,0]
meas = lcdata[:,1]
the = lcdata[:,2]
err = lcdata[:,3]

tbvfilelist = glob.glob("./tbv[0-9][0-9]_[0-9][0-9].out")
nfiles = len(tbvfilelist)
npl = nfiles

if npl == 0:
  print("Error: no tbvXX_YY.out files found in this directory")
  exit()

f, axes = plt.subplots(npl, 1, figsize=(5,3*npl))
axes = list(axes)

transtimes = [[] for i in range(npl)]
for i in range(nfiles):
  data = np.loadtxt(tbvfilelist[i])
  tt = data[:,1]
  transtimes[i] = tt 

phasewidth = 0.4
collisionwidth = phasewidth #0.15

for i in range(nfiles):
  phases = []
  fluxes = []
  othertts = transtimes[:i] + transtimes[i+1:]
  othertts = np.array(othertts).flatten()
  thistts = np.array(transtimes[i])
  for tti in thistts:
    if min(abs(othertts - tti)) > collisionwidth: 
      trange = np.where(np.abs(time - tti) < phasewidth)[0]
      phases.append(time[trange] - tti)
      fluxes.append(meas[trange])
  phases = np.hstack(phases)
  fluxes = np.hstack(fluxes)
  axes[i].scatter(phases, fluxes, s=0.01, c='gray', alpha=0.5)
 
  binwidth = 1./1440. * 10.
  nbins = int(2*phasewidth / binwidth)
  binedges = np.arange(nbins+1, dtype=np.float)*binwidth - phasewidth 
  bincenters = np.arange(nbins, dtype=np.float)*binwidth - phasewidth + binwidth/2. 

  phasesort = np.argsort(phases)
  phases = phases[phasesort]
  fluxes = fluxes[phasesort]

  j=0
  k=0
  mbinned = np.zeros(nbins)
  while j < len(phases):
    mbinvals = []
    while phases[j] < binedges[k+1]:
      mbinvals.append(fluxes[j])
      j += 1
      if j >= len(phases):
        break
    if len(mbinvals) > 0:
      mbinned[k] = np.mean(mbinvals)
    k += 1
    if k >= nbins:
      break

  axes[i].scatter(bincenters, mbinned, s=1.0, c='k') 
  axes[i].set_xlim((-phasewidth, phasewidth))
  axes[i].set_ylim((min(fluxes), max(fluxes)))

plt.xlabel('Phase (days)')
#plt.ylabel('Normalized Flux')
f.tight_layout()
f.savefig('PhaseFolded.png')

