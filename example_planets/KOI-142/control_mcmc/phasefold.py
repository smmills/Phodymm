import numpy as np
import matplotlib.pyplot as plt
import glob

pdict = {0:'b',1:'c',2:'d'}
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

tbvfilelist = sorted(glob.glob("./tbv[0-9][0-9]_[0-9][0-9].out"))
nfiles = len(tbvfilelist)
npl = nfiles

if npl == 0:
  print("Error: no tbvXX_YY.out files found in this directory")
  exit()

f, axes = plt.subplots(npl, 1, figsize=(5,3*npl))
try:
  axes = list(axes)
except:
  axes = [axes]

transtimes = [[] for i in range(npl)]
nums = [[] for i in range(npl)]
for i in range(nfiles):
  data = np.loadtxt(tbvfilelist[i])
  tt = data[:,1]
  transtimes[i] = tt 
  nn = data[:,0]
  nums[i] = nn

phasewidth = [0.4 for i in range(nfiles)]
for i in range(nfiles):
  if len(transtimes[i]) > 1:
    meanper, const = np.linalg.lstsq(np.vstack([nums[i], np.ones(len(nums[i]))]).T, transtimes[i], rcond=None)[0] 
    print("Deatils for planet %s:\n\tNumber of transits=%i\n\tMean period=%f days"%(pdict.get(i),len(nums[i]),meanper))
    # 3x the duration of an edge-on planet around the sun
    phasewidth[i] = min(3.*(13./24.) * ((meanper/365.25)**(1./3.)) ,1) # not too wide
collisionwidth = [pwi for pwi in phasewidth] 
#collisionwidth = 0.15*np.ones(nfiles)

for i in range(nfiles):
  phases = []
  fluxes = []
  model_fluxes = []
  othertts = transtimes[:i] + transtimes[i+1:]
  if len(othertts) > 0:
    othertts = np.hstack(np.array(othertts))
  thistts = np.array(transtimes[i])
  print(len(thistts))
  for tti in thistts:
    if len(othertts) == 0:
      trange = np.where(np.abs(time - tti) < phasewidth[i])[0]
      phases.append(time[trange] - tti)
      fluxes.append(meas[trange])
      model_fluxes.append(the[trange])
    elif min(abs(othertts - tti)) > collisionwidth[i]: 
      trange = np.where(np.abs(time - tti) < phasewidth[i])[0]
      phases.append(time[trange] - tti)
      fluxes.append(meas[trange])
      model_fluxes.append(the[trange])
  phases = np.hstack(phases)
  fluxes = np.hstack(fluxes)
  model_fluxes = np.hstack(model_fluxes)
  axes[i].scatter(phases, fluxes, s=0.01, c='gray', alpha=0.5,rasterized=True,zorder=1)
 
  binwidth = 1./1440. * 10.
  nbins = int(2*phasewidth[i] / binwidth)
  binedges = np.arange(nbins+1, dtype=np.float)*binwidth - phasewidth[i] 
  bincenters = np.arange(nbins, dtype=np.float)*binwidth - phasewidth[i] + binwidth/2. 

  phasesort = np.argsort(phases)
  phases = phases[phasesort]
  fluxes = fluxes[phasesort]
  model_fluxes = model_fluxes[phasesort]

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

  axes[i].scatter(bincenters, mbinned, s=2.0, c='k',label='binned',zorder=3) 
  axes[i].plot(phases, model_fluxes, c='blue',label='model '+str(pdict.get(i)),zorder=2,lw=1,rasterized=True)
  axes[i].set_xlim((-phasewidth[i], phasewidth[i]))
  axes[i].set_ylabel('Normalized Flux')
  try:
      axes[i].set_ylim((min(fluxes), max(fluxes)))
  except:
      pass
  axes[i].legend()

plt.xlabel('Phase (days)')
f.tight_layout()
f.savefig('PhaseFolded.png',dpi=300,format='png')
