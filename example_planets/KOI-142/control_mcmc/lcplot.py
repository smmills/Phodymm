### Edit this range to plot a different segment of data
trange = [1312,1335] # for publication
# trange = [160,190] # to inspect feature at 200 days
#

import numpy as np
import matplotlib.pyplot as plt
import glob

pdict = {0:'b',1:'c',2:'d'}
colorlist = ['b', 'r', 'g', 'y', 'c', 'm', 'midnightblue', 'yellow'] 


tbvfilelist = sorted(glob.glob("./tbv[0-9][0-9]_[0-9][0-9].out"))
nfiles = len(tbvfilelist)
npl = nfiles
transittimes = [[] for i in range(npl)]
nums = [[] for i in range(npl)]
for i in range(nfiles):
  data = np.loadtxt(tbvfilelist[i])
  tt = data[:,1]
  transittimes[i] = tt 
  nn = data[:,0]
  nums[i] = nn    
phasewidth = [0.4 for i in range(nfiles)]
for i in range(nfiles):
  if len(transittimes[i]) > 1:
    meanper, const = np.linalg.lstsq(np.vstack([nums[i], np.ones(len(nums[i]))]).T, transittimes[i], rcond=None)[0] 
    print("Deatils for planet %s:\n\tNumber of transits=%i\n\tMean period=%f days"%(pdict.get(i),len(nums[i]),meanper))
    # 3x the duration of an edge-on planet around the sun
    phasewidth[i] = min(3.*(13./24.) * ((meanper/365.25)**(1./3.)) ,1) # not too wide
collisionwidth = [pwi for pwi in phasewidth] 
#collisionwidth = 0.15*np.ones(nfiles)
  


def plotlc(time, meas, the, err, trange, outputname=None, autoadjust=True):

  f = plt.figure()
  f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
  a0.scatter(time, meas, s=0.1, c='k',rasterized=True) 
  a0.plot(time, the, marker="None", c=colorlist[0],rasterized=True)
  
  a1.scatter(time, meas-the, s=0.1, c='k')
  #a1.errorbar(time, meas, yerr=err, marker="None", linestyle="None", c='k')
  
  inrange = np.where((time > trange[0]) & (time < trange[1]))
  if autoadjust and (np.any(inrange) == False):
    while (np.any(inrange) == False):
      trange = [tr+50. for tr in trange]
      inrange = np.where((time > trange[0]) & (time < trange[1]))
    print("Warning: trange changed to [%7.2f, %7.2f]\n" % (trange[0], trange[1]))

  maxf0 = np.max(meas[inrange])
  minf0 = np.min(meas[inrange])
  yrange0 = [minf0, maxf0]
  maxf1 = np.max(meas[inrange]-the[inrange])
  minf1 = np.min(meas[inrange]-the[inrange])
  yrange1 = [1.1*minf1, 1.1*maxf1]
  
  a0.set_xlim((trange[0], trange[1]))
  a0.set_ylim((yrange0[0], yrange0[1]))
  a1.set_xlim((trange[0], trange[1]))
  a1.set_ylim((yrange1[0], yrange1[1]))
  a0.set_ylabel('Normalized Flux')
  a1.set_ylabel('Residual')
  
  ylim = a0.get_ylim()
  yspan = ylim[1]-ylim[0]
  
  i=0 
  for fname in sorted(glob.glob("./tbv[0-9][0-9]_[0-9][0-9].out")):
    i+=1
    data = np.loadtxt(fname)
    tt = data[:,1]
    for tti in tt:
      a0.plot([tti, tti], [ylim-0.05*yspan, ylim], c=colorlist[i], marker="None",rasterized=True) 
  
  plt.xlabel('Time (days)')
  plt.tight_layout()
  savestr = 'lcplot'
  if outputname is not None:
    savestr += '_'
    savestr += outputname
    savestr += '.png'
  f.savefig(savestr, dpi=180, format='png')

####

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
####
#plotlc(time, meas, the, err, [50., 1570.], outputname='AllData')
for planet_num, planet in enumerate(transittimes):
  print("Planet:", planet)
  if planet_num >= 2:
    for transit_time in planet:
      print("transit_time:", transit_time)
      if (transit_time < max(time)) and (transit_time > min(time)):
        trange = [transit_time - collisionwidth[planet_num], transit_time + collisionwidth[planet_num]]
        if planet_num==2:
          trange=[1335-48,1335+48]
        inrange = np.where((time > trange[0]) & (time < trange[1]))[0]
        if len(inrange) > 0:
          plotlc(time[inrange], meas[inrange], the[inrange], err[inrange], trange, outputname='zoom'+str(planet_num)+'_'+"{:04d}".format(int(transit_time)), autoadjust=False)




