import sys
import os
import pandas as pd
import numpy as np
from numpy import sin, cos, pi, arccos, arctan2
from numpy.random import randint
import corner
import matplotlib.pyplot as plt
plt.switch_backend('agg')

if len(sys.argv) == 1:
  print("ERROR: You must pass this script a .in file")
  print("e.g.")
  print("$ python demcmc_quick_analyze.py kepler36_longcadence.in")
  exit()

fname = sys.argv[1]

burnin = 0
if len(sys.argv) >= 3:
  burnin = int(sys.argv[2])
burnin //= 100

n_sample = 10
if len(sys.argv) == 4:
  n_sample = int(sys.argv[3])


f=open(fname)
lines=f.readlines()
f.close()

def get_in_value(linenumber):
  line = lines[linenumber]
  line = line.split()
  return line[2]

runname = get_in_value(8)
#runname = runname[:-1]
demcmcfile='demcmc_'+runname+'.out'
nchain = int(get_in_value(14))
npl = int(get_in_value(10)) - 1
parlist = lines[91+npl]
npar = len(parlist.split())
ndead=1

# lines per output in demcmc_ file
nper=npl+npar+ndead
# string to save files with
#savestr = demcmcfile.partition('demcmc_')[2]
#savestr = savestr.partition('.out')[0]
savestr = runname

print(nper, npl, npar, ndead)

f2 = open(demcmcfile)
mcmc_lines = f2.readlines()
f2.close()

n_links = len(mcmc_lines)//nper
draws = randint(low=0,high=n_links, size=n_sample) * 11

def get_draw(draw):
  lines = mcmc_lines[draw:draw+nper]
  return lines
  
tags = [str(int(x/11)) for x in draws]
  
def write_to_text(lines, tag):
    filename = runname+tag+".pldin"
    f = open("sample_pldin/"+filename,"w")
    f.writelines("%s" % line for line in lines)
    f.close()
    f2 = open("sample_pldin/lightcurve_runscript_3pl_rvs.sh","a")
    f2.write(" ./lcout koi142_3pl_rvs.in %s -rv0=koi142_rvs.txt \n" % filename)
    f2.write(" mv tbv00_01.out tbv00_01_%s.out \n" % tag)
    f2.write(" mv tbv00_02.out tbv00_02_%s.out \n" % tag)
    f2.write(" mv tbv00_03.out tbv00_03_%s.out \n" % tag)
    f2.close()
    return
    
def se_to_ecc(line):
    """
    Convert from Phodymm demcmc coordinates to pldin coordinates
    input: line from demcmc.  should have units:
        pparamlist=["Planet", "Period", "T$_0,$", r"$\sqrt{e}\cos \omega$", r"$\sqrt{e}\sin \omega$", "i", r"$\Omega$", "M$_{jup,}$", "R$_p$/R$_s$"]
    output: line with values of [Planet P T_0 e i Om om Mp RpRs]
    """
    print("%i pars" %len(line))
#    assert len(pars==npar+1)
    pars = line.split('\t')
    pl, P, T0, secw, sesw, i, Om, Mp, RpRs = pars
    e = str(float(secw)**2. + float(sesw)**2.)
    w = str(arctan2(float(sesw),float(secw))* 180/pi) 
    new_line = '\t'.join([pl, P, T0, e, i, Om, w, Mp, RpRs])
    return new_line

        
    
for draw in draws:
    lines = get_draw(draw)
    for i,line in enumerate(lines[0:npl]):
        lines[i] = se_to_ecc(line)
    tag = str(int(draw/11))
    header = ["planet	period (d) 	T0 (d)	e	i (deg)	Omega (deg)	omega(deg)	mp (mjup)	rpors\n"]
    lines = header+lines
    write_to_text(lines,tag)
    
print("Sampled %i draws from the posterior" % n_sample)



