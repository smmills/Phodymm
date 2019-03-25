./lightcurve_runscript_3pl_rvs.sh &
python lcplot.py
python omc.py
rm tbv00_03.out
rm tbv00_02.out
python phasefold.py
