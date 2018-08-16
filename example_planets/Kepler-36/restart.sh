#!/bin/bash

## usage:$ ./restart.sh demcmc.sbatch kid009632895.in [rv1.txt] [rv2.txt]
##       $ sbatch demcmc.sbatch.res

NSTARPAR=5

SBATCHF=$1
INFILE=$2
RVFILE0=$3
RVFILE1=$4

OUTSTR=$(grep "string outstr=" $INFILE | awk '{print $3}')
OUTFILE="demcmc_$OUTSTR.out"
NBODIES=$(grep "int nbodies=" $INFILE | awk '{print $3}')
NPL=$(($NBODIES-1))
BSQSTR="bestchisq_$OUTSTR.pldin"
REGEN=$(tac $OUTFILE | grep ', 0,' -m 1 | tac | awk '{print $6}')
GAMMAF="gamma_$OUTSTR.txt"

RESTARTF="$OUTFILE.res"
RESTARTBSQ="$BSQSTR.res"
RESTARTGAMMA="$GAMMAF.res"
RESTARTSBATCH="$SBATCHF.res.sh"

tac $OUTFILE | grep ', 0,' -m 1 -B 9999 -C $(($NPL+$NSTARPAR)) | tac > $RESTARTF
tac mcmc_bestchisq_$OUTSTR.aei | grep 'planet' -m 1 -B 9999 | tac > $RESTARTBSQ
tac $GAMMAF | grep $REGEN -m 1 | tac > $RESTARTGAMMA

if [[ ! -s $RESTARTGAMMA ]] ; then 
echo "ERROR!!"
fi ;


####cat $SBATCHF | grep '#SBATCH --ntasks=' -B 9999 > $RESTARTSBATCH
echo "#!/bin/bash " > $RESTARTSBATCH
echo " " >> $RESTARTSBATCH
touch empty.txt

if [ -z "$RVFILE1" ]
then
  if [ -z "$RVFILE0" ] 
  then
    ##echo mpirun ./demcmc $INFILE empty.txt $RESTARTF $RESTARTBSQ $RESTARTGAMMA >> $RESTARTSBATCH
    echo mpirun -np 20 -output-filename outf/sbatch.o --tag-output demcmc $INFILE empty.txt $RESTARTF $RESTARTBSQ $RESTARTGAMMA >> $RESTARTSBATCH 
  else 
    echo mpirun -np 20 -output-filename outf/sbatch.o --tag-output demcmc $INFILE empty.txt -rv0=$RVFILE0 $RESTARTF $RESTARTBSQ $RESTARTGAMMA >> $RESTARTSBATCH 
    ##echo mpirun ./demcmc $INFILE empty.txt -rv0=$RVFILE0 $RESTARTF $RESTARTBSQ $RESTARTGAMMA >> $RESTARTSBATCH
  fi
else
  ##echo mpirun ./demcmc $INFILE empty.txt -rv0=$RVFILE0 -rv1=$RVFILE1 $RESTARTF $RESTARTBSQ $RESTARTGAMMA >> $RESTARTSBATCH
  echo mpirun -np 20 -output-filename outf/sbatch.o --tag-output demcmc $INFILE empty.txt -rv0=$RVFILE0 -rv1=$RVFILE1 $RESTARTF $RESTARTBSQ $RESTARTGAMMA >> $RESTARTSBATCH 
fi

chmod +x $RESTARTSBATCH

