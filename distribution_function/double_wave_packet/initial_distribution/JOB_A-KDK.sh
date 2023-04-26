#!/bin/bash
#============ QSUB Options ============
#SBATCH -p gr20001b
#SBATCH -t 10:00:00
#SBATCH --rsc p=72:t=2:c=1
#--- p : # of processes
#--- t : # of threads
#--- c : # of cores (= t)
#SBATCH -o %x.%j.out
#============ Shell Script ============

set -x

# automatically
# export OMP_NUM_THREADS=$SLURM_DPC_THREADS

date
srun ./a.out
date