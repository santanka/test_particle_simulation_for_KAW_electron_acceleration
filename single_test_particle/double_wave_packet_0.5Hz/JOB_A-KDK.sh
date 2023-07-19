#!/bin/bash
#============ QSUB Options ============
#SBATCH -p gr20001b
#SBATCH -t 10:00:00
#SBATCH --rsc p=1:t=36:c=36
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