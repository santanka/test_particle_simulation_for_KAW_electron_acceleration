#!/bin/bash
#PJM -L "rscunit=ito-a"
#PJM -L "rscgrp=hp-s-dbg"
#PJM -L "vnode=4"
#PJM -L "vnode-core=36"
#PJM --no-stging
#PJM -j
#PJM -X

source /home/etc/intel.sh
module load oneapi/2022.3.1

NUM_NODES=$PJM_VNODES
NUM_CORES=36
NUM_PROCS=1
NUM_THREADS=36

export I_MPI_PERHOST=`expr $NUM_CORES / $NUM_THREADS`
export I_MPI_FABRICS=shm:ofi
export I_MPI_PIN_DOMAIN=omp
export I_MPI_PIN_CELL=core

export OMP_NUM_THREADS=$NUM_THREADS
export KMP_STACKSIZE=8m
export KMP_AFFINITY=compact

export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=/bin/pjrsh
export I_MPI_HYDRA_HOST_FILE=${PJM_O_NODEINF}

date
mpiexec.hydra -n $NUM_PROCS ./a.out
date
