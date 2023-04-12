#!/bin/bash
#PJM -L "rscunit=ito-a"
#PJM -L "rscgrp=ito-ss"
#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -j
#PJM -X

module load oneapi/2022.3.1
export OMP_NUM_THREADS=36
./a.out