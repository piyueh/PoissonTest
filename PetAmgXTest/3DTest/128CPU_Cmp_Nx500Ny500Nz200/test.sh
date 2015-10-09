#!/bin/bash

export LM_LICENSE_FILE=/opt/AmgX/amgx_trial-exp-Nov-01-2015.lic
export LD_LIBRARY_PATH=/opt/AmgX/lib:/opt/PETSc/lib:/opt/cuda65/lib64:/usr/lib
export PETAMGX_DIR=/home/pychuang/Dropbox/MyGit/PoissonTest/PetAmgXTest

mpiexec -display-map -n 1 -map-by ppr:1:node ${PETAMGX_DIR}/bin/PetAmgX -caseName test -platform CPU -Nx 10 -Ny 10 -Nz 10 -cfgFileName solversPetscOptions.info -optFileName perf.txt
