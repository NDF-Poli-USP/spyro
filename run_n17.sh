#!/bin/bash

module purge

source /home/public/app/firedrake_arm/firedrake/bin/activate
module unload python-3.8.8-gcc-9.2.0-ohzcine
module unload openmpi3/3.1.4

export LD_LIBRARY_PATH=$LD_LIBRAY_PATH:/home/public/app/firedrake_arm/segyio/lib64/

export FIREDRAKE_CACHE_DIR=~/tmp10
export PYOP2_CACHE_DIR=~/tmp10
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=~/tmp10
export EXEC_SPYRO=forward_mm.py

python $EXEC_SPYRO