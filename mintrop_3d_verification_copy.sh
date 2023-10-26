#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --partition=intel_large
#SBATCH --time=20:00:00
#SBATCH --job-name=results3d/verification
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --exclusive

echo -e "\n## Job started at $(date +'%d-%m-%Y as %T') #####################\n"
echo -e "\n## Jobs activated by $USER: \n"
squeue -a --user=$USER
echo -e "\n## Execution node:         $(hostname -s) \n"
echo -e "\n## Number of tasks per job: $SLURM_NTASKS \n"
#########################################
##------- Start of job     ----- #
#########################################
## Configure the execution environment


##gCreate a file to store hostname allocated
export HOSTFILE=$SLURM_SUBMIT_DIR/host-$SLURM_JOBID
module purge

module load gnu8/8.3.0
module load libtool-2.4.6-gcc-8.3.0-r7ax6ye
module load flex-2.6.4-gcc-8.3.0-fscrodi
module load zlib-1.2.11-gcc-8.3.0-dp3wfr2
module load boost-1.73.0-gcc-8.3.0-udmhvms
module load bison-3.4.2-gcc-8.3.0-axgm7v3
module load automake-1.16.2-gcc-8.3.0-mtwjidl
module load autoconf-2.69-gcc-8.3.0-itfodnu
module load cmake-3.16.2-gcc-8.3.0-kj4453d
module load libffi-3.2.1-gcc-8.3.0-5zm7fbr
module load cgal-5.0.2-gcc-8.3.0-oihcbja
module load gmp-6.1.2-gcc-8.3.0-mw35eu2
module load mpfr-4.0.2-gcc-8.3.0-qh3hdx2
module load netlib-lapack-3.8.0-gcc-8.3.0-wilxs67
module load openblas-0.3.9-gcc-8.3.0-e3xr7kn
module load git-2.25.0-intel-20.0.166-sqqkrvq

module load python-3.6.8-gcc-8.3.0-ak4nasp



. /home/public/app/firedrake_gnu_042022/firedrake/bin/activate

export FIREDRAKE_CACHE_DIR=~/tmp_amd7
export PYOP2_CACHE_DIR=~/tmp_amd7
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=~/tmp_amd7

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun hostname > $HOSTFILE
## Information about the entry and exit of the job
echo -e "\n## Diretorio de submissao do job:   $SLURM_SUBMIT_DIR \n"

mpiexec -n 20 python dof_3d_node_propagation_copy.py


echo -e "\n## Job finished on $(date +'%d-%m-%Y as %T') ###################"
rm $HOSTFILE

rm -rf $FIREDRAKE_CACHE_DIR