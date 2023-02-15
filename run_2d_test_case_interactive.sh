#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --partition=intel_interactive
#SBATCH --time=1:00:00
#SBATCH --job-name=2d_test_case/shdebug_interactive
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

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
module load firedrake/20220516


export FIREDRAKE_CACHE_DIR=~/tmp_amd7
export PYOP2_CACHE_DIR=~/tmp_amd7
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=~/tmp_amd7

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun hostname > $HOSTFILE
## Information about the entry and exit of the job
echo -e "\n## Diretorio de submissao do job:   $SLURM_SUBMIT_DIR \n"

mpiexec -n 5 python forward_2d_interactive.py

echo -e "\n## Job finished on $(date +'%d-%m-%Y as %T') ###################"
rm $HOSTFILE
