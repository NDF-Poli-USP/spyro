#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=25
#SBATCH --partition=intel_large
#SBATCH --time=1-24:00:00
#SBATCH --job-name=intel_strong_scalling_presentation/test1
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

module load firedrake

srun hostname > $HOSTFILE
## Information about the entry and exit of the job
echo -e "\n## Diretorio de submissao do job:   $SLURM_SUBMIT_DIR \n"

mpiexec -n 100 python benchmark_forward_3d.py 100
mpiexec -n 80 python benchmark_forward_3d.py 100
mpiexec -n 60 python benchmark_forward_3d.py 100
mpiexec -n 40 python benchmark_forward_3d.py 100
mpiexec -n 20 python benchmark_forward_3d.py 100
mpiexec -n 10 python benchmark_forward_3d.py 100



echo -e "\n## Job finished on $(date +'%d-%m-%Y as %T') ###################"
rm $HOSTFILE

rm -rf $FIREDRAKE_CACHE_DIR
