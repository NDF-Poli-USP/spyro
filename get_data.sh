#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=intel_interactive
#SBATCH --time=0:05:00
#SBATCH --job-name=amd_strong_scalling/just_get_data
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

srun hostname > $HOSTFILE
## Information about the entry and exit of the job
echo -e "\n## Diretorio de submissao do job:   $SLURM_SUBMIT_DIR \n"

# The folder containing the files to process
folder="amd_strong_scalling"

# Loop over every file in the folder
for file in $folder/intel_test1_overthurst.2322*.out; do

    # Check if the file is a regular file (not a directory)
    if [ -f "$file" ]; then

        # Run the Python script on the file
        python get_test_data_separated.py "$file"

    fi

done

echo -e "\n## Job finished on $(date +'%d-%m-%Y as %T') ###################"
rm $HOSTFILE

rm -rf $FIREDRAKE_CACHE_DIR
