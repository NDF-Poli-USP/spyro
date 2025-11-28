#!/bin/bash
#SBATCH --partition=intel_large
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=6:30:00
#SBATCH --job-name=int_queijo_minas_sem_filtro_f5
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --exclusive


echo -e "\n## Job started at $(date +'%d-%m-%Y as %T') #####################\n"
echo -e "\n## Jobs activated by $USER: \n"
squeue -a --user=$USER
echo -e "\n## Execution node:         $(hostname -s) \n"
echo -e "\n## Number of tasks per job: $SLURM_NTASKS \n"

module purge
module load apptainer

# --- Create a unique overlay for this job ---
OVERLAY_FILE="/tmp/ext3_overlay_$SLURM_JOBID.img"
apptainer overlay create --size 1024 "$OVERLAY_FILE"

# --- Run your container using this overlay ---
apptainer run --overlay "$OVERLAY_FILE" -e devimg.sif mpiexec -n $SLURM_NTASKS python3 queijo_minas_com_pao_de_queijo.py

# --- Clean up the overlay file so it won't block other jobs ---
rm -f "$OVERLAY_FILE"

echo -e "\n## Job finished on $(date +'%d-%m-%Y as %T') ###################"