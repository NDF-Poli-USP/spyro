#!/usr/bin/env bash
# Four-arm checkpointed elastic FWI comparison on Mintrop.

#SBATCH --job-name=fwi3d-elastic
#SBATCH --partition=intel_large
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --array=1-4%4
#SBATCH --output=slurm_live_logs/%x.%A_%a.out
#SBATCH --error=slurm_live_logs/%x.%A_%a.err

set -euo pipefail

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HERE"
mkdir -p slurm_live_logs

if [[ -n "${SLURM_JOB_ID:-}" && -z "${APPTAINER_CONTAINER:-}" ]]; then
    module purge
    if [[ "${SLURM_JOB_PARTITION:-}" == amd_* ]]; then
        module load apptainer/1.3.1-gcc-14.1.0-a3l6k3y
    else
        module load apptainer
    fi
    IMAGE="${APPTAINER_IMAGE:-$HOME/images/firedrake-2026.4.0.sif}"
    DEPS="${SPYRO_DEPS:-$HOME/envs/spyro-firedrake-2026.4.0/site-packages}"
    CACHE="/tmp/pyop2_fwi3d_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "$CACHE"
    trap 'rm -rf "$CACHE"' EXIT
    container_env=(
        --env "PYTHONPATH=$HERE/../../..:$DEPS"
        --env "PYOP2_CACHE_DIR=$CACHE"
        --env "APPTAINER_CONTAINER=1"
        --env "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
        --env "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
        --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
        --env "SLURM_NTASKS=${SLURM_NTASKS:-16}"
        --env "FWI3D_RESULTS_ROOT=${FWI3D_RESULTS_ROOT:-$HERE/fwi_3d_results/elastic_inclusion}"
    )
    while IFS='=' read -r name value; do
        case "$name" in
            FWI3D_*) container_env+=(--env "$name=$value") ;;
        esac
    done < <(env)
    exec apptainer exec --cleanenv --pwd "$HERE" \
        "${container_env[@]}" \
        "$IMAGE" bash "$HERE/run_elastic_fwi_3d_mintrop.slurm"
fi

case "${SLURM_ARRAY_TASK_ID:-1}" in
    1) METHOD=lmvm_h1 ;;
    2) METHOD=blmvm_h1 ;;
    3) METHOD=blmvm ;;
    4) METHOD=qn_latent ;;
    *) echo "Invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS=1
export FWI3D_METHOD="$METHOD"
export FWI3D_RUN_ID="${FWI3D_RUN_ID:-${SLURM_ARRAY_JOB_ID:-manual}_${METHOD}}"
export PYTHONPATH="$HERE/../../..:${PYTHONPATH:-}"

echo "method=$METHOD ranks=${SLURM_NTASKS:-16} snapshots=${FWI3D_CHECKPOINT_SNAPSHOTS:-32}"
mpiexec --bind-to core -n "${SLURM_NTASKS:-16}" \
    python3 "$HERE/driver_elastic_fwi_3d.py"
