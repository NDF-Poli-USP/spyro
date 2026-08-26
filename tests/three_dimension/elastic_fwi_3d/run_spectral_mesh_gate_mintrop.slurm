#!/usr/bin/env bash
# One-source trace comparison for the structured extruded spectral mesh.

#SBATCH --job-name=fwi3d-mesh
#SBATCH --partition=amd_large_2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_live_logs/%x.%j.out
#SBATCH --error=slurm_live_logs/%x.%j.err

set -euo pipefail

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HERE"
mkdir -p slurm_live_logs fwi_3d_results/mesh_gate

if [[ -n "${SLURM_JOB_ID:-}" && -z "${APPTAINER_CONTAINER:-}" ]]; then
    module purge
    module load apptainer/1.3.1-gcc-14.1.0-a3l6k3y
    IMAGE="${APPTAINER_IMAGE:-$HOME/images/firedrake-2026.4.0.sif}"
    DEPS="${SPYRO_DEPS:-$HOME/envs/spyro-firedrake-2026.4.0/site-packages}"
    CACHE="/tmp/pyop2_fwi3d_mesh_${SLURM_JOB_ID}"
    mkdir -p "$CACHE"
    trap 'rm -rf "$CACHE"' EXIT
    exec apptainer exec --cleanenv --pwd "$HERE" \
        --env "PYTHONPATH=$HERE/../../..:$DEPS" \
        --env "PYOP2_CACHE_DIR=$CACHE" \
        --env "APPTAINER_CONTAINER=1" \
        --env "FWI3D_MESH_GATE_OUTPUT=$HERE/fwi_3d_results/mesh_gate/${SLURM_JOB_ID}.json" \
        --env "FWI3D_MESH_GATE_EDGE=${FWI3D_MESH_GATE_EDGE:-0.05}" \
        --env "FWI3D_MESH_GATE_REFERENCE_EDGE=${FWI3D_MESH_GATE_REFERENCE_EDGE:-0.04}" \
        --env "FWI3D_MESH_GATE_RTOL=${FWI3D_MESH_GATE_RTOL:-0.05}" \
        "$IMAGE" bash "$HERE/run_spectral_mesh_gate_mintrop.slurm"
fi

export OMP_NUM_THREADS=1
export PYTHONPATH="$HERE/../../..:${PYTHONPATH:-}"
mpiexec --bind-to core -n 4 python3 "$HERE/validate_spectral_mesh_3d.py"
