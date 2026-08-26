# Elastic FWI 3D with checkpointing

This protocol is the 3D bridge for the four 2D elastic controls:

| `FWI3D_METHOD` | Coordinates | Optimizer | Explicit penalty |
|---|---|---|---|
| `lmvm_h1` | bounded sigmoid | PETSc TAO-LMVM | physical H1 |
| `blmvm_h1` | physical box | PETSc TAO-BLMVM | physical H1 |
| `blmvm` | physical box | PETSc TAO-BLMVM | none |
| `qn_latent` | bounded sigmoid | persistent external L-BFGS | none |

All arms use the automated elastic adjoint and a `MixedCheckpointSchedule`
stored in RAM. The synthetic truth is a co-located volumetric Vp/Vs anomaly;
it is a controlled 3D bridge, not yet a claim of field-scale generality.

## Local gate

```bash
source /Users/ddolci/dev_code/venv-firedrake/bin/activate
FWI3D_SMOKE=1 FWI3D_METHOD=qn_latent \
python tests/three_dimension/elastic_fwi_3d/driver_elastic_fwi_3d.py
```

## Mintrop

Submit from this directory after checking out the integration branch:

```bash
cd tests/three_dimension/elastic_fwi_3d
sbatch --array=4 --export=ALL,FWI3D_MAX_ITERATIONS=1 \
  run_elastic_fwi_3d_mintrop.slurm
```

That first submission is the production-resolution memory gate: it uses all
eight sources but stops after one QN iteration. Inspect `MaxRSS` and the saved
`summary.json`; only then submit the complete array with
`sbatch run_elastic_fwi_3d_mintrop.slurm`.

The array has four tasks, one method per node and eight MPI ranks. With eight
sources, automatic parallelism assigns one source to each rank. Override the
production discretization or checkpoint budget at submission time, for
example:

```bash
sbatch --partition=amd_large \
  --export=ALL,FWI3D_EDGE=0.10,FWI3D_CHECKPOINT_SNAPSHOTS=24 \
  run_elastic_fwi_3d_mintrop.slurm
```

Results are written incrementally under
`fwi_3d_results/elastic_inclusion/<method>/<run-id>/`. Each run contains
`configuration.json`, `convergence.csv`, `summary.json`, and `models.h5`.

Before the full array, run the gradient parity test:

```bash
python -m pytest -q tests/on_one_core/test_elastic_checkpointing_3d.py
```
