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

Submit from this directory after checking out the integration branch. First,
compare the production mesh against a finer one using one source:

```bash
sbatch --partition=amd_large_2 --nodelist=n19 \
  run_spectral_mesh_gate_mintrop.slurm
```

Then run one complete-acquisition BLMVM+H1 iteration as the checkpointing and
memory gate:

```bash
cd tests/three_dimension/elastic_fwi_3d
sbatch --array=2 --export=ALL,FWI3D_MAX_ITERATIONS=1 \
  run_elastic_fwi_3d_mintrop.slurm
```

That first submission is the production-resolution memory gate: it uses all
16 sources but stops after one BLMVM+H1 iteration. Inspect `MaxRSS` and the saved
`summary.json`; only then submit the complete array with
`sbatch run_elastic_fwi_3d_mintrop.slurm`.

The production mesh is a structured quadrilateral base extruded in depth,
with hexahedral fourth-order spectral elements and GLL quadrature. With
`edge=0.05 km`, it has 2.88 elements and 11.52 GLL intervals per minimum
wavelength at 5 Hz using the lower Vs bound. The acquisition has a 4x4 source
grid and a 9x9 receiver grid on the top surface.

The array has four tasks, one method per node and 64 MPI ranks. With 16
sources, automatic parallelism assigns four spatial ranks to each source. The
distributed VertexOnlyMesh receiver path restores all 81 vector traces in the
original acquisition order before objective evaluation. Override the
production discretization or checkpoint budget at submission time, for
example:

```bash
sbatch --partition=amd_large_2 --nodelist=n19 \
  --export=ALL,FWI3D_EDGE=0.05,FWI3D_CHECKPOINT_SNAPSHOTS=32 \
  run_elastic_fwi_3d_mintrop.slurm
```

Results are written incrementally under
`fwi_3d_results/elastic_inclusion/<method>/<run-id>/`. Each run contains
`configuration.json`, `convergence.csv`, `summary.json`, `models.h5`, and
`models.latest.json`. By default, every accepted iteration atomically replaces
`models.h5`; an interrupted write therefore leaves the preceding checkpoint
intact. `models.latest.json` records which iteration the HDF5 file contains.
Set `FWI3D_MODEL_SAVE_EVERY=N` only when checkpoint I/O needs to be reduced.

Before the full array, run the gradient parity test:

```bash
python -m pytest -q tests/on_one_core/test_elastic_checkpointing_3d.py
```
