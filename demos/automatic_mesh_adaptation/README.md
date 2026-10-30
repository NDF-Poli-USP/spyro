# Automatic mesh adaptation to a trace-error tolerance

`marmousi_automatic_loop.py` builds a sequence of KMV meshes until the
estimated relative trace error of the worst shot falls below
`theta * tol`. The estimate is a two-level pilot: the forward problem is
solved on the mesh and on its uniform refinement with the same time step,
and the receiver traces are compared in a time window.

Three strategies choose the next mesh (module `spyro.adaptivity`):

| strategy   | next mesh                                                                 |
|------------|---------------------------------------------------------------------------|
| `cpw`      | uniform cells per wavelength, raised by the predicted budget growth       |
| `gradient` | budget redistributed by the smoothed squared velocity gradient            |
| `adjoint`  | budget redistributed by the dual-weighted residual of the trace error     |

The adjoint indicator uses spyro's automated adjoint: the refined forward
solve is recorded on the pyadjoint tape, the windowed trace functional is
differentiated, and the adjoint of each time step is read from the tape. The
residual of the prolonged coarse trajectory, weighted by those adjoints, is
localized on the refined cells. The pilot reports the identity
`sum(eta_K) - sum(f(lambda)) = -2 J`, exact to round-off.

## Running

The velocity window is the `.npz` of the KMV4 calibration study
(`vp[ix, idepth]` in km/s, `x_origin`, `h`):

```bash
mpiexec -n 4 python marmousi_automatic_loop.py --model marmousi_window.npz \
    --strategy adjoint --tol 0.05 --output results/adjoint_tol0p05
```

`--strategy cpw` and `--strategy gradient` run the same loop. The report
`report_<strategy>.json` lists every pilot (mesh, DOFs, error per shot,
growth, cost in DOF-steps and wall time) and the accepted mesh.

## Notes

- Positions are local: `x` starts at the left end of the absorbing pad,
  because spyro requires `0 <= x <= length_x`.
- Boundary conditions are spyro's first-order non-reflecting conditions on
  the bottom and sides, with a natural top boundary.
- The indicator is accumulated during the reverse sweep, so no adjoint
  trajectory is kept. Without checkpointing the tape still keeps every
  refined time step; `--snapshots N` checkpoints it, keeping `N` steps in
  memory and recomputing the forward in between.
- `--accumulate-every k` accumulates the indicator every `k` steps (cheaper,
  approximate); the default `1` is exact.

AI assistance: implemented with Claude (Anthropic).
