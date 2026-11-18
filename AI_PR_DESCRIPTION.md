# Fix outdated tutorials (#186)

> Temporary, untracked file required by AGENTS.md. **Status: Phase 1 (notebooks fixed) is waiting for human review. The nbval tests and CI job (Phases 2–3) have not been added yet.**

AI assistance: Claude Code (Claude Opus 5.5) diagnosed the failures, edited the notebooks and ran them locally. A human must review every change before a PR is opened.

## Summary

The notebooks in `notebook_tutorials/` had drifted from spyro's current API. Ten of the twelve failed when run (only `acoustic_plots` and `elastic_temporal_convergence` passed). This change updates them to the current API without touching anything under `spyro/`. Where a notebook hit a spyro bug, it works around it inside the notebook, and the bug is listed below.

### Changes per notebook

| Notebook | Main changes |
|---|---|
| all `simple_forward*`, `premade_useful_examples`, `altering_time_integration`, `elastic_forward` | Renamed mesh keys `Lz/Lx/Ly` to `length_z/length_x/length_y`. Removed ignored `visualization` keys. |
| `simple_forward_exercises` | Fixed the non-answer cells: `create_segy` signature, `copy.deepcopy`, the SeismicMesh `set_mesh` inputs (velocity file and minimum velocity), `plot_mesh_sizes(mesh, ...)`, and an explicit `set_initial_velocity_model(new_file=...)`. `final_time` 5.0 → 2.0 s (spyro issue F). Cleared the stale outputs, which came from an answered run of the old API. |
| `simple_forward_exercises_answers.ipynb` (new) | Exercises with the blanks filled in. Answers taken from the deleted `186-some-tutorials-outdated` branch and checked against the old HTML. |
| `simple_forward_with_overthrust` | API fixes only (`mesh_type: "file"`, explicit velocity load). **Not run:** its data files are not in the repo. |
| `meshing` | Fixed the `rectangle_mesh`/`box_mesh` helpers (undefined `Lx`/`length_*`). `FiniteElement("KMV", ...)` without `variant`. `plot_mesh_sizes(mesh, ...)`. `firedrake.pyplot` import. Cells needing `cut_overthrust_binary.bin` are tagged `nbval-skip`; the deliberately failing cell is also tagged `raises-exception`. |
| `premade_useful_examples` | Mesh keys, plus `abc_type: None` where `status` is False (otherwise PML is inherited from the example defaults). `deepcopy` per object. The overthrust cell is tagged `nbval-skip`. |
| `altering_time_integration` | Manual time loop rewritten to the current scheme (`rhs_no_pml_source`, `solver.solve()`, `u_np1`). `current_sources`. Solver built after choosing `dt`. `round(T/dt)+1`. Exact eigenvalue for `dt` (spyro issue E). Updated outdated markdown. The receiver data now match `wave.forward_solve()` exactly (max abs diff 0.0). |
| `fwi_tutorial` | `velocity_to_grid` signature, `set_real_model`, `os.makedirs("shots")`, `.msh` output, undefined `radius`, references to scripts that don't exist. The in-memory observed data are passed directly (spyro issue C). `SPYRO_NOTEBOOK_TEST` switch. |
| `developer_fwi_tutorial` | Removed the premature `reset_pressure()`. Misfit is now `spyro.utils.compute_functional`, consistent with `gradient_solve`. The gradient passed to SciPy is the derivative (mass matrix × L2 gradient). Combined objective and gradient (`jac=True`). Directional finite-difference check (0.01 % error locally). Regularisation assembled as a 0-form. Added `kernelspec`. `SPYRO_NOTEBOOK_TEST` switch. |
| `elastic_forward` | Mesh keys. `np.abs` in the colour limits. |
| `elastic_plots` | Removed the `comm=` argument from `export_scalar_field`. |
| `elastic_temporal_convergence` | Print typo. |
| `acoustic_plots` | `firedrake.pyplot` import. `amplitude` moved to `acquisition`. |

`SPYRO_NOTEBOOK_TEST=1` is used only by the two FWI notebooks: fewer shots, larger `dt`, one iteration, no pvd output. All the other notebooks run at their real settings in under about a minute each.

## Validation actually run (local, Firedrake 2026.4.1, 1 CPU core)

- **Baseline (unmodified notebooks, `nbconvert --execute`):** 2 of 12 pass.
- **After the fixes, full settings:**

  | Notebook | Result |
  |---|---|
  | simple_forward | pass, 25 s |
  | simple_forward_exercises_answers | pass, 71 s |
  | meshing (data cells skipped) | pass, 7 s |
  | premade_useful_examples (data cell skipped) | pass, 10 s |
  | altering_time_integration | pass, 20 s |
  | elastic_forward | pass, 47 s |
  | elastic_plots | pass, 13 s |
  | elastic_temporal_convergence | pass, 37 s |
  | acoustic_plots | pass, 8 s |

- **developer_fwi_tutorial, test mode:** pass, 152 s. Full mode was not run (estimated about an hour or more on one core).
- **fwi_tutorial, test mode:** every cell except the `spyro_mesh` one was validated by running the notebook as a script with that single mesh call swapped for a `firedrake_mesh` (about 14 min). The `spyro_mesh` cell itself could not be validated: gmsh's Python module is not installed here, and with the PyPI gmsh 4.15 wheel even the existing `tests/on_one_core/test_mesh_tags.py` segfaults. Full mode was not run (hours).
- **Not run:** `simple_forward_with_overthrust` (missing data), and the exercises notebook (intentional blanks).
- Stored outputs of the nine fast notebooks were refreshed from a second full-settings run; all passed. `simple_forward_exercises_answers.html` was regenerated from the executed answers notebook with `jupyter nbconvert --to html`.
- `make lint`: passes. No `.py` files were changed.

## Spyro issues found (not fixed here; scope is notebooks, tests and CI only)

- **A.** Example models silently fall back to the default 1×1 km domain when given the legacy `Lz`/`Lx` keys.
- **B.** With PML, a velocity model given only through `synthetic_data.real_velocity_file` is never loaded. `Wave.forward_solve` only calls `_initialize_model_parameters` for no-ABC/NRBC (`spyro/solvers/wave.py:207-208`), and the PML layer then reads `c = None`.
- **C.** Spatial parallelism with several shots: `ensemble_load` (`spyro/io/parallelism_wrappers.py:235-240`) loads every shot into the same attribute, so `load_real_shot_record` keeps only the last shot. `calculate_misfit` then indexes `real_shot_record[snum]` as time rows, so the misfit is silently wrong.
- **D.** `plot_shots` always writes `plot_of_shot[0].pdf` to the working directory.
- **E.** `get_and_set_maximum_dt(estimate_max_eigenvalue=True)` is not conservative. On the `altering_time_integration` mesh it gives 1.90 ms with fraction 0.8, above the exact limit of about 1.58 ms, and `forward_solve` blows up.
- **F.** PML late-time growth. The `simple_forward_exercises` setup becomes unstable at about 3.6 s on the structured mesh (with `dt` 0.5 ms and 0.4 ms alike) and at about 2.6 s on the SeismicMesh mesh, where the stable `dt` is 2.5 ms. The same run without PML is stable to 5 s. The old stored outputs show 5 s used to work, so this may be a regression.
- **G.** `simple_forward_exercises`: the SEG-Y written from the padded PML mesh is read by the SeismicMesh sizing function as if it covered the unpadded box, so the sizing field is slightly squeezed. Velocity interpolation is consistent.

## Remaining concerns for the reviewer

- `final_time` in the exercises was reduced from 5.0 to 2.0 s only to avoid issue F.
- The test-mode `fwi_tutorial` run with the stand-in mesh made no model update: the start velocity equals `vmin` = 2.5. Whether the full tutorial (with `spyro_mesh` and its gradient mask) recovers the anomalies is unverified.
- Any run with absorbing boundaries imports `netgen` (through `HABCMesh`), so the CI image must provide it.
- The exercises notebook no longer stores outputs; the answers notebook does.
- The deleted branch `186-some-tutorials-outdated` still exists as a stale remote-tracking ref in this clone (`0ad10a35`). Only its answers were reused.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
