# Plan: Issue #186 — Some tutorials outdated

## Context
The notebooks in `notebook_tutorials/` have drifted from spyro's current API, and nothing tests them, so the drift went unnoticed. The issue has been flagged for the AI-assisted long-standing-issue initiative. The maintainer's comment of 2026-11-18 sets the scope:
- change only the tutorials, test files and CI;
- make every notebook run against the current API;
- **stop for a human review of the notebooks before adding tests**;
- then test every notebook with nbval and run those tests in CI next to the demo tests (not on every PR).

Decisions already taken with the user:
- Notebooks that need the missing overthrust data are skipped in tests. `meshes/cut_overthrust.msh`, `velocity_models/cut_overthrust.hdf5` and `velocity_models/cut_overthrust_binary.bin` are not in the repo or its history. A follow-up will deal with the data.
- Notebook CI is a **new job** on the `firedrakeproject/firedrake:2026.4.0` image (local env is 2026.4.1).
- Runtime: **env-var test mode**. Notebooks keep realistic parameters, and `SPYRO_NOTEBOOK_TEST=1` swaps in cheap values.
- Exercises: add `simple_forward_exercises_answers.ipynb` with the blanks filled in and test it. Regenerate the `.html` from it. The exercises notebook keeps its blanks and is excluded from nbval.

## Prior work: the deleted branch can be recovered
This clone still has a stale remote-tracking ref, `origin/186-some-tutorials-outdated` (head `0ad10a35`, Oct 2025, 9 commits on a base from Aug 2025). It is useful as a reference, not something to merge:
- **Reuse:** `notebook_tutorials/simple_forward_exercises_answers.ipynb`. All blanks are filled, and the answers agree with today's audit. Take it with `git show <ref>:<path>` and then bring it up to date: mesh keys, `plot_mesh_sizes`, `deepcopy`, PML velocity workaround, and data paths relative to `notebook_tutorials/` instead of `../velocity_models/`.
- **Reuse as a check:** its `simple_forward*.ipynb` diffs, to compare against my fixes.
- **Do not reuse:**
  - `tests/tutorials/test_*.py` are placeholder tests that `pass`.
  - `nbval_sanitize.cfg` strips almost every line (any number with six decimals, any line with "memory", "mesh", "/"...), so it hides real regressions.
  - The changes to `setup.py` (that file no longer exists) and `conftest.py` are out of scope.
  - Its separate `notebook-tests.yml` contradicts the CI decision above.
- Suggest that you push the ref back to GitHub, or keep it locally, so it is not lost when the clone is pruned. I will not push it myself.

**Hard constraint:** no edits under `spyro/`. Where a notebook hits a spyro bug, work around it inside the notebook and report the bug in `AI_PR_DESCRIPTION.md`. No commits, no pushes.

## Phase 1 — Make every notebook run (static audit done, all 12 read)

**Common pattern.** Each heavy notebook gets one small code cell after the imports:
```python
import os
TEST_MODE = os.environ.get("SPYRO_NOTEBOOK_TEST", "0") == "1"
```
Cost parameters then read `0.1 if TEST_MODE else 0.02`. A short markdown line says the test mode exists for CI only.

**Cross-cutting fixes:**
- `dictionary["mesh"]` `"Lz"/"Lx"/"Ly"` → `"length_z"/"length_x"/"length_y"`. Current code reads them at `spyro/io/model_parameters.py:321-322` and in the example defaults `spyro/examples/rectangle.py`. Affects simple_forward, simple_forward_exercises, simple_forward_with_overthrust, premade_useful_examples, altering_time_integration, elastic_forward.
- `spyro.plots.plot_mesh_sizes(mesh, title_str=..., show=True)`: the mesh is now the first positional argument (`spyro/plots/mesh_plots.py:22`).
- Remove `fwi_velocity_model_output` and `velocity_model_filename`, and move `amplitude` from `time_axis` to `acquisition`. These keys are ignored silently today.
- Prefer `from firedrake.pyplot import triplot, tricontourf`.

**Per notebook (main fixes):**
| Notebook | Fixes |
|---|---|
| simple_forward | Mesh keys. Test mode: coarser `edge_length`. |
| simple_forward_exercises (+ new `_answers.ipynb`) | Mesh keys. `create_segy(function, V, grid_spacing, filename)` per `spyro/io/segy_io.py:139`. `write_function_to_grid` returns one array. Mesh adaptation needs `velocity_model` and `minimum_velocity`. `copy.deepcopy` instead of `dict.copy()`. Fill blanks in the answers notebook. Fix `plot_mesh_sizes`. Call `set_initial_velocity_model(new_file=...)` before a PML solve (spyro bug B below). `os.makedirs("velocity_models", exist_ok=True)`. Test mode: shorter `final_time`. |
| simple_forward_with_overthrust | Mesh keys, `"mesh_type": "file"`, `set_initial_velocity_model(new_file=...)` before the PML solve. Whole notebook excluded from nbval (data missing). |
| meshing | Fix the `rectangle_mesh`/`box_mesh` helpers (undefined `Lx`/`length_*`; mirror `spyro/meshing/firedrake_based_wrappers.py`). `FiniteElement("KMV", cell, degree)` without `variant`. Fix `plot_mesh_sizes`. Tag the deliberately failing SeismicMesh cell `raises-exception`. Tag cells that need `cut_overthrust_binary.bin` with `nbval-skip`. |
| premade_useful_examples | Mesh keys plus `"abc_type": None` where `status` is False (otherwise PML is inherited). `copy.deepcopy`. Tag overthrust cells `nbval-skip`. |
| altering_time_integration | Mesh keys plus `"abc_type": None`. Call `get_and_set_maximum_dt` before `matrix_building`. `excitations.current_sources = [source_id]`. Rewrite the manual time loop to the current scheme: `rhs_no_pml_source().assign(apply_source(rhs_forcing, step))`, `wave.solver.solve()`, `wave.u_np1`, as in `spyro/solvers/time_integration_central_difference.py:85-114`. `round(T/dt)+1`. Update outdated markdown (abstract methods, `_initialize_model_parameters`). |
| fwi_tutorial | `velocity_to_grid(velocity_function, mesh_parameters, grid_spacing, output)` per `spyro/utils/velocity_to_grid.py:11`. `output_filename` `.msh`, not `.vtk`. `os.makedirs("shots", exist_ok=True)`. `set_real_model`. Define `radius` at module level. Work around spyro bug C by reusing the in-memory shot record. Remove references to `.py` scripts that don't exist. Test mode: 2 shots, larger dt, gradient sampling > 1, `maxiter=1`, `forward_output=False`. |
| developer_fwi_tutorial | Remove the premature `reset_pressure()` (no `hasattr` guard, per AGENTS.md). Use `spyro.utils.compute_functional` for the misfit so it is consistent with the gradient. Fix the FD check comparison (directional/Taylor test). Regularisation: assemble a 0-form. Combine functional and gradient with `jac=True`. Add a `kernelspec`. Test mode as fwi_tutorial. |
| elastic_forward | Mesh keys. Shot-plot vmin/vmax from the receiver data (or omitted). Test mode: coarser `edge_length`. |
| elastic_plots | Drop `comm=` from `export_scalar_field`, per `spyro/io/segy_io.py:250`. |
| elastic_temporal_convergence | Fix the "second direction" print typo. Test mode: two dts and a coarser mesh. |
| acoustic_plots | `firedrake.pyplot` imports. `time_index` off-by-one. |

**Loop:**
1. Run each notebook with `jupyter nbconvert --to notebook --execute` from `notebook_tutorials/`, writing to the scratchpad. Run it twice: once with full values where that is feasible, once with `SPYRO_NOTEBOOK_TEST=1`.
2. Fix the failing cell against the current source, not memorised APIs.
3. Repeat until every notebook executes.
4. Re-execute in place with full values so the stored outputs are fresh, unless that takes hours (FWI). Otherwise store the test-mode outputs and say so.

Limitation: `gmsh` is not installed locally, so the `spyro_mesh` path in fwi_tutorial can only be validated in CI (which apt-installs gmsh), or after you install it.

**→ STOP: hand the notebooks to you for review (issue step 4). Do not start Phase 2 until you approve.**

## Phase 2 — nbval tests (after approval)
- Run with `pytest --nbval-lax`: any cell error fails, and only cells marked `nbval-check-output` are compared. Mark a few deterministic outputs, such as DOF counts and the printed convergence-error order.
- Add `tests/notebook_tests/nbval_sanitize.cfg` for paths, memory addresses and timing lines.
- Exclude `simple_forward_exercises.ipynb` (blanks) and `simple_forward_with_overthrust.ipynb` (data) with `--ignore`. Data cells elsewhere use `nbval-skip` tags.
- Command: `SPYRO_NOTEBOOK_TEST=1 python -m pytest --nbval-lax --nbval-sanitize-with tests/notebook_tests/nbval_sanitize.cfg notebook_tutorials/ --ignore=...`
- nbval runs the kernel with cwd = the notebook's folder (`nbval/plugin.py:266`). The notebooks create the folders they need.
- Verify at implementation time that the env var reaches the kernel; it is inherited via jupyter_client.

## Phase 3 — CI
Add a `notebook-tests` job to `.github/workflows/demo_and_notebook_tests.yml`:
- same triggers as the demo job (push to main, weekly cron);
- image `firedrakeproject/firedrake:2026.4.0`;
- the same dependency block as the demo job, plus `pip3 install nbval ipykernel`;
- runs the Phase 2 command.

Also drop "Notebook support incoming" from the workflow name.

## Files
- **Modified:** all `notebook_tutorials/*.ipynb` (overthrust gets API fixes only); `notebook_tutorials/simple_forward_exercises_answers.html` (regenerated); `.github/workflows/demo_and_notebook_tests.yml`.
- **New:** `notebook_tutorials/simple_forward_exercises_answers.ipynb`; `tests/notebook_tests/nbval_sanitize.cfg`; untracked `AI_PR_DESCRIPTION.md`.
- **Not touched:** anything under `spyro/`, `Makefile`, `pyproject.toml`, `.gitignore`.

## Spyro bugs found (report only; need your permission to fix)
- **A.** Example dictionaries silently keep 1×1 km defaults when given the legacy `Lz`/`Lx` keys. This is misleading rather than broken.
- **B.** With PML, a velocity model given only via `synthetic_data.real_velocity_file` is never loaded: `spyro/solvers/wave.py:207-208` and `spyro/pml/pml_nsnc.py:271-272` lead to `c=None`.
- **C.** Spatial parallelism with several shots: `ensemble_load` overwrites the shot records (`spyro/io/parallelism_wrappers.py:235-240`), so the FWI misfit is silently wrong.
- **D.** `export_scalar_field`/`plot_shots` always write `plot_of_shot[0].pdf` into the cwd. This is a minor nuisance.

## Verification
- Each notebook executes cleanly via nbconvert, in full mode where feasible and in test mode; record runtimes.
- `SPYRO_NOTEBOOK_TEST=1 pytest --nbval-lax ...` passes locally, and the run time is reported.
- Workflow YAML is validated: parse it with `python -c "import yaml; yaml.safe_load(...)"`. It can only truly run on GitHub.
- `make lint` still passes (no `.py` changes expected). Notebooks are excluded from ruff and black.
- Write `AI_PR_DESCRIPTION.md` with the PR text, the actual commands and results, what could not be checked (gmsh path, the CI run itself) and the spyro bugs above.
