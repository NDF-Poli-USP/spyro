---
agent_command: copilot --allow-all
test_command: python3 -m pytest tests/on_one_core/test_gradient_*
max_iterations: 50
max_minutes: 120
backoff_ms: 5000
rotate_bytes: 150000
git:
  branch: ''
  commit: true
  commit_message: 'loopy: {change_type} {task_summary}'
phase_defaults:
  stop_on: all_checked
  test_command: pytest -q
phases:
  - id: config-entrypoint
    title: Config and run entrypoint
    stop_on: all_checked
    test_command: pytest -q
  - id: forward-misfit
    title: Forward modeling and misfit
    stop_on: all_checked
    test_command: pytest -q
  - id: adjoint-optimization
    title: Adjoint gradient optimization
    stop_on: all_checked
    test_command: pytest -q
  - id: checkpoints-artifacts
    title: Checkpointing logging artifacts
    stop_on: all_checked
    test_command: pytest -q
  - id: validation-hardening
    title: Validation and reproducibility
    stop_on: all_checked
    test_command: pytest -q
---

# Plan

<!-- loopy:seed # PRD: Automated Gradient Solver for Full-Waveform Inversion (Firedrake Adjoint)

## Problem Statement
Researchers need a reproducible, automated way to compute and apply Firedrake-adjoint gradients for full-waveform inversion, because manual gradient/update workflows are slow, error-prone, and hard to validate.

## Goals
- Deliver an end-to-end MVP that runs forward simulation, adjoint gradient computation, and model updates automatically from a single configuration.
- Ensure numerical correctness and experiment reproducibility through built-in validation, logging, and checkpointing.

## Non-Goals
- Production-scale 3D distributed inversion on large HPC clusters.
- Building a GUI or interactive dashboard.
- Supporting multi-physics inversion beyond the acoustic formulation in the tutorial.

## Users & Context
- Primary user: Computational geophysics researcher running inversion experiments in Python.
- Secondary user(s): Research engineer maintaining reproducible workflows and CI checks.
- Environment: Internal Python CLI/notebook workflow on workstation or small HPC node.

## Scope
- In scope:
  - Tutorial-based acoustic FWI implementation using Firedrake + firedrake-adjoint.
  - Automated gradient-based optimization loop for velocity model updates.
  - Validation (Taylor test), logging, checkpoint/resume, and output artifacts.
- Out of scope:
  - New PDE formulations not in tutorial baseline.
  - Real-time visualization service.
  - Cloud orchestration and job scheduling infrastructure.

## Requirements
### Functional
- [F1] Provide a single run entrypoint (CLI or script) that accepts a config file for mesh, time setup, source/receiver geometry, observed data path, and optimizer settings.
- [F2] Implement forward modeling (tutorial-equivalent) that computes synthetic traces and scalar objective/misfit \(J\).
- [F3] Define inversion control parameter(s) (at minimum velocity model) and compute \(\nabla J\) using firedrake-adjoint automatic differentiation.
- [F4] Implement iterative optimization (default L-BFGS-B or configurable gradient-based optimizer) with max iterations, stopping tolerance, and line-search/step controls.
- [F5] Persist per-iteration metrics: iteration index, objective value, gradient norm, step size, and elapsed time.
- [F6] Save model and optimizer checkpoints at configurable cadence; support resume from last checkpoint without restarting from iteration 0.
- [F7] Provide a gradient verification mode using a Taylor test and report pass/fail with measured convergence order.
- [F8] Write final artifacts to output directory: final model, convergence history (CSV/JSON), and misfit-vs-iteration plot.
- [F9] Fail loudly with actionable errors when config/data are invalid (missing files, incompatible dimensions, unstable time step).

### Non-Functional
- [N1] Performance: On a reference 2D benchmark (assumed <=50k cells, <=10 shots), complete 20 optimization iterations within 60 minutes on a 16-core CPU node.
- [N2] Security/Privacy: Operate fully offline on local files; no outbound network calls; do not log sensitive filesystem paths beyond user-specified workspace.
- [N3] Accessibility: CLI outputs must be plain-text and machine-readable (CSV/JSON logs) with no color-only signaling.

## User Stories (MVP)
- As a geophysics researcher, I want to run inversion from one config command, so that I can reproduce experiments quickly.
- As a researcher, I want automatic adjoint gradients and optimizer updates, so that I avoid manual derivations and coding errors.
- As a research engineer, I want a Taylor-test mode, so that I can verify gradient correctness before long runs.
- As a user, I want checkpoint/resume support, so that interrupted long jobs can continue without losing progress.
- As a user, I want convergence artifacts saved automatically, so that I can compare runs and report results.

## Success Metrics
- At least 95% of benchmark runs complete without manual code edits across 10 repeated executions within 30 days of MVP release.
- On the reference synthetic case, objective value decreases by >=30% within 20 iterations from the initial model.
- Taylor-test mode reports convergence order >=1.9 on the reference setup.
- Resume-from-checkpoint reproduces the same next-iteration objective within relative error <=1e-6 versus uninterrupted run.

## Risks & Mitigations
- Adjoint mismatch from non-differentiable operations → Restrict to differentiable operations in taped region and enforce Taylor-test checks.
- Unstable inversion updates (divergence/noise amplification) → Add step control, optional regularization, and early-stop criteria.
- Excess memory/runtime from adjoint tape → Use checkpointing, control annotation scope, and benchmark-driven defaults.
- Configuration complexity causing user errors → Provide strict schema validation and clear error messages with offending field names.

## Open Questions
- Is MVP strictly 2D, or must 3D support be included from day one?
- Is observed data synthetic-only for MVP, or must field-data ingestion be supported?
- Which optimizer is mandatory for acceptance (L-BFGS-B only vs multiple options)?
- Should this be integrated into an existing package/module structure or shipped as a standalone example pipeline?
- What is the required reference hardware/profile for official performance targets?

## Assumptions
- MVP follows the Firedrake tutorial’s acoustic FWI formulation as the technical baseline.
- Primary control variable is velocity (or equivalent squared slowness representation) on a 2D mesh.
- Users already have compatible Firedrake and firedrake-adjoint environments installed.
- Initial validation uses synthetic data generated from a known “true” model.
- Execution is non-interactive batch style (script/CLI), not GUI-driven. -->

## Phase: config-entrypoint
<!-- loopy:phase config-entrypoint -->

- [ ] add: define inversion config schema for required mesh, time, geometry, observed data, optimizer, checkpoint, and output fields — Acceptance: loading a complete sample config succeeds and omitting any required field returns an error naming that field.
- [ ] implement: parse YAML and JSON config files into one normalized runtime object — Acceptance: equivalent YAML and JSON configs produce identical normalized values.
- [ ] add: expose a single CLI run command that accepts --config — Acceptance: invoking the command with a valid config exits 0 and starts a non-interactive run.
- [ ] implement: validate file and directory existence during startup — Acceptance: missing mesh or observed-data paths fail fast with a non-zero exit and actionable message.
- [ ] implement: validate stable time-step bounds before simulation starts — Acceptance: an unstable dt is rejected before forward solve with a message containing dt and the allowed bound.
- [ ] verify: persist a resolved configuration snapshot into the run output directory — Acceptance: each run writes a config snapshot that can be reused unchanged for reproduction.

## Phase: forward-misfit
<!-- loopy:phase forward-misfit -->

- [ ] implement: build a tutorial-equivalent acoustic forward modeling function callable from config — Acceptance: forward execution produces synthetic traces for every configured shot and receiver.
- [ ] add: load observed data into the internal trace structure with strict shape checks — Acceptance: incompatible shot, receiver, or time dimensions fail with expected versus actual sizes.
- [ ] implement: compute residual traces between synthetic and observed data — Acceptance: residual arrays match the synthetic/observed indexing and lengths exactly.
- [ ] implement: compute scalar objective J from residual traces — Acceptance: J is finite and matches a hand-checked fixture value within defined tolerance.
- [ ] update: execute forward modeling plus initial objective evaluation in the main pipeline — Acceptance: iteration 0 logs and persists initial J before any model update.
- [ ] verify: enforce deterministic initial forward/misfit results for fixed inputs — Acceptance: two identical runs produce initial J values with relative difference <= 1e-12.

## Phase: adjoint-optimization
<!-- loopy:phase adjoint-optimization -->

- [ ] add: define velocity model as the primary firedrake-adjoint control variable — Acceptance: control creation succeeds from configured initial model and is connected to the taped functional.
- [ ] implement: tape the misfit functional for automatic differentiation — Acceptance: firedrake-adjoint returns a gradient field without manual derivative code.
- [ ] implement: compute and expose gradient norm each iteration — Acceptance: every optimization iteration records a finite gradient norm value.
- [ ] add: implement optimizer factory with L-BFGS-B default and config override support — Acceptance: missing optimizer key selects L-BFGS-B, and a valid override selects the requested optimizer.
- [ ] implement: add configurable line-search or step-size controls to updates — Acceptance: changing step-control config measurably changes accepted step sizes.
- [ ] implement: enforce stopping rules for max iterations and tolerance — Acceptance: run terminates on the first satisfied criterion and logs the stop reason.
- [ ] verify: satisfy objective-reduction target on reference synthetic case — Acceptance: objective decreases by at least 30% from iteration 0 to iteration 20.

## Phase: checkpoints-artifacts
<!-- loopy:phase checkpoints-artifacts -->

- [ ] add: capture per-iteration metrics (iteration, objective, gradient norm, step size, elapsed time) in memory — Acceptance: each completed iteration appends exactly one record with all required fields.
- [ ] implement: persist convergence history to CSV — Acceptance: CSV contains one row per iteration with required columns in stable order.
- [ ] implement: persist convergence history to JSON — Acceptance: JSON contains all iterations and numeric values matching the CSV output.
- [ ] add: save model and optimizer checkpoints at configurable cadence — Acceptance: with checkpoint_every=N, checkpoints exist at N, 2N, and final iteration.
- [ ] implement: resume from latest checkpoint without resetting iteration index — Acceptance: resumed run starts at last_saved_iteration + 1.
- [ ] add: export final model artifact to output directory — Acceptance: successful run writes final model file to documented path.
- [ ] add: generate misfit-versus-iteration plot artifact — Acceptance: successful run writes a readable plot file in the output directory.
- [ ] verify: match resumed versus uninterrupted next-iteration objective — Acceptance: relative error between resumed and uninterrupted next objective is <= 1e-6.

## Phase: validation-hardening
<!-- loopy:phase validation-hardening -->

- [ ] implement: add dedicated Taylor-test CLI mode — Acceptance: Taylor-test mode executes gradient verification without entering the full optimization loop.
- [ ] add: compute measured Taylor convergence order with pass/fail thresholding — Acceptance: output reports numeric order and marks pass only when order >= configured threshold.
- [ ] implement: validate geometry and data compatibility against mesh and receiver setup — Acceptance: mismatches fail early with an error naming the offending field.
- [ ] update: standardize actionable error formatting for config and runtime validation failures — Acceptance: each failure message includes error type, context, and suggested fix.
- [ ] update: constrain log path reporting to user workspace-relative paths — Acceptance: logs avoid unrelated absolute paths outside configured workspace.
- [ ] verify: keep CLI output plain text and artifacts machine-readable — Acceptance: no color-only signaling appears and produced CSV/JSON parse successfully in automated checks.
- [ ] add: provide a benchmark run target for the reference 2D case — Acceptance: benchmark command runs 20 iterations and writes runtime summary metrics.
- [ ] verify: run inversion workflow fully offline using local files only — Acceptance: integration check passes with outbound network disabled.
