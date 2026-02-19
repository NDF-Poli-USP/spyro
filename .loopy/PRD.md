# PRD: Automated Gradient Solver for Full-Waveform Inversion (Firedrake Adjoint)

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
- [F1] Activate the python enviroment source /Users/ddolci/my_venv/bin/activate.
- [F2] Provide a single run entrypoint (CLI or script) that accepts a config file for mesh, time setup, source/receiver geometry, observed data path, and optimizer settings.
- [F3] Implement forward modeling (tutorial-equivalent) that computes synthetic traces and scalar objective/misfit \(J\).
- [F4] Define inversion control parameter(s) (at minimum velocity model) and compute \(\nabla J\) using firedrake-adjoint automatic differentiation.
- [F5] Implement iterative optimization (default L-BFGS-B or configurable gradient-based optimizer) with max iterations, stopping tolerance, and line-search/step controls.
- [F6] Persist per-iteration metrics: iteration index, objective value, gradient norm, step size, and elapsed time.
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
- Execution is non-interactive batch style (script/CLI), not GUI-driven.
