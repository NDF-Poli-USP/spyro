# Loopy Plan Prompt

Timestamp: 2026-02-19T19:00:48.079Z

You are in PLANNING mode.
Goal: update the plan only. Do NOT implement anything. No code edits. No commits.

## Context
## Plan seed (PRD) (--generate-prd)
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
- Execution is non-interactive batch style (script/CLI), not GUI-driven.





## AGENTS
● Todo: Added: Scan project configs

● Glob "{README.md,Makefile,setup.cfg,setup.py,pyproject.toml,tox.ini,noxfile.py,requirements*.txt}"
  └ 14 files found

● Read setup.cfg
  └ 12 lines read

● Read Makefile
  └ 31 lines read

● Read README.md
  └ 259 lines read

● Grep "entry_points|console_scripts|scripts|cmdclass" (setup.py)
  └ No matches found

● Grep "\b(pytest|tox|nox|ruff|flake8|pylint|black|mypy|pyright|coverage|lint|typecheck)\b"
  └ 152 lines found

● Read setup.py
  └ 18 lines read

● Read .github/workflows/python-tests.yml
  └ 121 lines read

● Read conftest.py
  └ 55 lines read

● Read .github/workflows/lint.yml
  └ 27 lines read

● Glob "demos/**/*.py"
  └ No matches found

● List directory demos
  └ 1 file found

● Read docker/README.md
  └ 39 lines read

● Glob "AGENTS.md"
  └ No matches found

● Read demos/README.md
  └ 6 lines read

● Grep "pip install -e|python -m pip install -e|pytest|make lint|flake8" (README.md)
  └ 2 lines found

● Grep "\[tool\.|mypy|pyright|typecheck|ruff" in pyproject.toml
  └ No matches found

● Grep "if __name__ == ['\"]__main__['\"]"
  └ 39 files found

● Todo: Started: scan-configs

✗ Create session plan file
  $ cat > /Users/ddolci/.copilot/session-state/f375062a-b486-4b12-9bd1-3469575962a6/plan.md
  <<'EOF'
  # Plan
  1. Confirm build/run, test, and lint commands from repository docs and CI.
  2. Validate command availability with low-risk checks where feasible....
  Permission denied and could not request permission from user

● Todo: Querying

✗ Sanity-check build/lint/test commands
  $ python3 setup.py --help >/dev/null 2>&1; echo "setup_py_status=$?" && make -n lint; python3 -m
  pytest --version 2>/dev/null || true
  Permission denied and could not request permission from user

● Read cleanup.sh
  └ 28 lines read

● Glob "tests/on_one_core/*.py"
  └ 34 files found

● Glob "tests/integration/*.py"
  └ 1 file found

● Todo: Started: verify-commands

● Todo: Querying

✗ Create AGENTS.md
  Failed to apply patch: Error: Permission denied: Create file
  /Users/ddolci/usp_postdoc/spyro/AGENTS.md

● Todo: Completed: write-agents-md

● Todo: Querying

# AGENTS

## Build & Run
- In a Firedrake-enabled Python environment, install locally: `python -m pip install -e .`
- Build package artifacts: `python3 setup.py sdist bdist_wheel`
- This project is a Python library (no standalone CLI); run it from your own Python script/notebook via `import spyro`.

## Validation
- Tests: `MPLBACKEND=Agg pytest --maxfail=1`
- Lint: `make lint`
- Typecheck: Not configured in this repository.

## Operational Notes
- Full parallel validation requires MPI (`mpiexec`) and Firedrake dependencies (see `tests/parallel/`).

## Requirements
Study these sources before planning:
- `specs/*` (requirements)
- `src/*` (current implementation)
Use subagents for study and investigation; use only one subagent for tests.

## Plan
Compare specs against code. Produce a prioritized plan that closes gaps.
If the existing plan is wrong or stale, replace it.
Keep tasks atomic, testable, and outcome-focused.
Do not assume anything is missing; search first.
If acceptance criteria are subjective, add judge tests (see `loopy add-judge`).
- Phases use a two-gate completion model: Gate 1 = all tasks checked, Gate 2 = test_command passes. Tests are only run after all tasks in a phase are checked.
- The stop_on field is deprecated. All phases follow the two-gate model automatically.
- If a task is impossible to complete, mark it as skipped with [~] or [-] and include the reason in the task text.
- If a task is blocked by external factors after 3+ consecutive failures, mark it as [!] with a BLOCKED reason (e.g., `[!] task description — BLOCKED: reason`). Blocked tasks are excluded from phase gates.

## Current Plan
---
agent_command: copilot
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

## Phase: config-entrypoint
<!-- loopy:phase config-entrypoint -->

- [ ] add: define inversion config schema for required mesh, time, geometry, observed data, optimizer, checkpoint, and output fields — Acceptance: loading a complete sample config succeeds and omitting any required field returns an error naming that field.
- [ ] implement: parse YAML and JSON config files into one normalized runtime object — Acceptance: equivalent YAML and JSON configs produce identical normalized values.
- [ ] add: expose a single CLI run command that accepts --config — Acceptance: invoking the command with a valid config exits 0 and starts a non-interactive run.
- [ ] implement: validate file and directory existence during startup — Acceptance: missing mesh or observed-data paths fail fast with a non-zero exit and actionable message.
- [ ] implement: validate stable time-step bounds before simulation starts — Acceptance: an unstable dt is rejected before forward solve with a message containing dt and the allowed bound.
- [ ] verify: persist a resolved configuration snapshot into the run output directory — Acceptance: each run writes a config snapshot that can be reused unchanged for reproduction.

## Guardrails
# Loopy Guardrails

## Signs

## Output Rules
- Plan only.
- No implementation steps.
- No commits.
- Keep tasks small and unambiguous.
