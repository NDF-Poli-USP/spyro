# Loopy Build Prompt

Timestamp: 2026-02-18T01:43:02.918Z
Iteration: 1

You are in BUILDING mode.
Goal: complete exactly one task from the current plan.

## Context






## AGENTS
# AGENTS

## Build & Run
- Requires an active Firedrake environment with MPI available.
- Install for development: `python -m pip install -e .`
- Build distributable artifacts: `python setup.py sdist bdist_wheel`
- No standalone CLI entrypoint is defined; run repository scripts/tests directly from the repo root.

## Validation
- Tests: `MPLBACKEND=Agg pytest --maxfail=1`
- Lint: `make lint`
- Typecheck: Not configured in this repository.

## Operational Notes
- CI also runs MPI-heavy validation (e.g., `mpiexec -n 6 python -m pytest test_integration/`).
- `./cleanup.sh` is used in CI between test phases to remove generated artifacts.

## Current Task

- [ ] analysis: identify top-churn modules from git history — Acceptance: docs/architecture/churn-baseline.md lists modules covering at least 20% of recent churn.


## Plan
---
agent_command: copilot
test_command: pytest ...
max_iterations: 50
max_minutes: 120
backoff_ms: 5000
rotate_bytes: 150000
git:
  branch: dolci/use_variational_solver
  commit: true
  commit_message: 'loopy: {change_type} {task_summary}'
phase_defaults:
  stop_on: all_checked
  test_command: pytest -q
phases:
  - id: architecture-baseline
    title: Define target boundaries
    stop_on: all_checked
    test_command: pytest -q
  - id: shared-foundations
    title: Standardize shared concerns
    stop_on: tests_pass
    test_command: pytest -q
  - id: churn-module-refactor
    title: Refactor high-churn modules
    stop_on: tests_pass
    test_command: pytest -q
  - id: ci-architecture-gates
    title: Enforce architecture checks
    stop_on: tests_pass
    test_command: pytest -q
  - id: docs-rollout-metrics
    title: Document and track adoption
    stop_on: all_checked
    test_command: pytest -q
---

# Plan

## Phase: architecture-baseline
<!-- loopy:phase architecture-baseline -->

- [ ] analysis: identify top-churn modules from git history — Acceptance: docs/architecture/churn-baseline.md lists modules covering at least 20% of recent churn.
- [ ] architecture: define target layers and module map — Acceptance: docs/architecture/target-architecture.md contains named layers and a module-to-layer mapping for all scoped modules.
- [ ] architecture: define allowed dependency matrix — Acceptance: target architecture doc includes explicit allowed/disallowed layer dependencies with examples.
- [ ] ownership: assign module owners and backups — Acceptance: architecture doc includes ownership table with primary and secondary owner for each scoped module.
- [ ] governance: define refactor scope and non-goals — Acceptance: architecture doc includes in-scope/out-of-scope section aligned to PRD requirements.
- [ ] verification: capture baseline cycle and boundary violations — Acceptance: docs/architecture/architecture-baseline-report.md records current cycle count and boundary violation count for scoped modules.

## Guardrails
# Loopy Guardrails

## Signs

## Task Rules
- Use subagents to study specs/code; use only one subagent for tests.
- Do not assume functionality is missing; search first.
- If the plan is wrong or stale, switch to plan mode and regenerate it.
- If acceptance criteria are subjective, add and run judge tests (see `loopy add-judge`).
- If you discover new run/test commands, update AGENTS.md.
- Complete all unchecked tasks in the current phase before tests will be run.
- Mark a task checkbox as [x] when the implementation is done. The test_command runs automatically after all phase tasks are checked.
- If a task is impossible or should be skipped, mark it with [~] or [-] and explain the reason inline.
- If a task is blocked by external factors after 3+ consecutive failures, mark it as [!] with a reason: `[!] task — BLOCKED: reason`. Blocked tasks do not block phase advancement.
- If the test command fails after all tasks are checked, fix the failures before the phase can advance. Do not move on with broken tests.
- If the same task has failed for 3+ consecutive iterations, reassess your approach: read the error output carefully, consider reverting recent changes, or switch to plan mode to re-scope the task.

## Built-in Rules
- Phases follow a two-gate lifecycle: Gate 1 = all tasks checked [x] (or skipped [~]/[-] or blocked [!]), Gate 2 = test_command passes.
- The test_command is NOT executed until every task in the current phase is checked. Focus on completing tasks first.
- Focus on one task at a time. Do not check multiple boxes in a single iteration.
- Never cycle back to a previous phase. Phases are sequential and one-directional.

## Instructions
- Don't assume something is unimplemented; search first.
- Update AGENTS.md only for operational learnings.
- No stubs or placeholder implementations.
- Follow the plan checklist in LOOPY_PLAN.md.
- Update plan checkboxes as you complete items.
- Record any new guardrails if you detect repetition or drift.
- Keep changes focused and maintain repo state.
- Complete all unchecked tasks in the current phase before tests will be run.
- Mark a task [x] when the implementation is done. The test_command runs automatically after all phase tasks are checked.
- If a task should be skipped, mark it with [~] or [-] and note the reason.
- If a task is blocked by external factors after 3+ consecutive failures, mark it as [!] with a reason: `[!] task — BLOCKED: reason`. Blocked tasks do not block phase advancement.
- If tests fail after all tasks are checked, fix the failures first.
- If the same task has failed for 3+ consecutive iterations, reassess your approach.
- **Complete only the Current Task in this iteration.**
