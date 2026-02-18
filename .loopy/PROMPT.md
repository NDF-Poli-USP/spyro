# Loopy Plan Prompt

Timestamp: 2026-02-18T01:36:25.707Z

You are in PLANNING mode.
Goal: update the plan only. Do NOT implement anything. No code edits. No commits.

## Context
## Plan seed (PRD) (--generate-prd)
# PRD: Code Architecture Improvement (MVP)

## Problem Statement
The codebase architecture has weak module boundaries and inconsistent patterns, increasing the cost and risk of implementing and maintaining changes.

## Goals
- Reduce change complexity by defining and enforcing clear module boundaries.
- Improve maintainability and testability without changing user-visible behavior.

## Non-Goals
- Rewriting core algorithms or introducing new end-user features.

## Users & Context
- Primary user: Software engineers maintaining and extending the codebase.
- Secondary user(s): QA/release engineers and new contributors onboarding to the project.
- Environment: Internal repository and CI pipeline (Python-based backend/library).

## Scope
- In scope: Define target architecture and module ownership.
- In scope: Refactor high-churn modules to align with target boundaries.
- In scope: Add automated architecture checks to CI (imports/cycles/layer rules).
- In scope: Update developer documentation for architecture and contribution patterns.
- Out of scope: Product feature development unrelated to architecture.
- Out of scope: Full rewrite of low-churn legacy modules.
- Out of scope: Infrastructure/platform migration (cloud, database, runtime).

## Requirements
### Functional
- [F1] Produce a target architecture spec with named layers/modules, allowed dependencies, and ownership; store in repo docs.
- [F2] Refactor at least the top 20% highest-churn modules to comply with the target boundaries.
- [F3] Eliminate circular dependencies in scoped modules; CI must fail on new cycles.
- [F4] Standardize shared concerns (configuration, logging, error handling, utilities) into dedicated modules with no duplication in scoped areas.
- [F5] Preserve existing public APIs/CLIs for scoped modules; any breaking change requires a deprecation path documented in changelog.
- [F6] Add/adjust automated tests so behavior remains equivalent for all refactored flows.

### Non-Functional
- [N1] Performance: No more than 5% regression in existing benchmark/test runtime for scoped components.
- [N2] Security/Privacy: No reduction in current security posture; no secrets in code/logs; existing access controls remain intact.
- [N3] Accessibility: Architecture docs and dev workflows must be usable with keyboard-only CLI and plain-text rendering.

## User Stories (MVP)
- As a maintainer, I want clear module boundaries, so that I can implement changes without cross-module side effects.
- As a reviewer, I want CI to detect boundary violations, so that architectural drift is blocked early.
- As a new contributor, I want concise architecture docs, so that I can onboard and contribute safely.
- As a release engineer, I want behavior-preserving refactors with test coverage, so that releases remain stable.

## Success Metrics
- 90%+ of scoped modules pass architecture-rule checks in CI within 6 weeks.
- Circular dependency count in scoped modules reaches 0 and stays at 0 for 4 consecutive weeks.
- Median PR cycle time for scoped modules decreases by at least 25% within 8 weeks of rollout.
- Architecture-related regression defects in scoped modules decrease by at least 30% over the next 2 release cycles.

## Risks & Mitigations
- Incomplete architecture rules cause inconsistent refactors → Define and approve rules before refactoring begins.
- Regressions during refactor → Require parity tests and phased rollout per module.
- Team adoption resistance → Provide short migration guide, examples, and CI feedback with actionable messages.
- Scope creep into full rewrite → Enforce churn-based prioritization and strict out-of-scope rules.

## Open Questions
- Which modules are highest priority for the first refactor wave?
- What is the required delivery timeline (sprint count/release target)?
- Which architecture validation tool(s) are preferred in CI?

## Assumptions
- Existing tests and CI are available and can be extended.
- Architecture work is limited to internal maintainability, not feature changes.
- A small cross-functional team (2–4 engineers) is available for at least one release cycle.
- Current runtime/platform stack remains unchanged during this initiative.





## AGENTS
● Todo: Added: Inspect build and validation commands

● Read setup.cfg
  └ 12 lines read

● Read Makefile
  └ 31 lines read

● Read setup.py
  └ 18 lines read

● Read README.md
  └ 259 lines read

● Glob "**/{tox.ini,pyproject.toml,requirements*.txt,environment*.yml,noxfile.py}" in
/Users/ddolci/usp_postdoc/spyro
  └ 1 file found

● Glob ".github/workflows/*.{yml,yaml}" in /Users/ddolci/usp_postdoc/spyro
  └ 3 files found

● Grep "(pytest|unittest|nose|tox|nox|ruff|flake8|black|isort|mypy|pyright|pylint|make test|make
lint|make typecheck|coverage)" (/Users/ddolci/usp_postdoc/spyro)
  └ 6 lines found

● Read .github/workflows/python-tests.yml
  └ 121 lines read

● Read .github/workflows/lint.yml
  └ 27 lines read

● Read .github/workflows/build_firedrake_main.yml
  └ 116 lines read

● Read conftest.py
  └ 55 lines read

● List directory demos
  └ 2 files found

● List directory test_integration
  └ 2 files found

● Glob "tests/**/*.py" in /Users/ddolci/usp_postdoc/spyro
  └ 62 files found

● Glob "demos/**/*.py" in /Users/ddolci/usp_postdoc/spyro
  └ 4 files found

● Read demos/README.md
  └ 6 lines read

● Read demos/with_automatic_differentiation/run_forward_ad.py
  └ 46 lines read

● Read demos/with_automatic_differentiation/run_fwi_ad.py
  └ 112 lines read

● Grep "argparse|click\.command|typer\.Typer" (/Users/ddolci/usp_postdoc/spyro)
  └ No matches found

● Grep "pytest\.ini|tool\.pytest\.ini_options" (/Users/ddolci/usp_postdoc/spyro)
  └ No matches found

● Grep "if __name__ == ['\"]__main__['\"]" (/Users/ddolci/usp_postdoc/spyro)
  └ 39 lines found

● Todo: Started

✗ Create session plan file
  $ cat > /Users/ddolci/.copilot/session-state/b1e4ddb6-0032-48cd-82ec-eba2fb05a6fa/plan.md
  <<'EOF'
  # Plan
  1. Confirm reliable build/run/test/lint/typecheck commands from repo docs and CI.
  2. Create AGENTS.md with required structure and only confident commands....
  Permission denied and could not request permission from user

✗ Create AGENTS.md
  Failed to apply patch: Error: Permission denied: Create file
  /Users/ddolci/usp_postdoc/spyro/AGENTS.md

● Todo: Completed: draft-agents-md

● Glob "AGENTS.md" in /Users/ddolci/usp_postdoc/spyro
  └ No matches found

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
test_command: pytest ...
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

## Output Rules
- Plan only.
- No implementation steps.
- No commits.
- Keep tasks small and unambiguous.
