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

<!-- loopy:seed # PRD: Code Architecture Improvement (MVP)

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
- Current runtime/platform stack remains unchanged during this initiative. -->

## Phase: architecture-baseline
<!-- loopy:phase architecture-baseline -->

- [ ] analysis: identify top-churn modules from git history — Acceptance: docs/architecture/churn-baseline.md lists modules covering at least 20% of recent churn.
- [ ] architecture: define target layers and module map — Acceptance: docs/architecture/target-architecture.md contains named layers and a module-to-layer mapping for all scoped modules.
- [ ] architecture: define allowed dependency matrix — Acceptance: target architecture doc includes explicit allowed/disallowed layer dependencies with examples.
- [ ] ownership: assign module owners and backups — Acceptance: architecture doc includes ownership table with primary and secondary owner for each scoped module.
- [ ] governance: define refactor scope and non-goals — Acceptance: architecture doc includes in-scope/out-of-scope section aligned to PRD requirements.
- [ ] verification: capture baseline cycle and boundary violations — Acceptance: docs/architecture/architecture-baseline-report.md records current cycle count and boundary violation count for scoped modules.

## Phase: shared-foundations
<!-- loopy:phase shared-foundations -->

- [ ] refactor: add shared configuration module interface — Acceptance: scoped modules read configuration through the shared module and no direct environment access remains in scoped paths.
- [ ] refactor: add shared logging module wrapper — Acceptance: scoped modules use the shared logger wrapper and duplicated logger setup code is removed from scoped paths.
- [ ] refactor: add shared error handling module — Acceptance: scoped modules use shared error classes and duplicate exception definitions are removed from scoped paths.
- [ ] refactor: add shared utilities module for common helpers — Acceptance: duplicated helper functions in scoped paths are consolidated into the shared utilities module.
- [ ] compatibility: preserve public import/API contracts for shared concerns — Acceptance: existing public imports continue to work and compatibility tests pass.
- [ ] tests: add parity tests for shared concern behavior — Acceptance: tests assert equivalent configuration, logging, and error behavior before and after refactor and pass.

## Phase: churn-module-refactor
<!-- loopy:phase churn-module-refactor -->

- [ ] planning: select wave-1 modules meeting top-churn threshold — Acceptance: docs/architecture/refactor-wave-1.md lists selected modules with cumulative churn at or above 20%.
- [ ] refactor: align wave-1 module imports to layer rules — Acceptance: architecture checker reports zero boundary violations for each wave-1 module.
- [ ] refactor: remove circular dependencies in wave-1 modules — Acceptance: cycle detection output shows zero cycles in scoped wave-1 modules.
- [ ] compatibility: add deprecation shims for moved public symbols — Acceptance: existing API/CLI entry points remain functional and moved symbols emit documented deprecation warnings.
- [ ] tests: add regression tests for refactored module flows — Acceptance: critical flows in each wave-1 module are covered by tests that pass in CI.
- [ ] performance: verify scoped runtime regression stays within limit — Acceptance: benchmark comparison report shows no more than 5% runtime regression for scoped components.

## Phase: ci-architecture-gates
<!-- loopy:phase ci-architecture-gates -->

- [ ] tooling: configure architecture rule checker in repository — Acceptance: committed checker config encodes layer dependency and cycle rules from target architecture.
- [ ] ci: add architecture rule job to pull-request workflow — Acceptance: PR CI executes architecture checks and publishes actionable violation messages.
- [ ] ci: enforce fail-on-new-cycle policy — Acceptance: a synthetic cycle introduced in scoped code causes CI architecture job to fail.
- [ ] ci: enforce fail-on-boundary-violation policy — Acceptance: a synthetic disallowed import in scoped code causes CI architecture job to fail.
- [ ] developer-experience: add local architecture check command — Acceptance: one documented local command reproduces CI architecture check results.
- [ ] security: verify architecture job logs redact sensitive values — Acceptance: CI logs from architecture checks contain no matched secret patterns from existing secret scan rules.

## Phase: docs-rollout-metrics
<!-- loopy:phase docs-rollout-metrics -->

- [ ] docs: update architecture guide with boundaries and ownership — Acceptance: guide includes layers, allowed dependencies, and ownership table in plain-text-friendly format.
- [ ] docs: add contributor migration workflow for refactors — Acceptance: contributor guide includes step-by-step refactor procedure and reviewer checklist.
- [ ] docs: add API deprecation and changelog rules — Acceptance: changelog policy documents required deprecation path when public symbols move.
- [ ] metrics: add weekly architecture compliance report definition — Acceptance: report source records scoped-module architecture-check pass rate with 90% target.
- [ ] metrics: add weekly cycle-count trend report definition — Acceptance: report source records scoped-module cycle count and supports 4-week zero-cycle tracking.
- [ ] metrics: add PR cycle-time and regression-defect tracking definition — Acceptance: report source defines median PR cycle time and architecture-regression defect metrics for scoped modules.
