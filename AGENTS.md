# AI-assisted contributions in Spyro

For most AI-assisted contributions: declare that AI was used, for what, and which agent.
AI agents can be used to facilitate, but it should not be guiding your work or substituting you.

A human must submit the PR, understand every change, and answer reviewer questions themselves. Instead of relying on AI to answer reviewer questions.

AI should not be used to close issues labelled 'good first issue'. These issues are entry points for newcomers.

## Core Working Rules

* **Do Not Trust Memorized Firedrake API Shapes:** Firedrake changes over time. An LLM's trained
  knowledge reflects a snapshot that may already be stale.
* **Avoid Duplication:** Avoid unnecessary code duplication. Prefer reusing or extending nearby logic
  when it keeps behavior clear and local. Do not add speculative abstractions or broad refactors unless
  explicitly requested.

## Coding Style And Conventions

* **Class Attributes:** Every attribute a class can hold must be declared in one visible place, either
  initialized in the constructor (`__init__`) or, for state that is expensive or unnecessary to compute
  eagerly, declared as a `functools.cached_property`. Avoid discovering an attribute's existence via
  `hasattr`/`setattr`/`getattr` scattered across methods: laziness is fine, ad hoc laziness is not.
* **Docstrings Are Always `numpydoc`:** Every docstring you write or touch — public API, private helper,
  Cython function in `firedrake/cython/*.pyx`, test helper — must be `numpydoc`, using its section
  headings (`Parameters`, `Returns`, `Raises`, `Notes`).  Much of spyro predates the convention, so matching the neighbouring docstrings
  is precisely the wrong instinct. Give every parameter and every return value its
  `numpydoc` entry, however small the helper.
* **Type Hints:** New code should include type hints on function/method signatures.

## Testing Requirements

* **Pull Requests:** All PRs must include comprehensive tests demonstrating that the new feature works
  or the bug is fixed.

## Pull Request Expectations

* All changes are expected to arrive through GitHub Pull Requests.
* Keep diffs reviewable and focused.
* Before concluding work, ensure `make lint` passes, and verify that the relevant subset of the
  pytest test suite succeeds locally.


## Anti-Patterns

These must be avoided when writing code, and flagged when reviewing it.

### Branching On Discretization Or Execution State

### Using `hasattr` As A Setup Guard

WRONG — Using `hasattr` to infer whether one-time setup has already run, by probing for state that
setup is expected to have built:

```python
class KSPWrapper:
    def solve(self, pc, b, x):
        # Anti-pattern: hasattr stands in for "has `_ksp` been built yet?"
        if not hasattr(self, "_ksp"):
            self._ksp = PETSc.KSP().create(comm=pc.comm)
            self._ksp.setOperators(*pc.getOperators())
        self._ksp.solve(b, x)
```

RIGHT — Declare a boolean attribute that describes the state directly, and check that instead

# TODO: most of these topics come from the Firedrake repo and we need to ask them and cite it before merging in main

## AI-Assisted Resolution of Long-Standing Issues

This section applies to the initiative to address long-standing issues using AI assistance. It supplements, and does not override, the existing AI-assisted contribution rules above.

### Eligibility and Scope

* Only issues that have been open for at least one year are eligible.
* Only project researchers/devs that have a good understanding of the codebase may initiate work under this initiative.
* Issues labelled `good first issue` must not be selected.
* A postdoc, PhD candidate, or Developer must assess the issue's relevance, define the intended outcome, and establish clear acceptance criteria before implementation begins. Ideally a clear issue resolution plan should be added to the issue as a comment.
* Ambiguous, obsolete, or insufficiently specified issues must be clarified before implementation. AI agents must not invent requirements or independently expand the scope.
* Work should be undertaken periodically when spare AI credits or tokens are available, without displacing higher-priority project work.
* Each effort must target a single issue and produce a focused, reviewable proposed solution.

### Implementation and Validation

* Follow all coding conventions and testing requirements defined in this file.
* Investigate the current implementation and relevant documentation before proposing changes. Do not rely on memorized APIs, particularly for Firedrake.
* Add or update tests that demonstrate the reported problem and verify the proposed fix.
* Preserve existing behavior and public APIs unless a change is explicitly required and approved.
* Avoid unrelated refactoring, speculative abstractions, unnecessary dependencies, and modifications outside the issue's scope.
* Run `make lint` and the relevant subset of the pytest suite. Report the actual results and clearly identify any checks that could not be completed.
* Scientific and numerical changes require particular scrutiny of mathematical correctness, edge cases, and potential regressions.

### Local-Only Changes

During AI-assisted implementation:

* Do not create Git commits or push changes to any remote.
* Do not open a pull request or modify GitHub issues, comments, labels, or repository settings.
* Do not overwrite, discard, or incorporate unrelated pre-existing working-tree changes.
* Do not weaken or remove existing tests to make the implementation pass.
* Do not claim that tests or checks succeeded unless they were actually executed successfully.

All proposed changes must remain local for human review.

### Required Deliverables

Before handing the work back to the researcher, the AI agent must:

1. Implement the proposed solution to the open issue and add or update relevant tests and documentation.
2. Create an temporary untracked human readable `AI_PR_DESCRIPTION.md` file in the repository root containing a proposed PR description, a summary of the changes, the tests actually executed and their results, and any remaining concerns.
3. Provide a concise summary of the diagnosis, implementation, modified files, validation, and limitations.

The researcher must independently review the complete diff, understand every change, and verify the proposed solution before deciding whether to submit a pull request.

AI-generated code is a proposal, not an approved contribution. The normal contribution and pull-request review process remains mandatory. The researcher has to answer all comments on the PR.
