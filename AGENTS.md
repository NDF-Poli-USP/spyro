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
