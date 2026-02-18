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
