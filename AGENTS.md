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
