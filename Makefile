VERSION=$(shell python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")

.PHONY: default install-dev tag upload publish clean-pyc clean-root clean-data clean-output clean format lint

include makefiles/profiling.mk

default:
	@echo "\"make publish\"?"

install-dev:
	python3 -m pip install -e ".[dev]"

tag:

	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
	curl -H "Authorization: token `cat $(HOME)/.github-access-token`" -d '{"tag_name": "v$(VERSION)"}' https://api.github.com/repos/NDF-Poli-USP/spyro/releases

upload:
	# Make sure we're on the main branch
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi

	rm -rf dist/*
	python3 -m build
	twine upload dist/*

publish: tag upload

clean-pyc:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf build/*
	@rm -rf spyro.egg-info/
	@rm -rf dist/

clean-root:
	@rm -f *.msh *.vtk *.png *.vtu *.pvtu *.pvd *.npy *.pdf *.dat *.segy *.hdf5
	@rm -rf asn*/ bsn*/

clean-data:
	@rm -f shots/*.dat
	@rm -f *.txt
	@rm -rf property_fields/
	@rm -rf profilers/
	@rm -rf test_case/

clean-output:
	@rm -rf velocity_models/test*
	@rm -rf results/*
	@rm -rf control_*/ gradient*/ initial_velocity_model/ output*/ vp_end*/ test_debug*/

clean: clean-pyc clean-root clean-data clean-output

format:
	@if [ -n "$(FILE)" ]; then \
		black "$(FILE)"; \
		ruff check "$(FILE)" --fix; \
		black "$(FILE)"; \
		pydocstringformatter "$(FILE)"; \
		black "$(FILE)"; \
	else \
		black .; \
		ruff check . --fix; \
		black .; \
		pydocstringformatter .; \
		black .; \
	fi

lint:
	python3 -m flake8 spyro tests test_integration

