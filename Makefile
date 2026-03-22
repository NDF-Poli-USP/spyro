VERSION=$(shell python3 -c "from configparser import ConfigParser; p = ConfigParser(); p.read('setup.cfg'); print(p['metadata']['version'])")

default:
	@echo "\"make publish\"?"

tag:

	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
	curl -H "Authorization: token `cat $(HOME)/.github-access-token`" -d '{"tag_name": "v$(VERSION)"}' https://api.github.com/repos/krober10nd/spyro/releases

upload:
	# Make sure we're on the main branch
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi

	rm -rf dist/*
	python3 setup.py sdist bdist_wheel
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

clean-output:
	@rm -rf velocity_models/test*
	@rm -rf results/*
	@rm -rf control_*/ gradient*/ initial_velocity_model/ output*/ vp_end*/ test_debug*/

clean: clean-pyc clean-root clean-data clean-output

format:
	autopep8 --in-place --global-config setup.cfg --recursive .

lint:
	flake8 setup.py spyro/ tests/integration/*.py tests/on_one_core/*.py
