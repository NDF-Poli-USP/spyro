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

clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf build/*
	@rm -rf spyro.egg-info/
	@rm -rf dist/

format:
	isort -rc spyro/ test/*.py
	black spyro/ test/*.py

black:
	black .

lint:
	isort --check . spyro/ setup.py test/*.py
	black --check spyro/ setup.py test/*.py
	flake8 setup.py spyro/ test/*.py
