# Единая команда проверки: make check  (== python check.py). Под Windows без make: python check.py
PY ?= python

.PHONY: check quick test demo baselines trajectories docker-build docker-check

check:
	$(PY) check.py

quick:
	$(PY) check.py --quick

test:
	$(PY) -m unittest discover -s tests/generator -p "test_gen_*.py"
	$(PY) -m unittest discover -s tests/env -p "test_env_*.py"
	$(PY) -m unittest discover -s tests/grader -p "test_grader_*.py"

demo:
	$(PY) -m env.demo

baselines:
	$(PY) -m env.runner --splits train,dev --mode eval --report-md reports/baselines_latest.md

trajectories:
	$(PY) check.py --quick --skip-tests --update

docker-build:
	docker build -t audit-env .

docker-check: docker-build
	docker run --rm --network none --cpus 2 --memory 4g audit-env
