.PHONY: help setup test test-all export inspect clean

help:
	@echo 'make setup     install the package and its dev dependencies'
	@echo 'make test      run the test suite'
	@echo 'make test-all  run the test suite including the slow end-to-end grid'
	@echo 'make export    generate every task as JSONL into data/'
	@echo 'make inspect   start the inspector on http://127.0.0.1:5000'
	@echo 'make clean     remove build artefacts and caches (leaves data/)'

setup:
	uv sync

test:
	uv run pytest -q

test-all:
	uv run pytest -q -m "" -n auto

export:
	uv run arggym export-all

inspect:
	uv run arggym inspect

clean:
	rm -rf dist .pytest_cache *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
