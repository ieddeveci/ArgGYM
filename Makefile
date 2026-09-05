.PHONY: help setup test test-all freeze eval inspect clean

help:
	@echo 'make setup     install the package and its dev dependencies'
	@echo 'make test      run the test suite'
	@echo 'make test-all  run the test suite including the slow end-to-end grid'
	@echo 'make freeze    freeze the standard taskset into data/taskset.jsonl'
	@echo 'make eval      run + score one model (MODEL=..., default stub)'
	@echo 'make inspect   start the inspector on http://127.0.0.1:5000'
	@echo 'make clean     remove build artefacts and caches (leaves data/)'

setup:
	uv sync

test:
	uv run pytest -q

test-all:
	uv run pytest -q -m "" -n auto

freeze:
	uv run arggym freeze -c tasksets/standard.yaml -o data/taskset.jsonl

# `make eval MODEL=claude-openrouter`. Scoring is a second step on purpose, so
# it can be rerun against these generations whenever the scorer moves.
MODEL ?= stub
eval:
	uv run python -m evals.run taskset=data/taskset.jsonl model=$(MODEL)
	uv run python -m evals.score $$(ls -td outputs/runs/* | head -1)

inspect:
	uv run arggym inspect

clean:
	rm -rf dist .pytest_cache *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
