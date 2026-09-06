.PHONY: help setup test test-all freeze eval inspect clean

help:
	@echo 'make setup     install the package and its dev dependencies'
	@echo 'make test      run the test suite'
	@echo 'make test-all  run the test suite including the slow end-to-end grid'
	@echo 'make freeze    freeze the standard taskset into data/taskset.jsonl'
	@echo 'make eval      run + score one model (MODEL=..., TEMPLATE=..., ELICITATION=...)'
	@echo 'make inspect   start the inspector on http://127.0.0.1:5000'
	@echo 'make clean     remove build artefacts and caches (leaves data/)'

setup:
	uv sync

# `-n auto` takes the logical core count, so this adapts from a laptop to the
# 128-thread box without a number to keep in sync. Bare `pytest` stays serial
# on purpose: xdist breaks pdb and makes `-x` behave oddly, and debugging one
# test is what a bare invocation is for.
test:
	uv run pytest -q -n auto

test-all:
	uv run pytest -q -m "" -n auto

freeze:
	uv run arggym freeze -c tasksets/standard.yaml -o data/taskset.jsonl

# `make eval MODEL=claude-openrouter`. Scoring is a second step on purpose, so
# it can be rerun against these generations whenever the scorer moves.
#
# The run directory is derived from the config rather than read back off the
# filesystem: picking the most recently modified directory scores the wrong run
# whenever two of these are in flight. It is `run_id` from
# `evals/conf/config.yaml` spelled out, so both halves have to be told the same
# three things -- hardcoding the template and the elicitation here made
# `make eval ELICITATION=cot` generate one directory and score another that
# does not exist.
MODEL ?=
TEMPLATE ?= xml_tags
ELICITATION ?= none
eval:
	@test -n "$(MODEL)" || { echo 'usage: make eval MODEL=<a config under evals/conf/model>'; \
	  echo 'available:'; ls evals/conf/model | sed 's/.yaml$$/  /;s/^/  /'; exit 1; }
	uv run python -m evals.run taskset=data/taskset.jsonl model=$(MODEL) \
	  template=$(TEMPLATE) elicitation=$(ELICITATION)
	uv run python -m evals.score outputs/runs/$(MODEL)__$(TEMPLATE)__$(ELICITATION)

inspect:
	uv run arggym inspect

clean:
	rm -rf dist .pytest_cache *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
