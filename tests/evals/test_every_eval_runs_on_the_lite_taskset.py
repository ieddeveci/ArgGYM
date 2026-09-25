"""Every eval runs on `data/taskset-lite.jsonl`.

120 items per task are enough for a per-task ranking, at a fifth of the
standard taskset's cost, so the config default and every entry point that names
a taskset for an eval name lite. The standard file still ships and `make freeze`
still builds it; an eval that silently picked it up would be five times the
bill and would not pool with the rest of the sweep.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
LITE = "data/taskset-lite.jsonl"
STANDARD = "data/taskset.jsonl"

#: Code that runs or scores an eval. A string literal naming the standard file
#: here is a default, an override or a usage line that points an eval at it.
PYTHON = sorted([*(ROOT / "evals").glob("*.py"), *(ROOT / "hpc" / "vllm").glob("*.py"),
                 *(ROOT / "scripts").glob("*.py")])
SHELL = sorted([*(ROOT / "hpc" / "vllm").glob("*.sh"),
                *(ROOT / "hpc" / "vllm").glob("*.sbatch"), ROOT / "Makefile"])


def test_the_config_default_is_lite():
    cfg = yaml.safe_load((ROOT / "evals" / "conf" / "config.yaml").read_text())
    assert cfg["taskset"] == LITE


@pytest.mark.parametrize("path", PYTHON, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_eval_code_names_the_standard_taskset(path):
    # String literals only, and not docstrings: a comment may cite the standard
    # file as an upper bound on prompt length, which lite, a subset of it,
    # respects, and a docstring may show how to re-score a run generated on it.
    tree = ast.parse(path.read_text())
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                       ast.AsyncFunctionDef))
                  and ast.get_docstring(node, clean=False) is not None}
    # Nor a path mapped to the make target that freezes it (evals/run.py's
    # FREEZE_TARGETS): that says how to rebuild the file, not to evaluate on it.
    freeze_keys = {id(k) for d in ast.walk(tree) if isinstance(d, ast.Dict)
                   for k, v in zip(d.keys, d.values)
                   if isinstance(v, ast.Constant) and str(v.value).startswith("make freeze")}
    bad = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and STANDARD in n.value and id(n) not in docstrings | freeze_keys]
    assert not bad, f"{path.name} names {STANDARD} at lines {bad}; evals run on {LITE}"


@pytest.mark.parametrize("path", SHELL, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_eval_command_overrides_the_taskset_to_standard(path):
    bad = [i for i, line in enumerate(path.read_text().splitlines(), 1)
           if re.search(r"taskset[=\s]+\S*" + re.escape(STANDARD), line)]
    assert not bad, f"{path.name} runs an eval on {STANDARD} at lines {bad}"


def test_make_eval_runs_on_lite():
    assert f"taskset={LITE}" in (ROOT / "Makefile").read_text()
