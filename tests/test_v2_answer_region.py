"""Every v2 answer parser must read the LAST complete answer region.

A reasoning model routinely drafts an answer mid-chain-of-thought and then
revises it. What it submits is the revision, so a parser that takes the first
region reports the model wrong on work it had already corrected -- and does so
selectively for the models that think longest, which is exactly the population
under comparison.
"""
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parent.parent / "ArgGYM_v2"


def _load(mod_path: str, name: str):
    """Load a v2 module by file path.

    v1 and v2 both have top-level `core` and `tasks` packages, so a plain import
    resolves to whichever tree is first on sys.path -- which in a full suite run
    depends on what another test imported. Loading by path removes the ambiguity.
    V2 still goes on sys.path so the module's own `aspic.*` imports resolve.
    """
    import importlib.util
    if str(V2) not in sys.path:
        sys.path.insert(0, str(V2))
    spec = importlib.util.spec_from_file_location(name, V2 / mod_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_scoring = _load("core/scoring.py", "_v2_core_scoring")
answer_region, parse_answer = _scoring.answer_region, _scoring.parse_answer


def test_takes_last_complete_region():
    txt = ("[answer]\nfirst: justified\n[/answer]\n"
           "on reflection that was wrong\n"
           "[answer]\nsecond: overruled\n[/answer]")
    assert "second" in answer_region(txt)
    assert "first" not in answer_region(txt)


def test_single_region_is_unchanged():
    assert answer_region("[answer]\na1: justified\n[/answer]").strip() == "a1: justified"


def test_unclosed_final_region_falls_back_to_the_last_complete_one():
    """A truncated generation can leave a dangling '[answer]' with no closer.
    The last *complete* region is still the model's best submitted answer."""
    txt = "[answer]\ngood: justified\n[/answer]\nrethinking...\n[answer]\ntrunc"
    assert "good" in answer_region(txt)


def test_no_region_is_none_not_empty():
    """None and '' must stay distinguishable: no region is a format failure,
    an empty region is a submitted-but-blank answer."""
    assert answer_region("I forgot the tags") is None
    assert answer_region("") is None
    assert answer_region(None) is None
    assert answer_region("[answer][/answer]") == ""


def test_whitespace_and_case_tolerated():
    assert answer_region("[ ANSWER ]\nx: justified\n[ / answer ]") is not None


def test_parse_answer_scores_the_revision():
    txt = ("[answer]\n[premise: -a1]\n[/answer]\n"
           "wait, wrong literal\n"
           "[answer]\n[premise: -b2]\n[/answer]")
    ops = parse_answer(txt).ops
    assert [o.content for o in ops] == ["-b2"]


@pytest.mark.parametrize("name", [
    "status_query", "perturbation", "claim_chain", "defeat_diagnosis",
    "formalization",
])
def test_every_task_parser_uses_the_shared_region(name):
    """Guards against a future copy-paste of the first-region regex.

    Read by path, not by import: v1 and v2 both define a top-level `tasks`
    package, so whichever is imported first in a suite run wins and an
    import-based check would silently inspect the wrong tree.
    """
    src = (V2 / "tasks" / f"{name}.py").read_text()
    assert "answer_region" in src, f"v2 {name} does not use the shared region reader"
    assert "_ANSWER = re.compile" not in src, f"v2 {name} reintroduced a local region regex"
