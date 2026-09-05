"""A prompt may not state a rule the row gives the scorer no way to apply.

The bloat rule is the case that motivated this. `score_item` applies it only
`if minimum`, and every construction prompt states it flatly. All six generators
do set `min_directives`, so every shipped item was fine -- but that was a
property of six generators rather than something checked, and nothing would have
noticed a seventh task, or a level, that stopped setting it (#21).

`freeze` is the right place for the check: it is the step that decides what
ships, and it already refuses a cell whose reference does not score 1.0.
"""
from __future__ import annotations

import pytest

from arggym.core.freeze import PromptClaimUnsupported, freeze
from arggym.core.prompting import MINIMALITY
from arggym.core.spec import SeedPolicy, TasksetSpec

CHEAP = dict(levels=(3,), orderings=("last_link_elitist",), seeds=SeedPolicy(take=1))


def spec(task):
    return TasksetSpec(tasks=(task,), **CHEAP)


def test_a_construction_question_states_the_rule_and_carries_the_minimum(tmp_path):
    out = tmp_path / "t.jsonl"
    freeze(spec("preference_construction"), str(out), verbose=False)
    rows = [r for r in _rows(out)]
    assert rows and all(MINIMALITY in r["question"] for r in rows)
    assert all(r["metadata"]["gold"]["min_directives"] >= 1 for r in rows)


def test_a_query_question_makes_no_such_promise(tmp_path):
    """The rule is not universal, so the check must not demand it everywhere."""
    out = tmp_path / "t.jsonl"
    freeze(spec("status_query"), str(out), verbose=False)
    rows = list(_rows(out))
    assert rows and not any(MINIMALITY in r["question"] for r in rows)


def test_the_export_refuses_a_promise_the_row_cannot_keep(tmp_path, monkeypatch):
    """The failure the invariant exists for: the sentence stays, the value goes."""
    import arggym.core.freeze as fz

    real = fz.fill_cell

    def strip(task, level, ordering, spec):
        got, report = real(task, level, ordering, spec)
        for r in got:
            r["metadata"]["gold"]["min_directives"] = None
        return got, report

    monkeypatch.setattr(fz, "fill_cell", strip)
    with pytest.raises(PromptClaimUnsupported, match="min_directives"):
        freeze(spec("preference_construction"), str(tmp_path / "t.jsonl"), verbose=False)


def _rows(path):
    import json
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if "question" in d:
            yield d
