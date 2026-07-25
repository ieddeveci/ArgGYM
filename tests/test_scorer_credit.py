"""A correct answer must not lose credit for something the task never forbade.

Each case here is a scorer that read more into the answer text than the task's
answer spec asked for, and docked or zeroed a correct submission because of it.
The benchmark under-reports ability when that happens, which is worse than
scoring generously: the error correlates with how a model phrases itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aspic_gym import create_dataset, score_content  # noqa: E402


def _first(task, level, seed=2, **want):
    ds = create_dataset(task, size=20, level=level, seed=seed,
                        with_content=want.pop("with_content", False))
    for i in range(len(ds)):
        md = ds[i]["metadata"]
        if all(md.get(k) == v for k, v in want.items()):
            return ds[i]
    raise AssertionError(f"no {task} item matching {want}")


def test_robustness_tolerates_prose_that_names_no_atom():
    """Atom names were matched unanchored, so "Rule2" yielded the atom "e2" and
    "claim3" yielded "m3" -- phantom tokens that the scorer then treated as a
    contract violation and zeroed the answer for."""
    e = _first("robustness", level=2, variant="single", mode="symbolic")
    gold = e["answer"]
    assert score_content(gold, e) == 1.0
    assert score_content(gold + " -- it feeds Rule2 and claim3", e) == 1.0


def test_preference_economy_counts_only_the_preferences():
    """The economy score divides by the set it minimised -- the preferences the
    answer added. A restated premise is not a preference and must not dilute it."""
    e = _first("preference_construction", level=6, seed=5)
    gold = e["answer"]
    restated = next(o for o in e["metadata"]["ops"] if o["kind"] == "premise")
    noisy = gold + f"\n[premise: {restated['content']}]"
    assert score_content(gold, e) == 1.0
    assert score_content(noisy, e) == 1.0


def test_status_query_self_contradiction_loses_the_claim():
    """Assigning one claim two different statuses earns no credit for it, however
    the second assignment is phrased -- by claim number or by naming the claim."""
    e = _first("status_query", level=4, seed=7)
    md = e["metadata"]
    gold = e["answer"]
    assert score_content(gold, e) == 1.0

    flip = "overruled" if md["gold_statuses"][0] != "OVERRULED" else "justified"
    per_claim = 1.0 / len(md["queries"])
    by_number = score_content(gold + f"\n1: {flip}", e)
    by_name = score_content(gold + f"\n{md['queries'][0]}: {flip}", e)
    assert by_number == by_name == 1.0 - per_claim


def test_one_line_answer_is_not_read_as_self_contradiction():
    """Several claims on one line is a formatting choice, not a contradiction."""
    e = _first("status_query", level=4, seed=7)
    gold = e["answer"]
    assert score_content(", ".join(gold.splitlines()), e) == 1.0
