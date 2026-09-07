"""A reversed preference must cost structural credit.

`[prefer_rule: a > b]` and `[prefer_rule: b > a]` are different directives, so the shape metric has
to separate them - it used to compare preferences by kind alone and scored a flipped answer 1.0.

The cell is an exhibit, not a sample: the flip below needs a rule preference and a premise
preference in one reference, and level 15 is the only exported level where every ordering and seed
carries both. Level 12 carries both at six of its eight cells, so a cell there would pass or raise
depending on which one it was.
"""
from __future__ import annotations

import pytest

from arggym.aspic.engine import Operation
from arggym.tasks import formalization

PREFS = {"prefer_rule", "prefer_premise"}


def render(ops):
    return "\n".join(formalization.render_op(o) for o in ops)


@pytest.fixture(scope="module")
def item():
    it = formalization.make_item(15, 0, formalization.WEAKEST_LINK)
    assert it is not None, "no item generated"
    kinds = {o.kind for o in it.reference_ops} & PREFS
    assert kinds == PREFS, \
        f"the flip is parametrized over both kinds and this item carries {kinds} (#118)"
    return it


def test_gold_answer_still_scores_one(item):
    result = formalization.score(render(item.reference_ops), item)
    assert result.score == pytest.approx(1.0), result.reason
    assert result.diagnostics["shape_f1"] == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["prefer_rule", "prefer_premise"])
def test_one_flipped_preference_lowers_shape_f1(item, kind):
    ops = list(item.reference_ops)
    i = next(j for j, o in enumerate(ops) if o.kind == kind)
    o = ops[i]
    ops[i] = Operation(kind=o.kind, stronger=o.weaker, weaker=o.stronger)

    result = formalization.score(render(ops), item)
    assert result.diagnostics["shape_f1"] < 1.0, f"flipping {formalization.render_op(o)} went unnoticed"
    assert result.diagnostics["directives_f1"] < 1.0
