"""Scorer accepts the notation the content prompt naturally elicits.

The content prompt displays statements in quotes and refers to rules as "Rule k",
so models echo both. The scorer must read those back, not reject them -- otherwise
a correct answer scores 0 for notation alone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aspic_gym import create_dataset, score_answer


def _find(task, mode, level, needle_in_gold=None):
    ds = create_dataset(task, seed=20260721, size=40, level=level, with_content=(mode == "content"))
    for i in range(40):
        e = ds[i]
        if needle_in_gold is None or needle_in_gold in e["answer"]:
            return e
    raise AssertionError("no matching item")


def test_quoted_statements_match():
    """[premise: "X"] must score the same as [premise: X]; the prompt quotes X."""
    e = _find("evidence_construction", "content", 1)
    gold = e["answer"]
    # wrap every statement inside a directive in quotes, as the model does
    import re
    quoted = re.sub(r"(\[(?:premise|axiom): )([^\]]+)\]", r'\1"\2"]', gold)
    assert quoted != gold  # sanity: we actually changed something
    assert score_answer(quoted, e) == score_answer(gold, e) == 1.0


def test_labeled_added_rule_resolves():
    """A model that labels its added rule and names it in a preference should not
    be rejected for the label."""
    # construct a minimal content theory by hand-checking parse acceptance:
    # gold-style unlabelled vs model-style labelled must parse to the same ops.
    from aspic_gym import _content_to_ops
    # atoms: p (pos/neg glosses), q
    class A(dict): pass
    atoms = {"a1": {"pos": "it rains", "neg": "it does not rain"},
             "a2": {"pos": "the ground is wet", "neg": "the ground is dry"}}
    base = []
    unlab = _content_to_ops("[defeasible: it rains => the ground is wet]", atoms, base)
    lab = _content_to_ops("[defeasible Rule 1: it rains => the ground is wet]", atoms, base)
    assert unlab is not None and lab is not None
    assert unlab[0].antecedents == lab[0].antecedents
    assert unlab[0].consequent == lab[0].consequent
