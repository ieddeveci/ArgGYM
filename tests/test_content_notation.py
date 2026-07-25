"""Scorer accepts the notation the content prompt naturally elicits.

The content prompt displays statements in quotes and refers to rules as "Rule k",
so models echo both. The scorer must read those back, not reject them -- otherwise
a correct answer scores 0 for notation alone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aspic_gym import create_dataset, score_content


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
    assert score_content(quoted, e) == score_content(gold, e) == 1.0


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


def test_lowercase_and_in_statement_not_split():
    """A single antecedent statement containing the word 'and' must parse as one
    condition, not be fragmented at the word into unresolvable pieces -> 0."""
    from aspic_gym import _content_to_ops
    atoms = {"a1": {"pos": "information asymmetry exists between buyers and sellers",
                    "neg": "no information asymmetry exists between buyers and sellers"},
             "a2": {"pos": "the market clears efficiently",
                    "neg": "the market does not clear efficiently"}}
    ops = _content_to_ops(
        "[defeasible: information asymmetry exists between buyers and sellers "
        "=> the market clears efficiently]", atoms, [])
    assert ops is not None                       # was None: split at ' and '
    assert ops[0].antecedents == ("a1",)         # one condition, not fragmented
    assert ops[0].consequent == "a2"


def test_uppercase_AND_still_joins_conditions():
    """The uppercase joiner must still split two conditions -- even when one of
    them itself contains a lowercase 'and'."""
    from aspic_gym import _content_to_ops
    atoms = {"a1": {"pos": "it rains", "neg": "it does not rain"},
             "a2": {"pos": "the ground is wet and slippery",
                    "neg": "the ground is not wet and slippery"},
             "a3": {"pos": "the road floods", "neg": "the road does not flood"}}
    ops = _content_to_ops(
        "[defeasible: it rains AND the ground is wet and slippery => the road floods]",
        atoms, [])
    assert ops is not None
    assert set(ops[0].antecedents) == {"a1", "a2"}   # split on AND, not on inner 'and'
    assert ops[0].consequent == "a3"


def test_lowercase_and_joins_conditions():
    """Content mode displays rules as "if A and B, then C", so a model that echoes
    that phrasing writes the joiner in lowercase. That must score the same as the
    uppercase gold -- otherwise the benchmark penalises its own phrasing."""
    from aspic_gym import _content_to_ops
    atoms = {"a1": {"pos": "it rains", "neg": "it does not rain"},
             "a2": {"pos": "the drain is blocked", "neg": "the drain is clear"},
             "a3": {"pos": "the road floods", "neg": "the road does not flood"}}
    ops = _content_to_ops(
        "[defeasible: it rains and the drain is blocked => the road floods]", atoms, [])
    assert ops is not None
    assert set(ops[0].antecedents) == {"a1", "a2"}
    assert ops[0].consequent == "a3"


def test_lowercase_joiner_keeps_statements_with_and_whole():
    """The lowercase joiner and a statement containing the word must coexist: a
    score that depended on whether a statement happens to contain 'and' would be a
    content-correlated bias in a benchmark that compares content to symbolic."""
    from aspic_gym import _content_to_ops
    atoms = {"a1": {"pos": "it rains", "neg": "it does not rain"},
             "a2": {"pos": "the ground is wet and slippery",
                    "neg": "the ground is not wet and slippery"},
             "a3": {"pos": "the road floods", "neg": "the road does not flood"}}
    ops = _content_to_ops(
        "[defeasible: it rains and the ground is wet and slippery => the road floods]",
        atoms, [])
    assert ops is not None
    assert set(ops[0].antecedents) == {"a1", "a2"}   # two conditions, a2 kept whole
    assert ops[0].consequent == "a3"


def test_content_score_is_joiner_case_invariant():
    """End to end: rewriting a gold's ' AND ' joiners to ' and ' must not change
    its score on any content-mode item."""
    ds = create_dataset("formalization", seed=20260721, size=25, level=5, with_content=True)
    checked = 0
    for i in range(25):
        e = ds[i]
        gold = e["answer"]
        if " AND " not in gold:
            continue
        checked += 1
        assert score_content(gold, e) == 1.0
        assert score_content(gold.replace(" AND ", " and "), e) == 1.0
    assert checked, "no multi-antecedent gold in the sample"


def test_formalization_negation_quoted_affirmative():
    """The formalization prompt now instructs models to write a negated statement
    as -"<affirmative>". That quoted form must score the same as the unquoted gold
    -<affirmative> (scorer strips quotes; the leading '-' resolves the contrary)."""
    import re
    e = None
    for lvl in (5, 3):
        try:
            e = _find("formalization", "content", lvl, needle_in_gold="=> -")
            break
        except AssertionError:
            continue
    assert e is not None, "no formalization/content item with a negated consequent found"
    gold = e["answer"]
    quoted = re.sub(r'(=>\s*)-([^\]\n]+)', r'\1-"\2"', gold)   # '=> -X' -> '=> -"X"'
    assert quoted != gold
    assert score_content(quoted, e) == score_content(gold, e) == 1.0


def test_formalization_symbolic_rule_numbering():
    """Symbolic formalization: a model may name rules 'Rule k' (the prompt's
    numbering) instead of dk/sk. That must score the same as the dk form."""
    import re
    e = _find("formalization", "symbolic", 3, needle_in_gold="prefer_rule: d")
    gold = e["answer"]
    # rewrite dK -> Rule K only inside preference directives (rule references)
    rulek = re.sub(r"\b d(\d+) ".replace(" ", ""), lambda m: "Rule " + m.group(1), gold)
    assert "Rule " in rulek and rulek != gold
    assert score_content(rulek, e) == score_content(gold, e) == 1.0
