"""`formalization` succeeds only when the theory behaves like the reference on every literal it names.

The question prints the statuses a correct formalization gives its queried literals. Matching
those alone is reachable without reading the text: one or two directives per printed status
reproduce them (#190). And a model that writes an undercut as a rebuttal usually matches them
too, while its theory makes the rebutting literal justified where the reference has no
argument for it at all. Success compares the status of every atom the reference names, both
polarities, UNSATISFIABLE included.
"""
from __future__ import annotations

import re

import pytest

from arggym.aspic.engine import Operation
from arggym.tasks import formalization

ORDERINGS = ["last_link_elitist", "last_link_democratic",
             "weakest_link_elitist", "weakest_link_democratic"]

STATUS_TEMPLATE = {
    "justified": lambda b: [f"[premise: {b}]"],
    "overruled": lambda b: [f"[premise: {b}]", f"[axiom: -{b}]"],
    "undecided": lambda b: [f"[premise: {b}]", f"[premise: -{b}]"],
}


def render(ops):
    return "\n".join(formalization.render_op(o) for o in ops)


def restate_the_question(item) -> str:
    """Directives built from the question's printed statuses and nothing else."""
    m = re.search(r"Under a correct formalization: (.*?)\.\n", item.prompt)
    lines = []
    for part in m.group(1).split("; "):
        lit, status = re.fullmatch(r"(-?\w+) is (\w+)", part).groups()
        lines += STATUS_TEMPLATE[status](lit)
    return "\n".join(lines)


def rename_rules(ops):
    names = {o.name: f"zz{i}" for i, o in enumerate(ops) if o.kind in ("defeasible", "strict")}

    def lit(x):
        neg = x.startswith("-")
        bare = x[1:] if neg else x
        return ("-" if neg else "") + names.get(bare, bare)

    out = []
    for o in ops:
        if o.kind in ("defeasible", "strict"):
            out.append(Operation(kind=o.kind, name=names[o.name],
                                 antecedents=tuple(lit(a) for a in o.antecedents),
                                 consequent=lit(o.consequent)))
        elif o.kind == "prefer_rule":
            out.append(Operation(kind=o.kind, stronger=names[o.stronger], weaker=names[o.weaker]))
        else:
            out.append(o)
    return out


def undercuts_as_rebuttals(ops):
    """`[defeasible u: c => -r]` written as `[defeasible u: c => -<conclusion of r>]`."""
    rules = {o.name: o for o in ops if o.kind in ("defeasible", "strict")}
    out = []
    for o in ops:
        if o.kind in ("defeasible", "strict") and o.consequent.startswith("-") \
                and o.consequent[1:] in rules:
            target = rules[o.consequent[1:]].consequent
            negated = target[1:] if target.startswith("-") else "-" + target
            out.append(Operation(kind=o.kind, name=o.name, antecedents=o.antecedents,
                                 consequent=negated))
        else:
            out.append(o)
    return out


@pytest.mark.parametrize("level", [1, 3, 6, 9, 12, 15])
@pytest.mark.parametrize("ordering", ORDERINGS)
def test_restating_the_printed_statuses_is_not_a_success(level, ordering):
    item = formalization.make_item(level, 0, ordering)
    result = formalization.score(restate_the_question(item), item)
    assert result.diagnostics["behavioural"] == pytest.approx(1.0), \
        "the template should reproduce every queried status; the test is not exercising #190"
    assert result.success is False, \
        f"an answer built from the question's printed statuses succeeded ({result.score})"
    assert result.diagnostics["mismatched_literals"]


@pytest.mark.parametrize("level", [1, 3, 6, 9, 12, 15])
@pytest.mark.parametrize("ordering", ORDERINGS)
def test_reference_with_renamed_rules_succeeds_at_one(level, ordering):
    item = formalization.make_item(level, 0, ordering)
    result = formalization.score(render(rename_rules(item.reference_ops)), item)
    assert result.success is True
    assert result.score == pytest.approx(1.0)
    assert not result.diagnostics.get("mismatched_literals")


# Cells whose reference carries an undercut and whose rebuttal rewrite keeps every
# queried status, so only the literals the question does not print can tell them apart.
@pytest.mark.parametrize("level,ordering", [
    (1, "last_link_elitist"),
    (3, "last_link_elitist"),
    (6, "last_link_democratic"),
    (12, "weakest_link_elitist"),
])
def test_an_undercut_written_as_a_rebuttal_is_not_a_success(level, ordering):
    item = formalization.make_item(level, 0, ordering)
    rewritten = undercuts_as_rebuttals(item.reference_ops)
    assert rewritten != list(item.reference_ops), "this cell's reference has no undercut"
    result = formalization.score(render(rewritten), item)
    assert result.diagnostics["behavioural"] == pytest.approx(1.0), \
        "the rewrite moved a queried status, so the queried check alone would catch it"
    assert result.success is False
    assert result.diagnostics["mismatched_literals"]
