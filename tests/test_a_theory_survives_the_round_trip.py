"""A serialized theory rebuilds into the theory it came from.

`docs/dataset-contract.md` section 2: a frozen row carries the theory so a
stored generation can be scored without the generator that produced it. That
only holds if the round trip is exact, and the one way it can silently fail is
the tuple in `Operation.antecedents` -- a list survives JSON, compares unequal,
and raises nothing.
"""
import json

import pytest

from arggym.aspic.engine import Operation
from arggym.core.serialize import (THEORY_SCHEMA, check_schema, op_from_dict,
                                   op_to_dict, ops_from_json, ops_to_json)

EVERY_KIND = [
    Operation(kind="premise", content="p"),
    Operation(kind="axiom", content="q"),
    Operation(kind="defeasible", name="r1", antecedents=("p", "q"), consequent="s"),
    Operation(kind="strict", name="r2", antecedents=("s",), consequent="t"),
    Operation(kind="prefer_rule", stronger="r1", weaker="r2"),
    Operation(kind="prefer_premise", stronger="p", weaker="q"),
]


@pytest.mark.parametrize("op", EVERY_KIND, ids=lambda o: o.kind)
def test_every_kind_comes_back_equal(op):
    assert op_from_dict(op_to_dict(op)) == op


def test_antecedents_come_back_as_a_tuple():
    # The failure this guards is silent: Operation is frozen, so a list here
    # compares unequal to the tuple it was built from without raising, and a
    # round-tripped theory would simply stop matching answers.
    back = op_from_dict(op_to_dict(EVERY_KIND[2]))
    assert isinstance(back.antecedents, tuple)


def test_a_theory_survives_being_written_to_a_file():
    text = json.dumps(ops_to_json(EVERY_KIND))
    assert ops_from_json(json.loads(text)) == EVERY_KIND


def test_the_row_carries_only_the_fields_its_kind_uses():
    # A premise has no antecedents and a preference has no consequent. Writing
    # the unused fields would put six nulls on every line of every taskset.
    assert op_to_dict(EVERY_KIND[0]) == {"kind": "premise", "content": "p"}
    assert op_to_dict(EVERY_KIND[4]) == {"kind": "prefer_rule",
                                         "stronger": "r1", "weaker": "r2"}


def test_an_unknown_kind_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        op_to_dict(Operation(kind="wishful", content="x"))
    with pytest.raises(ValueError):
        op_from_dict({"kind": "wishful", "content": "x"})


def test_a_theory_from_a_different_schema_is_refused():
    # Adding a field to Operation would make an old row deserialize with a
    # default and mean something new. Refusing is the only safe reading.
    check_schema(THEORY_SCHEMA)
    with pytest.raises(ValueError):
        check_schema(THEORY_SCHEMA + 1)
