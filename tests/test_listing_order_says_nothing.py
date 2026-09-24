"""Where a directive is listed says nothing about the answer.

A generator lists a theory as facts, then rules, then preferences. Each block has to
be shuffled, or its order is the order the generator built it in, and that order
tracks what each directive is for. Before #208 the preferences kept build order, so the
first listed preference of a `perturbation` theory named the rule the perturbation
flips, and the last listed one in `counter_argument` from level 9 was the decoy the
reference prefers. `semantics_query` appended its padding after the preferences, and
every padding claim is justified.

The test plays each of those shortcuts on a few cells: pick the first or the last
listed item and check how often it is the gold one, against the rate a uniformly
random pick among the same items gets on the same rows. A shuffled listing gives the
random rate; the shortcuts above were right on 85 to 100 percent of rows where a
random pick was right on 20 to 45.
"""
import re

import pytest

import arggym

PREFS = ("prefer_rule", "prefer_premise")
RULES = ("defeasible", "strict")
#: The two elitist orderings: the democratic ones cost 50 to 100 times as much to build
#: at the top levels of `counter_argument`, and the listing is ordering-blind.
ORDERINGS = ("last_link_elitist", "weakest_link_elitist")
SEEDS = 5

#: How far above the random rate a position may land before it counts as a tell. The
#: sample is 40 rows a task (120 for `semantics_query`); the shortcuts this guards against
#: sat 0.3 to 0.8 above it, a shuffled listing sits within about 0.1 of it.
MARGIN = 0.2

_TOK = re.compile(r"-?[A-Za-z_][A-Za-z_0-9]*")


def _perturbation(row):
    """Base preferences; gold if it names a rule whose conclusion changes status."""
    ops = row["metadata"]["base_ops"]
    changed = row["metadata"]["gold"]["gold"]
    names = {o["name"] for o in ops if o["kind"] in RULES
             and (o["consequent"] in changed or "-" + o["consequent"] in changed)}
    return [o["stronger"] in names or o["weaker"] in names
            for o in ops if o["kind"] in PREFS]


def _counter_argument(row):
    """Preferences; gold if it names a rule or literal the reference writes."""
    toks = set(_TOK.findall(row["reference_answer"]))
    toks |= {t.lstrip("-") for t in toks}
    return [o["stronger"] in toks or o["weaker"] in toks
            for o in row["metadata"]["base_ops"] if o["kind"] in PREFS]


def _semantics_query(row):
    """Every listed rule whose conclusion is asked; gold if justified under all it is asked."""
    asked = {}
    for claim, _sem, status in row["metadata"]["gold"]["gold"]:
        asked.setdefault(claim, set()).add(status)
    return [asked[o["consequent"]] == {"JUSTIFIED"}
            for o in row["metadata"]["base_ops"]
            if o["kind"] in RULES and o["consequent"] in asked]


#: task -> (levels sampled, the items a position picks from, marked gold or not)
CHECKS = {
    "perturbation": (range(6, 16, 3), _perturbation),
    "counter_argument": (range(9, 16, 2), _counter_argument),
    "semantics_query": (range(4, 16), _semantics_query),
}


def position_rates(rows, items_of):
    """(rows used, first-listed hits, last-listed hits, expected hits of a random pick)."""
    n = first = last = rand = 0
    for row in rows:
        hits = items_of(row)
        if len(hits) < 2:
            continue
        n += 1
        first += hits[0]
        last += hits[-1]
        rand += sum(hits) / len(hits)
    return n, first, last, rand


@pytest.mark.parametrize("task", sorted(CHECKS))
def test_no_listing_position_predicts_the_gold(task):
    levels, items_of = CHECKS[task]
    rows = []
    for level in levels:
        for ordering in ORDERINGS:
            ds = arggym.create(task, level=level, ordering=ordering, size=SEEDS)
            rows += [ds[i] for i in range(SEEDS)]
    n, first, last, rand = position_rates(rows, items_of)
    assert n >= 20, f"{task}: only {n} rows list two or more items"
    for pos, hits in (("first", first), ("last", last)):
        assert (hits - rand) / n <= MARGIN, (
            f"{task}: the {pos} listed item is gold on {hits}/{n} rows, "
            f"a random one on {rand:.1f}")
