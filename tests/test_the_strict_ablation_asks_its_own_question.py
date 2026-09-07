"""The strict ablation has to ask something its own baseline does not.

`counter_argument_strict` is `counter_argument` with one thing varied: strict rules are
permitted. So the theory is the control and must be shared, and the cost of the cheapest
answer is the treatment and must fall. `min_directives` is the property rather than the
text of the reference: it sets the bloat budget an answer is scored against, so two arms
with the same minimum are the same scoring problem however differently their references
read.

Over the 40 exported cells the strict arm cost exactly what the plain arm cost on 19 of
them, and 15 of those shipped the same reference answer as well (#37). The one-directive
strict counter-argument is consistent only where every chain reaches the target through a
defeasible rule -- otherwise that chain derives the target strictly, the answer derives
its contrary strictly, and the framework is inconsistent. So the generator breaks those
chains first, one undercut each, which is the cost law asserted below: 1 + k directives
for the k chains that reach the target strictly.

`k` is `n_strict_final`, on every cell. It used to be zero on a mid-chain-target item
whatever that item's strict rules said, because the strict flag named the chain's last
rule and the mid-chain target sits before it, so no chain reached the target strictly and
`[strict cs: <premise> -> -<target>]` was the whole answer on all 16 of them (#93). The
flag names the target-reaching rule now, so the law has one branch rather than two.

Whether either arm can be answered without reading the theory is asked next door, in
tests/test_neither_arm_answers_from_the_question_line_and_one_grep.py.

One test per cell, asserting everything about that cell, because comparing the arms means
building both and the level 12 and 15 weakest-link-democratic cells are the dearest items
the generator makes. Splitting the assertions across tests built them twice over.
"""
from __future__ import annotations

import pytest

from arggym.core.scoring import score_item
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import counter_argument as ca

CHEAP = [(lv, o, s) for lv in (3, 6) for o in ALL_ORDERINGS for s in SEEDS]
GRID = [(lv, o, s) for lv in LEVELS for o in ALL_ORDERINGS for s in SEEDS]
DEAR = [c for c in GRID if c not in CHEAP]
#: Where the draw may zero `n_strict` for the strict arm alone (#32), which is the only
#: place the two arms are allowed to build different theories.
DRAWN = (3, 4, 5)


def _the_ablation_holds(level: int, ordering: str, seed: int) -> None:
    plain = ca.make_item(level, seed, ordering, allow_strict=False)
    strict = ca.make_item(level, seed, ordering, allow_strict=True)
    where = f"L{level} {ordering} seed {seed}"
    assert plain is not None and strict is not None, where

    # The control. One theory, or the entry-level draw and nothing else.
    if strict.metadata["n_strict_final"] == plain.metadata["n_strict_final"]:
        assert strict.theory_text == plain.theory_text, (
            f"{where}: the two arms build different theories, so a score gap between "
            f"them cannot be read as an effect of permitting strict rules")
    else:
        assert level in DRAWN and strict.metadata["n_strict_final"] == 0, (
            f"{where}: the theories differ for something other than the entry-level "
            f"draw -- plain has {plain.metadata['n_strict_final']} strict-final chains "
            f"and strict has {strict.metadata['n_strict_final']}")

    # The treatment, and the law that says what it costs.
    assert strict.min_directives < plain.min_directives, (
        f"{where}: strict costs {strict.min_directives} against plain "
        f"{plain.min_directives}, so the wider answer space bought nothing here")
    assert strict.reference != plain.reference, (
        f"{where}: the two arms ship one answer between them:\n{plain.reference}")
    blocked = strict.metadata["n_strict_final"]
    assert strict.min_directives == 1 + blocked, (
        f"{where}: {strict.min_directives} directives against {blocked} chains reaching "
        f"the target strictly, and the answer is the strict counter-argument plus one "
        f"undercut each:\n{strict.reference}")

    # The gold. Several directives now, so it is no longer minimal by being one line.
    got = score_item(strict.reference, ca.as_score_input(strict))
    assert got.score == pytest.approx(1.0), (
        f"{where}: gold regrades at {got.score}: {got.reason}")


@pytest.mark.parametrize("level,ordering,seed", CHEAP)
def test_the_strict_arm_asks_its_own_question(level, ordering, seed):
    _the_ablation_holds(level, ordering, seed)


@pytest.mark.slow
@pytest.mark.parametrize("level,ordering,seed", DEAR)
def test_the_strict_arm_asks_its_own_question_at_every_exported_cell(level, ordering, seed):
    _the_ablation_holds(level, ordering, seed)
