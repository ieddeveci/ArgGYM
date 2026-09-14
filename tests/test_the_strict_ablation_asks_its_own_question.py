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

Sharing a theory is not sharing an answer, and `docs/dataset-card.md` reports how far
apart the two arms land. Submit the plain arm's own reference to the strict item and it is
legal there and reaches the same goal, so it scores -- but it pays economy for directives
the cheaper minimum no longer needs, and where it runs past twice that minimum the bloat
rule zeroes it instead. Two of the four things the card says about that are theorems and
need no sweep: an answer that is not bloated used at most twice the minimum, so efficiency
is at least 0.5 and the score at least 0.75, and the plain reference always costs more than
the strict minimum, so it never reaches 1.0. What is measured is the split -- which cells
bloat out and which merely pay -- and the top of the band, and those are what go stale.
They already did: #101 wrote the paragraph at 22 bloated of 35, #117 raised the strict
minimum from 1 to 1 + `n_strict_final` and every bloat budget with it, 22 became 10, and
the prose stayed (#133).

Whether either arm can be answered without reading the theory is asked next door, in
tests/test_neither_arm_answers_from_the_question_line_and_one_grep.py.

One test per cell, asserting everything about that cell, because comparing the arms means
building both and the level 12 and 15 weakest-link-democratic cells are the dearest items
the generator makes: 170 seconds to build both arms across the grid, of which those four
cells are 130 and the dearest single one is 54. Splitting the assertions across tests built
them twice over. Scoring one arm's reference against the other's item costs 0.1 seconds
over the whole grid, so the overlap rides along here rather than paying for the builds a
second time somewhere else.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from arggym.core.scoring import score_item
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import counter_argument as ca

CHEAP = [(lv, o, s) for lv in (3, 6) for o in ALL_ORDERINGS for s in SEEDS]
GRID = [(lv, o, s) for lv in LEVELS for o in ALL_ORDERINGS for s in SEEDS]
DEAR = [c for c in GRID if c not in CHEAP]
#: The two off-grid levels where the decoy is in play and the chain count is four, which
#: is where the arms collapsed into one answer (#109). A spec may name either; both build
#: both arms in seconds on every ordering, unlike the level 13 and 14 tail (#111).
DECOY = [(lv, o, s) for lv in (10, 11) for o in ALL_ORDERINGS for s in SEEDS]
#: Where the draw may zero `n_strict` for the strict arm alone (#32), which is the only
#: place the two arms are allowed to build different theories.
DRAWN = (3, 4, 5)

#: The cells where that draw actually fell, so the two arms publish different theories and
#: there is no cross-arm answer to score. Five of the forty, which is what leaves the
#: card's "byte for byte on 35 of their 40 cells".
DRAWN_APART = frozenset({
    (3, "last_link_elitist", 0),
    (3, "last_link_elitist", 1),
    (3, "last_link_democratic", 1),
    (3, "weakest_link_elitist", 1),
    (3, "weakest_link_democratic", 1),
})
#: Of the 35 cells that do share a theory, the ones where the plain arm's reference runs
#: past twice the strict minimum and is zeroed for bloat. Weakest-link-democratic at every
#: level, because its theories carry the most junctions and so the longest plain answers,
#: plus both level 6 weakest-link-elitist cells. The other 25 score in [0.75, 0.92].
BLOATED_ON_THE_STRICT_ITEM = frozenset({
    (3, "weakest_link_democratic", 0),
    (6, "weakest_link_elitist", 0),
    (6, "weakest_link_elitist", 1),
    (6, "weakest_link_democratic", 0),
    (6, "weakest_link_democratic", 1),
    (9, "weakest_link_democratic", 0),
    (9, "weakest_link_democratic", 1),
    (12, "weakest_link_democratic", 0),
    (12, "weakest_link_democratic", 1),
    (15, "weakest_link_democratic", 0),
})
#: The best the plain reference does on the strict item anywhere on the grid: 0.5 + 0.5 *
#: 5/6, at level 15 last-link. The floor of the band is a theorem, this is a measurement.
BEST_CROSS_SCORE = 0.9167

CARD = pathlib.Path(__file__).resolve().parent.parent / "docs" / "dataset-card.md"
CARD_PARAGRAPH = "**`counter_argument_strict` is an ablation of `counter_argument`"
#: Integers, and only integers: `0.75` must not read as a 0 and a 75.
AN_INTEGER = re.compile(r"(?<![\d.])\d+(?![\d.])")


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

    # The overlap, which is what docs/dataset-card.md reports over the grid. The two
    # tables are the card's three counts written as cells; the paragraph itself is
    # checked against them in test_the_card_prints_the_split_this_file_measures.
    cell = (level, ordering, seed)
    shares_a_theory = strict.theory_text == plain.theory_text
    assert shares_a_theory == (cell not in DRAWN_APART), (
        f"{where}: the arms {'share' if shares_a_theory else 'do not share'} a theory and "
        f"DRAWN_APART says otherwise, so the card's count of cells that publish one "
        f"theory has moved off 35")
    if not shares_a_theory:
        return

    cross = score_item(plain.reference, ca.as_score_input(strict))
    if cell in BLOATED_ON_THE_STRICT_ITEM:
        assert cross.score == 0.0 and cross.reason.startswith("bloated:"), (
            f"{where}: the plain reference used to be zeroed for bloat on the strict "
            f"item and now scores {cross.score} ({cross.reason}), so the card's split "
            f"has moved")
    else:
        assert cross.reason == "ok", (
            f"{where}: the plain reference no longer scores on the strict item at all "
            f"({cross.reason}); if it is bloat, the card's split has moved")
        assert 0.75 <= cross.score <= BEST_CROSS_SCORE, (
            f"{where}: the plain reference scores {cross.score} on the strict item, "
            f"outside the 0.75-0.92 band the card prints")


def test_the_card_prints_the_split_this_file_measures():
    """`docs/dataset-card.md` is the only place these counts are written as numbers.

    The card's ablation paragraph states three of them -- 35 cells publishing one theory,
    10 of those zeroed for bloat, 25 scored -- and every one of them is a measurement over
    the built grid that a generator change can move under the prose. It moved twice
    already, in the card here and in #94's visible-gap count, and nothing failed.

    Three ways to stop that, and this is the third. Asserting the counts as integers in
    this file leaves the number written twice, which is the bug being fixed rather than a
    guard against it: the sweep goes red, someone edits the integer, and the card is still
    wrong. Generating the paragraph from a computed value would make the card a build
    artefact, and a datasheet is read and edited as prose. So the card keeps the numbers
    and this file keeps the cells, which is the one thing the card does not say; the tables
    above are checked cell by cell in the sweep, and their sizes are checked against the
    card here. Neither copy can move without the other.

    It reads the paragraph rather than a line number, and asks only that the three counts
    appear in it as digits. Rewording is free; writing "ten" for 10 is not, and that is the
    intended trade -- a measured count spelled as a word is a count nothing can check.

    What this does not catch: the sweep that fills the tables is `slow` above level 6, so a
    change that moves only a level 9, 12 or 15 cell passes `pytest` and fails `make
    test-all`. Running the whole grid by default costs 170 seconds of build for 40 cells
    and 54 of them in one cell, against a 91-second suite.
    """
    text = CARD.read_text()
    start = text.find(CARD_PARAGRAPH)
    assert start >= 0, (
        f"{CARD} no longer has the ablation paragraph this test checks; it began "
        f"{CARD_PARAGRAPH!r}")
    paragraph = text[start:text.find("\n\n", start)]
    line = text.count("\n", 0, start) + 1

    shared = len(GRID) - len(DRAWN_APART)
    counts = {
        "cells publishing one theory": shared,
        "of those, zeroed for bloat": len(BLOATED_ON_THE_STRICT_ITEM),
        "of those, scored in the band": shared - len(BLOATED_ON_THE_STRICT_ITEM),
    }
    printed = {int(n) for n in AN_INTEGER.findall(paragraph)}
    missing = {name: n for name, n in counts.items() if n not in printed}
    assert not missing, (
        f"{CARD}:{line} does not print "
        + ", ".join(f"{n} ({name})" for name, n in missing.items())
        + f". The grid says {counts['cells publishing one theory']} of {len(GRID)} cells "
        f"publish one theory, {counts['of those, zeroed for bloat']} of those zero the "
        f"plain reference for bloat and {counts['of those, scored in the band']} score "
        f"it, so the paragraph needs rewriting:\n\n{paragraph}")


@pytest.mark.parametrize("level,ordering,seed", CHEAP)
def test_the_strict_arm_asks_its_own_question(level, ordering, seed):
    _the_ablation_holds(level, ordering, seed)


@pytest.mark.parametrize("level,ordering,seed", DECOY)
def test_the_strict_arm_asks_its_own_question_where_the_decoy_enters(level, ordering, seed):
    _the_ablation_holds(level, ordering, seed)


@pytest.mark.slow
@pytest.mark.parametrize("level,ordering,seed", DEAR)
def test_the_strict_arm_asks_its_own_question_at_every_exported_cell(level, ordering, seed):
    _the_ablation_holds(level, ordering, seed)
