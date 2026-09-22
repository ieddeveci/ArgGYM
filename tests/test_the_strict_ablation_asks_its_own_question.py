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
legal there and reaches the same goal, so it scores -- but it is docked on economy for
directives the cheaper minimum no longer needs, and where it runs past twice that minimum
the bloat rule zeroes it instead.

Two of the card's four claims about that are theorems, and are asserted here only as
bounds. An answer that survives the bloat rule used at most twice the minimum, so
efficiency is at least 0.5 and the score at least 0.75 -- which holds while
`BLOAT_FACTOR == 2` and while the minimum is truthy, since `scoring.py:287` gates the
bloat rule on `if minimum` and `:347` returns a flat 0.5 as `success_but_minimum_unknown`
without one. And the plain reference always costs more than the strict minimum, so it
never reaches 1.0 -- which needs the reference to cost exactly its own `min_directives`,
true by construction at `counter_argument.py:505`, where the item takes
`min_directives=len(lines)` from the lines it also ships as `reference`.

The other two claims are measurements, and they are what go stale. They already did: #101
wrote the paragraph at 22 bloated of 35, #117 raised the strict minimum from 1 to
1 + `n_strict_final` and every bloat budget with it, 22 became 10, and the prose stayed
(#133). So the split and the top of the band are pinned cell by cell below -- the top by
equality on the cells that reach it, not by an upper bound, or the real maximum could fall
to the floor with the card still printing 0.92.

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
#: Of the 35 cells that do share a theory, the ten where the plain arm's reference runs
#: past twice the strict minimum and is zeroed for bloat. Eight are weakest-link-
#: democratic, the family whose theories carry the most junctions and so the longest plain
#: answers; the other two are the level 6 weakest-link-elitist pair. Two weakest-link-
#: democratic cells are absent, for different reasons: (3, .., 1) is in DRAWN_APART and has
#: no cross-arm score at all, and (15, .., 1) shares a theory but spends 8 directives
#: against a budget of 10, so it is docked to 0.8125 rather than zeroed.
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
#: The same measurement on the two off-grid decoy levels, kept in its own set because the
#: card counts the one above: its length is the card's "the other 10", and a cell added
#: here would move a number the card makes about the exported grid alone.
#:
#: Weakest-link-democratic is zeroed at every level from 9 to 12. At 10 and 11 the plain
#: arm costs 10 directives against a strict minimum of 3, and the other three orderings
#: cost 4 or 5 and score in band. Level 10 read this way before #109; level 11 did not,
#: because the two arms cost 4 each there and shipped one answer between them, which is
#: the collapse #109 is about. The `n_chain - 2` cap leaves the plain arm a second
#: defeasible chain, the preference that keeps costs it directives the strict minimum no
#: longer needs, and the pair now splits like every other decoy level.
BLOATED_OFF_GRID = frozenset({
    (10, "weakest_link_democratic", 0),
    (10, "weakest_link_democratic", 1),
    (11, "weakest_link_democratic", 0),
    (11, "weakest_link_democratic", 1),
})
#: The cells that reach the top of the band, and the only ones allowed to: the plain
#: reference costs 6 there against a strict minimum of 5, so 0.5 + 0.5 * 5/6. Every other
#: scored cell is held strictly below, which together with these four makes the grid's
#: maximum an equality rather than a bound.
AT_THE_TOP_OF_THE_BAND = frozenset({
    (15, "last_link_elitist", 0),
    (15, "last_link_elitist", 1),
    (15, "last_link_democratic", 0),
    (15, "last_link_democratic", 1),
})
#: A theorem, given BLOAT_FACTOR == 2 and a truthy minimum. See the module docstring.
BAND_FLOOR = 0.75
#: A measurement. The card prints it rounded, and this is what it rounds to.
BEST_CROSS_SCORE = 0.9167

CARD = pathlib.Path(__file__).resolve().parent.parent / "docs" / "dataset-card.md"
CARD_PARAGRAPH = "**`counter_argument_strict` is an ablation of `counter_argument`"


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

    # The overlap, which is what docs/dataset-card.md reports over the grid. The three
    # tables above are the card's claims written as cells; the paragraph itself is checked
    # against them in test_the_card_prints_the_split_this_file_measures.
    cell = (level, ordering, seed)
    shares_a_theory = strict.theory_text == plain.theory_text
    assert shares_a_theory == (cell not in DRAWN_APART), (
        f"{where}: the arms {'share' if shares_a_theory else 'do not share'} a theory and "
        f"DRAWN_APART says otherwise, so the card's count of cells that publish one "
        f"theory has moved off {len(GRID) - len(DRAWN_APART)}")
    if not shares_a_theory:
        return

    cross = score_item(plain.reference, ca.as_score_input(strict))
    if cell in BLOATED_ON_THE_STRICT_ITEM or cell in BLOATED_OFF_GRID:
        assert cross.score == 0.0 and cross.reason.startswith("bloated:"), (
            f"{where}: the plain reference used to be zeroed for bloat on the strict "
            f"item and now scores {cross.score} ({cross.reason}), so the card's split "
            f"has moved")
        return

    assert cross.reason == "ok", (
        f"{where}: the plain reference no longer scores on the strict item at all "
        f"({cross.reason}); if it is bloat, the card's split has moved")
    if cell in AT_THE_TOP_OF_THE_BAND:
        assert cross.score == pytest.approx(BEST_CROSS_SCORE), (
            f"{where}: this is one of the cells that reach the top of the band and it "
            f"scores {cross.score} against {BEST_CROSS_SCORE}, so the card's "
            f"{BEST_CROSS_SCORE:.2f} is no longer what the grid reaches")
    else:
        assert BAND_FLOOR <= cross.score < BEST_CROSS_SCORE, (
            f"{where}: the plain reference scores {cross.score} on the strict item, "
            f"outside [{BAND_FLOOR}, {BEST_CROSS_SCORE}). The card prints that band with "
            f"the top reached only at {sorted(AT_THE_TOP_OF_THE_BAND)}")


def test_the_card_prints_the_split_this_file_measures():
    """`docs/dataset-card.md` is the only place these measurements are written as numbers.

    The card's ablation paragraph makes three of them -- 35 of the 40 cells publish one
    theory, 25 of those score the plain reference between 0.75 and 0.92, the other 10 zero
    it for bloat -- and every one can move under the prose when the generator changes. Two
    already did, here and in #94's visible-gap count, and nothing failed.

    Three ways to stop that, and this is the third. Asserting the counts as integers in
    this file leaves each number written twice, which is the bug being fixed rather than a
    guard against it: the sweep goes red, someone edits the integer, and the card is still
    wrong. Generating the paragraph from a computed value would make the card a build
    artefact, and a datasheet is read and edited as prose. So the card keeps the numbers and
    this file keeps the cells behind them, which is the one thing the card does not say; the
    tables above are checked cell by cell in the sweep, and the claims they add up to are
    checked against the card here. Neither copy can move without the other.

    Each claim is matched inside its own clause rather than looked up in the paragraph's
    bag of integers. Bag-of-integers passes on a paragraph with the split reversed, the
    same three numbers reassigned to each other's claims -- which is exactly what a change
    undoing #117's direction would produce. The cost is that the few words binding a number
    to its claim are fixed, and they are the words in the patterns below; the rest of the
    paragraph is free to be rewritten. A count spelled as a word rather than digits also
    fails, which is intended, since nothing can check that count.

    What this does not catch: the sweep filling the tables is `slow` above level 6, and
    nothing automated runs `slow`. CI runs `uv run pytest -q -n auto` and `pyproject.toml`
    sets `addopts = "-m 'not slow'"`, so 24 of the 40 cells -- including 5 of the 10 that
    bloat and all four that reach the top of the band -- are checked only when someone runs
    `make test-all` by hand. Running the whole grid by default costs 170 seconds of build
    against a 91-second suite, 54 of them in one cell.
    """
    text = CARD.read_text()
    start = text.find(CARD_PARAGRAPH)
    assert start >= 0, (
        f"{CARD} no longer has the ablation paragraph this test checks; it began "
        f"{CARD_PARAGRAPH!r}")
    paragraph = text[start:text.find("\n\n", start)]
    line = text.count("\n", 0, start) + 1
    # The card hard-wraps, so a clause spans lines and rewrapping moves where.
    unwrapped = re.sub(r"\s+", " ", paragraph)

    shared = len(GRID) - len(DRAWN_APART)
    bloated = len(BLOATED_ON_THE_STRICT_ITEM)
    required = {
        f"{shared} of {len(GRID)} cells publish one theory":
            rf"byte for byte on {shared} of their {len(GRID)} cells",
        f"{shared - bloated} of those score between {BAND_FLOOR:.2f} and "
        f"{BEST_CROSS_SCORE:.2f}":
            rf"on {shared - bloated} of them it scores between {BAND_FLOOR:.2f} and "
            rf"{BEST_CROSS_SCORE:.2f}",
        f"the other {bloated} are zeroed for bloat":
            rf"the other {bloated} it runs past twice the strict minimum",
    }
    missing = [claim for claim, pattern in required.items()
               if not re.search(pattern, unwrapped)]
    assert not missing, (
        f"{CARD}:{line} does not state " + "; ".join(missing)
        + ". Each of those is a measurement over the built grid, and the paragraph has to "
        f"carry it in the clause that claims it -- the patterns are in {__name__}. The "
        f"paragraph now reads:\n\n{paragraph}")


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
