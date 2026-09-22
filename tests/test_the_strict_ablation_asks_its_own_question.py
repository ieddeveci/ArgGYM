"""The strict ablation has to ask something its own baseline does not.

`counter_argument_strict` is `counter_argument` with one thing varied: strict rules are
permitted. So the theory is the control and must be shared, and the cost of the cheapest
answer is the treatment and must fall. `min_directives` is the property rather than the
text of the reference: it sets the bloat budget an answer is scored against, so two arms
with the same minimum are the same scoring problem however differently their references
read.

Over the 40 cells the grid exported then, the strict arm cost exactly what the plain arm
cost on 19 of them, and 15 of those shipped the same reference answer as well (#37). The
one-directive strict counter-argument is consistent only where every chain reaches the
target through a defeasible rule -- otherwise that chain derives the target strictly, the
answer derives its contrary strictly, and the framework is inconsistent. So the generator
breaks those chains first, one undercut each, which is the cost law asserted below: 1 + k
directives for the k chains that reach the target strictly.

`k` is `n_strict_final`, on every cell. It used to be zero on a mid-chain-target item
whatever that item's strict rules said, because the strict flag named the chain's last
rule and the mid-chain target sits before it, so no chain reached the target strictly and
`[strict cs: <premise> -> -<target>]` was the whole answer on all 16 of them (#93). The
flag names the target-reaching rule now, so the law has one branch rather than two.

The two arms shipping one answer between them is its own failure, and levels 10 and 11
are where it happened: the decoy is in play there and the chain count is four, which left
both arms costing 4 directives and writing the same reference (#109). The `n_chain - 2`
cap leaves the plain arm a second defeasible chain, the preference that keeps it costs
directives the strict minimum no longer needs, and the pair splits like every other level.
Those two levels used to be off the grid and were parametrized separately to reach them;
the grid exports them now, so the sweep covers them and `strict.reference !=
plain.reference` is asserted at every cell.

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

The other two claims are measurements, and they are what go stale. They already did twice:
#101 wrote the paragraph at 22 bloated of 35, #117 raised the strict minimum from 1 to
1 + `n_strict_final` and every bloat budget with it, 22 became 10, and the prose stayed
(#133); then the grid went from five levels to fifteen and 35 of 40 became 106 of 120. So
the split and the top of the band are pinned cell by cell below -- the top by equality on
the cells that reach it, not by an upper bound, or the real maximum could fall to the floor
with the card still printing 0.92.

Whether either arm can be answered without reading the theory is asked next door, in
tests/test_neither_arm_answers_from_the_question_line_and_one_grep.py.

One test per cell, asserting everything about that cell, because comparing the arms means
building both and splitting the assertions across tests built them twice over. Scoring one
arm's reference against the other's item costs 0.1 seconds over the whole grid, so the
overlap rides along here rather than paying for the builds a second time somewhere else.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from arggym.core.scoring import score_item
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import counter_argument as ca

GRID = [(lv, o, s) for lv in LEVELS for o in ALL_ORDERINGS for s in SEEDS]
#: The cost of the whole grid sits in one ordering, and it is the ordering rather than the
#: level that predicts it: building both arms at all 120 cells takes 357 seconds, of which
#: `weakest_link_democratic` is 328 and the dearest single cell, level 15 seed 0, is 63.
#: Every other ordering builds both arms in under 5 seconds at every level, so splitting on
#: the ordering puts 90 of the 120 cells in the default run for 30 seconds -- including all
#: four cells that reach the top of the band, which the old level-based split left to
#: `make test-all`. The 30 dear cells carry 26 of the 32 that bloat (#111).
DEAR_ORDERING = "weakest_link_democratic"
CHEAP = [c for c in GRID if c[1] != DEAR_ORDERING]
DEAR = [c for c in GRID if c[1] == DEAR_ORDERING]
#: Where the draw may zero `n_strict` for the strict arm alone (#32), which is the only
#: place the two arms are allowed to build different theories.
DRAWN = (3, 4, 5)

#: The cells where that draw actually fell, so the two arms publish different theories and
#: there is no cross-arm answer to score. Fourteen of the 120, which is what leaves the
#: card's "byte for byte on 106 of their 120 cells". Level 3 alone used to be exported out
#: of this band and the count was 5 of 40; levels 4 and 5 ship now and draw the same way.
DRAWN_APART = frozenset({
    (3, "last_link_elitist", 0),
    (3, "last_link_elitist", 1),
    (3, "last_link_democratic", 1),
    (3, "weakest_link_elitist", 1),
    (3, "weakest_link_democratic", 1),
    (4, "last_link_democratic", 1),
    (4, "weakest_link_elitist", 1),
    (4, "weakest_link_democratic", 1),
    (5, "last_link_elitist", 0),
    (5, "last_link_elitist", 1),
    (5, "last_link_democratic", 1),
    (5, "weakest_link_elitist", 0),
    (5, "weakest_link_elitist", 1),
    (5, "weakest_link_democratic", 0),
})
#: Of the 106 cells that do share a theory, the 32 where the plain arm's reference runs
#: past twice the strict minimum and is zeroed for bloat. The split reads off the ordering
#: almost entirely.
#:
#: Weakest-link-democratic carries 26 of them: it is the family whose theories hold the
#: most junctions, so its plain answers are the longest -- 7 to 14 directives against a
#: strict minimum of 1 to 5 -- and every one of its cells that shares a theory bloats
#: except (15, .., 1), which spends 8 against a budget of 10 and is docked to 0.8125.
#:
#: The other 6 are weakest-link-elitist, where the plain answer costs one or two more than
#: the strict minimum and so bloats only while that minimum is small enough for twice it
#: to be smaller still: 3 against a budget of 2 at levels 1 and 2, and 6 against 4 at level
#: 6. From level 7 the minimum is 3 or more and the same one-or-two-directive gap lands
#: inside the budget.
#:
#: No last-link cell bloats at any level. There the plain reference costs exactly one
#: directive more than the strict minimum, which is inside twice it from level 1 up.
#:
#: Levels 10 and 11 were off the old grid and kept in a set of their own so that a cell
#: added there could not move a number the card made about the exported grid. They are
#: exported now, so the set is gone and its four cells are here.
BLOATED_ON_THE_STRICT_ITEM = frozenset({
    (1, "weakest_link_elitist", 0),
    (1, "weakest_link_elitist", 1),
    (1, "weakest_link_democratic", 0),
    (1, "weakest_link_democratic", 1),
    (2, "weakest_link_elitist", 0),
    (2, "weakest_link_elitist", 1),
    (2, "weakest_link_democratic", 0),
    (2, "weakest_link_democratic", 1),
    (3, "weakest_link_democratic", 0),
    (4, "weakest_link_democratic", 0),
    (5, "weakest_link_democratic", 1),
    (6, "weakest_link_elitist", 0),
    (6, "weakest_link_elitist", 1),
    (6, "weakest_link_democratic", 0),
    (6, "weakest_link_democratic", 1),
    (7, "weakest_link_democratic", 0),
    (7, "weakest_link_democratic", 1),
    (8, "weakest_link_democratic", 0),
    (8, "weakest_link_democratic", 1),
    (9, "weakest_link_democratic", 0),
    (9, "weakest_link_democratic", 1),
    (10, "weakest_link_democratic", 0),
    (10, "weakest_link_democratic", 1),
    (11, "weakest_link_democratic", 0),
    (11, "weakest_link_democratic", 1),
    (12, "weakest_link_democratic", 0),
    (12, "weakest_link_democratic", 1),
    (13, "weakest_link_democratic", 0),
    (13, "weakest_link_democratic", 1),
    (14, "weakest_link_democratic", 0),
    (14, "weakest_link_democratic", 1),
    (15, "weakest_link_democratic", 0),
})
#: The cells that reach the top of the band, and the only ones allowed to: the plain
#: reference costs 6 there against a strict minimum of 5, so 0.5 + 0.5 * 5/6. Level 15
#: under last-link is where the ratio is best, because the strict minimum is 1 +
#: `n_strict_final` and `n_strict_final` peaks at 4 there while the last-link plain
#: reference stays one directive above it. The four last-link cells at levels 12 to 14 sit
#: just under, at 5/6 of a 5-directive minimum -- 0.9 against 0.9167 -- which is why the
#: top is asserted as an equality on these four and a strict inequality everywhere else.
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
    if cell in BLOATED_ON_THE_STRICT_ITEM:
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


def test_the_tables_cover_the_grid_and_nothing_else():
    """Cheap, and it catches the edit the sweep cannot.

    The sweep visits a cell and checks the tables say the right thing about it, so a cell
    named in a table that is not on the grid is never visited and never contradicted. That
    is how a table outlives the grid it was measured on: `BLOATED_OFF_GRID` held levels 10
    and 11 while the grid skipped them, and when the grid took them the two sets described
    the same cells with only one of them counted on the card.
    """
    grid = set(GRID)
    for name, table in (("DRAWN_APART", DRAWN_APART),
                        ("BLOATED_ON_THE_STRICT_ITEM", BLOATED_ON_THE_STRICT_ITEM),
                        ("AT_THE_TOP_OF_THE_BAND", AT_THE_TOP_OF_THE_BAND)):
        stray = sorted(table - grid)
        assert not stray, (
            f"{name} names {stray}, which the grid does not export, so the sweep never "
            f"checks those rows and the card counts them anyway")
    overlap = sorted(DRAWN_APART & BLOATED_ON_THE_STRICT_ITEM)
    assert not overlap, (
        f"{overlap} is both drawn apart and bloated; a cell with no shared theory has no "
        f"cross-arm score to bloat")
    assert not AT_THE_TOP_OF_THE_BAND & (DRAWN_APART | BLOATED_ON_THE_STRICT_ITEM)
    assert all(lv in DRAWN for lv, _, _ in DRAWN_APART), sorted(DRAWN_APART)


def test_the_card_prints_the_split_this_file_measures():
    """`docs/dataset-card.md` is the only place these measurements are written as numbers.

    The card's ablation paragraph makes three of them -- 106 of the 120 cells publish one
    theory, 74 of those score the plain reference between 0.75 and 0.92, the other 32 zero
    it for bloat -- and every one can move under the prose when the generator changes, or
    when the grid does. All three already have, twice: here and in #94's visible-gap count
    when #117 changed the minimum, and again when the grid went to fifteen levels. Nothing
    failed either time.

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

    What this does not catch: the sweep filling the tables is `slow` on one ordering, and
    nothing automated runs `slow`. CI runs `uv run pytest -q -n auto` and `pyproject.toml`
    sets `addopts = "-m 'not slow'"`, so the 30 `weakest_link_democratic` cells -- 26 of
    the 32 that bloat -- are checked only when someone runs `make test-all` by hand.
    Running them by default costs 328 seconds of build, 63 of them in one cell.
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


@pytest.mark.slow
@pytest.mark.parametrize("level,ordering,seed", DEAR)
def test_the_strict_arm_asks_its_own_question_under_the_dear_ordering(level, ordering, seed):
    _the_ablation_holds(level, ordering, seed)
