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
cap leaves the plain arm a second defeasible chain, the undercut that breaks it costs a
directive the strict minimum no longer needs, and the pair splits like every other level.
Those two levels used to be off the grid and were parametrized separately to reach them;
the grid exports them now, so the sweep covers them and `strict.reference !=
plain.reference` is asserted at every cell.

Sharing a theory is not sharing an answer, and `docs/dataset-card.md` reports how far
apart the two arms land. Submit the plain arm's own reference to the strict item and it is
legal there and reaches the same goal, so it scores -- but it is docked on economy for
directives the cheaper minimum no longer needs. It would be zeroed for bloat where it ran
past twice that minimum, and it does so on no cell: the plain minimum is `1 + n`, or `n`
from level 9, against the strict `1 + k` with `k >= 1`.

Two of the card's four claims about that are theorems, and are asserted here only as
bounds. An answer that survives the bloat rule used at most twice the minimum, so
efficiency is at least 0.5 and the score at least 0.75 -- which holds while
`BLOAT_FACTOR == 2` and while the minimum is truthy, since `scoring.py:287` gates the
bloat rule on `if minimum` and `:347` returns a flat 0.5 as `success_but_minimum_unknown`
without one. And the plain reference always costs more than the strict minimum, so it
never reaches 1.0 -- which needs the reference to cost exactly its own `min_directives`,
true by construction at `counter_argument.py:531`, where the item takes
`min_directives=len(lines)` from the lines it also ships as `reference`.

The other two claims are measurements, and they are what go stale. They already did twice:
#101 wrote the paragraph at 22 bloated of 35, #117 raised the strict minimum from 1 to
1 + `n_strict_final` and every bloat budget with it, 22 became 10, and the prose stayed
(#133); then the grid went from five levels to fifteen and 35 of 40 became 106 of 120;
then the entry-level draw went and every cell shared a theory (#167). So
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
#: the ordering puts 90 of the 120 cells in the default run for 30 seconds -- including six
#: of the eight cells that reach the top of the band, which the old level-based split left
#: to `make test-all` (#111).
DEAR_ORDERING = "weakest_link_democratic"
CHEAP = [c for c in GRID if c[1] != DEAR_ORDERING]
DEAR = [c for c in GRID if c[1] == DEAR_ORDERING]
#: The cells where the plain arm's reference runs past twice the strict minimum and is
#: zeroed for bloat. There are none. The plain reference costs `1 + n` at levels 1 to 8
#: and `n` from level 9, against a strict minimum of `1 + k` with `1 <= k <= n - 1`; the
#: widest ratio is level 6, 4 against 2, which lands exactly on the budget.
BLOATED_ON_THE_STRICT_ITEM: frozenset = frozenset()
#: The cells that reach the top of the band, and the only ones allowed to: the plain
#: reference costs 6 there against a strict minimum of 5, so 0.5 + 0.5 * 5/6. Level 15 is
#: where the ratio is best, because the strict minimum is 1 + `n_strict_final` and
#: `n_strict_final` peaks at 4 there while the plain minimum is the six chains. Levels 12
#: to 14 sit just under, at 4/5 of a 5-directive plain minimum -- 0.9 against 0.9167 --
#: which is why the top is asserted as an equality on these eight and a strict inequality
#: everywhere else.
AT_THE_TOP_OF_THE_BAND = frozenset(
    (15, ordering, seed) for ordering in ALL_ORDERINGS for seed in SEEDS)
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

    # The control. One theory, at every cell.
    assert strict.theory_text == plain.theory_text, (
        f"{where}: the two arms build different theories, so a score gap between them "
        f"cannot be read as an effect of permitting strict rules")

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
    # tables above are the card's claims written as cells; the paragraph itself is checked
    # against them in test_the_card_prints_the_split_this_file_measures.
    cell = (level, ordering, seed)
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
    for name, table in (("BLOATED_ON_THE_STRICT_ITEM", BLOATED_ON_THE_STRICT_ITEM),
                        ("AT_THE_TOP_OF_THE_BAND", AT_THE_TOP_OF_THE_BAND)):
        stray = sorted(table - grid)
        assert not stray, (
            f"{name} names {stray}, which the grid does not export, so the sweep never "
            f"checks those rows and the card counts them anyway")
    assert not AT_THE_TOP_OF_THE_BAND & BLOATED_ON_THE_STRICT_ITEM


def test_the_card_prints_the_split_this_file_measures():
    """`docs/dataset-card.md` is the only place these measurements are written as numbers.

    The card's ablation paragraph makes three of them -- all 120 cells publish one theory,
    120 of them score the plain reference between 0.75 and 0.92, 0 of them zero it for
    bloat -- and every one can move under the prose when the generator changes, or
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
    sets `addopts = "-m 'not slow'"`, so the 30 `weakest_link_democratic` cells are
    checked only when someone runs `make test-all` by hand.
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

    shared = len(GRID)
    bloated = len(BLOATED_ON_THE_STRICT_ITEM)
    required = {
        f"all {shared} cells publish one theory":
            rf"byte for byte on all {shared} of their cells",
        f"{shared - bloated} of those score between {BAND_FLOOR:.2f} and "
        f"{BEST_CROSS_SCORE:.2f}":
            rf"on {shared - bloated} of them it scores between {BAND_FLOOR:.2f} and "
            rf"{BEST_CROSS_SCORE:.2f}",
        f"{bloated} of them are zeroed for bloat":
            rf"on {bloated} of them it runs past twice the strict minimum",
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
