"""`stable` has to answer something `sceptical preferred` does not.

Both names read the same all/any/none formula off their own extension family
(`semantics_query.status_under`), so they answer differently only where the families
differ. No cluster built anything that made them differ: preferred equalled stable on 40
of 40 exported cells and 360 of 360 in a wider sweep, `no stable extension` never
occurred, and over every queried claim the two names disagreed zero times -- `stable` was
`sceptical preferred` under a second name (#77).

Odd defeat cycles are the whole of the difference. Dung's coherence result says a graph
with no odd-length cycle has its preferred and stable extensions equal, so a generator
that never builds one cannot separate them. What separates them cheaply is an odd cycle
that something else can kill: put an undercut ring on one branch of a contested pair and
the branch that survives is stable while the branch that keeps the ring alive is
preferred and not stable. An *unattacked* ring is the harder case -- it leaves no stable
extension at all -- and is scheduled thinly, because it answers `no stable extension` for
every claim in its item and would otherwise be most of what `stable` ever says.
"""
from __future__ import annotations

import collections
import inspect
import itertools
import random

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import semantics_query as sq

CELLS = [(lv, o, s) for lv in LEVELS for o in ALL_ORDERINGS for s in SEEDS]
ASKS_STABLE = [c for c in CELLS if sq.STABLE in sq.semantics_for(c[0])]
ASKS_BOTH = [c for c in ASKS_STABLE if sq.SCEPT_PREF in sq.semantics_for(c[0])]


@pytest.fixture(scope="module")
def grid():
    """The exported grid, built once: `make_item` walks up to 24 build seeds a cell."""
    return {c: sq.make_item(c[0], c[2], c[1]) for c in CELLS}


def cluster(unshielded: bool, seed: int = 0):
    rng = random.Random(seed)
    return sq._odd_cycle(iter(sq._names(seed, 12)), [0], rng, unshielded=unshielded)


def statuses(ops, ordering, claims):
    verifier = ASPICVerifier.from_operations(list(ops), ordering=ordering)
    preferred = verifier.preferred_conclusions()
    stable = verifier.stable_conclusions()
    cache = sq.SemCache(grounded=verifier.status_map(), preferred=preferred, stable=stable)
    return preferred, stable, {(c, s): sq.status_under(cache, c, s)
                               for c in claims for s in (sq.SCEPT_PREF, sq.STABLE)}


CLUSTER_DRAWS = list(itertools.product(ALL_ORDERINGS, range(6)))


@pytest.mark.parametrize("ordering,seed", CLUSTER_DRAWS)
def test_a_shielded_ring_makes_stable_decide_what_preferred_leaves_open(ordering, seed):
    """The cluster's claim, measured on the cluster alone, over both shapes it draws.

    The ring stands on one side of the contested pair, so the extension that takes the
    other side defeats every rule in the ring and is stable, while the extension that
    takes the ring's own side stays preferred and is not. Sceptical preferred sees two
    extensions and calls both literals undecided; stable sees one and decides them.
    """
    ops, (p, not_p) = cluster(unshielded=False, seed=seed)
    preferred, stable, status = statuses(ops, ordering, (p, not_p))
    assert len(preferred) == 2 and len(stable) == 1, (
        f"{len(preferred)} preferred and {len(stable)} stable extensions")
    assert status[(p, sq.SCEPT_PREF)] == status[(not_p, sq.SCEPT_PREF)] == "UNDECIDED"
    assert {status[(p, sq.STABLE)], status[(not_p, sq.STABLE)]} == {"JUSTIFIED", "OVERRULED"}


@pytest.mark.parametrize("ordering,seed", CLUSTER_DRAWS)
def test_a_ring_on_both_branches_leaves_no_stable_extension(ordering, seed):
    """The same builder, loaded on both sides: nothing kills the ring, so nothing is stable.

    This is the case #77 asked for. It is the rarer one here because it answers `no stable
    extension` for every claim in the item rather than about a claim.
    """
    ops, lits = cluster(unshielded=True, seed=seed)
    preferred, stable, status = statuses(ops, ordering, lits)
    assert preferred, "an odd cycle still has a preferred extension"
    assert stable == [], "the ring is attacked, so a stable extension survives"
    assert all(status[(c, sq.STABLE)] == "NO_STABLE_EXTENSION" for c in lits)


def test_the_separation_holds_under_either_reading_of_overruled(grid):
    """#36 may reread `overruled` as `the contrary is in every extension` (#29).

    The cluster's separation does not depend on which reading wins: the stable-overruled
    literal has its contrary in the single stable extension, so it is overruled either
    way, and each literal sits in one preferred extension of two, so sceptical preferred
    calls it undecided either way. Worth pinning, because the discriminating claims would
    otherwise be the first thing that reading breaks.
    """
    for cell in ASKS_BOTH:
        item = grid[cell]
        if item.metadata["unshielded_ring"]:
            continue
        verifier = ASPICVerifier.from_operations(list(item.base_ops), ordering=cell[1])
        stable = verifier.stable_conclusions()
        for (claim, sem), gold in item.gold.items():
            if sem != sq.STABLE or gold != "OVERRULED":
                continue
            contrary = claim[1:] if claim.startswith("-") else "-" + claim
            assert all(contrary in e for e in stable), (
                f"{claim} is overruled only under the reading this change does not rely on")


def test_the_two_extension_families_differ_wherever_stable_is_asked(grid):
    """The measurement #77 opened on, read off the theory: it was equal on 40 of 40."""
    for cell in ASKS_STABLE:
        verifier = ASPICVerifier.from_operations(list(grid[cell].base_ops), ordering=cell[1])
        preferred = {frozenset(e) for e in verifier.preferred_conclusions()}
        stable = {frozenset(e) for e in verifier.stable_conclusions()}
        assert preferred != stable, (
            f"{cell}: the two families are equal, so stable repeats sceptical preferred")


def test_every_item_asks_one_claim_under_both_names_and_answers_it_differently(grid):
    """A theory that separates the two and an item that shows it are different things.

    The trim in `build` keeps a fraction of the query pool and rarely drew one claim under
    both names, so an earlier cut of this change separated the families on 32 of 32 cells
    and surfaced it to a reader on 6 of 24. The cluster's literals are asked under both
    names outright.
    """
    for cell in ASKS_BOTH:
        item = grid[cell]
        asked = {(c, s) for c, s in item.queries}
        shown = [c for c, s in asked
                 if s == sq.STABLE and (c, sq.SCEPT_PREF) in asked
                 and item.gold[(c, sq.STABLE)] != item.gold[(c, sq.SCEPT_PREF)]]
        assert shown, (f"{cell} asks no claim under both names with different answers, so "
                       f"the item does not show what the theory separates")


def test_the_unattacked_ring_is_spread_over_levels_and_orderings():
    """One ordering per level, one level per ordering, over the levels that ask stable.

    Not keyed on the seed: `tasksets/standard.yaml` scans up to 40 seeds a cell and keeps
    the first two that build, so which seeds a cell ships is not fixed and a seed-keyed
    rule realises an uncontrolled fraction. Not keyed on the level alone either, which
    would confound it with the curriculum axis the report breaks out.
    """
    levels = sorted({lv for lv in LEVELS if sq.STABLE in sq.semantics_for(lv)})
    chosen = {lv: [o for o in ALL_ORDERINGS if sq.wants_unshielded_ring(lv, o)]
              for lv in levels}
    assert all(len(v) == 1 for v in chosen.values()), chosen
    flat = [v[0] for v in chosen.values()]
    assert len(set(flat)) == len(flat), f"an ordering carries the ring twice: {chosen}"
    for level in range(1, 16):
        if sq.STABLE in sq.semantics_for(level):
            continue
        assert not any(sq.wants_unshielded_ring(level, o) for o in ALL_ORDERINGS), (
            f"level {level} never asks about stable, so a ring there asks nothing")


def test_no_stable_extension_is_answered_without_becoming_the_answer(grid):
    """Both degenerate ends at once: it has to occur, and it has to stay a minority.

    The share is of what `stable` says, not of the whole item. Every stable query in an
    unattacked-ring item has the same answer, so left uncapped the status-balancing loop
    in `build` treats them as a bucket to draw from and hands one item four of them.
    """
    said = collections.Counter(status for item in grid.values()
                               for (_, sem), status in item.gold.items()
                               if sem == sq.STABLE)
    assert said["NO_STABLE_EXTENSION"], (
        "no exported item answers `no stable extension`, and the prompt promises it on "
        "every item that asks about stable")
    share = said["NO_STABLE_EXTENSION"] / sum(said.values())
    assert share < 0.2, (f"`no stable extension` is {share:.0%} of what stable says, so "
                         f"guessing it on every stable query pays that much")


def test_an_unattacked_ring_item_asks_stable_once(grid):
    """There, `no stable extension` is a fact about the theory, so a second query repeats."""
    for cell, item in grid.items():
        if not item.metadata["unshielded_ring"]:
            continue
        asked = [c for c, sem in item.queries if sem == sq.STABLE]
        assert len(asked) == 1, f"{cell} spends {len(asked)} queries on one fact"


def test_every_cluster_kind_the_menu_names_is_built():
    """`_odd` shipped for months called from an `else` the menu could not reach (#77).

    The menu is a literal inside `build`, so it is read from the source rather than
    restated here: a kind restated in a test is a kind the test cannot catch. The `else`
    now raises, so a kind added to the menu and forgotten in the dispatch stops the build.
    """
    src = inspect.getsource(sq.build)
    menu = {tok.strip().strip('"') for tok
            in src.split("_menu = [")[1].split("]")[0].replace("\n", "").split(",")}
    built = {kind for kind in menu | {"floating", "odd"} if f'kind == "{kind}"' in src}
    assert menu <= built, f"{sorted(menu - built)} is on the menu and never built"
    assert "odd" in built, "the odd cycle is unreachable again"
    assert "raise ValueError" in src, "an unknown cluster kind falls through silently"


@pytest.mark.parametrize("cell", CELLS)
def test_the_reference_still_scores_itself(cell):
    """`no stable extension` is a status the prompt already named and the parser already
    read (#76). This is the check that gold now written with it round-trips."""
    item = sq.make_item(cell[0], cell[2], cell[1])
    assert item is not None
    assert sq.score(item.reference, item).score == pytest.approx(1.0)
