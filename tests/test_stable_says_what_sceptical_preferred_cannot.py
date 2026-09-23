"""`stable` has to answer something `sceptical preferred` does not, without being guessable.

Both names read the same all/any/none formula off their own extension family
(`semantics_query.status_under`), so they answer differently only where the families
differ. No cluster built anything that made them differ: preferred equalled stable on all
40 cells the grid exported then and 360 of 360 in a wider sweep, `no stable extension`
never occurred, and over every queried claim the two names disagreed zero times --
`stable` was `sceptical preferred` under a second name (#77).

Odd defeat cycles are the whole of the difference. Dung's coherence result says a graph
with no odd-length cycle has its preferred and stable extensions equal, so a generator
that never builds one cannot separate them. What separates them cheaply is an odd cycle
that something else can kill: an undercut ring standing over one branch of a contested
pair, so the branch that survives is stable while the branch keeping the ring alive is
preferred and not stable.

Half of these tests exist because the first cut of that fix moved the degeneracy instead
of removing it. Asking both of the cluster's literals under both names put a complementary
pair in every item, and the cluster makes both undecided under sceptical preferred by
construction, so that column went to 88% one answer and `if c and -c are both asked,
answer undecided` was never wrong. A cluster that hands a solver free answers is the same
defect as #77 one column over, so the shares are asserted here rather than left to a
reader to notice.
"""
from __future__ import annotations

import collections
import itertools
import random

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import semantics_query as sq

CELLS = [(lv, o, s) for lv in LEVELS for o in ALL_ORDERINGS for s in SEEDS]
ASKS_STABLE = [c for c in CELLS if sq.STABLE in sq.semantics_for(c[0])]
ASKS_BOTH = [c for c in ASKS_STABLE if sq.SCEPT_PREF in sq.semantics_for(c[0])]
CLUSTER_DRAWS = list(itertools.product(ALL_ORDERINGS, range(6)))


@pytest.fixture(scope="module")
def grid():
    """The exported grid, built once: `make_item` walks up to 24 build seeds a cell."""
    return {c: sq.make_item(c[0], c[2], c[1]) for c in CELLS}


def cluster(unshielded: bool, seed: int):
    return sq._odd_cycle(iter(sq._names(seed, 12)), [0], random.Random(seed), unshielded)


def statuses(ops, ordering, claims):
    verifier = ASPICVerifier.from_operations(list(ops), ordering=ordering)
    preferred, stable = verifier.preferred_conclusions(), verifier.stable_conclusions()
    cache = sq.SemCache(grounded=verifier.status_map(), preferred=preferred, stable=stable)
    return preferred, stable, {(c, s): sq.status_under(cache, c, s)
                               for c in claims for s in (sq.SCEPT_PREF, sq.STABLE)}


@pytest.mark.parametrize("ordering,seed", CLUSTER_DRAWS)
def test_a_shielded_ring_makes_stable_decide_what_preferred_leaves_open(ordering, seed):
    """The cluster's claim, measured on the cluster alone.

    The ring stands over one side of the contested pair, so the extension taking the other
    side defeats the rule the ring hangs from and every rule in it, and is stable; the
    extension taking the ring's own side stays preferred and is not. Sceptical preferred
    sees two extensions and calls both literals undecided; stable sees one and decides
    them.
    """
    ops, _ = cluster(unshielded=False, seed=seed)
    pair = [o.content for o in ops if o.kind == "premise"]
    preferred, stable, status = statuses(ops, ordering, pair)
    assert len(preferred) == 2 and len(stable) == 1, (
        f"{len(preferred)} preferred and {len(stable)} stable extensions")
    assert {status[(c, sq.SCEPT_PREF)] for c in pair} == {"UNDECIDED"}
    assert {status[(c, sq.STABLE)] for c in pair} == {"JUSTIFIED", "OVERRULED"}


@pytest.mark.parametrize("ordering,seed", CLUSTER_DRAWS)
def test_a_ring_over_both_branches_leaves_no_stable_extension(ordering, seed):
    """The case #77 asked for: nothing kills either ring, so nothing is stable.

    It is the rarer one here because it answers `no stable extension` for every claim in
    the item rather than about a claim, so on many items it would be most of what stable
    says.
    """
    ops, _ = cluster(unshielded=True, seed=seed)
    pair = [o.content for o in ops if o.kind == "premise"]
    preferred, stable, status = statuses(ops, ordering, pair)
    assert preferred, "an odd cycle still has a preferred extension"
    assert stable == [], "a ring is attacked, so a stable extension survives"
    assert all(status[(c, sq.STABLE)] == "NO_STABLE_EXTENSION" for c in pair)


def test_the_two_extension_families_differ_wherever_stable_is_asked(grid):
    """The measurement #77 opened on, read off the theory: equal on all 40 cells then."""
    for cell in ASKS_STABLE:
        verifier = ASPICVerifier.from_operations(list(grid[cell].base_ops), ordering=cell[1])
        preferred = {frozenset(e) for e in verifier.preferred_conclusions()}
        stable = {frozenset(e) for e in verifier.stable_conclusions()}
        assert preferred != stable, (
            f"{cell}: the two families are equal, so stable repeats sceptical preferred")


def test_the_two_semantics_disagree_about_claims_the_items_actually_ask(grid):
    """#77's own metric: `status_under` differed on 0 of 75 queried claims."""
    disagree = total = 0
    for cell in ASKS_BOTH:
        item = grid[cell]
        cache = sq.semantics_cache(item.base_ops, item.metadata["semantics"], cell[1])
        for claim in sorted({c for c, _ in item.queries}):
            total += 1
            disagree += (sq.status_under(cache, claim, sq.SCEPT_PREF)
                         != sq.status_under(cache, claim, sq.STABLE))
    assert disagree / total > 0.25, (
        f"the two semantics disagree about {disagree} of {total} queried claims")


def test_some_items_ask_one_claim_under_both_names_and_answer_it_differently(grid):
    """A theory that separates the two and an item that shows it are different things.

    Not every item: a claim from another cluster carries the pairing on some of them, so
    that "asked under both names" does not by itself mean "the cluster's literal, therefore
    undecided under sceptical preferred". The coin is fair and the realised split is not --
    the cluster's own literal still carries the pairing on 174 of 213 items, 82%, because
    the retry loop accepts cluster-paired items more often and because the cluster's
    literal often survives the trim under both names anyway. So this bounds the pattern
    rather than removing it.
    """
    shown = 0
    for cell in ASKS_BOTH:
        item = grid[cell]
        asked = {(c, s) for c, s in item.queries}
        if any(s == sq.STABLE and (c, sq.SCEPT_PREF) in asked
               and item.gold[(c, sq.STABLE)] != item.gold[(c, sq.SCEPT_PREF)]
               for c, s in asked):
            shown += 1
    assert shown > len(ASKS_BOTH) // 2, (
        f"only {shown} of {len(ASKS_BOTH)} items show the difference to a reader")
    assert shown < len(ASKS_BOTH), (
        "every item pairs the cluster's own literal, which makes the pairing the answer")


def test_no_semantics_answers_the_grid_with_one_word(grid):
    """The defect #77 reports, checked on every column rather than on the two it names.

    `MAX_STATUS_SHARE` cannot see this: it pools all five semantics into one per-item
    ratio, and the required queries are exempt from its per-step check. The first cut of
    this change took sceptical preferred to 88% undecided while passing that gate, and
    these bounds are set to catch a regression of that size.

    They are not evidence that any column improved. Forty items give a column of about 35
    answers, and the standard error on a difference of two such shares is around 12 points,
    so a few points either way on this grid says nothing. Two known consequences of that:
    the sceptical-preferred share reads better than main here and worse at n = 14,000, and
    the generic 0.9 bound sits about 1.6 points above the credulous-preferred column, which
    is the one column this change measurably worsened (85.3% to 88.9% over 1,920 items,
    from the cluster taking a menu slot that would otherwise draw `_defeated`). The bound
    lets that through by design; `docs/dataset-card.md` reports it instead.
    """
    columns = collections.defaultdict(collections.Counter)
    for item in grid.values():
        for (_, sem), status in item.gold.items():
            columns[sem][status] += 1
    for sem, counts in sorted(columns.items()):
        top, n = counts.most_common(1)[0][1], sum(counts.values())
        assert top / n < 0.9, f"{sem} answers {top}/{n} = {top / n:.0%} with one word"
    for sem, limit in ((sq.SCEPT_PREF, 0.65), (sq.STABLE, 0.55)):
        counts = columns[sem]
        top, n = counts.most_common(1)[0][1], sum(counts.values())
        assert top / n < limit, f"{sem} is {top / n:.0%} one answer, over {limit:.0%}"


def test_the_cluster_literals_are_never_asked_where_their_answer_is_free(grid):
    """A contested pair is undecided under grounded and justified under credulous
    preferred whatever else the theory says, so asking it there is two given answers an
    item spends its questions on. It is asked under the two semantics it separates."""
    for cell, item in grid.items():
        literals = set(item.metadata["odd_cycle_claims"])
        for claim, sem in item.queries:
            assert not (claim in literals and sem in (sq.GROUNDED, sq.CRED_PREF)), (
                f"{cell} asks {claim} under {sem}, which the cluster settles for free")


#: Every cluster kind the generator can draw, and the function that builds it. Named here
#: rather than read out of `build`'s source, because a test that greps source passes a
#: dict dispatch that is broken and fails a refactor that is not.
BUILDERS = {"floating": "_floating", "odd": "_odd_cycle", "defeated": "_defeated",
            "settled": "_settled", "undermining": "_self_undermining",
            "junction": "_junction_cluster", "strict_axiom": "_strict_axiom",
            "negated_premise": "_negated_premise"}
MENU = [k for k in BUILDERS if k not in ("floating", "odd")]


def test_every_cluster_kind_reaches_a_builder_of_its_own(monkeypatch):
    """`_odd` was dispatched from an `else` the menu could not reach, so it shipped
    unbuilt for as long as it existed (#77).

    The `else` now falls to `_negated_premise`, so a kind added to the menu and forgotten
    in the dispatch would not crash -- it would quietly build the wrong cluster, which is
    the same defect wearing the same clothes. Counting which builder each draw reaches
    catches that: a forgotten kind shows up as one builder never called and
    `_negated_premise` absorbing its share, which on the grid runs 15% against a fair
    share of 17%.
    """
    calls = collections.Counter()
    for kind, name in BUILDERS.items():
        original = getattr(sq, name)

        def spy(*args, _kind=kind, _original=original, **kwargs):
            calls[_kind] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(sq, name, spy)
    for level, ordering, seed in CELLS:
        sq.make_item(level, seed, ordering)

    missing = [k for k in BUILDERS if not calls[k]]
    assert not missing, f"{missing} is dispatched to no builder of its own"
    drawn = sum(calls[k] for k in MENU)
    share = calls["negated_premise"] / drawn
    assert share < 0.25, (
        f"the else branch built {share:.0%} of the menu draws against a fair share of "
        f"{1 / len(MENU):.0%}, so it is absorbing a kind the dispatch forgot")


#: Which ordering carries the unattacked ring at each level that asks about stable, read
#: off `wants_unshielded_ring`. Ten levels ask (6 and up), and `ALL_ORDERINGS[level % 4]`
#: walks the four orderings one level at a time, so the ten split 3/3/2/2. Pinned as a
#: table because the shape of it is the thing under test and no formula states it more
#: clearly than the rows do.
RING = {
    6: "weakest_link_elitist", 7: "weakest_link_democratic",
    8: "last_link_elitist", 9: "last_link_democratic",
    10: "weakest_link_elitist", 11: "weakest_link_democratic",
    12: "last_link_elitist", 13: "last_link_democratic",
    14: "weakest_link_elitist", 15: "weakest_link_democratic",
}


def test_the_unattacked_ring_stays_thin_and_reaches_every_ordering():
    """One ordering per stable-asking level, and no ordering left out or left alone.

    The ring answers `no stable extension` for every claim in its item, so it has to be
    rare enough that the answer stays unguessable and spread enough that it does not land
    on one column of a report. One ordering per level is the rarity: three of the four
    orderings at every stable-asking level build an ordinary shielded ring, which caps the
    ring at a quarter of the stable rows. Reaching all four orderings is the spread: an
    ordering that carried none would have its `stable` column built from a different
    population than the rest, and a per-ordering mean would read that as an ordering effect.

    Not keyed on the seed, because `tasksets/standard.yaml` scans up to 40 seeds a cell and
    keeps the first ones that build, so a seed-keyed rule realises an uncontrolled fraction.

    The spread is even as well as complete: no ordering carries more than one ring above
    any other, and no two adjacent levels share one, so each ordering's rings sit four
    levels apart rather than in one block of the curriculum (#168).
    """
    asks = [lv for lv in range(1, 16) if sq.STABLE in sq.semantics_for(lv)]
    chosen = {lv: [o for o in ALL_ORDERINGS if sq.wants_unshielded_ring(lv, o)]
              for lv in asks}
    more_than_one = {lv: v for lv, v in chosen.items() if len(v) != 1}
    assert not more_than_one, (
        f"a stable-asking level does not carry the ring on exactly one ordering: "
        f"{more_than_one}")
    assert {lv: v[0] for lv, v in chosen.items()} == RING, chosen
    carried = collections.Counter(v[0] for v in chosen.values())
    missing = [o for o in ALL_ORDERINGS if not carried[o]]
    assert not missing, (
        f"{missing} carries the ring at no level, so its `stable` column is built from a "
        f"different population than the other orderings': {chosen}")
    assert max(carried.values()) * 2 <= len(asks), (
        f"one ordering carries {max(carried.values())} of the {len(asks)} rings, so most "
        f"of what `stable` answers under it comes from ring items and its column is a "
        f"measurement of the ring rather than of the ordering: {dict(carried)}")
    assert max(carried.values()) - min(carried.values()) <= 1, (
        f"the rings split {dict(carried)} across orderings, where {len(asks)} levels "
        f"allow a split within one of even")
    blocks = [lv for lv in asks[:-1] if RING[lv] == RING[lv + 1]]
    assert not blocks, (
        f"levels {blocks} hand the ring to the same ordering as the level after them, so "
        f"that ordering's rings sit in one stretch of the curriculum")
    for level in range(1, 16):
        if level in chosen:
            continue
        assert not any(sq.wants_unshielded_ring(level, o) for o in ALL_ORDERINGS), (
            f"level {level} never asks about stable, so a ring there asks nothing")


def test_the_release_ships_every_ordering_the_ring_is_balanced_over():
    """The balance above is over `ALL_ORDERINGS`, so it holds for the release only while
    the release ships all four. Drop one and its ring levels export no ring at all, which
    moves the ring's share of the `stable` rows without failing anything above.
    """
    from pathlib import Path

    from arggym.core.spec import load

    shipped = load(str(Path(__file__).resolve().parent.parent / "tasksets"
                       / "standard.yaml")).orderings
    assert set(shipped) == set(ALL_ORDERINGS), (
        f"tasksets/standard.yaml ships {shipped}, so the rings at levels "
        f"{sorted(lv for lv, o in RING.items() if o not in shipped)} are never exported; "
        f"re-read the ring's share of the stable rows on the release")


def test_the_grid_exports_every_level_that_carries_a_ring():
    """A ring at a level the grid skips is a ring no row carries.

    The table above is over the curriculum and the rows a report reads are the exported
    ones, so the balance argument holds only while the two agree. That is the same #30
    dependency the rest of this suite carries: a schedule keyed on the level and a grid
    that skips levels can disagree without either of them looking wrong.
    """
    assert set(RING) <= set(LEVELS), (
        f"levels {sorted(set(RING) - set(LEVELS))} carry a ring the grid never exports")


def test_no_stable_extension_is_answered_without_becoming_the_answer(grid):
    """Both degenerate ends at once: it has to occur, and it has to stay a minority.

    Every stable query in an unattacked-ring item has the same answer, so left uncapped
    the status-balancing loop in `build` treats them as a bucket to draw from and hands
    one item four of them.
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
    for cell, item in grid.items():
        if not item.metadata["unshielded_ring"]:
            continue
        asked = [c for c, sem in item.queries if sem == sq.STABLE]
        assert len(asked) == 1, f"{cell} spends {len(asked)} queries on one fact"


def test_every_reference_scores_itself_and_some_of_them_say_no_stable_extension(grid):
    """`no stable extension` is a status the prompt already named and the parser already
    read (#76). This is the check that gold now written with it round-trips."""
    for cell, item in grid.items():
        assert sq.score(item.reference, item).score == pytest.approx(1.0), cell
    assert any("no stable extension" in item.reference for item in grid.values())
