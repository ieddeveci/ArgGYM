"""`stable` has to answer something `sceptical preferred` does not, without being guessable.

Both names read the same all/any/none formula off their own extension family
(`semantics_query.status_under`), so they answer differently only where the families
differ. No cluster built anything that made them differ: preferred equalled stable on 40
of 40 exported cells and 360 of 360 in a wider sweep, `no stable extension` never
occurred, and over every queried claim the two names disagreed zero times -- `stable` was
`sceptical preferred` under a second name (#77).

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
    """The measurement #77 opened on, read off the theory: equal on 40 of 40."""
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

    Not every item: the claim asked under both names is the cluster's own only about half
    the time, and a claim from elsewhere carries the pairing on the rest. Asking the
    cluster's literal under both on every item made the pairing itself the answer, since
    the cluster makes it undecided under sceptical preferred by construction.
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
    this change took sceptical preferred to 88% undecided while passing that gate.
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


def test_the_unattacked_ring_is_balanced_over_levels_and_orderings():
    """One ordering per level and one level per ordering, over the levels asking stable.

    Balanced on both marginals a report breaks out: each ordering carries 2 of its 10 grid
    rows, each stable-asking level 2 of its 8. Only the level-by-ordering cell is
    confounded and nothing reports that. Not keyed on the seed, because
    `tasksets/standard.yaml` scans up to 40 seeds a cell and keeps the first two that
    build, so a seed-keyed rule realises an uncontrolled fraction.
    """
    levels = [lv for lv in LEVELS if sq.STABLE in sq.semantics_for(lv)]
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
