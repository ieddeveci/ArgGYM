"""The irredundance scan carries a call budget and reports when it runs out (issue #28).

`assert_irredundant` asks whether any proper subset of the reference answer still solves
the item. A full answer is 2^n calls, and `counter_argument` reached n=25 at level 12 under
`weakest_link_democratic`: 33.5 million verifier calls, 62 hours at the 6.7ms a call that
theory costs. The scan sat after a `minimal_subset_exact` that had finished in 0.2s, so the
budget the bounded search advertised did not bound the generator.

The result only ever reached `metadata["reference_irredundant"]`, never a gate, so a budget
cannot change which items exist or what they ask for. What it can change is whether the flag
tells the truth, which is why exhaustion reports None rather than a verdict.

These tests use a fake `holds` so they pin the search's contract rather than any theory.
"""
from __future__ import annotations

import itertools

from arggym.core.invariants import assert_irredundant


def test_irredundant_set_is_reported_true():
    """Nothing can be dropped: only the whole set holds."""
    full = ["a", "b", "c", "d"]
    verdict, calls = assert_irredundant(full, lambda sub: len(sub) == len(full))
    assert verdict is True
    assert calls == 2 ** len(full) - 2  # every proper non-empty subset


def test_a_droppable_element_is_reported_false():
    def holds(sub):
        return {"a", "b"} <= set(sub)

    verdict, _calls = assert_irredundant(["a", "b", "c"], holds)
    assert verdict is False


def test_exhausted_budget_reports_none_not_a_verdict():
    """The failure this issue is about: a scan that ran out must not read as clean."""
    full = list(range(20))
    verdict, calls = assert_irredundant(full, lambda sub: len(sub) == len(full),
                                        max_calls=100)
    assert verdict is None, "an unfinished scan claimed a result"
    assert calls == 100


def test_the_budget_is_never_exceeded():
    seen = []

    def holds(sub):
        seen.append(len(sub))
        return False

    _verdict, calls = assert_irredundant(list(range(20)), holds, max_calls=37)
    assert calls == 37
    assert len(seen) == 37


def test_the_scan_removes_elements_before_it_tries_small_subsets():
    """Downward, largest first.

    Both directions cover the same count, since C(n, k) equals C(n, n - k). What the order
    decides is what a budget that runs out has established: downward it is "no removal of
    this many or fewer works", upward it is "no subset this small suffices", which says
    almost nothing about a set already known to need every element.
    """
    full = list(range(12))
    sizes = []

    def holds(sub):
        sizes.append(len(sub))
        return False

    assert_irredundant(full, holds, max_calls=len(full) + 1)
    assert sizes[0] == len(full) - 1, f"first subset tested had size {sizes[0]}"
    assert sizes == sorted(sizes, reverse=True)


def test_a_full_scan_still_settles_the_question():
    """A budget large enough for 2^n leaves the old answer intact."""
    full = ["a", "b", "c", "d", "e"]
    for keep in itertools.chain.from_iterable(
            itertools.combinations(full, k) for k in (3, 4)):
        need = set(keep)
        verdict, _calls = assert_irredundant(full, lambda sub: need <= set(sub))
        assert verdict is False
