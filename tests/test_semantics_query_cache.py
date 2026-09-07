"""One theory is enumerated once, however many queries an item asks about.

Every query in a semantics_query item is asked about the same theory, so each semantics needs
exactly one extension enumeration. The generator used to build a fresh verifier and re-enumerate
per (claim, semantics) pair, which made high levels quadratic in the candidate count. These tests
count the enumerations rather than the seconds, so they stay meaningful on a loaded CI box.
"""
from __future__ import annotations

import collections

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import ASPICFramework
from arggym.core.build import Rejected
from arggym.tasks import semantics_query as sq

COUNTED = ("preferred_extensions", "stable_extensions", "eager_extension", "status_map")


@pytest.fixture
def counts(monkeypatch):
    calls = {name: 0 for name in COUNTED}
    calls["verifier"] = 0

    for name in COUNTED:
        original = getattr(ASPICFramework, name)

        def counting(self, _name=name, _original=original):
            calls[_name] += 1
            return _original(self)

        monkeypatch.setattr(ASPICFramework, name, counting)

    original_from_ops = ASPICVerifier.from_operations

    def counting_from_ops(*args, **kwargs):
        calls["verifier"] += 1
        return original_from_ops(*args, **kwargs)

    monkeypatch.setattr(ASPICVerifier, "from_operations", counting_from_ops)
    return calls


def first_built_item(level: int, counts: dict, ordering: str = sq.LAST_LINK):
    """Build until a seed yields an item, and report the counts of that build alone."""
    reasons = []
    for seed in range(40):
        for name in counts:
            counts[name] = 0
        item = sq.build(level, seed, ordering)
        # Not `is not None`: a Rejected satisfies that, so every test downstream of
        # this helper would pass on a generator that builds nothing.
        if not isinstance(item, Rejected):
            return item, dict(counts)
        reasons.append(item.reason)
    pytest.fail(f"no level {level} item in 40 seeds: {collections.Counter(reasons)}")


# The enumeration each semantics needs, so a test can ask for the schedule's own count
# rather than name levels. semantics_for decides which of these a level requests, and
# these tests hold whatever that schedule says.
ENUMERATION = {
    sq.SCEPT_PREF: "preferred_extensions",
    sq.CRED_PREF: "preferred_extensions",
    sq.STABLE: "stable_extensions",
    sq.EAGER: "eager_extension",
    sq.GROUNDED: "status_map",
}

TOP_LEVEL = max(sq.SEMANTICS_BY_LEVEL)


def test_top_level_enumerates_each_requested_semantics_once(counts):
    item, seen = first_built_item(TOP_LEVEL, counts)
    assert len(item.gold) >= 2
    wanted = {ENUMERATION[s] for s in sq.semantics_for(TOP_LEVEL)}
    for name in wanted:
        assert seen[name] == 1, f"{name} was re-enumerated per query"
    # Two constructions guard the eager argument budget; the gold loop reuses the second.
    assert seen["verifier"] <= 2


@pytest.mark.parametrize("level", sorted(sq.SEMANTICS_BY_LEVEL))
def test_unrequested_semantics_are_never_enumerated(level, counts):
    _item, seen = first_built_item(level, counts)
    wanted = {ENUMERATION[s] for s in sq.semantics_for(level)}
    for name in COUNTED:
        if name not in wanted:
            assert seen[name] == 0, f"{name} enumerated at level {level}, which does not ask for it"


@pytest.mark.parametrize("level", [1, 4, 8, 12])
def test_enumeration_is_flat_in_the_claim_count(level, counts):
    """A build enumerates at most once per requested semantics, never once per claim."""
    _item, seen = first_built_item(level, counts)
    total = sum(seen[name] for name in COUNTED)
    assert total <= len(sq.semantics_for(level)), seen
