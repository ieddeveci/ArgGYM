"""One theory is enumerated once, however many queries an item asks about.

Every query in a semantics_query item is asked about the same theory, so each semantics needs
exactly one extension enumeration. The generator used to build a fresh verifier and re-enumerate
per (claim, semantics) pair, which made high levels quadratic in the candidate count. These tests
count the enumerations rather than the seconds, so they stay meaningful on a loaded CI box.
"""
from __future__ import annotations

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import ASPICFramework
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
    for seed in range(40):
        for name in counts:
            counts[name] = 0
        item = sq.build(level, seed, ordering)
        if item is not None:
            return item, dict(counts)
    pytest.fail(f"no level {level} item in 40 seeds")


def test_level_14_enumerates_each_semantics_once(counts):
    item, seen = first_built_item(14, counts)
    assert len(item.gold) >= 2
    assert seen["preferred_extensions"] == 1, "preferred was re-enumerated per query"
    assert seen["stable_extensions"] == 1
    assert seen["eager_extension"] == 1
    assert seen["status_map"] == 1
    # Two constructions guard the eager argument budget; the gold loop reuses the second.
    assert seen["verifier"] <= 2


def test_unrequested_semantics_are_never_enumerated(counts):
    item, seen = first_built_item(8, counts)
    assert sq.STABLE not in sq.semantics_for(8) and sq.EAGER not in sq.semantics_for(8)
    assert seen["preferred_extensions"] == 1
    assert seen["stable_extensions"] == 0
    assert seen["eager_extension"] == 0
    assert seen["verifier"] == 1


@pytest.mark.parametrize("level", [1, 4, 8, 12])
def test_enumeration_is_flat_in_the_claim_count(level, counts):
    """A build enumerates at most once per requested semantics, never once per claim."""
    _item, seen = first_built_item(level, counts)
    total = sum(seen[name] for name in COUNTED)
    assert total <= len(sq.semantics_for(level)), seen
