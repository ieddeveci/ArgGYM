"""Adding a task cannot silently miss the export.

The pre-contract export dispatched through an if/elif ladder whose default
failure is a forgotten branch -- that is the mechanism behind #23, where two
tasks were absent from it. The registry replaces the ladder with a table, so
these tests check the table's own properties instead: every task builds, every
variant is registered separately, and every task declares how it is checked.
`test_registry_matches_inspector_modes.py` holds the registry to the one other
list of tasks still alive, the inspector's.
"""
import pytest

from arggym.core import registry


def test_every_registered_module_imports_and_can_build():
    for name in registry.task_names():
        spec = registry.get(name)
        item = spec.make_item(3, 0, "last_link_elitist")
        # A task may legitimately reject a cell, but it must not raise and must
        # not be missing its module.
        assert item is None or hasattr(item, "prompt"), name


def test_the_three_variants_are_separate_tasks_over_two_modules():
    # counter_argument_strict differs from counter_argument by a flag, and
    # attack/defence/attack_defense are three modes of one builder. The variant
    # is part of the task identity, so it belongs in the registry rather than in
    # a caller's if/elif.
    assert registry.get("counter_argument").variant == {"allow_strict": False}
    assert registry.get("counter_argument_strict").variant == {"allow_strict": True}
    assert registry.get("counter_argument_strict").module == "counter_argument"
    modes = {registry.get(n).variant["mode"] for n in ("attack", "defence", "attack_defense")}
    assert modes == {"attack", "defence", "attack_defense"}


def test_every_task_declares_how_its_answer_is_checked():
    kinds = {registry.VERIFIED, registry.GRADED, registry.EXACT}
    shapes = {registry.OPERATION_LIST, registry.LABEL_MAP,
              registry.ORDERED_SEQUENCE, registry.RECORD_LIST}
    for name in registry.task_names():
        spec = registry.get(name)
        assert spec.checker in kinds, name
        assert spec.answer_shape in shapes, name


def test_the_six_engine_checked_tasks_are_the_ones_with_no_oracle():
    # These are graded by running the engine on theory plus answer, so any
    # directive set reaching the goals is correct and a string comparison
    # against the reference is wrong. docs/dataset-contract.md section 2.
    verified = {n for n in registry.task_names()
                if registry.get(n).checker == registry.VERIFIED}
    assert verified == {"preference_construction", "counter_argument",
                        "counter_argument_strict", "attack", "defence",
                        "attack_defense"}


def test_an_unknown_task_says_what_is_known():
    with pytest.raises(KeyError, match="status_query"):
        registry.get("status_queries")
