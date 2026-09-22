"""A taskset is defined by a file, not by whatever the defaults were.

`docs/dataset-contract.md` section 7. The export grid is a function default the
CLI cannot override today, so a published taskset is identified by a commit
rather than by an input anyone can read.
"""
import json

import pytest

from arggym.core.spec import LEVELS, SeedPolicy, TasksetSpec, from_dict, load


def test_a_minimal_spec_gets_the_evaluated_grid():
    s = TasksetSpec(tasks=("status_query",))
    assert s.levels == LEVELS == tuple(range(1, 16))
    assert len(s.orderings) == 4
    assert len(s.cells) == 1 * 15 * 4


def test_a_cell_asks_for_items_and_bounds_the_search():
    # take, not stop: a cell that rejects most of its seeds must fail loudly
    # rather than ship short. The frozen v2 taskset has a cell that rejected
    # 12 of 20 seeds.
    p = SeedPolicy(start=0, take=2, scan_limit=40)
    assert (p.take, p.scan_limit) == (2, 40)


def test_a_bound_below_the_ask_could_never_be_filled():
    with pytest.raises(ValueError, match="never be filled"):
        SeedPolicy(take=10, scan_limit=4)


def test_an_unknown_task_is_refused_when_the_spec_is_read():
    with pytest.raises(ValueError, match="unknown task"):
        TasksetSpec(tasks=("status_queries",))


def test_an_unknown_ordering_is_refused():
    with pytest.raises(ValueError, match="unknown ordering"):
        TasksetSpec(tasks=("status_query",), orderings=("last_link",))


def test_a_typo_in_a_spec_file_is_an_error_not_a_no_op():
    # Silently ignoring an unknown key means a misspelled setting changes
    # nothing and the generator gets blamed.
    with pytest.raises(ValueError, match="unknown spec field"):
        from_dict({"tasks": ["status_query"], "orderigns": ["last_link_elitist"]})


def test_a_bare_string_where_a_list_belongs_is_refused():
    # tasks: status_query in YAML is a string, and iterating it would register
    # twelve one-character task names.
    with pytest.raises(ValueError, match="must be a list"):
        from_dict({"tasks": "status_query"})


def test_a_non_full_profile_says_why_it_is_refused():
    # The axis is real but under-built: two tasks take no profile and only one
    # mixes it into its seed, so the seed would not name the item.
    with pytest.raises(ValueError, match="not exportable yet"):
        TasksetSpec(tasks=("status_query",), profile="P_S")


def test_a_level_that_is_not_an_integer_is_refused():
    # levels=(3.5,) reached the generators: claim_chain raised TypeError on a
    # slice and status_query built an item, so one spec both crashed and shipped.
    with pytest.raises(ValueError, match="levels must be integers"):
        TasksetSpec(tasks=("status_query",), levels=(3.5,))
    # bool is an int subclass, so isinstance would have read `[true]` as level 1.
    with pytest.raises(ValueError, match="levels must be integers"):
        TasksetSpec(tasks=("status_query",), levels=(True,))


def test_the_build_acceptance_guard_takes_a_share_and_is_off_by_default():
    # Seed acceptance is 1.00 on every cell of the standard grid, so the seed
    # guard cannot fire there; the candidate guard can, and a spec that does not
    # name it keeps the old behaviour (#114).
    assert TasksetSpec(tasks=("status_query",)).min_build_acceptance == 0.0
    with pytest.raises(ValueError, match="min_build_acceptance"):
        TasksetSpec(tasks=("status_query",), min_build_acceptance=1.5)


def test_a_spec_round_trips_through_a_file(tmp_path):
    s = TasksetSpec(tasks=("status_query", "claim_chain"), levels=(3, 6),
                    seeds=SeedPolicy(take=4, scan_limit=20))
    p = tmp_path / "taskset.json"
    p.write_text(json.dumps(s.to_dict()))
    assert load(str(p)) == s


def test_the_checked_in_spec_describes_todays_grid():
    # The standard spec replaces the function defaults in core/export.py, so it
    # has to name the same grid: 12 tasks x 15 levels x 4 orderings x 2 seeds.
    from pathlib import Path

    s = load(str(Path(__file__).resolve().parent.parent / "tasksets" / "standard.yaml"))
    assert len(s.tasks) == 12
    assert s.levels == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    assert len(s.orderings) == 4
    assert len(s.cells) * s.seeds.take == 1440
    # The guard thresholds describe the grid too, and nothing else here reads
    # them: set min_build_acceptance to 0.9 and every test in the suite still
    # passes while the next `make freeze` dies on the first cell it reaches that
    # falls under. The 0.1 is set against the thinnest cell of the grid,
    # semantics_query at L12 under last-link democratic, which keeps 2
    # candidates of 14 (#114).
    assert s.min_build_acceptance == 0.1


def test_the_checked_in_spec_matches_this_tree():
    # The version fields are constraints, not records: a spec naming a pyarg
    # this tree does not have would produce a taskset the spec does not name.
    from pathlib import Path

    from arggym.core.spec import check_versions

    check_versions(load(str(Path(__file__).resolve().parent.parent
                            / "tasksets" / "standard.yaml")))
