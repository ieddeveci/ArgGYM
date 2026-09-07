"""A discarded build candidate says why, and the export counts every one.

Issue #72. `build` used to answer a bare `None`; `claim_chain.build` alone
refuses at five distinct sites and the manifest recorded all five as
`build_returned_none`.

The count that matters is per build call, not per failed seed. A sweep of 720
grid seeds found `make_item` returning `None` for none of them, so recording
failed seeds alone would ship a feature that prints nothing, while
`semantics_query` at level 12 quietly spends 30 candidates on 4 items. That gap
is what `test_the_export_counts_what_a_filled_cell_discarded` pins: a cell at
100% seed acceptance whose build acceptance is a third.
"""
from __future__ import annotations

import pytest

from arggym.core import freeze as F
from arggym.core.build import Rejected, retry
from arggym.core.spec import SeedPolicy, TasksetSpec
from arggym.tasks import claim_chain as cc
from arggym.tasks import semantics_query as smq

CHEAP = TasksetSpec(tasks=("claim_chain",), levels=(3,),
                    orderings=("last_link_elitist",),
                    seeds=SeedPolicy(start=0, take=2, scan_limit=10))


def test_retry_returns_the_item_and_counts_what_it_discarded():
    def candidate(k):
        return "item" if k == 2 else Rejected("no_junction" if k else "too_thin")

    report = retry(candidate, 10)
    assert report.item == "item"
    assert report.calls == 3, "the accepted candidate is counted too"
    assert report.reasons == {"too_thin": 1, "no_junction": 1}


def test_a_seed_that_never_builds_reports_every_reason():
    report = retry(lambda k: Rejected("too_thin"), 4)
    assert report.item is None
    assert report.calls == 4
    assert report.reasons == {"too_thin": 4}


def test_a_bare_none_is_a_missed_return_site_not_an_empty_cell():
    # Sixty-six `return None` sites became `return Rejected(...)`; one missed
    # would otherwise hand an item of None to the row builder.
    with pytest.raises(TypeError, match="must say why"):
        retry(lambda k: None, 2)


def test_make_item_still_answers_none_rather_than_a_sentinel(monkeypatch):
    # Around fifty call sites in tests/ and arggym/inspector.py are shaped
    # `is not None`, and a truthy sentinel passes every one of them.
    monkeypatch.setattr(cc, "build", lambda *a, **kw: Rejected("hand_rejected"))
    assert cc.make_item(3, 0) is None
    assert cc.make_item_report(3, 0).reasons == {"hand_rejected": 14}


def test_the_export_counts_what_a_filled_cell_discarded(monkeypatch, tmp_path):
    """Every seed succeeds, and two candidates in three are still thrown away.

    Seed acceptance reads 1.00 here, which is what it reads on all 240 cells of
    the standard grid. The build-level numbers are the ones carrying a signal.
    """
    real, seen = cc.build, {"n": 0}

    def flaky(*a, **kw):
        seen["n"] += 1
        return real(*a, **kw) if seen["n"] % 3 == 0 else Rejected("hand_rejected")

    monkeypatch.setattr(cc, "build", flaky)
    manifest = F.freeze(CHEAP, str(tmp_path / "t.jsonl"), verbose=False)
    cell = manifest["cells"]["claim_chain|L3|last_link_elitist"]

    assert cell["n"] == 2
    assert cell["acceptance_rate"] == 1.0
    assert cell["seeds_skipped"] == []
    assert cell["build_calls"] == 6
    assert cell["build_rejections"] == {"hand_rejected": 4}
    assert cell["build_acceptance_rate"] == pytest.approx(2 / 6, abs=1e-4)
    assert manifest["n_skipped"] == 0
    assert manifest["n_rejected"] == 4


def test_a_seed_that_runs_out_of_tries_lists_the_reasons_it_hit(monkeypatch, tmp_path):
    real = cc.build

    def refuse_first_seed(level, seed, *a, **kw):
        # make_item mixes `seed * 71 + k`, so build seeds under 71 are spec seed 0.
        return Rejected("first_seed_refused") if seed < 71 else real(level, seed, *a, **kw)

    monkeypatch.setattr(cc, "build", refuse_first_seed)
    manifest = F.freeze(CHEAP, str(tmp_path / "t.jsonl"), verbose=False)
    cell = manifest["cells"]["claim_chain|L3|last_link_elitist"]
    assert [s["seed"] for s in cell["seeds_skipped"]] == [0]
    skipped = cell["seeds_skipped"][0]
    assert skipped["tries"] == 14
    assert skipped["reason"] == "first_seed_refused"
    assert skipped["reasons"] == {"first_seed_refused": 14}
    assert cell["reason_counts"] == {"first_seed_refused": 1}


def test_a_generator_names_reasons_of_its_own():
    """Not a hand-made string: the thinnest cell on the grid, refusing for cause.

    Level 12 semantics_query builds a theory over MAX_DIRECTIVES most of the time
    and throws it away. Before this, the item came out and the four discarded
    candidates behind it left no trace.
    """
    report = smq.make_item_report(12, 0, "last_link_democratic")
    assert report.item is not None
    assert report.calls > 1, "the cell the issue names discarded nothing"
    assert report.reasons["theory_over_max_directives"] >= 1
