"""Every mode generates an item whose reference answer the engine verifies.

This is the property the benchmark rests on: gold is never asserted, it is scored. The test drives
the inspector's HTTP layer because that is the one place where all twelve modes are reachable
through a single call - each task module has its own make_item and score signatures.
"""
from __future__ import annotations

import pytest

from arggym import inspector

MODES = [
    "attack", "attack_defense", "defence",
    "claim_chain", "counter_argument", "counter_argument_strict",
    "defeat_diagnosis", "formalization", "perturbation",
    "preference_construction", "semantics_query", "status_query",
]


@pytest.fixture(scope="module")
def client():
    return inspector.app.test_client()


@pytest.mark.parametrize("mode", MODES)
def test_reference_answer_verifies(client, mode):
    item = client.post("/api/generate",
                       json={"mode": mode, "level": 8, "seed": 1}).get_json()
    assert item["ok"], item.get("error")
    assert item["ref_score"] == pytest.approx(1.0), f"{mode}: gold does not verify"

    graded = client.post("/api/grade",
                         json={"token": item["token"], "answer": item["reference"]}).get_json()
    assert graded["ok"], graded.get("error")
    score = graded["result"]["score"]
    assert score == pytest.approx(1.0), f"{mode}: gold regrades at {score}"


def test_exportable_tasks_are_importable():
    from arggym.core import export
    assert export._EXPORTABLE, "no exportable tasks registered"
