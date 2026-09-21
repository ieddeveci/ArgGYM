from __future__ import annotations

import importlib.metadata as md

import arggym
import pytest

from rl.generate import _build_quota_pool, _frozen_index


def test_generated_domain_seed_namespace_is_disjoint_from_frozen():
    try:
        assert md.version("python-argumentation") == "2.0.2"
    except (md.PackageNotFoundError, AssertionError):
        pytest.skip("procedural generation requires pinned python-argumentation==2.0.2")

    spec = arggym.load_spec("tasksets/standard.yaml")
    _m, _rows, forbidden = _frozen_index("data/taskset.jsonl")
    quotas = {("status_query", 3, "last_link_elitist"): 1}
    rows, manifest = _build_quota_pool(
        quotas=quotas,
        split="RL_TEST_SPLIT",
        master_seed_namespace="arggym-rl-unit-test-v1",
        profile=spec.profile,
        forbidden=forbidden,
        bands={"easy": [1,2,3,4,5], "medium": [6,7,8,9,10], "hard": [11,12,13,14,15]},
        max_candidates_per_item=80,
        curriculum_stage="test",
    )
    assert len(rows) == 1
    assert manifest["split_namespace"] == "RL_TEST_SPLIT"
    coord = (rows[0]["task"], rows[0]["level"], rows[0]["ordering"], rows[0]["seed"])
    assert coord not in forbidden["coordinate"]
