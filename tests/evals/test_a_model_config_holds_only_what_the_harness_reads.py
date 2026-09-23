"""A key in a model config that `build_solver` never reads is a setting nobody gets.

Twenty configs carried a top-level `timeout_s` and `retries`. `build_solver`
takes both from `cfg.endpoint`, so `timeout_s: 30` in one of them ran at 5400
while the file said otherwise (#163).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evals.client import Endpoint

CONFIGS = sorted((Path(__file__).resolve().parents[2] / "evals" / "conf" / "model")
                 .glob("*.yaml"))

#: What `evals/run.py:build_solver` reads off `cfg.model`, plus the `name` that
#: `run_id` is built from.
READ = {"name", "model", "base_url", "api_key_env", "sampling", "extra_body"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_key_in_a_model_config_is_read(path):
    cfg = yaml.safe_load(path.read_text())
    assert set(cfg) <= READ, f"{path.name} sets {sorted(set(cfg) - READ)}, which nothing reads"
    # `run_id` is built from `name` and `make eval` finds the run directory by the
    # file's stem, so the two must agree.
    assert cfg["name"] == path.stem
    # And every sampling key is one the client forwards rather than refuses.
    Endpoint(model=cfg["model"], sampling=cfg.get("sampling") or {}).check()


def test_the_timeout_outlasts_the_largest_cap_at_a_slow_decode():
    """The cap, not the clock, has to end a long generation.

    A request the clock stops is an API error and leaves the denominator; one the
    cap stops is a truncation the score can see. So `endpoint.timeout_s` must let
    the largest cap finish at 10 tokens/s, a deliberately slow single request.
    """
    root = Path(__file__).resolve().parents[2]
    conf = yaml.safe_load((root / "evals" / "conf" / "config.yaml").read_text())
    timeout = conf["endpoint"]["timeout_s"]
    caps = {}
    for p in CONFIGS:
        s = yaml.safe_load(p.read_text()).get("sampling") or {}
        caps[p.stem] = s.get("max_tokens", s.get("max_completion_tokens", 0))
    worst = max(caps, key=caps.get)
    assert timeout >= caps[worst] / 10, f"{worst}'s cap {caps[worst]} outlasts {timeout}s"
    assert Endpoint(model="m").timeout_s == timeout, "the client default lags the config"


LEVELS = ("minimal", "low", "medium", "high", "xhigh")


def test_the_configs_of_one_model_differ_only_in_their_effort_level():
    """`<config>-<level>` files are one model at several levels, and nothing else.

    They are written as separate files, so a sampling change made to one and not
    its siblings would turn a comparison across levels into a comparison across
    settings.
    """
    groups = {}
    for p in CONFIGS:
        for level in LEVELS:
            if p.stem.endswith(f"-{level}"):
                groups.setdefault(p.stem[: -len(level) - 1], {})[level] = p

    def masked(node, level):
        if isinstance(node, dict):
            return {k: masked(v, level) for k, v in node.items() if k != "name"}
        return "<level>" if node == level else node

    assert groups, "no per-level configs found"
    for base, files in groups.items():
        shapes = {lv: masked(yaml.safe_load(p.read_text()), lv) for lv, p in files.items()}
        first = next(iter(shapes.values()))
        for lv, shape in shapes.items():
            assert shape == first, f"{base}-{lv} differs from its siblings beyond the level"
            assert "<level>" in str(shape), f"{base}-{lv} never sends its level"


@pytest.mark.parametrize("path", [p for p in CONFIGS if p.stem != "stub"],
                         ids=lambda p: p.stem)
def test_no_real_model_decodes_greedily(path):
    """Every card we cite recommends sampling, and reasoning APIs refuse temperature 0."""
    t = (yaml.safe_load(path.read_text()).get("sampling") or {}).get("temperature")
    assert t is None or t > 0
