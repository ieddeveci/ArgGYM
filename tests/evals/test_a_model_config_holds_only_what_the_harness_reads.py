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
    # And every sampling key is one the client forwards rather than refuses.
    Endpoint(model=cfg["model"], sampling=cfg.get("sampling") or {}).check()
