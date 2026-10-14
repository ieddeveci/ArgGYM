"""One YAML file holds every setting of a run; `key.sub=value` arguments override it.

    python -m rl.launch rl/configs/qwen3-4b.yaml grpo.learning_rate=2e-5 data.levels=[1,2,3]

An override is parsed as YAML, so `2e-5` is a float, `[1,2]` a list and `true` a
bool. A key the file does not already have is refused: a typo would otherwise
train with the default it meant to change, and the run would record the typo as
if it had been applied.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, Sequence

import yaml


class _Loader(yaml.SafeLoader):
    """YAML 1.1 reads `1e-5` as a string, because it wants a dot in the mantissa.

    A learning rate written the way everyone writes one would then reach
    GRPOConfig as text. This loader reads it as the float it is.
    """


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"))


def _parse(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass


def load(path: str, overrides: Sequence[str] = ()) -> Dict[str, Any]:
    with open(path) as f:
        cfg = _parse(f.read())
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path} does not hold a mapping")
    return apply(cfg, overrides)


def apply(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"override {item!r} is not key=value")
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            if not isinstance(node.get(p), dict):
                raise SystemExit(f"override {key!r}: {p!r} is not a section of the config")
            node = node[p]
        if leaf not in node:
            raise SystemExit(f"override {key!r}: no such key; known here: {sorted(node)}")
        node[leaf] = _parse(raw)
    return cfg


def run_name(cfg: Dict[str, Any]) -> str:
    """Model, loss, learning rate and seed, which is what tells two curves apart.

    Prefixed with the config's name when that is not the model's (`smoke-...`).
    """
    model = cfg["model"]["name"].split("/")[-1].lower()
    g = cfg["grpo"]
    name = f"{model}-{g['loss_type']}-lr{g['learning_rate']:g}-s{cfg['seed']}"
    return name if cfg["name"] == model else f"{cfg['name']}-{name}"
