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
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]


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

    Led by the config's name, with the model added when the name lacks it:
    `qwen3.5-4b-curriculum-dapo-...`, `smoke-qwen3-4b-dapo-...`.
    """
    model = cfg["model"]["name"].split("/")[-1].lower()
    g = cfg["grpo"]
    head = cfg["name"] if model in cfg["name"] else f"{cfg['name']}-{model}"
    return f"{head}-{g['loss_type']}-lr{g['learning_rate']:g}-s{cfg['seed']}"


#: What an eval model config may set. Any other key would change how the eval
#: lane samples without dev evaluation following, so it is refused, not ignored.
_SAMPLING = {"max_tokens", "temperature", "top_p", "seed", "presence_penalty"}
_EXTRA_BODY = {"top_k", "min_p", "repetition_penalty", "chat_template_kwargs"}


def eval_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The eval lane's model config that `eval_model` names, as training reads it.

    One file decides the chat-template arguments (thinking on or off) of the
    training prompts and the sampling and token cap of dev evaluation, so a run
    cannot drift from how the final eval asks the same model.
    """
    path = ROOT / "evals/conf/model" / f"{cfg['eval_model']}.yaml"
    m = yaml.safe_load(path.read_text())
    if m["model"] != cfg["model"]["name"]:
        raise SystemExit(f"{path} serves {m['model']!r}; the run trains {cfg['model']['name']!r}")
    sampling, extra = m.get("sampling") or {}, m.get("extra_body") or {}
    stray = sorted(set(sampling) - _SAMPLING) + sorted(set(extra) - _EXTRA_BODY)
    if stray:
        raise SystemExit(f"{path} sets {stray}, which dev evaluation does not reproduce")
    return {"name": m["name"], "model": m["model"],
            "temperature": sampling["temperature"], "top_p": sampling.get("top_p", 1.0),
            "top_k": extra.get("top_k", 0), "min_p": extra.get("min_p", 0.0),
            "presence_penalty": sampling.get("presence_penalty", 0.0),
            "repetition_penalty": extra.get("repetition_penalty", 1.0),
            "max_tokens": sampling["max_tokens"],
            "chat_template_kwargs": dict(extra.get("chat_template_kwargs") or {})}


def dev_sampling(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """What dev evaluation samples under: `VLLMGeneration` attributes, plus
    `presence_penalty`, which `rl.train.dev_sampling_swap` sends as a dev-only
    generation kwarg.

    `dev_max_tokens` lowers the cap for the smoke config only; a real run takes
    the eval config's `max_tokens`.
    """
    em = eval_model(cfg)
    return {"temperature": em["temperature"], "top_p": em["top_p"], "top_k": em["top_k"],
            "min_p": em["min_p"], "repetition_penalty": em["repetition_penalty"],
            "presence_penalty": em["presence_penalty"],
            "max_completion_length": cfg.get("dev_max_tokens") or em["max_tokens"]}
