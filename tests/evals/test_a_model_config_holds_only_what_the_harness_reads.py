"""A key in a model config that `build_solver` never reads is a setting nobody gets.

Twenty configs carried a top-level `timeout_s` and `retries`. `build_solver`
takes both from `cfg.endpoint`, so `timeout_s: 30` in one of them ran at 5400
while the file said otherwise (#163).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from evals.client import Endpoint

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((ROOT / "evals" / "conf" / "model").glob("*.yaml"))
HPC_MODELS = ROOT / "hpc" / "vllm" / "models"

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


# ---------------------------------------------------------------------------
# One naming rule: `<provider>-<model>[-<level>]`, derived from fields the config
# already holds, so no name is chosen by hand (#179).
# ---------------------------------------------------------------------------

#: The endpoint, read off the key the config sends.
PROVIDER = {"VLLM_API_KEY": "hf", "OPENROUTER_API_KEY": "openrouter",
            "OPENAI_API_KEY": "openai", "GEMINI_API_KEY": "aistudio"}

#: The stub stands in for an endpoint in tests; it is not a model.
NAMED = [p for p in CONFIGS if p.stem != "stub"]


def level_sent(cfg: dict):
    """The effort level a config sends, through whichever knob its provider reads.

    `chat_template_kwargs.enable_thinking` is a switch, not a level, and is not
    read here.
    """
    sampling = cfg.get("sampling") or {}
    extra = cfg.get("extra_body") or {}
    google = (extra.get("extra_body") or {}).get("google") or {}
    return (sampling.get("reasoning_effort")
            or (extra.get("reasoning") or {}).get("effort")
            or (google.get("thinking_config") or {}).get("thinking_level"))


def base_name(cfg: dict) -> str:
    """`<provider>-<model>`: the checkpoint name after the last `/`, lowercased."""
    return f"{PROVIDER[cfg['api_key_env']]}-{cfg['model'].rsplit('/', 1)[-1].lower()}"


@pytest.mark.parametrize("path", NAMED, ids=lambda p: p.stem)
def test_a_config_is_named_by_its_provider_model_and_level(path):
    cfg = yaml.safe_load(path.read_text())
    base, level = base_name(cfg), level_sent(cfg)
    tail = f"-{level}" if level else ""
    # An RL fine-tune keeps its base model's name and adds `-rl-<tag>`; its
    # serving profile pins the checkpoint.
    assert names_fit(path.stem, cfg), \
        f"{path.stem} should be {base + tail} (or {base}-rl-<tag>{tail})"
    assert re.fullmatch(r"[a-z0-9][a-z0-9.-]*", path.stem), path.stem
    if path.stem != base + tail:
        profile = HPC_MODELS / f"{path.stem[: len(path.stem) - len(tail)]}.yaml"
        assert (yaml.safe_load(profile.read_text()) or {}).get("revision"), \
            f"{path.stem} is an RL fine-tune, so {profile.name} must pin its revision"


def names_fit(stem: str, cfg: dict) -> bool:
    """`<provider>-<model>[-<level>]`, or `<provider>-<model>-rl-<tag>[-<level>]`."""
    base, level = base_name(cfg), level_sent(cfg)
    tail = f"-{level}" if level else ""
    return stem == base + tail or bool(re.fullmatch(
        re.escape(base) + r"-rl-[a-z0-9][a-z0-9.-]*" + re.escape(tail), stem))


def test_the_rule_takes_the_rl_form_and_refuses_an_invented_name():
    """The rule itself, on names no config carries."""
    qwen = {"api_key_env": "VLLM_API_KEY", "model": "Qwen/Qwen3-8B"}
    assert names_fit("hf-qwen3-8b", qwen)
    assert names_fit("hf-qwen3-8b-rl-arggym-40k", qwen)
    assert not names_fit("hf-qwen3-8b-arggym-40k", qwen)
    gemma = {"api_key_env": "GEMINI_API_KEY", "model": "gemma-4-31b-it",
             "extra_body": {"extra_body": {"google": {"thinking_config":
                                                      {"thinking_level": "high"}}}}}
    assert names_fit("aistudio-gemma-4-31b-it-high", gemma)
    assert not names_fit("gemma4-31b-aistudio", gemma)
    assert not names_fit("aistudio-gemma-4-31b-it", gemma)


#: Every level each model's provider accepts, as the configs cite them. A model
#: not listed sends no level (a thinking on/off switch is not a level).
VALID_LEVELS = {
    # https://huggingface.co/Qwen/Qwen3.8-27B ("xhigh (default)", "medium", "low")
    "Qwen/Qwen3.8-27B": {"low", "medium", "xhigh"},
    # https://huggingface.co/openai/gpt-oss-20b and .../gpt-oss-120b
    # ("Low / Medium / High")
    "openai/gpt-oss-20b": {"low", "medium", "high"},
    "openai/gpt-oss-120b": {"low", "medium", "high"},
    # https://developers.openai.com/api/docs/models/gpt-5
    "gpt-5": {"minimal", "low", "medium", "high"},
    "openai/gpt-5": {"minimal", "low", "medium", "high"},
    # https://openrouter.ai/docs/use-cases/reasoning-tokens (`max` is xhigh,
    # `none` turns thinking off)
    "anthropic/claude-sonnet-4.5": {"minimal", "low", "medium", "high", "xhigh"},
    # https://ai.google.dev/gemini-api/docs/thinking
    "gemini-2.5-pro": {"low", "medium", "high"},
    "gemini-3.5-flash-lite": {"minimal", "low", "medium", "high"},
    "gemini-3.6-flash": {"minimal", "low", "medium", "high"},
    "gemini-3.7-flash": {"low", "medium", "high"},
    "gemini-3.8-flash": {"low", "medium", "high"},
    # https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api ("high" for
    # enabled, "minimal" for disabled)
    "gemma-4-31b-it": {"minimal", "high"},
    "gemma-4-26b-a4b-it": {"minimal", "high"},
}


def test_every_model_with_levels_has_a_config_at_each_level_and_no_other():
    sent: dict = {}
    for p in NAMED:
        cfg = yaml.safe_load(p.read_text())
        # Keyed by the endpoint as well: one checkpoint through two providers is
        # two sets of configs.
        sent.setdefault((cfg["api_key_env"], cfg["model"]), []).append(level_sent(cfg))
    for (key, model), levels in sorted(sent.items()):
        if model in VALID_LEVELS:
            assert sorted(levels) == sorted(VALID_LEVELS[model]), \
                f"{model} via {key}: configs at {sorted(map(str, levels))}, " \
                f"provider accepts {sorted(VALID_LEVELS[model])}"
        else:
            assert levels == [None] * len(levels), \
                f"{model} via {key} sends a level but has no level set listed"
    assert {m for _, m in sent} >= set(VALID_LEVELS), "a listed model has no config"
