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
#: `run_id` is built from and the `base_model` an RL fine-tune names, which
#: `evals/run.py:manifest` records and the request never carries.
READ = {"name", "model", "base_model", "base_url", "api_key_env", "sampling",
        "extra_body"}


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


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_no_config_caps_its_output_above_the_roster_cap(path):
    """Every config shares one output cap, 65,536, so every model gets the same budget.

    65,536 is the Gemini API's output limit, thinking included, so no larger cap
    can hold across the roster. A config may sit below it only where its context
    or its provider's output limit leaves less; `hpc/vllm/verify_bundle.py` checks
    the hf-* configs sit exactly at min(65,536, room).
    """
    s = yaml.safe_load(path.read_text()).get("sampling") or {}
    cap = s.get("max_tokens", s.get("max_completion_tokens", 0))
    assert cap <= 65536, f"{path.stem} caps its output at {cap}, above the roster's 65,536"


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
# One naming rule: `<provider>-<model>[-<level>|-nothink]`, derived from fields
# the config already holds, so no name is chosen by hand (#179).
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


def thinking_off(cfg: dict) -> bool:
    """Whether the config turns a thinking switch off, which its name says as `-nothink`."""
    kwargs = (cfg.get("extra_body") or {}).get("chat_template_kwargs") or {}
    return kwargs.get("enable_thinking") is False


def slug(model_id: str) -> str:
    """The checkpoint name after the last `/`, lowercased."""
    return model_id.rsplit("/", 1)[-1].lower()


def expected_name(cfg: dict) -> str:
    """The name the rule gives a config, with `<tag>` standing for an RL tag.

    `<provider>-<model>[-<level>|-nothink]`, or for an RL fine-tune, which
    declares the checkpoint it was trained from in `base_model` and sends its own
    repo as `model`, `<provider>-<base model>-rl-<tag>[-<level>|-nothink]`. A
    thinking switch left on adds nothing to the name.
    """
    level = level_sent(cfg)
    tail = f"-{level}" if level else ("-nothink" if thinking_off(cfg) else "")
    provider = PROVIDER[cfg["api_key_env"]]
    if cfg.get("base_model"):
        return f"{provider}-{slug(cfg['base_model'])}-rl-<tag>{tail}"
    return f"{provider}-{slug(cfg['model'])}{tail}"


def names_fit(stem: str, cfg: dict) -> bool:
    """Whether `stem` is the name the rule gives `cfg`."""
    if cfg.get("base_model") and cfg["base_model"] == cfg["model"]:
        return False  # the base checkpoint itself is not a fine-tune
    head, tagged, tail = expected_name(cfg).partition("<tag>")
    if not tagged:
        return stem == head
    return bool(re.fullmatch(re.escape(head) + r"[a-z0-9][a-z0-9.-]*" + re.escape(tail),
                             stem))


@pytest.mark.parametrize("path", NAMED, ids=lambda p: p.stem)
def test_a_config_is_named_by_its_provider_model_and_level(path):
    cfg = yaml.safe_load(path.read_text())
    assert names_fit(path.stem, cfg), f"{path.stem} should be {expected_name(cfg)}"
    assert re.fullmatch(r"[a-z0-9][a-z0-9.-]*", path.stem), path.stem


def test_the_rule_names_an_rl_fine_tune_after_its_base_model():
    """On configs no file carries: an RL fine-tune, its base, and hosted Gemma."""
    base = {"api_key_env": "VLLM_API_KEY", "model": "Qwen/Qwen3-8B"}
    assert names_fit("hf-qwen3-8b", base)
    assert not names_fit("hf-qwen3-8b-rl-arggym-40k", base)  # no base_model, no -rl-
    assert not names_fit("hf-qwen3-8b-arggym-40k", base)

    rl = {"api_key_env": "VLLM_API_KEY", "model": "someorg/ArgGYM-Qwen3-8B-GRPO",
          "base_model": "Qwen/Qwen3-8B"}
    assert names_fit("hf-qwen3-8b-rl-arggym-40k", rl)
    assert not names_fit("hf-arggym-qwen3-8b-grpo", rl)  # named after the repo
    assert not names_fit("hf-qwen3-8b", rl)  # a fine-tune must say so
    assert not names_fit("hf-qwen3-8b-rl-arggym-40k",
                         {**rl, "model": "Qwen/Qwen3-8B"})  # the base is no fine-tune

    leveled = {**rl, "sampling": {"reasoning_effort": "low"}}
    assert names_fit("hf-qwen3-8b-rl-arggym-40k-low", leveled)
    assert not names_fit("hf-qwen3-8b-rl-arggym-40k", leveled)

    gemma = {"api_key_env": "GEMINI_API_KEY", "model": "gemma-4-31b-it",
             "extra_body": {"extra_body": {"google": {"thinking_config":
                                                      {"thinking_level": "high"}}}}}
    assert names_fit("aistudio-gemma-4-31b-it-high", gemma)
    assert not names_fit("gemma4-31b-aistudio", gemma)
    assert not names_fit("aistudio-gemma-4-31b-it", gemma)


def test_nothink_names_a_thinking_switch_turned_off_and_nothing_else():
    """`-nothink` is in the name exactly when `enable_thinking` is false."""
    def cfg(**kwargs):
        return {"api_key_env": "VLLM_API_KEY", "model": "Qwen/Qwen3.5-9B",
                "extra_body": {"chat_template_kwargs": kwargs}}
    assert names_fit("hf-qwen3.5-9b-nothink", cfg(enable_thinking=False))
    assert not names_fit("hf-qwen3.5-9b", cfg(enable_thinking=False))
    assert names_fit("hf-qwen3.5-9b", cfg(enable_thinking=True))
    assert not names_fit("hf-qwen3.5-9b-nothink", cfg(enable_thinking=True))
    # Left unset, the template's own default decides, which the name cannot say.
    assert not names_fit("hf-qwen3.5-9b-nothink", cfg())


def test_the_thinking_on_half_of_a_pair_sends_the_switch():
    """A bare name whose `-nothink` sibling exists must send `enable_thinking: true`.

    Left unset, the template decides, and the Gemma 4 and Qwen3.5-0.8B/2B
    templates think only when the switch is sent true: the pair would run
    thinking off twice.
    """
    stems = {p.stem: p for p in CONFIGS}
    pairs = [s[: -len("-nothink")] for s in stems if s.endswith("-nothink")]
    assert pairs, "no thinking-off configs found"
    for bare in pairs:
        assert bare in stems, f"{bare}-nothink has no thinking-on sibling"
        cfg = yaml.safe_load(stems[bare].read_text())
        kwargs = (cfg.get("extra_body") or {}).get("chat_template_kwargs") or {}
        assert kwargs.get("enable_thinking") is True, f"{bare} must send enable_thinking true"


def test_a_synthetic_rl_config_is_checked_like_any_other(tmp_path):
    """The same test the shipped configs go through, on a file written here."""
    body = {"model": "someorg/ArgGYM-Qwen3-8B-GRPO", "base_model": "Qwen/Qwen3-8B",
            "base_url": "http://127.0.0.1:8000/v1", "api_key_env": "VLLM_API_KEY",
            "sampling": {"max_tokens": 32768, "temperature": 0.6}}
    good = tmp_path / "hf-qwen3-8b-rl-arggym-40k.yaml"
    good.write_text(yaml.safe_dump({"name": good.stem, **body}))
    test_every_key_in_a_model_config_is_read(good)
    test_a_config_is_named_by_its_provider_model_and_level(good)

    bad = tmp_path / "hf-arggym-qwen3-8b-grpo.yaml"
    bad.write_text(yaml.safe_dump({"name": bad.stem, **body}))
    with pytest.raises(AssertionError):
        test_a_config_is_named_by_its_provider_model_and_level(bad)


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
        # An RL fine-tune takes the levels of the model it was trained from.
        model = cfg.get("base_model") or cfg["model"]
        sent.setdefault((cfg["api_key_env"], cfg["model"], model), []).append(level_sent(cfg))
    for (key, _, model), levels in sorted(sent.items()):
        if model in VALID_LEVELS:
            assert sorted(levels) == sorted(VALID_LEVELS[model]), \
                f"{model} via {key}: configs at {sorted(map(str, levels))}, " \
                f"provider accepts {sorted(VALID_LEVELS[model])}"
        else:
            assert levels == [None] * len(levels), \
                f"{model} via {key} sends a level but has no level set listed"
    assert {m for _, _, m in sent} >= set(VALID_LEVELS), "a listed model has no config"
