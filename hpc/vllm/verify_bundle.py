#!/usr/bin/env python3
"""Static checks that the serving profiles and the eval configs fit together.

A serving profile (`hpc/vllm/models/<profile>.yaml`) says how to serve; an eval
config (`evals/conf/model/<config>.yaml`) says what each request carries,
including the checkpoint id. No fact lives in both. What is checked is what the
two layers must agree on: which config each profile serves, whether the request
fits the server, and the request fields that only work with a given server
setting. `MODEL_MATRIX.csv`, the evaluation roster, restates a few of those facts
for reading at a glance, so its columns are checked against them.
"""
from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import yaml
from pairing import (CONFIGS, EFFORT_LEVELS, NO_THINKING, PROFILES, ROOT, level_of,
                     profile_of)

# Tokens held back for the prompt out of a serving profile's max_model_len. What
# is left is the room for the output, and every hf-* eval config's max_tokens is
# min(ROSTER_CAP, room), so token budgets match wherever the context allows. The
# check below requires exactly that value.
#
# Measured on 2026-09-24 against the shipped data/taskset.jsonl (7,200 rows), each
# prompt composed as evals/prompt.py:compose does with the xml_tags template and
# the cot elicitation (the default, and the longer of the two shipped
# conditions), chat-templated with each hf-* config's own tokenizer and the
# chat_template_kwargs and reasoning_effort it sends, generation prompt
# included. The longest is 5,785 tokens, under Mistral-7B-Instruct-v0.3's
# tokenizer, on claim_chain/L15/last_link_democratic/s4; every other tokenizer's
# longest is 4,695 to 5,374, on a level-15 claim_chain row. Mistral-Small-3.2
# ships no Hugging Face chat template and was not measured. The reserve is that
# figure rounded up to the next power of two, which leaves 2,407 tokens for
# prompts that grow when a task states another rule. Re-measure when a generator
# changes its question text.
PROMPT_RESERVE = 8192

# Every eval config caps its output at 65,536 tokens at most, less where its
# context or provider limit is smaller. 65,536 is the Gemini API's output limit,
# thinking included, so no larger cap can hold across the roster. The hosted
# configs' caps are checked in tests/evals, against their providers' limits.
ROSTER_CAP = 65536

errors: list[str] = []

for path in (
    "hpc/vllm/arggym-vllm.def",
    "hpc/vllm/check_runtime.sh",
    "hpc/vllm/setup_runtime.sbatch",
    "hpc/vllm/submit_truba.sh",
    "hpc/vllm/truba_vllm.sbatch",
    "hpc/vllm/preflight_truba.sh",
    "hpc/vllm/run_vllm_arggym.py",
    "hpc/vllm/pairing.py",
    "hpc/vllm/requirements-vllm.txt",
):
    if not (ROOT / path).is_file():
        errors.append(f"missing required deployment file: {path}")


def load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        errors.append(f"{path.relative_to(ROOT)} is not a YAML object")
        return {}
    return data


profiles = {p.stem: load(p) for p in PROFILES.glob("*.yaml")}
configs = {p.stem: load(p) for p in CONFIGS.glob("hf-*.yaml")}
names = set(profiles)


# ---------------------------------------------------------------------------
# Pairing, by stem: <profile>; <profile>-<level> for a model with levels; and
# <profile> with <profile>-nothink for a model whose thinking is a switch.
# ---------------------------------------------------------------------------

#: What a thinking-on/off pair may set differently: the card's per-mode sampling.
MODE_SAMPLING = {"temperature", "top_p", "top_k", "min_p", "presence_penalty",
                 "repetition_penalty"}


def thinking_sent(cfg: dict):
    return ((cfg.get("extra_body") or {}).get("chat_template_kwargs") or {}).get(
        "enable_thinking")


def without_mode(cfg: dict) -> dict:
    """A config without its name, its per-mode sampling values and enable_thinking."""
    sampling = {k: v for k, v in (cfg.get("sampling") or {}).items()
                if k not in MODE_SAMPLING}
    extra = {k: v for k, v in (cfg.get("extra_body") or {}).items()
             if k not in MODE_SAMPLING}
    extra["chat_template_kwargs"] = {
        k: v for k, v in (extra.get("chat_template_kwargs") or {}).items()
        if k != "enable_thinking"}
    return {**{k: v for k, v in cfg.items() if k != "name"},
            "sampling": sampling, "extra_body": extra}


served: dict[str, list[str]] = {name: [] for name in profiles}
for stem in sorted(configs):
    profile = profile_of(stem, names)
    if profile is None:
        errors.append(f"eval config without a serving profile: {stem}")
    else:
        served[profile].append(stem)

for profile, stems in sorted(served.items()):
    if not stems:
        errors.append(f"serving profile without an eval config: {profile}")
    # Either one level-less config (with its thinking-off sibling, if any) or one
    # per level, never both: a leftover level-less file would run at whatever
    # effort the template defaults to.
    levelled = [s for s in stems if level_of(s, names)]
    if profile in stems and levelled:
        errors.append(f"{profile}: has both a level-less config and "
                      + ", ".join(levelled))
    nothink = f"{profile}-{NO_THINKING}"
    if nothink in stems and profile not in stems:
        errors.append(f"{nothink}: a thinking-off config needs its thinking-on {profile}")
    # One server, one checkpoint: the configs a profile serves must all name the
    # same model, and per-level configs must differ in nothing but the level. A
    # thinking-on/off pair may differ in its sampling values too, since the cards
    # recommend different sampling per mode, but in nothing else: same cap, same
    # seed. And its thinking-on half must say so, because the Gemma 4 and the
    # small Qwen3.5 templates think only when enable_thinking is sent true.
    if nothink in stems and profile in stems:
        if thinking_sent(configs[profile]) is not True:
            errors.append(f"{profile}: its sibling {nothink} turns thinking off, so it "
                          f"must send enable_thinking true")
        if without_mode(configs[profile]) != without_mode(configs[nothink]):
            errors.append(f"{nothink}: differs from {profile} in more than sampling "
                          f"values and enable_thinking")
    models = {configs[s].get("model") for s in stems}
    if len(models) > 1:
        errors.append(f"{profile}: its configs name different models {sorted(map(str, models))}")
    if len(levelled) > 1:
        def without_level(cfg: dict) -> dict:
            sampling = {k: v for k, v in (cfg.get("sampling") or {}).items()
                        if k != "reasoning_effort"}
            return {**{k: v for k, v in cfg.items() if k != "name"}, "sampling": sampling}
        first = without_level(configs[levelled[0]])
        for s in levelled[1:]:
            if without_level(configs[s]) != first:
                errors.append(f"{s}: differs from {levelled[0]} in more than the effort level")


# ---------------------------------------------------------------------------
# Each config against its profile.
# ---------------------------------------------------------------------------

for name in sorted(configs):
    profile = profile_of(name, names)
    if profile is None:
        continue
    p, e = profiles[profile], configs[name]

    if e.get("name") != name:
        errors.append(f"{name}: name field differs from the file name")

    level = level_of(name, names)
    if level is not None:
        effort = (e.get("sampling") or {}).get("reasoning_effort")
        if effort != level:
            errors.append(f"{name}: file names effort {level}, config sends {effort}")
    # `-nothink` is the name of `enable_thinking: false`, and only of it.
    thinking = thinking_sent(e)
    if name.endswith(f"-{NO_THINKING}") != (thinking is False):
        errors.append(f"{name}: enable_thinking {thinking}; a config sends false "
                      f"exactly when its name ends in -{NO_THINKING}")

    sampling = e.get("sampling") or {}
    max_tokens = sampling.get("max_tokens", sampling.get("max_completion_tokens"))
    try:
        cap = min(ROSTER_CAP, int(p["max_model_len"]) - PROMPT_RESERVE)
        if int(max_tokens) != cap:
            errors.append(f"{name}: max_tokens {max_tokens}, but min({ROSTER_CAP}, "
                          f"{profile}'s max_model_len {p['max_model_len']} - "
                          f"{PROMPT_RESERVE} prompt reserve) is {cap}")
    except (KeyError, TypeError, ValueError):
        errors.append(f"{name}: needs max_tokens, and {profile} needs max_model_len")


# ---------------------------------------------------------------------------
# Each profile on its own: what the runner and submit_truba.sh require.
# ---------------------------------------------------------------------------

for profile, p in sorted(profiles.items()):
    # A profile is named `hf-<model>`, where <model> is the checkpoint name its
    # configs send, lowercased. An RL fine-tune sends its own repo as `model`
    # and declares what it was trained from in `base_model`; its profile is
    # `hf-<base model>-rl-<tag>`. Never with a level: pairing.py reads a
    # trailing level off a config's stem, so a profile ending in one would pair
    # with the wrong file.
    if not re.fullmatch(r"hf-[a-z0-9][a-z0-9.-]*", profile):
        errors.append(f"{profile}: a profile stem is hf- and then [a-z0-9.-]")
    if any(profile.endswith(f"-{lv}") for lv in EFFORT_LEVELS):
        errors.append(f"{profile}: a profile stem carries no effort level")
    for stem in served.get(profile, []):
        model, base = str(configs[stem].get("model")), configs[stem].get("base_model")
        if base:
            want = "hf-" + str(base).rsplit("/", 1)[-1].lower() + "-rl-"
            if base == model:
                errors.append(f"{stem}: base_model is the model it serves, so it is "
                              f"no fine-tune; drop base_model")
            elif not re.fullmatch(re.escape(want) + r"[a-z0-9][a-z0-9.-]*", profile):
                errors.append(f"{profile}: serves a fine-tune of {base}, so should be "
                              f"named {want}<tag>")
        else:
            want = "hf-" + model.rsplit("/", 1)[-1].lower()
            if profile != want:
                errors.append(f"{profile}: serves {model}, so should be named {want}")
    # The lane serves BF16 weights; run_vllm_arggym.py refuses anything else
    # at job start, which is a queue wait too late to find out.
    if p.get("dtype") != "bfloat16":
        errors.append(f"{profile}: dtype is not bfloat16")
    for key in ("tensor_parallel_size_h100", "tensor_parallel_size_h200"):
        if p.get(key) not in (1, 2, 3, 4):
            errors.append(f"{profile}: {key} must be 1..4")


# ---------------------------------------------------------------------------
# Request fields that only work with a given server setting.
# ---------------------------------------------------------------------------

# Ministral-3-3B's card serves it with Mistral's own formats and parser.
m = profiles.get("hf-ministral-3-3b-reasoning-2512")
if m is not None:
    joined = " ".join(map(str, m.get("extra_args", [])))
    for arg in ("--tokenizer-mode mistral", "--config-format mistral",
                "--load-format mistral"):
        if arg not in joined:
            errors.append(f"hf-ministral-3-3b-reasoning-2512: missing server arg {arg}")
    if m.get("reasoning_parser") != "mistral":
        errors.append("hf-ministral-3-3b-reasoning-2512: reasoning_parser must be mistral")

# Gemma 4 thinks when the request sets enable_thinking, and the gemma4 parser
# can only split the thought out if the special tokens that delimit it survive.
# It has no effort levels to split its config by: a profile serves a thinking-on
# config and at most a thinking-off one, and the card's one sampling setting "across
# all use cases" (huggingface.co/google/gemma-4-31B-it) leaves the pair differing
# in enable_thinking alone.
for profile in sorted(p for p in profiles if p.startswith("hf-gemma-4-")):
    if profiles[profile].get("reasoning_parser") != "gemma4":
        errors.append(f"{profile}: reasoning parser must be gemma4")
    # Gemma 4 takes image, audio and video; a limit left unset stays at 999 and
    # keeps the vision tower and its encoder cache. Serve the text model alone.
    gemma_args = list(map(str, profiles[profile].get("extra_args", [])))
    if "--language-model-only" not in gemma_args or "--limit-mm-per-prompt" in gemma_args:
        errors.append(f"{profile}: serve text only with --language-model-only, "
                      "not --limit-mm-per-prompt")
    stems = served.get(profile, [])
    if profile not in stems or not set(stems) <= {profile, f"{profile}-{NO_THINKING}"}:
        errors.append(f"{profile}: takes a thinking-on config and at most a "
                      f"-{NO_THINKING} one, no effort levels")
        continue
    if thinking_sent(configs[profile]) is not True:
        errors.append(f"{profile}: expected chat_template_kwargs.enable_thinking true")
    for stem in stems:
        if (configs[stem].get("extra_body") or {}).get("skip_special_tokens") is not False:
            errors.append(f"{stem}: skip_special_tokens should be false")
    if len(stems) == 2:
        def without_switch(cfg: dict) -> dict:
            extra = dict(cfg.get("extra_body") or {})
            extra["chat_template_kwargs"] = {
                k: v for k, v in (extra.get("chat_template_kwargs") or {}).items()
                if k != "enable_thinking"}
            return {**{k: v for k, v in cfg.items() if k != "name"}, "extra_body": extra}
        if without_switch(configs[stems[0]]) != without_switch(configs[stems[1]]):
            errors.append(f"{profile}: its two configs differ in more than enable_thinking")

# Qwen presence penalties, as each model card recommends for the mode the
# config runs in: thinking mode on general tasks (text tasks on the 0.8B and 2B
# cards), and for Qwen3.5 thinking off, the card's non-thinking setting for
# reasoning tasks (text tasks on 0.8B and 2B). The cards differ -- the 3.6-27B
# and 3.8-27B cards say 0.0 in thinking mode, the others 1.5, and three Qwen3.5
# cards say 2.0 with thinking off -- and #130 was on whether a penalty helps, so
# the value is pinned here as well as in the configs, which cite the card line.
# Keyed by config, or by profile for every config that profile serves.
expected_presence_penalty = {
    "hf-qwen3.6-27b": 0.0,
    "hf-qwen3.6-35b-a3b": 1.5,
    "hf-qwen3.8-27b": 0.0,
    "hf-qwen3.5-0.8b": 1.5, "hf-qwen3.5-0.8b-nothink": 2.0,
    "hf-qwen3.5-2b": 1.5, "hf-qwen3.5-2b-nothink": 2.0,
    "hf-qwen3.5-4b": 1.5, "hf-qwen3.5-4b-nothink": 1.5,
    "hf-qwen3.5-9b": 1.5, "hf-qwen3.5-9b-nothink": 1.5,
    "hf-qwen3.5-27b": 1.5, "hf-qwen3.5-27b-nothink": 1.5,
    "hf-qwen3.5-35b-a3b": 1.5, "hf-qwen3.5-35b-a3b-nothink": 1.5,
    "hf-qwen3.5-122b-a10b": 1.5, "hf-qwen3.5-122b-a10b-nothink": 2.0,
}
for name in sorted(configs):
    want = expected_presence_penalty.get(
        name, expected_presence_penalty.get(profile_of(name, names)))
    if want is not None:
        actual = (configs[name].get("sampling") or {}).get("presence_penalty")
        if actual != want:
            errors.append(f"{name}: expected presence_penalty {want}, got {actual}")


# ---------------------------------------------------------------------------
# MODEL_MATRIX.csv: the evaluation roster. It is the only home of `params_b` and
# `tier`; every other column restates a profile or a config, so it is checked
# against them rather than trusted.
# ---------------------------------------------------------------------------

matrix = ROOT / "MODEL_MATRIX.csv"
if not matrix.is_file():
    errors.append("missing MODEL_MATRIX.csv")
else:
    with matrix.open(newline="") as fh:
        for row in csv.DictReader(fh):
            pattern = row["config"]
            m = re.fullmatch(r"(.+)-<([^>]+)>", pattern)
            stems = [f"{m.group(1)}-{lv}" for lv in m.group(2).split("|")] if m else [pattern]
            for stem in stems:
                if stem not in configs:
                    errors.append(f"MODEL_MATRIX.csv: {pattern} names no eval config {stem}")
                    continue
                profile = profile_of(stem, names)
                p = profiles.get(profile, {})
                if configs[stem].get("model") != row["hf_model"]:
                    errors.append(f"MODEL_MATRIX.csv: {stem} hf_model {row['hf_model']} "
                                  f"but the config sends {configs[stem].get('model')}")
                for col, key in (("default_gpus", "tensor_parallel_size_h100"),
                                 ("dtype", "dtype"), ("max_model_len", "max_model_len"),
                                 ("reasoning_parser", "reasoning_parser")):
                    want = "" if p.get(key) is None else str(p.get(key))
                    if row[col] != want:
                        errors.append(f"MODEL_MATRIX.csv: {stem} {col} {row[col]!r} "
                                      f"but {profile} says {want!r}")


# ---------------------------------------------------------------------------
# The deployment scripts: syntax, and the two mistakes they once made.
# ---------------------------------------------------------------------------

for shell_file in ("check_runtime.sh", "setup_runtime.sbatch", "submit_truba.sh",
                   "truba_vllm.sbatch", "preflight_truba.sh"):
    path = ROOT / "hpc" / "vllm" / shell_file
    if path.is_file():
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if proc.returncode:
            errors.append(f"shell syntax error in {shell_file}: {proc.stderr.strip()}")

runner = ROOT / "hpc" / "vllm" / "run_vllm_arggym.py"
if runner.is_file():
    text = runner.read_text(encoding="utf-8")
    # The container has no uv; the runner must call the interpreter it runs in.
    if "uv run" in text or '"uv", "run"' in text or "sys.executable" not in text:
        errors.append("run_vllm_arggym.py must call sys.executable, not uv")

job = ROOT / "hpc" / "vllm" / "truba_vllm.sbatch"
if job.is_file():
    text = job.read_text(encoding="utf-8")
    if "exec --nv" not in text:
        errors.append("GPU job must execute the image with Apptainer --nv")
    if "pip install" in text or "python -m venv" in text:
        errors.append("GPU job must not install packages or create a venv")


if errors:
    print("FAILED")
    for error in errors:
        print(" -", error)
    raise SystemExit(1)

print(f"PASS: {len(profiles)} serving profiles and their {len(configs)} eval configs")
