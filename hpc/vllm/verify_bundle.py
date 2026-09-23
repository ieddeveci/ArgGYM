#!/usr/bin/env python3
"""Static checks that the serving profiles and the eval configs fit together.

A serving profile (`hpc/vllm/models/<profile>.yaml`) says how to serve; an eval
config (`evals/conf/model/<config>.yaml`) says what each request carries,
including the checkpoint id. No fact lives in both, so there is nothing to
compare copies of. What is checked is what the two layers must agree on:
which config each profile serves, whether the request fits the server, and the
request fields that only work with a given server setting.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
from pairing import CONFIGS, PROFILES, ROOT, level_of, profile_of

# Tokens held back for the prompt out of a serving profile's max_model_len. Every
# hf-* eval config's max_tokens must fit in what is left: the check below
# requires max_tokens + PROMPT_RESERVE <= max_model_len. A config's cap is the
# smaller of that room and the longest output its model card recommends.
#
# Measured on 2026-09-23 against tasksets/standard.yaml at commit 91307ae (7,070
# of its 7,200 rows: 13 slow high-level cells of counter_argument,
# counter_argument_strict and preference_construction had not finished
# building, and those tasks' prompts run 3,000+ characters shorter than the
# longest at every level), each prompt composed as evals/prompt.py:compose does
# with the xml_tags template and the cot elicitation (the HPC runner's default,
# and the longer of the two shipped conditions), chat-templated with each served
# model's own tokenizer, generation prompt included. The longest is 4,773
# tokens, under Mistral-7B-Instruct-v0.3's tokenizer, on
# status_query/L15/weakest_link_elitist/s6 (8,552 characters); every other
# tokenizer's longest is 3,708 to 4,428, on a level-15 claim_chain or
# status_query row. The reserve is that figure rounded up to the next power of
# two, which leaves 3,419 tokens for prompts that grow when a task states
# another rule. Re-measure when a generator changes its question text.
PROMPT_RESERVE = 8192

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
# Pairing, by stem: <profile>, or <profile>-<level> for a model with levels.
# ---------------------------------------------------------------------------

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
    # Either one level-less config or one per level, never both: a leftover
    # level-less file would run at whatever effort the template defaults to.
    if profile in stems and len(stems) > 1:
        errors.append(f"{profile}: has both a level-less config and "
                      + ", ".join(s for s in stems if s != profile))
    # One server, one checkpoint: the configs a profile serves must all name the
    # same model, and per-level configs must differ in nothing but the level.
    models = {configs[s].get("model") for s in stems}
    if len(models) > 1:
        errors.append(f"{profile}: its configs name different models {sorted(map(str, models))}")
    if len(stems) > 1:
        def without_level(cfg: dict) -> dict:
            sampling = {k: v for k, v in (cfg.get("sampling") or {}).items()
                        if k != "reasoning_effort"}
            return {**{k: v for k, v in cfg.items() if k != "name"}, "sampling": sampling}
        first = without_level(configs[stems[0]])
        for s in stems[1:]:
            if without_level(configs[s]) != first:
                errors.append(f"{s}: differs from {stems[0]} in more than the effort level")


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

    sampling = e.get("sampling") or {}
    max_tokens = sampling.get("max_tokens", sampling.get("max_completion_tokens"))
    try:
        # At most, not exactly: a cap may sit below what the context leaves
        # when the model's card recommends a shorter output.
        if int(max_tokens) + PROMPT_RESERVE > int(p["max_model_len"]):
            errors.append(f"{name}: max_tokens {max_tokens} + {PROMPT_RESERVE} prompt "
                          f"reserve exceeds {profile}'s max_model_len {p['max_model_len']}")
    except (KeyError, TypeError, ValueError):
        errors.append(f"{name}: needs max_tokens, and {profile} needs max_model_len")


# ---------------------------------------------------------------------------
# Each profile on its own: what the runner and submit_truba.sh require.
# ---------------------------------------------------------------------------

for profile, p in sorted(profiles.items()):
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
m = profiles.get("hf-ministral3-3b-reasoning")
if m is not None:
    joined = " ".join(map(str, m.get("extra_args", [])))
    for arg in ("--tokenizer-mode mistral", "--config-format mistral",
                "--load-format mistral"):
        if arg not in joined:
            errors.append(f"hf-ministral3-3b-reasoning: missing server arg {arg}")
    if m.get("reasoning_parser") != "mistral":
        errors.append("hf-ministral3-3b-reasoning: reasoning_parser must be mistral")

# Gemma 4 thinks when the request sets enable_thinking, and the gemma4 parser
# can only split the thought out if the special tokens that delimit it survive.
# It has no effort levels to split its config by.
for profile in ("hf-gemma4-31b-it", "hf-gemma4-26b-a4b-it"):
    if profile not in profiles:
        continue
    if profiles[profile].get("reasoning_parser") != "gemma4":
        errors.append(f"{profile}: reasoning parser must be gemma4")
    if served.get(profile) != [profile]:
        errors.append(f"{profile}: takes exactly one config, no effort levels")
        continue
    extra = configs[profile].get("extra_body") or {}
    if (extra.get("chat_template_kwargs") or {}).get("enable_thinking") is not True:
        errors.append(f"{profile}: expected chat_template_kwargs.enable_thinking true")
    if extra.get("skip_special_tokens") is not False:
        errors.append(f"{profile}: skip_special_tokens should be false")

# Qwen 3.6/3.8 presence penalties, as each model card recommends for thinking
# mode on general tasks. The two 3.6 cards differ -- only the 35B-A3B card sets
# 1.5 -- and #130 is still open on whether 1.5 helps, so the value is pinned
# here as well as in the configs (huggingface.co/Qwen/Qwen3.6-35B-A3B,
# huggingface.co/Qwen/Qwen3.6-27B, huggingface.co/Qwen/Qwen3.8-27B).
expected_presence_penalty = {
    "hf-qwen3.6-27b": 0.0,
    "hf-qwen3.6-35b-a3b": 1.5,
    "hf-qwen3.8-27b": 0.0,
}
for name in sorted(configs):
    profile = profile_of(name, names)
    if profile in expected_presence_penalty:
        actual = (configs[name].get("sampling") or {}).get("presence_penalty")
        if actual != expected_presence_penalty[profile]:
            errors.append(f"{name}: expected presence_penalty "
                          f"{expected_presence_penalty[profile]}, got {actual}")


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
