#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "hpc" / "vllm" / "models"
E = ROOT / "evals" / "conf" / "model"

ALLOWED_SAMPLING = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
    "n",
    "reasoning_effort",
}

# Tokens held back for the prompt out of a serving profile's max_model_len. Every
# hf-* eval config's max_tokens must fit in what is left: the check below
# requires max_tokens + PROMPT_RESERVE <= max_model_len. A config's cap is the
# smaller of that room and the longest output its model card recommends.
#
# Measured on 2026-09-23 against tasksets/standard.yaml at commit 91307ae (7,070
# of its 7,200 rows: 13 slow high-level cells of counter_argument,
# counter_argument_strict and preference_construction had not finished
# building, and those tasks' prompts run 3,000+ characters shorter than the
# longest at every level), each
# prompt composed as evals/prompt.py:compose does with the xml_tags template and
# the cot elicitation (the HPC runner's default, and the longer of the two
# shipped conditions), chat-templated with each served model's own tokenizer,
# generation prompt included. The longest is 4,773 tokens, under
# Mistral-7B-Instruct-v0.3's tokenizer, on status_query/L15/weakest_link_elitist/s6
# (8,552 characters); every other tokenizer's longest is 3,708 to 4,428, on a
# level-15 claim_chain or status_query row. The reserve is that
# figure rounded up to the next power of two, which leaves 3,419 tokens for
# prompts that grow when a task states another rule. Re-measure when a generator
# changes its question text.
PROMPT_RESERVE = 8192

errors: list[str] = []

required = [
    ROOT / "hpc/vllm/arggym-vllm.def",
    ROOT / "hpc/vllm/check_runtime.sh",
    ROOT / "hpc/vllm/setup_runtime.sbatch",
    ROOT / "hpc/vllm/submit_truba.sh",
    ROOT / "hpc/vllm/truba_vllm.sbatch",
    ROOT / "hpc/vllm/preflight_truba.sh",
    ROOT / "hpc/vllm/run_vllm_arggym.py",
    ROOT / "hpc/vllm/requirements-vllm.txt",
]

for path in required:
    if not path.is_file():
        errors.append(
            f"missing required deployment file: {path.relative_to(ROOT)}"
        )


profiles = {
    p.stem: yaml.safe_load(p.read_text(encoding="utf-8"))
    for p in P.glob("*.yaml")
}

endpoints = {
    p.stem: yaml.safe_load(p.read_text(encoding="utf-8"))
    for p in E.glob("hf-*.yaml")
}

# A model with reasoning-effort levels has one eval config per level, named
# `<profile>-<level>`, and all of them are served by the one profile.
EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh")


def level_of(stem: str) -> str | None:
    """The effort level a config's name carries, if it names one of a profile."""
    for level in EFFORT_LEVELS:
        if stem.endswith(f"-{level}") and stem[: -len(level) - 1] in profiles:
            return level
    return None


def profile_of(stem: str) -> str | None:
    if stem in profiles:
        return stem
    level = level_of(stem)
    return stem[: -len(level) - 1] if level else None


# ---------------------------------------------------------------------------
# Model profile / endpoint pairing
# ---------------------------------------------------------------------------

served: dict[str, list[str]] = {name: [] for name in profiles}
for stem in sorted(endpoints):
    profile = profile_of(stem)
    if profile is None:
        errors.append(f"endpoint without a matching profile: {stem}")
    else:
        served[profile].append(stem)

for name, stems in sorted(served.items()):
    if not stems:
        errors.append(f"profile without a matching endpoint: {name}")
    # Either one level-less config or one per level, never both: a leftover
    # base file would run at whatever effort the template defaults to.
    if name in stems and len(stems) > 1:
        errors.append(
            f"{name}: has both a level-less config and per-level configs "
            + ", ".join(s for s in stems if s != name)
        )


# ---------------------------------------------------------------------------
# Generic model validation
# ---------------------------------------------------------------------------

for name in sorted(endpoints):
    profile = profile_of(name)
    if profile is None:
        continue
    p = profiles[profile]
    e = endpoints[name]

    if not isinstance(p, dict):
        errors.append(f"{name}: serving profile is not a YAML object")
        continue

    if not isinstance(e, dict):
        errors.append(f"{name}: endpoint config is not a YAML object")
        continue

    if p.get("endpoint_config") != profile:
        errors.append(f"{name}: endpoint_config mismatch")

    if e.get("name") != name:
        errors.append(f"{name}: name field differs from the file name")

    level = level_of(name)
    if level is not None:
        effort = (e.get("sampling") or {}).get("reasoning_effort")
        if effort != level:
            errors.append(
                f"{name}: file names effort {level}, config sends {effort}"
            )

    if p.get("hf_model") != e.get("model"):
        errors.append(f"{name}: model ID mismatch")

    if p.get("dtype") != "bfloat16":
        errors.append(f"{name}: dtype is not bfloat16")

    # -----------------------------------------------------------------------
    # Endpoint sampling validation
    # -----------------------------------------------------------------------

    sampling = e.get("sampling", {})

    if not isinstance(sampling, dict):
        errors.append(f"{name}: sampling must be a YAML object")
        sampling = {}

    bad = set(sampling) - ALLOWED_SAMPLING

    if bad:
        errors.append(
            f"{name}: unsupported sampling keys {sorted(bad)}"
        )

    max_tokens = sampling.get(
        "max_tokens",
        sampling.get("max_completion_tokens"),
    )

    if max_tokens is None:
        errors.append(
            f"{name}: endpoint must define max_tokens or max_completion_tokens"
        )
    else:
        try:
            max_tokens_i = int(max_tokens)
            max_model_len_i = int(p["max_model_len"])

            # At most, not exactly: a cap may sit below what the context
            # leaves when the model's card recommends a shorter output, and
            # the server keeps the model's context either way.
            if max_tokens_i + PROMPT_RESERVE > max_model_len_i:
                errors.append(
                    f"{name}: max_tokens {max_tokens_i} + {PROMPT_RESERVE} "
                    f"prompt reserve exceeds max_model_len {max_model_len_i}"
                )

        except (KeyError, TypeError, ValueError):
            errors.append(
                f"{name}: invalid max_tokens/max_model_len configuration"
            )

    # -----------------------------------------------------------------------
    # TRUBA GPU profile validation
    # -----------------------------------------------------------------------

    for key in (
        "tensor_parallel_size_h100",
        "tensor_parallel_size_h200",
    ):
        try:
            tp = int(p.get(key, 0))
        except (TypeError, ValueError):
            tp = 0

        if tp not in (1, 2, 3, 4):
            errors.append(f"{name}: invalid {key}")


# ---------------------------------------------------------------------------
# Model-specific validation
# ---------------------------------------------------------------------------

# Ministral reasoning profile.
ministral_name = "hf-ministral3-3b-reasoning"

if ministral_name in profiles:
    m = profiles[ministral_name]

    joined = " ".join(
        map(str, m.get("extra_args", []))
    )

    for required_arg in (
        "--tokenizer-mode mistral",
        "--config-format mistral",
        "--load-format mistral",
    ):
        if required_arg not in joined:
            errors.append(
                f"{ministral_name}: missing required server arg "
                f"{required_arg}"
            )

    if m.get("reasoning_parser") != "mistral":
        errors.append(
            f"{ministral_name}: reasoning_parser must be mistral"
        )


# Gemma 4 reasoning profiles.
for name in (
    "hf-gemma4-31b-it",
    "hf-gemma4-26b-a4b-it",
):
    if name not in profiles and name not in endpoints:
        continue

    # Gemma 4 thinks or does not; there is no level to split it by.
    if any(profile_of(s) == name and s != name for s in endpoints):
        errors.append(f"{name}: Gemma 4 takes no effort levels")

    if profiles.get(name, {}).get("reasoning_parser") != "gemma4":
        errors.append(
            f"{name}: reasoning parser must be gemma4"
        )

    # Gemma 4 has no reasoning-effort levels: its chat template reads
    # `enable_thinking` and nothing else.
    if (
        endpoints.get(name, {})
        .get("extra_body", {})
        .get("chat_template_kwargs", {})
        .get("enable_thinking")
        is not True
    ):
        errors.append(
            f"{name}: expected chat_template_kwargs.enable_thinking true"
        )

    if (
        endpoints.get(name, {})
        .get("extra_body", {})
        .get("skip_special_tokens")
        is not False
    ):
        errors.append(
            f"{name}: skip_special_tokens should be false"
        )


# Qwen 3.6/3.8 presence penalties, as each model card recommends for thinking
# mode on general tasks. The two 3.6 cards differ: only the 35B-A3B card sets
# 1.5 (huggingface.co/Qwen/Qwen3.6-35B-A3B, huggingface.co/Qwen/Qwen3.6-27B,
# huggingface.co/Qwen/Qwen3.8-27B).
expected_presence_penalty = {
    "hf-qwen3.6-27b": 0.0,
    "hf-qwen3.6-35b-a3b": 1.5,
    "hf-qwen3.8-27b": 0.0,
}

for name in sorted(endpoints):
    expected_pp = expected_presence_penalty.get(profile_of(name))
    if profile_of(name) not in expected_presence_penalty:
        continue

    actual = (
        endpoints[name]
        .get("sampling", {})
        .get("presence_penalty")
    )

    if actual != expected_pp:
        errors.append(
            f"{name}: expected presence_penalty "
            f"{expected_pp}, got {actual}"
        )


# ---------------------------------------------------------------------------
# Runtime dependency validation
# ---------------------------------------------------------------------------

requirements_path = ROOT / "hpc/vllm/requirements-vllm.txt"

if requirements_path.is_file():
    req_text = requirements_path.read_text(encoding="utf-8")

    if "vllm==0.29.0" not in req_text:
        errors.append(
            "requirements-vllm.txt must pin vllm==0.29.0"
        )

    if "transformers==5.17.0" not in req_text:
        errors.append(
            "requirements-vllm.txt must pin transformers==5.17.0"
        )


# ---------------------------------------------------------------------------
# Apptainer definition validation
# ---------------------------------------------------------------------------

definition_path = ROOT / "hpc/vllm/arggym-vllm.def"

if definition_path.is_file():
    def_text = definition_path.read_text(encoding="utf-8")

    if "vllm/vllm-openai:v0.29.0-cu129" not in def_text:
        errors.append(
            "Apptainer definition must use "
            "vllm/vllm-openai:v0.29.0-cu129"
        )

    if "sys.version_info[:2] != (3, 12)" not in def_text:
        errors.append(
            "Apptainer definition does not enforce Python 3.12"
        )

    if "python3" not in def_text or "\n    python -" in def_text:
        errors.append(
            "Apptainer definition must use python3 explicitly"
        )


# ---------------------------------------------------------------------------
# ArgGYM runner validation
# ---------------------------------------------------------------------------

runner_path = ROOT / "hpc/vllm/run_vllm_arggym.py"

if runner_path.is_file():
    runner_text = runner_path.read_text(encoding="utf-8")

    if '"uv", "run"' in runner_text or "uv run" in runner_text:
        errors.append(
            "run_vllm_arggym.py still depends on uv"
        )

    if "sys.executable" not in runner_text:
        errors.append(
            "run_vllm_arggym.py must use sys.executable"
        )

    forbidden_size_policy_tokens = (
        ">40B total parameters refused",
        "allow_over_40b_mxfp4",
        "allow_over_40b_bf16",
    )
    for token in forbidden_size_policy_tokens:
        if token in runner_text:
            errors.append(
                f"run_vllm_arggym.py still contains obsolete model-size policy token: {token}"
            )


# ---------------------------------------------------------------------------
# Shell syntax validation
# ---------------------------------------------------------------------------

shell_files = [
    ROOT / "hpc/vllm/check_runtime.sh",
    ROOT / "hpc/vllm/setup_runtime.sbatch",
    ROOT / "hpc/vllm/submit_truba.sh",
    ROOT / "hpc/vllm/truba_vllm.sbatch",
    ROOT / "hpc/vllm/preflight_truba.sh",
]

for shell_file in shell_files:
    if not shell_file.is_file():
        continue

    proc = subprocess.run(
        ["bash", "-n", str(shell_file)],
        capture_output=True,
        text=True,
    )

    if proc.returncode:
        errors.append(
            f"shell syntax error in {shell_file.name}: "
            f"{proc.stderr.strip()}"
        )


# ---------------------------------------------------------------------------
# TRUBA submission helper validation
# ---------------------------------------------------------------------------

submit_path = ROOT / "hpc/vllm/submit_truba.sh"

if submit_path.is_file():
    submit_text = submit_path.read_text(encoding="utf-8")

    for token in (
        "kolyoz-cuda",
        "afterok",
        "16 * GPUS",
        "SETUP_GPUS=1",
        "SETUP_CPUS=16",
    ):
        if token not in submit_text:
            errors.append(
                "submit_truba.sh missing expected TRUBA "
                f"orchestration token: {token}"
            )


# ---------------------------------------------------------------------------
# GPU job validation
# ---------------------------------------------------------------------------

job_path = ROOT / "hpc/vllm/truba_vllm.sbatch"

if job_path.is_file():
    job_text = job_path.read_text(encoding="utf-8")

    if "exec --nv" not in job_text:
        errors.append(
            "GPU job must execute the image with Apptainer --nv"
        )

    if "pip install" in job_text or "python -m venv" in job_text:
        errors.append(
            "GPU job must not install packages or create a venv"
        )


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------

if errors:
    print("FAILED")

    for error in errors:
        print(" -", error)

    raise SystemExit(1)


print(
    f"PASS: {len(profiles)} model profiles and their "
    f"{len(endpoints)} endpoint configs validated"
)
print(
    "PASS: TRUBA Apptainer deployment closure validated statically"
)