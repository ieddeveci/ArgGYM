#!/usr/bin/env python3
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "hpc" / "vllm" / "models"
E = ROOT / "evals" / "conf" / "model"

ALLOWED_SAMPLING = {
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "seed",
    "stop", "presence_penalty", "frequency_penalty", "logprobs",
    "top_logprobs", "n", "reasoning_effort",
}

errors = []
profiles = {p.stem: yaml.safe_load(p.read_text()) for p in P.glob("*.yaml")}
endpoints = {p.stem: yaml.safe_load(p.read_text()) for p in E.glob("*.yaml")}

if set(profiles) != set(endpoints):
    errors.append("profile/endpoint name sets differ")

for name in sorted(set(profiles) & set(endpoints)):
    p = profiles[name]
    e = endpoints[name]
    if p["endpoint_config"] != name:
        errors.append(f"{name}: endpoint_config mismatch")
    if p["hf_model"] != e["model"]:
        errors.append(f"{name}: model ID mismatch")
    if p["dtype"] != "bfloat16":
        errors.append(f"{name}: dtype is not bfloat16")
    if float(p["params_b"]) > 40:
        errors.append(f"{name}: exceeds 40B total parameter policy")
    bad = set(e.get("sampling", {})) - ALLOWED_SAMPLING
    if bad:
        errors.append(f"{name}: unsupported sampling keys {sorted(bad)}")
    if int(e["sampling"]["max_tokens"]) + 4096 > int(p["max_model_len"]):
        errors.append(f"{name}: max_tokens + 4096 prompt reserve exceeds max_model_len")
    for key in ("tensor_parallel_size_h100", "tensor_parallel_size_h200"):
        if int(p.get(key, 0)) not in (1, 2, 3, 4):
            errors.append(f"{name}: invalid {key}")

# Model-specific invariants.
m = profiles["hf-ministral3-3b-reasoning"]
joined = " ".join(map(str, m.get("extra_args", [])))
for required in ("--tokenizer-mode mistral", "--config-format mistral",
                 "--load-format mistral"):
    if required not in joined:
        errors.append(f"Ministral missing required server args: {required}")
if m.get("reasoning_parser") != "mistral":
    errors.append("Ministral reasoning_parser must be mistral")

for name in ("hf-gemma4-31b-it", "hf-gemma4-26b-a4b-it"):
    if profiles[name].get("reasoning_parser") != "gemma4":
        errors.append(f"{name}: reasoning parser must be gemma4")
    if endpoints[name].get("sampling", {}).get("reasoning_effort") != "medium":
        errors.append(f"{name}: expected medium reasoning_effort")
    if endpoints[name].get("extra_body", {}).get("skip_special_tokens") is not False:
        errors.append(f"{name}: skip_special_tokens should be false")

for name in ("hf-qwen3.6-27b", "hf-qwen3.6-35b-a3b"):
    if endpoints[name]["sampling"].get("presence_penalty") != 1.5:
        errors.append(f"{name}: expected presence_penalty 1.5")

if errors:
    print("FAILED")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print(f"PASS: {len(profiles)} model profiles/endpoints validated")
