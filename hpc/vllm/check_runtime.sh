#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---fast}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGGYM_ROOT="${ARGGYM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNTIME_DIR="${ARGGYM_RUNTIME_DIR:-/arf/scratch/$USER/arggym_runtime}"
IMAGE="${ARGGYM_IMAGE:-$RUNTIME_DIR/arggym-vllm-0.29.0.sif}"
STAMP="${ARGGYM_FINGERPRINT_FILE:-$RUNTIME_DIR/fingerprint.sha256}"

container_bin() {
    if [[ -n "${CONTAINER_BIN:-}" ]]; then
        command -v "$CONTAINER_BIN" >/dev/null 2>&1 || return 1
        printf '%s\n' "$CONTAINER_BIN"
    elif command -v apptainer >/dev/null 2>&1; then
        printf '%s\n' apptainer
    elif command -v singularity >/dev/null 2>&1; then
        printf '%s\n' singularity
    else
        return 1
    fi
}

fingerprint() {
    local files=(
        "$ARGGYM_ROOT/hpc/vllm/arggym-vllm.def"
        "$ARGGYM_ROOT/hpc/vllm/requirements-vllm.txt"
        "$ARGGYM_ROOT/pyproject.toml"
    )
    if [[ -f "$ARGGYM_ROOT/uv.lock" ]]; then
        files+=("$ARGGYM_ROOT/uv.lock")
    fi

    local f
    for f in "${files[@]}"; do
        [[ -f "$f" ]] || {
            echo "missing fingerprint input: $f" >&2
            return 1
        }
    done

    sha256sum "${files[@]}" | sha256sum | awk '{print $1}'
}

fast_check() {
    [[ -s "$IMAGE" ]] || { echo "runtime image missing: $IMAGE" >&2; return 1; }
    [[ -s "$STAMP" ]] || { echo "runtime fingerprint missing: $STAMP" >&2; return 1; }

    local expected installed bin
    expected="$(fingerprint)" || return 1
    installed="$(tr -d '[:space:]' < "$STAMP")"
    [[ "$expected" == "$installed" ]] || {
        echo "runtime fingerprint is stale" >&2
        return 1
    }

    bin="$(container_bin)" || {
        echo "Apptainer/Singularity command not found" >&2
        return 1
    }
    "$bin" inspect "$IMAGE" >/dev/null
}

deep_check() {
    fast_check
    local bin
    bin="$(container_bin)"
    "$bin" exec \
        --bind "$ARGGYM_ROOT:$ARGGYM_ROOT" \
        --pwd "$ARGGYM_ROOT" \
        "$IMAGE" \
        python3 - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 required, found {sys.version.split()[0]}")

import importlib.metadata as md

import hydra
import openai
import torch
import tqdm
import transformers
import vllm
import yaml
import arggym
import evals.run
import evals.score

def base_version(version: str) -> str:
    return version.split("+", 1)[0]

vllm_version = md.version("vllm")
transformers_version = md.version("transformers")
torch_version = md.version("torch")

if base_version(vllm_version) != "0.29.0":
    raise SystemExit(f"vLLM mismatch: {vllm_version}")
if base_version(transformers_version) != "5.17.0":
    raise SystemExit(f"transformers mismatch: {transformers_version}")

print("ArgGYM runtime deep check: PASS")
print("Python:", sys.version.split()[0])
print("vLLM:", vllm_version)
print("Transformers:", transformers_version)
print("PyTorch:", torch_version)
PY
}

case "$MODE" in
    --print-fingerprint)
        fingerprint
        ;;
    --fast)
        fast_check
        ;;
    --deep)
        deep_check
        ;;
    *)
        echo "usage: $0 [--fast|--deep|--print-fingerprint]" >&2
        exit 2
        ;;
esac
