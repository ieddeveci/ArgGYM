#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUNTIME_DIR="${ARGGYM_GRPO_RUNTIME_DIR:-/arf/scratch/$USER/arggym_grpo_runtime}"
IMAGE="${ARGGYM_GRPO_IMAGE:-$RUNTIME_DIR/arggym-grpo-vllm0.28.0-trl1.13.0.sif}"
HF_HOME="${HF_HOME:-/arf/scratch/$USER/arggym_hf_cache}"

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_BIN=singularity
else
    echo 'ERROR: Apptainer/Singularity unavailable' >&2
    exit 1
fi

[[ -f "$IMAGE" ]] || { echo "ERROR: runtime missing: $IMAGE" >&2; exit 1; }
[[ $# -gt 0 ]] || { echo "usage: $0 COMMAND [ARGS...]" >&2; exit 2; }

mkdir -p "$HF_HOME"
ARGS=(exec --bind "$ROOT:$ROOT" --bind "$HF_HOME:$HF_HOME" --pwd "$ROOT")
if [[ -n "${ARGGYM_MODEL_PATH:-}" ]]; then
    [[ -d "$ARGGYM_MODEL_PATH" ]] || { echo "ERROR: ARGGYM_MODEL_PATH not found: $ARGGYM_MODEL_PATH" >&2; exit 3; }
    ARGS+=(--bind "$ARGGYM_MODEL_PATH:$ARGGYM_MODEL_PATH")
fi

export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_ARGGYM_ROOT="$ROOT"
export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export APPTAINERENV_HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export APPTAINERENV_PYTHONHASHSEED=0
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
if [[ -n "${ARGGYM_MODEL_PATH:-}" ]]; then
    export APPTAINERENV_ARGGYM_MODEL_PATH="$ARGGYM_MODEL_PATH"
fi

"$CONTAINER_BIN" "${ARGS[@]}" "$IMAGE" "$@"
