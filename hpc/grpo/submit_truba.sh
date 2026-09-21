#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/rl/qwen3_14b_smoke.yaml}"
ACCOUNT="${ACCOUNT:-romer}"
RUNTIME_DIR="${ARGGYM_GRPO_RUNTIME_DIR:-/arf/scratch/$USER/arggym_grpo_runtime}"
IMAGE="$RUNTIME_DIR/arggym-grpo-vllm0.28.0-trl1.13.0.sif"
mkdir -p outputs/slurm

command -v sbatch >/dev/null 2>&1 || { echo 'ERROR: run from TRUBA cuda-ui' >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: missing config $CONFIG" >&2; exit 2; }

EXPORT="ALL,ARGGYM_ROOT=$ROOT,ARGGYM_GRPO_RUNTIME_DIR=$RUNTIME_DIR"
if [[ -n "${HF_TOKEN:-}" ]]; then
    EXPORT="$EXPORT,HF_TOKEN=$HF_TOKEN"
fi

BUILD_JOB=""
if [[ ! -f "$IMAGE" ]]; then
    BUILD_JOB="$(sbatch --parsable -A "$ACCOUNT" --export="$EXPORT" hpc/grpo/build_runtime.sbatch)"
    echo "GRPO runtime build job: $BUILD_JOB"
fi
DEP=()
[[ -z "$BUILD_JOB" ]] || DEP=(--dependency="afterok:$BUILD_JOB")
TRAIN_JOB="$(sbatch --parsable -A "$ACCOUNT" "${DEP[@]}" --export="$EXPORT" \
    hpc/grpo/truba_grpo.sbatch "$CONFIG")"
echo "GRPO training job: $TRAIN_JOB"
[[ -z "$BUILD_JOB" ]] || echo "dependency: afterok:$BUILD_JOB"
echo "config: $CONFIG"
echo "monitor: squeue -u $USER"
