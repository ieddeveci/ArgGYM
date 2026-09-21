#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ACCOUNT="${ACCOUNT:-romer}"
RUNTIME_DIR="${ARGGYM_GRPO_RUNTIME_DIR:-/arf/scratch/$USER/arggym_grpo_runtime}"
IMAGE="$RUNTIME_DIR/arggym-grpo-vllm0.28.0-trl1.13.0.sif"
mkdir -p outputs/slurm

command -v sbatch >/dev/null 2>&1 || { echo 'ERROR: run from TRUBA cuda-ui' >&2; exit 1; }

if [[ -f "$IMAGE" ]]; then
    echo "Runtime already exists: $IMAGE"
    [[ -f "$IMAGE.sha256" ]] && cat "$IMAGE.sha256"
    exit 0
fi

EXPORT="ALL,ARGGYM_ROOT=$ROOT,ARGGYM_GRPO_RUNTIME_DIR=$RUNTIME_DIR"
if [[ -n "${HF_TOKEN:-}" ]]; then
    EXPORT="$EXPORT,HF_TOKEN=$HF_TOKEN"
fi
JOB="$(sbatch --parsable -A "$ACCOUNT" --export="$EXPORT" hpc/grpo/build_runtime.sbatch)"
echo "GRPO runtime build job: $JOB"
echo "monitor: squeue -j $JOB"
echo "after it completes successfully, verify: $IMAGE"
