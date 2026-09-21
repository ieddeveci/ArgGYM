#!/usr/bin/env bash
#SBATCH --job-name=arggym-grpo
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH -p kolyoz-cuda
#SBATCH -A romer
#SBATCH -C H200
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=2-23:30:00

set -euo pipefail
CONFIG="${1:-configs/rl/qwen3_14b_smoke.yaml}"
: "${ARGGYM_ROOT:?export ARGGYM_ROOT before sbatch}"
cd "$ARGGYM_ROOT"

RUNTIME_DIR="${ARGGYM_GRPO_RUNTIME_DIR:-/arf/scratch/$USER/arggym_grpo_runtime}"
IMAGE="${ARGGYM_GRPO_IMAGE:-$RUNTIME_DIR/arggym-grpo-vllm0.28.0-trl1.13.0.sif}"
HF_HOME="${HF_HOME:-/arf/scratch/$USER/arggym_hf_cache}"
mkdir -p outputs/slurm outputs/rl "$HF_HOME"

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_BIN=singularity
else
    echo 'ERROR: Apptainer/Singularity unavailable' >&2
    exit 1
fi

[[ -f "$IMAGE" ]] || { echo "ERROR: training image missing: $IMAGE" >&2; exit 1; }
: "${ARGGYM_MODEL_PATH:?export ARGGYM_MODEL_PATH to a pinned local Qwen3-14B snapshot}"

export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_ARGGYM_ROOT="$ARGGYM_ROOT"
export APPTAINERENV_ARGGYM_MODEL_PATH="$ARGGYM_MODEL_PATH"
export APPTAINERENV_ARGGYM_RL_RUN_DIR="$ARGGYM_ROOT/outputs/rl/slurm-${SLURM_JOB_ID}"
export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export APPTAINERENV_HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export APPTAINERENV_PYTHONHASHSEED=0
export APPTAINERENV_TOKENIZERS_PARALLELISM=false

EXTRA_BIND_ARGS=()
if [[ -n "${ARGGYM_LIGER_SITE:-}" ]]; then
    [[ -d "$ARGGYM_LIGER_SITE" ]] || {
        echo "ERROR: ARGGYM_LIGER_SITE missing: $ARGGYM_LIGER_SITE" >&2
        exit 1
    }
    EXTRA_BIND_ARGS+=(--bind "$ARGGYM_LIGER_SITE:$ARGGYM_LIGER_SITE")
    export APPTAINERENV_PYTHONPATH="$ARGGYM_LIGER_SITE"
    echo "External Liger site: $ARGGYM_LIGER_SITE"
fi

nvidia-smi
IMAGE_SHA256="$(sha256sum "$IMAGE" | awk '{print $1}')"
echo "$IMAGE_SHA256  $IMAGE"
export APPTAINERENV_ARGGYM_GRPO_IMAGE="$IMAGE"
export APPTAINERENV_ARGGYM_GRPO_IMAGE_SHA256="$IMAGE_SHA256"

"$CONTAINER_BIN" exec --nv \
    --bind "$ARGGYM_ROOT:$ARGGYM_ROOT" \
    --bind "$HF_HOME:$HF_HOME" \
    --bind "$ARGGYM_MODEL_PATH:$ARGGYM_MODEL_PATH" \
    "${EXTRA_BIND_ARGS[@]}" \
    --pwd "$ARGGYM_ROOT" \
    "$IMAGE" \
    bash hpc/grpo/run_node.sh "$CONFIG"
