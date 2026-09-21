#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_node.sh CONFIG [--resume-from-checkpoint PATH]}"
shift
ROOT="${ARGGYM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

readarray -t CFG_FIELDS < <(python3 - "$CONFIG" <<'PY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1]))
print(cfg['model']['name'])
print(cfg['model']['revision'])
print(cfg['generation']['max_model_length'])
print(cfg['vllm'].get('gpu_memory_utilization', 0.90))
PY
)
MODEL_NAME="${CFG_FIELDS[0]}"
MODEL_REVISION="${CFG_FIELDS[1]}"
MAX_MODEL_LEN="${CFG_FIELDS[2]}"
GPU_MEMORY_UTILIZATION="${CFG_FIELDS[3]}"
MODEL_SOURCE="${ARGGYM_MODEL_PATH:-$MODEL_NAME}"
PORT="${VLLM_PORT:-8000}"
RUN_LOG_DIR="${ARGGYM_RL_RUN_DIR:-outputs/rl/slurm-${SLURM_JOB_ID:-local}}"
mkdir -p "$RUN_LOG_DIR"

export VLLM_SERVER_BASE_URL="http://127.0.0.1:${PORT}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# Fails on wrong frozen suite, wrong model snapshot, prompt overflow, data
# collisions, batch arithmetic, or a locked main config before loading models.
python3 -m rl.preflight --config "$CONFIG" --world-size 3

REVISION_ARGS=()
if [[ ! -e "$MODEL_SOURCE" ]]; then
    REVISION_ARGS=(--revision "$MODEL_REVISION")
fi

# GPU 0 is rollout-only. Dev mode exposes weight-transfer endpoints, so bind to
# localhost. TRL pushes updated policy weights after optimizer steps.
CUDA_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 \
vllm serve "$MODEL_SOURCE" \
    "${REVISION_ARGS[@]}" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --generation-config vllm \
    --weight-transfer-config '{"backend":"nccl"}' \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1 \
    > "$RUN_LOG_DIR/vllm-server.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python3 hpc/grpo/wait_for_vllm.py --url "$VLLM_SERVER_BASE_URL/v1/models" --timeout 600

# GPUs 1-3: data-parallel LoRA trainer. 1*3*8=24 completion samples per
# optimizer update, divisible by G=8 => 3 unique prompt groups per update.
CUDA_VISIBLE_DEVICES=1,2,3 \
accelerate launch --num_processes 3 --mixed_precision bf16 \
    -m rl.train_grpo --config "$CONFIG" "$@"
