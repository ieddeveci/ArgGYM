#!/bin/bash
#SBATCH -A romer
#SBATCH -p kolyoz-cuda
#SBATCH --job-name=arggym-pre-transfer
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/arf/scratch/futan/ArgGYM/logs/transfer/pre_custom_%j.out
#SBATCH --error=/arf/scratch/futan/ArgGYM/logs/transfer/pre_custom_%j.err

set -euo pipefail

ROOT=/arf/scratch/futan/ArgGYM
SIF=/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif
MODEL=/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
OUT=$ROOT/outputs/transfer/pre_rl_qwen3_14b_v3/custom

mkdir -p "$OUT" "$ROOT/logs/transfer"
cd "$ROOT"

export OPENAI_API_KEY=none

apptainer exec \
  --nv \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  vllm serve "$MODEL" \
    --served-model-name Qwen/Qwen3-14B \
    --dtype bfloat16 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --generation-config vllm \
    --reasoning-parser qwen3 \
    --host 127.0.0.1 \
    --port 8000 \
    > "$ROOT/logs/transfer/vllm_pre_${SLURM_JOB_ID}.log" \
    2>&1 &

SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0

for _ in $(seq 1 180); do
    if python3 - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen(
    "http://127.0.0.1:8000/health",
    timeout=2,
).read()
PY
    then
        READY=1
        break
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "vLLM server did not become healthy"
    exit 1
fi

echo "VLLM PRE SERVER READY"

PIDS=()

for BENCH in \
    multilogieval \
    rulearena \
    logiqa2 \
    finereason
do
    RUN_DIR="$OUT/$BENCH"
    ENDPOINT="$ROOT/configs/transfer/endpoints/qwen3_14b_pre_v3_${BENCH}.json"

    apptainer exec \
      --bind /arf/scratch/futan:/arf/scratch/futan \
      "$SIF" \
      bash -lc "
        set -euo pipefail
        cd '$ROOT'
        export PYTHONPATH='$ROOT'
        export OPENAI_API_KEY=none

        /usr/bin/python3 \
          -m transfer.run_custom_transfer \
          --benchmark '$BENCH' \
          --run-dir '$RUN_DIR' \
          --endpoint-config '$ENDPOINT'
      " \
      > "$ROOT/logs/transfer/pre_${BENCH}_${SLURM_JOB_ID}.log" \
      2>&1 &

    PIDS+=("$!")
done

FAILED=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "At least one runner did not finish successfully."
    echo "Resubmit this same job: completed IDs will be resumed/skipped."
    exit 1
fi

for BENCH in \
    multilogieval \
    rulearena \
    logiqa2 \
    finereason
do
    apptainer exec \
      --bind /arf/scratch/futan:/arf/scratch/futan \
      "$SIF" \
      bash -lc "
        set -euo pipefail
        cd '$ROOT'
        export PYTHONPATH='$ROOT'

        /usr/bin/python3 \
          -m transfer.score_custom_transfer \
          --benchmark '$BENCH' \
          --run-dir '$OUT/$BENCH'
      "
done

echo "PRE-RL CUSTOM TRANSFER: COMPLETE"
