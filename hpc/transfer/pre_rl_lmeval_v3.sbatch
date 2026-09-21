#!/bin/bash
#SBATCH -A romer
#SBATCH -p kolyoz-cuda
#SBATCH --job-name=arggym-pre-lmeval
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/arf/scratch/futan/ArgGYM/logs/transfer/pre_lmeval_%x_%j.out
#SBATCH --error=/arf/scratch/futan/ArgGYM/logs/transfer/pre_lmeval_%x_%j.err

set -euo pipefail

BENCH="${1:?Usage: sbatch pre_rl_lmeval_v3.sbatch <gsm8k|mmlu_pro|bbh>}"

ROOT=/arf/scratch/futan/ArgGYM
SIF=/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif
VENV=/arf/scratch/futan/arggym_transfer_runtime/lm_eval_venv_20260920
LM=/arf/scratch/futan/arggym_transfer_sources/lm_eval
CACHE=/arf/scratch/futan/arggym_transfer_runtime/hf_transfer_cache_20260920

MODEL=/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18

TASKS=$ROOT/transfer/lm_eval_tasks/pinned_20260920_qwen3
BASE_OUT=$ROOT/outputs/transfer/pre_rl_qwen3_14b_v3/lm_eval

case "$BENCH" in
    gsm8k)
        TASK=arggym_gsm8k_cot_pinned
        ;;
    mmlu_pro)
        TASK=arggym_mmlu_pro_pinned
        ;;
    bbh)
        TASK=arggym_bbh_pinned
        ;;
    *)
        echo "ERROR: unknown benchmark: $BENCH" >&2
        exit 2
        ;;
esac

OUT=$BASE_OUT/$BENCH

mkdir -p "$OUT" "$ROOT/logs/transfer"
cd "$ROOT"

# These are execution metadata only. They do not alter the frozen protocol.
{
    echo "slurm_job_id=$SLURM_JOB_ID"
    echo "benchmark=$BENCH"
    echo "task=$TASK"
    echo "model=$MODEL"
    echo "lm_eval=$LM"
    echo "task_bundle=$TASKS"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "protocol_lock:"
    sha256sum \
      "$ROOT/outputs/transfer/protocol_freeze/transfer_protocol_lock_20260920.json"
    echo
    echo "task_manifest:"
    sha256sum \
      "$TASKS/manifest.json"
} > "$OUT/run_environment.txt"

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_len=40960,enable_thinking=True,think_end_token=</think>"

echo "========================================"
echo "PRE-RL LM-EVAL"
echo "benchmark: $BENCH"
echo "task:      $TASK"
echo "job:       $SLURM_JOB_ID"
echo "node:      $SLURMD_NODENAME"
echo "========================================"

nvidia-smi

apptainer exec \
  --nv \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env \
    PYTHONPATH="$LM:$ROOT" \
    HF_HOME="$CACHE" \
    HF_HUB_CACHE="$CACHE/hub" \
    HUGGINGFACE_HUB_CACHE="$CACHE/hub" \
    HF_DATASETS_CACHE="$CACHE/datasets" \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    "$VENV/bin/python" \
    -m lm_eval \
      --model vllm \
      --model_args "$MODEL_ARGS" \
      --tasks "$TASK" \
      --include_path "$TASKS" \
      --batch_size auto \
      --apply_chat_template \
      --log_samples \
      --output_path "$OUT"

echo
echo "PRE-RL LM-EVAL COMPLETE: $BENCH"
date -u +%Y-%m-%dT%H:%M:%SZ
