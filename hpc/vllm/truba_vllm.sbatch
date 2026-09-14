#!/usr/bin/env bash
#SBATCH --job-name=arggym-vllm
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROFILE="${1:?profile required}"
cd "${SLURM_SUBMIT_DIR}"

# submit_truba.sh creates this before sbatch is called; keep this for
# interactive/direct launches as well.
mkdir -p outputs/slurm outputs/vllm_logs

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export VLLM_API_KEY="${VLLM_API_KEY:-none}"
export VLLM_BIN="${VLLM_BIN:-$HOME/venvs/arggym-vllm/bin/vllm}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Avoid localhost:8000 collisions when several one-GPU jobs share a node.
PORT="${VLLM_PORT:-$((20000 + (SLURM_JOB_ID % 20000)))}"

ARGS=(
  "$PROFILE"
  --template xml_tags
  --elicitation cot
  --port "$PORT"
)

if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi

uv run python hpc/vllm/run_vllm_arggym.py "${ARGS[@]}"
