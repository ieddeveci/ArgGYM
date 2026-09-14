#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:?profile required}"
GPU_TYPE="${GPU_TYPE:-H100}"

case "$GPU_TYPE" in
  H100|H200) ;;
  *) echo "GPU_TYPE must be H100 or H200" >&2; exit 2 ;;
esac

PROFILE_FILE="hpc/vllm/models/${PROFILE}.yaml"
[[ -f "$PROFILE_FILE" ]] || { echo "Unknown profile: $PROFILE" >&2; exit 2; }

# Slurm opens stdout/stderr before the job script runs, so this directory
# must exist before sbatch is submitted.
mkdir -p outputs/slurm outputs/vllm_logs

DEFAULT_GPUS="$(
  uv run python - "$PROFILE" "$GPU_TYPE" <<'PY'
import sys, yaml
from pathlib import Path

profile, gpu = sys.argv[1], sys.argv[2].lower()
cfg = yaml.safe_load(
    (Path("hpc/vllm/models") / f"{profile}.yaml").read_text(encoding="utf-8")
)
print(cfg.get(f"tensor_parallel_size_{gpu}", cfg["tensor_parallel_size"]))
PY
)"

GPUS="${TP_SIZE:-$DEFAULT_GPUS}"
[[ "$GPUS" =~ ^[1-4]$ ]] || { echo "TP_SIZE/GPUS must be 1..4" >&2; exit 2; }
CPUS=$((16 * GPUS))

sbatch \
  -p kolyoz-cuda \
  -N 1 \
  -n 1 \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="${CPUS}" \
  -C "${GPU_TYPE}" \
  --export="ALL,TP_SIZE=${GPUS},GPU_TYPE=${GPU_TYPE}" \
  hpc/vllm/truba_vllm.sbatch "$PROFILE"
