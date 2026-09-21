#!/usr/bin/env bash
#SBATCH --job-name=arggym-vllm
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROFILE="${1:-${MODEL_PROFILE:-}}"
[[ -n "$PROFILE" ]] || { echo "ERROR: model profile is missing" >&2; exit 2; }
: "${ARGGYM_ROOT:?ARGGYM_ROOT was not exported by submit_truba.sh}"
cd "$ARGGYM_ROOT"

SCRIPT_DIR="$ARGGYM_ROOT/hpc/vllm"
RUNTIME_DIR="${ARGGYM_RUNTIME_DIR:-/arf/scratch/$USER/arggym_runtime}"
IMAGE="${ARGGYM_IMAGE:-$RUNTIME_DIR/arggym-vllm-0.29.0.sif}"

mkdir -p outputs/slurm outputs/vllm_logs outputs/runs

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_BIN=singularity
else
    echo "ERROR: Apptainer/Singularity is unavailable on this GPU node." >&2
    exit 1
fi
export CONTAINER_BIN

"$SCRIPT_DIR/check_runtime.sh" --deep

"$CONTAINER_BIN" exec \
    --bind "$ARGGYM_ROOT:$ARGGYM_ROOT" \
    --pwd "$ARGGYM_ROOT" \
    "$IMAGE" \
    python3 hpc/vllm/verify_bundle.py

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: nvidia-smi is unavailable inside the allocated GPU job." >&2
    exit 1
}
nvidia-smi

HF_HOME="${HF_HOME:-/arf/scratch/$USER/arggym_hf_cache}"
mkdir -p "$HF_HOME"

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export VLLM_API_KEY="${VLLM_API_KEY:-none}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-0}"

# APPTAINERENV_* is the supported way to force these values into the image
# even when the base Docker image defines its own environment.
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_PYTHONHASHSEED="$PYTHONHASHSEED"
export APPTAINERENV_TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"
export APPTAINERENV_VLLM_API_KEY="$VLLM_API_KEY"
export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINERENV_VLLM_ENABLE_CUDA_COMPATIBILITY="$VLLM_ENABLE_CUDA_COMPATIBILITY"
if [[ -n "${HF_TOKEN:-}" ]]; then
    export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
fi

EXPECTED_GPUS="${TP_SIZE:-1}"
"$CONTAINER_BIN" exec --nv \
    --bind "$ARGGYM_ROOT:$ARGGYM_ROOT" \
    --bind "$HF_HOME:$HF_HOME" \
    --pwd "$ARGGYM_ROOT" \
    "$IMAGE" \
    python3 - "$EXPECTED_GPUS" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Apptainer runtime")
count = torch.cuda.device_count()
if count < expected:
    raise SystemExit(f"expected {expected} visible GPU(s), found {count}")
print("CUDA runtime check: PASS")
print("visible GPUs:", count)
for i in range(count):
    print(i, torch.cuda.get_device_name(i))
PY

RUNTIME_SHA256="$(sha256sum "$IMAGE" | awk '{print $1}')"
export APPTAINERENV_ARGGYM_RUNTIME_SHA256="$RUNTIME_SHA256"
export APPTAINERENV_ARGGYM_RUNTIME_IMAGE="$IMAGE"

# Keep localhost ports distinct if Slurm co-locates independent one-GPU jobs.
PORT="${VLLM_PORT:-$((20000 + (SLURM_JOB_ID % 20000)))}"
ARGS=(
    "$PROFILE"
    --template "${TEMPLATE:-xml_tags}"
    --elicitation "${ELICITATION:-cot}"
    --port "$PORT"
)
if [[ "${RESUME:-0}" == "1" ]]; then
    ARGS+=(--resume)
fi
if [[ "${SKIP_SCORE:-0}" == "1" ]]; then
    ARGS+=(--skip-score)
fi

echo "Starting ArgGYM evaluation"
echo "Profile: $PROFILE"
echo "Runtime image: $IMAGE"
echo "Runtime SHA256: $RUNTIME_SHA256"
echo "HF cache: $HF_HOME"
echo "vLLM port: $PORT"

"$CONTAINER_BIN" exec --nv \
    --bind "$ARGGYM_ROOT:$ARGGYM_ROOT" \
    --bind "$HF_HOME:$HF_HOME" \
    --pwd "$ARGGYM_ROOT" \
    "$IMAGE" \
    python3 hpc/vllm/run_vllm_arggym.py "${ARGS[@]}"
