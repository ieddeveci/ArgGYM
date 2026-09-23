#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
    export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGGYM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ARGGYM_ROOT"

# The argument is the eval config to run, a stem under evals/conf/model/. For
# most models that is also the serving profile's name (hf-qwen3-8b). A model
# with reasoning-effort levels has one config per level and no level-less one,
# so it is named with its level (hf-qwen3.8-27b-medium), and the profile is
# the name with the level taken off.
ENDPOINT_CONFIG="${1:?usage: $0 EVAL_CONFIG   (e.g. hf-qwen3-8b, hf-qwen3.8-27b-medium)}"
PROFILE="$ENDPOINT_CONFIG"
for level in minimal low medium high xhigh; do
    if [[ ! -f "$SCRIPT_DIR/models/${PROFILE}.yaml" && "$ENDPOINT_CONFIG" == *"-$level" ]]; then
        PROFILE="${ENDPOINT_CONFIG%-"$level"}"
    fi
done
GPU_TYPE="${GPU_TYPE:-H100}"
ACCOUNT="${ACCOUNT:-$USER}"
SETUP_TIME="${SETUP_TIME:-02:00:00}"
# The kolyoz-cuda maximum (TRUBA_DIRECTIVE.md). One job runs the whole taskset,
# and a large reasoning model can need more than three days for it. A job that
# hits the wall loses only its in-flight generations: submit the same config
# again with RESUME=1 and it continues from the rows already written, as many
# times as the run needs.
GPU_TIME="${GPU_TIME:-3-00:00:00}"

case "$GPU_TYPE" in
    H100|H200) ;;
    *)
        echo "ERROR: GPU_TYPE must be H100 or H200" >&2
        exit 2
        ;;
esac

command -v sbatch >/dev/null 2>&1 || {
    echo "ERROR: sbatch is not available. Submit from TRUBA cuda-ui." >&2
    exit 1
}

PROFILE_FILE="$SCRIPT_DIR/models/${PROFILE}.yaml"

[[ -f "$PROFILE_FILE" ]] || {
    echo "ERROR: unknown model profile: $PROFILE" >&2
    exit 2
}

[[ -f "evals/conf/model/${ENDPOINT_CONFIG}.yaml" ]] || {
    echo "ERROR: no eval config evals/conf/model/${ENDPOINT_CONFIG}.yaml" >&2
    ls evals/conf/model/ | sed -n "s/^\(${PROFILE}-.*\)\.yaml$/  one of: \1/p" >&2
    exit 2
}

[[ -f pyproject.toml ]] || {
    echo "ERROR: incomplete ArgGYM checkout; pyproject.toml is missing." >&2
    exit 2
}

mkdir -p outputs/slurm outputs/vllm_logs outputs/runs

profile_scalar() {
    local key="$1"

    awk -F':' -v key="$key" '
        $1 == key {
            sub(/^[^:]*:[[:space:]]*/, "", $0)
            gsub(/[[:space:]\r]+$/, "", $0)
            print $0
            exit
        }
    ' "$PROFILE_FILE"
}

GPU_KEY="tensor_parallel_size_$(printf '%s' "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')"

DEFAULT_GPUS="$(profile_scalar "$GPU_KEY")"

if [[ -z "$DEFAULT_GPUS" ]]; then
    DEFAULT_GPUS="$(profile_scalar tensor_parallel_size)"
fi

GPUS="${TP_SIZE:-$DEFAULT_GPUS}"

[[ "$GPUS" =~ ^[1-4]$ ]] || {
    echo "ERROR: tensor-parallel/GPU count must be 1..4, got '$GPUS'" >&2
    exit 2
}

GPU_CPUS=$((16 * GPUS))

# ARF-ACC (kolyoz-cuda) requires 16 CPU cores per requested H100/H200 GPU.
# Setup also runs on ARF-ACC because ARF and ARF-ACC are separate Slurm systems;
# this keeps setup -> evaluation in one controller so afterok is valid.
SETUP_GPUS=1
SETUP_CPUS=16

EXPORT_COMMON="ALL,ARGGYM_ROOT=$ARGGYM_ROOT,ARGGYM_RUNTIME_DIR=${ARGGYM_RUNTIME_DIR:-/arf/scratch/$USER/arggym_runtime}"

# Explicitly propagate Hugging Face authentication through Slurm.
if [[ -n "${HF_TOKEN:-}" ]]; then
    EXPORT_COMMON="$EXPORT_COMMON,HF_TOKEN=$HF_TOKEN"
    EXPORT_COMMON="$EXPORT_COMMON,HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"
    EXPORT_COMMON="$EXPORT_COMMON,APPTAINERENV_HF_TOKEN=$HF_TOKEN"
    EXPORT_COMMON="$EXPORT_COMMON,APPTAINERENV_HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"
fi

SETUP_JOB=""

if ARGGYM_ROOT="$ARGGYM_ROOT" "$SCRIPT_DIR/check_runtime.sh" --fast >/dev/null 2>&1; then
    echo "Runtime check: READY"
else
    echo "Runtime check: MISSING OR STALE"

    SETUP_JOB="$(
        sbatch --parsable \
            -A romer \
            -p kolyoz-cuda \
            -N 1 -n 1 \
            --gres="gpu:${SETUP_GPUS}" \
            --cpus-per-task="$SETUP_CPUS" \
            -C "$GPU_TYPE" \
            --time="$SETUP_TIME" \
            --export="$EXPORT_COMMON" \
            "$SCRIPT_DIR/setup_runtime.sbatch"
    )"

    echo "Runtime setup job: $SETUP_JOB"
fi

DEPENDENCY_ARGS=()

if [[ -n "$SETUP_JOB" ]]; then
    DEPENDENCY_ARGS=(--dependency="afterok:$SETUP_JOB")
fi

GPU_EXPORT="$EXPORT_COMMON,MODEL_PROFILE=$PROFILE,ENDPOINT_CONFIG=$ENDPOINT_CONFIG,TP_SIZE=$GPUS,GPU_TYPE=$GPU_TYPE,TEMPLATE=${TEMPLATE:-xml_tags},ELICITATION=${ELICITATION:-cot},RESUME=${RESUME:-0},SKIP_SCORE=${SKIP_SCORE:-0}"

GPU_JOB="$(
    sbatch --parsable \
        -A romer \
        -p kolyoz-cuda \
        -N 1 -n 1 \
        --gres="gpu:${GPUS}" \
        --cpus-per-task="$GPU_CPUS" \
        -C "$GPU_TYPE" \
        --time="$GPU_TIME" \
        "${DEPENDENCY_ARGS[@]}" \
        --export="$GPU_EXPORT" \
        "$SCRIPT_DIR/truba_vllm.sbatch" "$PROFILE"
)"

echo "GPU evaluation job: $GPU_JOB"

if [[ -n "$SETUP_JOB" ]]; then
    echo "GPU job dependency: afterok:$SETUP_JOB"
    echo "First-time setup uses one $GPU_TYPE allocation because ARF-ACC is a separate Slurm system."
fi

echo "Model profile: $PROFILE"
echo "Eval config: $ENDPOINT_CONFIG"
echo "GPU type/count: $GPU_TYPE x $GPUS"
echo "CPU cores for GPU job: $GPU_CPUS"
echo "Account: $ACCOUNT"
echo "Monitor with: squeue -u $USER"