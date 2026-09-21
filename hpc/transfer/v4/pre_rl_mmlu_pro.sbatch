#!/bin/bash
#SBATCH -A romer
#SBATCH -p kolyoz-cuda
#SBATCH --job-name=arggym-v4-pre-mmlu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=/arf/scratch/futan/ArgGYM/logs/transfer/v4/pre_mmlu_pro_%j.out
#SBATCH --error=/arf/scratch/futan/ArgGYM/logs/transfer/v4/pre_mmlu_pro_%j.err

set -eo pipefail

ROOT=/arf/scratch/futan/ArgGYM

SIF=/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif

MODEL=/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18

LM=/arf/scratch/futan/arggym_transfer_sources/lm_eval

VENV=/arf/scratch/futan/arggym_transfer_runtime/lm_eval_venv_20260920

CACHE=/arf/scratch/futan/arggym_transfer_runtime/hf_transfer_cache_20260920

TASKS=$ROOT/transfer/lm_eval_tasks/pinned_20260920_v4

FREEZE=$ROOT/outputs/transfer/protocol_freeze/v4/lm_eval

OUT=$ROOT/outputs/transfer/pre_rl_qwen3_14b_v4/mmlu_pro

TASK=arggym_mmlu_pro_pinned

EXPECTED_LM_REV=d6de81643928d653435c431bae19945d41d32520

EXPECTED_SIF_SHA=a6317095fc2cb21d225bdd5a6d1f7822a4d57ac6430e5dfecd6206aa2aa8a957

EXPECTED_TEMPLATE_SHA=f2a7e42d7eed52fcc90dd50dc1eb57ad4651fe19092d2b39ebdbd81ee6007412

EXPECTED_TREE_SHA=4c15be054fc1b645e51b7f3443ce6528f54eebd7f62a24551099c1626a72f8d6

EXPECTED_FREEZE_SHA=fbf850875e4bd30ea81f126a0567931b67af44e1709d6ffbe4b7eeca722ecde6

EXPECTED_PROMPT_DETAIL_SHA=df2bd69937a633325f5ce8a4082dcab118628b03b3d852439872e72227939dbd

EXPECTED_PROMPT_SUMMARY_SHA=fac25dc1839c11bc1718d0938b0e5c270ff6fe9bd038c4e1ae840bcefceee7d9

mkdir -p \
  "$OUT" \
  "$ROOT/logs/transfer/v4"

cd "$ROOT"

check_sha() {
    path="$1"
    expected="$2"

    actual=$(sha256sum "$path" | awk '{print $1}')

    if [ "$actual" != "$expected" ]; then
        echo "HASH DRIFT: $path"
        echo "expected=$expected"
        echo "actual=$actual"
        exit 20
    fi
}

check_sha \
  "$SIF" \
  "$EXPECTED_SIF_SHA"

check_sha \
  "$TASKS/mmlu_pro/_default_template_yaml" \
  "$EXPECTED_TEMPLATE_SHA"

check_sha \
  "$FREEZE/lm_eval_v4_freeze_manifest.json" \
  "$EXPECTED_FREEZE_SHA"

check_sha \
  "$FREEZE/prompt_token_counts.jsonl" \
  "$EXPECTED_PROMPT_DETAIL_SHA"

check_sha \
  "$FREEZE/prompt_budget_audit.json" \
  "$EXPECTED_PROMPT_SUMMARY_SHA"

ACTUAL_LM_REV=$(git -C "$LM" rev-parse HEAD)

if [ "$ACTUAL_LM_REV" != "$EXPECTED_LM_REV" ]; then
    echo "LM-EVAL REVISION DRIFT"
    echo "expected=$EXPECTED_LM_REV"
    echo "actual=$ACTUAL_LM_REV"
    exit 21
fi

python3 - \
  "$TASKS" \
  "mmlu_pro" \
  "$EXPECTED_TREE_SHA" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

tasks = Path(sys.argv[1])
subdir = sys.argv[2]
expected = sys.argv[3]

root = tasks / subdir

files = []

for p in sorted(
    x for x in root.rglob("*")
    if x.is_file()
    and x.name != "manifest.json"
):
    h = hashlib.sha256(
        p.read_bytes()
    ).hexdigest()

    files.append({
        "path":
            p.relative_to(tasks).as_posix(),
        "size":
            p.stat().st_size,
        "sha256":
            h,
    })

blob = (
    json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n"
).encode()

actual = hashlib.sha256(
    blob
).hexdigest()

if actual != expected:
    raise SystemExit(
        "MMLU-PRO TREE HASH DRIFT\n"
        f"expected={expected}\n"
        f"actual={actual}"
    )

print(
    "MMLU-PRO TREE HASH: PASS",
    actual,
)
PY

cat > "$OUT/runtime_identity_${SLURM_JOB_ID}.txt" <<EOF
benchmark=mmlu_pro
task=$TASK
model=$MODEL
lm_eval_revision=$ACTUAL_LM_REV
freeze_manifest_sha256=$EXPECTED_FREEZE_SHA
prompt_detail_sha256=$EXPECTED_PROMPT_DETAIL_SHA
prompt_summary_sha256=$EXPECTED_PROMPT_SUMMARY_SHA
task_tree_sha256=$EXPECTED_TREE_SHA
slurm_job_id=$SLURM_JOB_ID
EOF

export HF_HOME="$CACHE"
export HF_HUB_CACHE="$CACHE/hub"
export HUGGINGFACE_HUB_CACHE="$CACHE/hub"
export HF_DATASETS_CACHE="$CACHE/datasets"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_len=40960,enable_thinking=True,think_end_token=</think>"

echo "STARTING OFFICIAL V4 PRE: MMLU-PRO"
echo "task=$TASK"
echo "output=$OUT"

apptainer exec \
  --nv \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env \
    PYTHONPATH="$LM:$ROOT" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
    HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
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

echo "OFFICIAL V4 PRE COMPLETE: MMLU-PRO"
