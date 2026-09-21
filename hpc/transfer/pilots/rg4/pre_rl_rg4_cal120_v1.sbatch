#!/bin/bash
#SBATCH -A romer
#SBATCH -p kolyoz-cuda
#SBATCH --job-name=rg4-cal120-v1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/arf/scratch/futan/ArgGYM/logs/transfer/pilots/rg4_cal120_v1_%j.out
#SBATCH --error=/arf/scratch/futan/ArgGYM/logs/transfer/pilots/rg4_cal120_v1_%j.err

set -eo pipefail

ROOT=/arf/scratch/futan/ArgGYM
SIF=/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif
MODEL=/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
RG=/arf/scratch/futan/arggym_transfer_sources/reasoning_gym_v4
RGVENV=/arf/scratch/futan/arggym_transfer_runtime/reasoning_gym_v4_py312_venv/venv
EXPECTED_RG_REV=49b07130b3fcd12f2d064bba7c43869543a0e7e7

DATA=$ROOT/data/transfer/pilots/rg4/rg4_cal120_v1.jsonl
MANIFEST=$ROOT/data/transfer/pilots/rg4/rg4_cal120_v1.manifest.json
OUT=$ROOT/outputs/transfer/pilots/rg4_cal120_v1

mkdir -p "$OUT" "$ROOT/logs/transfer/pilots"
cd "$ROOT"

ACTUAL_RG_REV=$(git -C "$RG" rev-parse HEAD)
if [ "$ACTUAL_RG_REV" != "$EXPECTED_RG_REV" ]; then
    echo "RG revision drift expected=$EXPECTED_RG_REV actual=$ACTUAL_RG_REV"
    exit 21
fi

python3 - "$DATA" "$MANIFEST" <<'PY'
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json
import sys

data = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
rows = [
    json.loads(x)
    for x in data.read_text(encoding="utf-8").splitlines()
    if x.strip()
]
h = sha256(data.read_bytes()).hexdigest()

print("data rows:", len(rows))
print("data sha256:", h)

if len(rows) != 120:
    raise SystemExit(f"pilot row-count drift: {len(rows)}")
if h != manifest["data_sha256"]:
    raise SystemExit("pilot data/manifest hash mismatch")
if manifest["reasoning_gym_revision"] != "49b07130b3fcd12f2d064bba7c43869543a0e7e7":
    raise SystemExit("manifest RG revision mismatch")

tasks = Counter(x["task"] for x in rows)
levels = Counter(x["difficulty"] for x in rows)

expected_tasks = {
    "propositional_logic": 30,
    "knights_knaves": 30,
    "course_schedule": 30,
    "zebra_puzzles": 30,
}
expected_levels = {
    "easy": 40,
    "medium": 40,
    "hard": 40,
}

if dict(tasks) != expected_tasks:
    raise SystemExit(f"task counts mismatch: {dict(tasks)}")
if dict(levels) != expected_levels:
    raise SystemExit(f"difficulty counts mismatch: {dict(levels)}")

print("preflight structure: PASS")
PY

PORT=$((18000 + (SLURM_JOB_ID % 40000)))
SERVER_LOG=$ROOT/logs/transfer/pilots/vllm_rg4_cal120_v1_${SLURM_JOB_ID}.log
RUNNER_LOG=$ROOT/logs/transfer/pilots/rg4_cal120_v1_${SLURM_JOB_ID}.runner.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

apptainer exec \
  --nv \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  vllm serve "$MODEL" \
    --served-model-name Qwen/Qwen3-14B \
    --dtype bfloat16 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --generation-config vllm \
    --reasoning-parser qwen3 \
    --host 127.0.0.1 \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for _ in $(seq 1 180); do
    if python3 -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=2).read()" \
      >/dev/null 2>&1
    then
        READY=1
        break
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "vLLM server did not become healthy"
    tail -n 100 "$SERVER_LOG" || true
    exit 30
fi

echo "RG4 CAL120 VLLM READY port=$PORT"

apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env PYTHONPATH="$RG:$ROOT" \
  "$RGVENV/bin/python" \
  "$ROOT/transfer/pilots/rg4/rg4_pilot_v1.py" \
    run \
    --data-jsonl "$DATA" \
    --out-dir "$OUT" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    > "$RUNNER_LOG" 2>&1

cat "$OUT/metrics.json"
echo "RG4 CAL120 COMPLETE"
