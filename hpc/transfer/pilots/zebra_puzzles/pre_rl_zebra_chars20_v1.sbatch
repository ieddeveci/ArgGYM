#!/bin/bash
#SBATCH -A romer
#SBATCH -p kolyoz-cuda
#SBATCH --job-name=zebra-chars20-v1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/arf/scratch/futan/ArgGYM/logs/transfer/pilots/zebra_chars20_v1_%j.out
#SBATCH --error=/arf/scratch/futan/ArgGYM/logs/transfer/pilots/zebra_chars20_v1_%j.err

set -eo pipefail

ROOT=/arf/scratch/futan/ArgGYM
SIF=/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif
MODEL=/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
RG=/arf/scratch/futan/arggym_transfer_sources/reasoning_gym_v4
RGVENV=/arf/scratch/futan/arggym_transfer_runtime/reasoning_gym_v4_py312_venv/venv

EXPECTED_RG_REV=49b07130b3fcd12f2d064bba7c43869543a0e7e7
EXPECTED_DATA_SHA=d1215d163587181b496cd1b5b546c48a8978950c42c80bee616a551b45c54148

DATA=$ROOT/data/transfer/pilots/zebra_puzzles/zebra_chars20_v1.jsonl
MANIFEST=$ROOT/data/transfer/pilots/zebra_puzzles/zebra_chars20_v1.manifest.json
OUT=$ROOT/outputs/transfer/pilots/zebra_chars20_v1
RUNNER=$ROOT/transfer/pilots/zebra_puzzles/run_zebra_chars20_v1.py

mkdir -p "$OUT" "$ROOT/logs/transfer/pilots"
cd "$ROOT"

ACTUAL_RG_REV=$(git -C "$RG" rev-parse HEAD)

if [ "$ACTUAL_RG_REV" != "$EXPECTED_RG_REV" ]; then
    echo "RG revision drift expected=$EXPECTED_RG_REV actual=$ACTUAL_RG_REV"
    exit 21
fi

python3 - "$DATA" "$MANIFEST" "$EXPECTED_DATA_SHA" <<'PY'
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import sys

data_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
expected_sha = sys.argv[3]

rows = [
    json.loads(x)
    for x in data_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

manifest = json.loads(
    manifest_path.read_text(
        encoding="utf-8"
    )
)

actual_sha = sha256(
    data_path.read_bytes()
).hexdigest()

print("data rows:", len(rows))
print("data sha256:", actual_sha)

if len(rows) != 20:
    raise SystemExit(
        f"expected 20 rows, got {len(rows)}"
    )

if actual_sha != expected_sha:
    raise SystemExit(
        f"dataset SHA drift: {actual_sha}"
    )

if manifest["data_sha256"] != expected_sha:
    raise SystemExit(
        "manifest SHA mismatch"
    )

if manifest["reasoning_gym_revision"] != \
   "49b07130b3fcd12f2d064bba7c43869543a0e7e7":
    raise SystemExit(
        "manifest RG revision mismatch"
    )

if any(
    r["task"] != "zebra_puzzles"
    for r in rows
):
    raise SystemExit(
        "unexpected task"
    )

counts = Counter(
    r["difficulty"]
    for r in rows
)

if counts != Counter({
    "c6": 10,
    "c7": 10,
}):
    raise SystemExit(
        f"bad counts: {dict(counts)}"
    )

print("level counts:", dict(counts))
print("preflight structure: PASS")
PY

PORT=$((18000 + (SLURM_JOB_ID % 40000)))

SERVER_LOG=$ROOT/logs/transfer/pilots/vllm_zebra_chars20_v1_${SLURM_JOB_ID}.log
RUNNER_LOG=$ROOT/logs/transfer/pilots/zebra_chars20_v1_${SLURM_JOB_ID}.runner.log

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

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
}
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

echo "ZEBRA CHARS20 VLLM READY port=$PORT"

apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env PYTHONPATH="$RG:$ROOT" \
  "$RGVENV/bin/python" \
  "$RUNNER" \
  "http://127.0.0.1:${PORT}/v1" \
  > "$RUNNER_LOG" 2>&1

cat "$OUT/metrics.json"

echo "ZEBRA CHARS20 COMPLETE"
