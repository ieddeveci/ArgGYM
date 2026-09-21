#!/bin/bash
#SBATCH -A romer
#SBATCH -p kolyoz-cuda
#SBATCH --job-name=course-density20-v1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/arf/scratch/futan/ArgGYM/logs/transfer/pilots/course_density20_v1_%j.out
#SBATCH --error=/arf/scratch/futan/ArgGYM/logs/transfer/pilots/course_density20_v1_%j.err

set -eo pipefail

ROOT=/arf/scratch/futan/ArgGYM
SIF=/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif
MODEL=/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
RG=/arf/scratch/futan/arggym_transfer_sources/reasoning_gym_v4
RGVENV=/arf/scratch/futan/arggym_transfer_runtime/reasoning_gym_v4_py312_venv/venv

EXPECTED_RG_REV=49b07130b3fcd12f2d064bba7c43869543a0e7e7
EXPECTED_DATA_SHA=9248f45f391657ba6a0ce0cca4568fc0d70c9554c02f347114952bc1eb890be6

DATA=$ROOT/data/transfer/pilots/course_schedule/course_density20_v1.jsonl
MANIFEST=$ROOT/data/transfer/pilots/course_schedule/course_density20_v1.manifest.json
OUT=$ROOT/outputs/transfer/pilots/course_density20_v1
RUNNER=$ROOT/transfer/pilots/course_schedule/run_course_density20_v1.py

mkdir -p "$OUT" "$ROOT/logs/transfer/pilots"
cd "$ROOT"

ACTUAL_RG_REV=$(git -C "$RG" rev-parse HEAD)

if [ "$ACTUAL_RG_REV" != "$EXPECTED_RG_REV" ]; then
    echo "RG revision drift expected=$EXPECTED_RG_REV actual=$ACTUAL_RG_REV"
    exit 21
fi

python3 - "$DATA" "$MANIFEST" "$EXPECTED_DATA_SHA" <<'PY'
from collections import Counter, defaultdict
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
    r["task"] != "course_schedule"
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
    "p34": 10,
    "p45": 10,
}):
    raise SystemExit(
        f"bad counts: {dict(counts)}"
    )

balance = defaultdict(Counter)

for row in rows:
    solvable = bool(
        row["entry"]["metadata"]["solvable"]
    )

    label = "true" if solvable else "false"

    balance[
        row["difficulty"]
    ][label] += 1

for level in ("p34", "p45"):
    if balance[level] != Counter({
        "true": 5,
        "false": 5,
    }):
        raise SystemExit(
            f"{level} bad balance: "
            f"{dict(balance[level])}"
        )

print("level counts:", dict(counts))
print(
    "label balance:",
    {
        k: dict(v)
        for k, v in sorted(balance.items())
    },
)
print("preflight structure: PASS")
PY

PORT=$((18000 + (SLURM_JOB_ID % 40000)))

SERVER_LOG=$ROOT/logs/transfer/pilots/vllm_course_density20_v1_${SLURM_JOB_ID}.log
RUNNER_LOG=$ROOT/logs/transfer/pilots/course_density20_v1_${SLURM_JOB_ID}.runner.log

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

echo "COURSE DENSITY20 VLLM READY port=$PORT"

apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env PYTHONPATH="$RG:$ROOT" \
  "$RGVENV/bin/python" \
  "$RUNNER" \
  "http://127.0.0.1:${PORT}/v1" \
  > "$RUNNER_LOG" 2>&1

cat "$OUT/metrics.json"

echo "COURSE DENSITY20 COMPLETE"
