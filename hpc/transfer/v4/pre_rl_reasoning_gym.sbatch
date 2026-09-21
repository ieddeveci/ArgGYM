#!/bin/bash
#SBATCH -A romer
#SBATCH -p kolyoz-cuda
#SBATCH --job-name=arggym-v4-pre-rg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=/arf/scratch/futan/ArgGYM/logs/transfer/v4/pre_reasoning_gym_%j.out
#SBATCH --error=/arf/scratch/futan/ArgGYM/logs/transfer/v4/pre_reasoning_gym_%j.err

set -eo pipefail

ROOT=/arf/scratch/futan/ArgGYM
SIF=/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif

MODEL=/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18

RG=/arf/scratch/futan/arggym_transfer_sources/reasoning_gym_v4
RGVENV=/arf/scratch/futan/arggym_transfer_runtime/reasoning_gym_v4_py312_venv/venv

BASE_ENDPOINT=$ROOT/configs/transfer/v4/qwen3_14b_pre.json

OUT=$ROOT/outputs/transfer/pre_rl_qwen3_14b_v4/reasoning_gym

EXPECTED_RG_REV=49b07130b3fcd12f2d064bba7c43869543a0e7e7
EXPECTED_DATA_SHA=c005eca94a9bd9c8ba1c2a3660349e300daddd101cf427bb61f573f5ccd31ca6

EXPECTED_VERIFIER_SHA=dff4274f6fb07894d2baeaa401af0c5ffbc08d3db864c0ac82ac8836af80d695
EXPECTED_RUNNER_SHA=a145600d89045be67a301f0afdf19e7416d76ff18cb47b23867ae624aa35818d
EXPECTED_SCORER_SHA=c5f405bcca020665e0cdbe0a9cb18994cb2a7f71c1363ea81ac6a2f11a18a14f

mkdir -p \
  "$OUT" \
  "$ROOT/logs/transfer/v4"

cd "$ROOT"

check_hash() {
    path="$1"
    expected="$2"

    got=$(sha256sum "$path" | awk '{print $1}')

    if [ "$got" != "$expected" ]; then
        echo "HASH DRIFT: $path"
        echo "expected=$expected"
        echo "actual=$got"
        exit 20
    fi
}

check_hash \
  "$ROOT/transfer/v4/verify_reasoning_gym_freeze.py" \
  "$EXPECTED_VERIFIER_SHA"

check_hash \
  "$ROOT/transfer/v4/run_reasoning_gym.py" \
  "$EXPECTED_RUNNER_SHA"

check_hash \
  "$ROOT/transfer/v4/score_reasoning_gym.py" \
  "$EXPECTED_SCORER_SHA"

check_hash \
  "$ROOT/data/transfer/v4/frozen/reasoning_gym/reasoning_gym_v4_1000.jsonl" \
  "$EXPECTED_DATA_SHA"

ACTUAL_RG_REV=$(git -C "$RG" rev-parse HEAD)

if [ "$ACTUAL_RG_REV" != "$EXPECTED_RG_REV" ]; then
    echo "Reasoning Gym source revision drift"
    echo "expected=$EXPECTED_RG_REV"
    echo "actual=$ACTUAL_RG_REV"
    exit 21
fi

echo "RG source revision: PASS"

# Verify exact frozen rows before touching the GPU.
apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env PYTHONPATH="$ROOT" \
  /usr/bin/python3 \
  "$ROOT/transfer/v4/verify_reasoning_gym_freeze.py"

# Verify all ten registered native scorers against all 1,000
# frozen gold answers. No instances are regenerated here.
apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env PYTHONPATH="$RG:$ROOT" \
  "$RGVENV/bin/python" - <<'PY'
import json
from pathlib import Path
import reasoning_gym

p = Path(
    "/arf/scratch/futan/ArgGYM/"
    "data/transfer/v4/frozen/reasoning_gym/"
    "reasoning_gym_v4_1000.jsonl"
)

rows = [
    json.loads(line)
    for line in p.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if len(rows) != 1000:
    raise RuntimeError(
        f"expected 1000 frozen rows, got {len(rows)}"
    )

scorers = {}

for row in rows:
    entry = row["entry"]
    name = entry["metadata"]["source_dataset"]

    if name not in scorers:
        scorers[name] = (
            reasoning_gym.get_score_answer_fn(name)
        )

    score = float(
        scorers[name](
            answer=entry["answer"],
            entry=entry,
        )
    )

    if score != 1.0:
        raise RuntimeError(
            f"native gold self-score failure: "
            f"{row['id']} score={score}"
        )

print("RG NATIVE GOLD SELF-SCORE: 1000/1000 PASS")
PY

# Local transport port only. This does not alter any
# scientific generation parameter.
PORT=$((18000 + (SLURM_JOB_ID % 40000)))

RUNTIME_ENDPOINT=$OUT/runtime_endpoint_${SLURM_JOB_ID}.json

python3 - \
  "$BASE_ENDPOINT" \
  "$RUNTIME_ENDPOINT" \
  "$PORT" <<'PY'
from pathlib import Path
import json
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
port = int(sys.argv[3])

obj = json.loads(
    src.read_text(encoding="utf-8")
)

obj["base_url"] = (
    f"http://127.0.0.1:{port}/v1"
)

dst.write_text(
    json.dumps(
        obj,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "RG local port: $PORT"
echo "RG runtime endpoint: $RUNTIME_ENDPOINT"

SERVER_LOG=$ROOT/logs/transfer/v4/vllm_pre_reasoning_gym_${SLURM_JOB_ID}.log
RUNNER_LOG=$ROOT/logs/transfer/v4/pre_reasoning_gym_${SLURM_JOB_ID}.runner.log

export OPENAI_API_KEY=none
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

apptainer exec \
  --nv \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
  vllm serve "$MODEL" \
    --served-model-name Qwen/Qwen3-14B \
    --dtype bfloat16 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --generation-config vllm \
    --reasoning-parser qwen3 \
    --host 127.0.0.1 \
    --port "$PORT" \
    > "$SERVER_LOG" \
    2>&1 &

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

echo "VLLM V4 PRE SERVER READY: reasoning_gym"

if ! apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env \
    PYTHONPATH="$ROOT" \
    OPENAI_API_KEY=none \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
  /usr/bin/python3 \
  -m transfer.v4.run_reasoning_gym \
    --run-dir "$OUT" \
    --endpoint-config "$RUNTIME_ENDPOINT" \
    > "$RUNNER_LOG" \
    2>&1
then
    echo "Reasoning Gym generation failed"
    tail -n 100 "$RUNNER_LOG" || true
    exit 40
fi

echo "RG GENERATION COMPLETE"

apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env \
    PYTHONPATH="$RG:$ROOT" \
  "$RGVENV/bin/python" \
  -m transfer.v4.score_reasoning_gym \
    --run-dir "$OUT"

echo "OFFICIAL V4 PRE COMPLETE: reasoning_gym"
