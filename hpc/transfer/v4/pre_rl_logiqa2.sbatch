#!/bin/bash
#SBATCH -A romer
#SBATCH -p kolyoz-cuda
#SBATCH --job-name=arggym-v4-pre-logiqa2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=/arf/scratch/futan/ArgGYM/logs/transfer/v4/pre_logiqa2_%j.out
#SBATCH --error=/arf/scratch/futan/ArgGYM/logs/transfer/v4/pre_logiqa2_%j.err

set -eo pipefail

ROOT=/arf/scratch/futan/ArgGYM
SIF=/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif
MODEL=/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18

BENCH=logiqa2

BASE_ENDPOINT=$ROOT/configs/transfer/v4/qwen3_14b_pre.json
OUT=$ROOT/outputs/transfer/pre_rl_qwen3_14b_v4/$BENCH

EXPECTED_RUNNER_SHA=196a95021786683c909e7453902dd77e400d9ef86ff1b0d8138c8120c5958507
EXPECTED_SCORER_SHA=efb230ea79ad45627e17796d2f189b85eb977c5e812f5ebb5d0c08662c7002c9
EXPECTED_VERIFIER_SHA=7ae816f2966896553cbc797522208e9375597ad73e79a8c93cf826580570d941

mkdir -p "$OUT" "$ROOT/logs/transfer/v4"
cd "$ROOT"

export OPENAI_API_KEY=none
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
  "$ROOT/transfer/v4/run_custom_transfer.py" \
  "$EXPECTED_RUNNER_SHA"

check_hash \
  "$ROOT/transfer/v4/score_custom_transfer.py" \
  "$EXPECTED_SCORER_SHA"

check_hash \
  "$ROOT/transfer/v4/verify_custom_freeze.py" \
  "$EXPECTED_VERIFIER_SHA"

# Full frozen scientific-input verification before starting a GPU server.
apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env PYTHONPATH="$ROOT" \
  /usr/bin/python3 \
  "$ROOT/transfer/v4/verify_custom_freeze.py"

# ---------------------------------------------------------
# Runtime endpoint:
#
# Sampling/thinking/model configuration comes byte-for-byte
# from the frozen base endpoint. Only localhost TCP port is
# operationally varied so independent Slurm jobs may safely
# share one physical node.
#
# Once created, this file is reused on resubmission so the
# runner's resumable identity remains identical.
# ---------------------------------------------------------

RUNTIME_ENDPOINT="$OUT/runtime_endpoint.json"

if [ -f "$RUNTIME_ENDPOINT" ]; then
    PORT=$(
      python3 - \
        "$BASE_ENDPOINT" \
        "$RUNTIME_ENDPOINT" <<'PYENDPOINT'
from pathlib import Path
import json
import re
import sys

base_path = Path(sys.argv[1])
runtime_path = Path(sys.argv[2])

base = json.loads(base_path.read_text())
runtime = json.loads(runtime_path.read_text())

base_scientific = dict(base)
runtime_scientific = dict(runtime)

base_scientific.pop("base_url", None)
runtime_url = runtime_scientific.pop("base_url", None)

if runtime_scientific != base_scientific:
    raise SystemExit(
        "Existing runtime endpoint differs scientifically "
        "from frozen base endpoint"
    )

match = re.fullmatch(
    r"http://127\.0\.0\.1:(\d+)/v1",
    str(runtime_url),
)

if not match:
    raise SystemExit(
        f"Invalid runtime base_url: {runtime_url!r}"
    )

port = int(match.group(1))

if not (1024 <= port <= 65535):
    raise SystemExit(
        f"Invalid runtime port: {port}"
    )

print(port)
PYENDPOINT
    )
else
    PORT=$((18000 + (SLURM_JOB_ID % 40000)))

    python3 - \
      "$BASE_ENDPOINT" \
      "$RUNTIME_ENDPOINT" \
      "$PORT" <<'PYENDPOINT'
from pathlib import Path
import json
import sys

base_path = Path(sys.argv[1])
runtime_path = Path(sys.argv[2])
port = int(sys.argv[3])

obj = json.loads(base_path.read_text())

obj["base_url"] = (
    f"http://127.0.0.1:{port}/v1"
)

runtime_path.write_text(
    json.dumps(
        obj,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print(
    "runtime endpoint:",
    runtime_path,
)
print(
    "runtime port:",
    port,
)
PYENDPOINT
fi

echo "BENCH=$BENCH"
echo "PORT=$PORT"
echo "RUNTIME_ENDPOINT_SHA=$(sha256sum "$RUNTIME_ENDPOINT" | awk '{print $1}')"

SERVER_LOG=$ROOT/logs/transfer/v4/vllm_pre_${BENCH}_${SLURM_JOB_ID}.log
RUNNER_LOG=$ROOT/logs/transfer/v4/pre_${BENCH}_${SLURM_JOB_ID}.runner.log

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

echo "VLLM V4 PRE SERVER READY: $BENCH port=$PORT"

if ! apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  bash -lc "
    set -eo pipefail
    cd '$ROOT'

    export PYTHONPATH='$ROOT'
    export OPENAI_API_KEY=none
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    /usr/bin/python3 \
      -m transfer.v4.run_custom_transfer \
      --benchmark '$BENCH' \
      --run-dir '$OUT' \
      --endpoint-config '$RUNTIME_ENDPOINT'
  " \
  > "$RUNNER_LOG" \
  2>&1
then
    echo "V4 runner failed: $BENCH"
    tail -n 100 "$RUNNER_LOG" || true
    exit 40
fi

echo "V4 GENERATION COMPLETE: $BENCH"

apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  bash -lc "
    set -eo pipefail
    cd '$ROOT'

    export PYTHONPATH='$ROOT'

    /usr/bin/python3 \
      -m transfer.v4.score_custom_transfer \
      --benchmark '$BENCH' \
      --run-dir '$OUT'
  "

echo "OFFICIAL V4 PRE COMPLETE: $BENCH"
