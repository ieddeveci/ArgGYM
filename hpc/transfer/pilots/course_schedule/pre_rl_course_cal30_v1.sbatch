#!/bin/bash
#SBATCH -A romer
#SBATCH -p kolyoz-cuda
#SBATCH --job-name=course-cal30-v1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/arf/scratch/futan/ArgGYM/logs/transfer/pilots/course_cal30_v1_%j.out
#SBATCH --error=/arf/scratch/futan/ArgGYM/logs/transfer/pilots/course_cal30_v1_%j.err

set -eo pipefail

ROOT=/arf/scratch/futan/ArgGYM
SIF=/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif
MODEL=/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
RG=/arf/scratch/futan/arggym_transfer_sources/reasoning_gym_v4
RGVENV=/arf/scratch/futan/arggym_transfer_runtime/reasoning_gym_v4_py312_venv/venv
EXPECTED_RG_REV=49b07130b3fcd12f2d064bba7c43869543a0e7e7

DATA=$ROOT/data/transfer/pilots/course_schedule/course_cal30_v1.jsonl
MANIFEST=$ROOT/data/transfer/pilots/course_schedule/course_cal30_v1.manifest.json
OUT=$ROOT/outputs/transfer/pilots/course_cal30_v1
RUNNER=$ROOT/transfer/pilots/rg4/rg4_pilot_v4.py

mkdir -p "$OUT" "$ROOT/logs/transfer/pilots"
cd "$ROOT"

# Host-only revision check: do not call git from inside the SIF.
ACTUAL_RG_REV=$(git -C "$RG" rev-parse HEAD)
if [ "$ACTUAL_RG_REV" != "$EXPECTED_RG_REV" ]; then
    echo "RG revision drift expected=$EXPECTED_RG_REV actual=$ACTUAL_RG_REV"
    exit 21
fi

python3 - "$DATA" "$MANIFEST" <<'PY'
from pathlib import Path
from collections import Counter, defaultdict
from hashlib import sha256
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

if len(rows) != 30:
    raise SystemExit(f"pilot row-count drift: {len(rows)}")
if h != manifest["data_sha256"]:
    raise SystemExit("pilot data/manifest hash mismatch")
if manifest["reasoning_gym_revision"] != "49b07130b3fcd12f2d064bba7c43869543a0e7e7":
    raise SystemExit("manifest RG revision mismatch")
if any(r["task"] != "course_schedule" for r in rows):
    raise SystemExit("non-course_schedule task present")

levels = Counter(r["difficulty"] for r in rows)
if levels != Counter({"easy": 10, "medium": 10, "hard": 10}):
    raise SystemExit(f"difficulty counts mismatch: {dict(levels)}")

balance = defaultdict(Counter)
for r in rows:
    meta = r["entry"]["metadata"]
    label = "true" if bool(meta["solvable"]) else "false"
    balance[r["difficulty"]][label] += 1

for level in ("easy", "medium", "hard"):
    if balance[level] != Counter({"true": 5, "false": 5}):
        raise SystemExit(
            f"label balance mismatch {level}: {dict(balance[level])}"
        )

print("label balance:", {k: dict(v) for k, v in sorted(balance.items())})
print("preflight structure: PASS")
PY

if [ ! -f "$RUNNER" ]; then
    echo "missing runner: $RUNNER"
    exit 22
fi

PORT=$((18000 + (SLURM_JOB_ID % 40000)))
SERVER_LOG=$ROOT/logs/transfer/pilots/vllm_course_cal30_v1_${SLURM_JOB_ID}.log
RUNNER_LOG=$ROOT/logs/transfer/pilots/course_cal30_v1_${SLURM_JOB_ID}.runner.log

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

echo "COURSE CAL30 VLLM READY port=$PORT"

# rg4_pilot_v4.py's run command hardcodes a 120-row suite. This wrapper
# reuses only its validated request/extraction helpers for this isolated
# 30-row course-schedule calibration job.
apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env PYTHONPATH="$RG:$ROOT" \
  "$RGVENV/bin/python" - \
    "$DATA" "$OUT" "http://127.0.0.1:${PORT}/v1" "$RUNNER" \
    > "$RUNNER_LOG" 2>&1 <<'PY'
from collections import defaultdict
from pathlib import Path
import importlib.util
import json
import sys

import reasoning_gym

data_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
base_url = sys.argv[3]
runner_path = Path(sys.argv[4])

spec = importlib.util.spec_from_file_location("rg4v4", runner_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

rows = m.read_jsonl(data_path)
if len(rows) != 30:
    raise RuntimeError(f"expected 30 rows, got {len(rows)}")

out_dir.mkdir(parents=True, exist_ok=True)
gen_path = out_dir / "generations.jsonl"

existing = {}
if gen_path.exists():
    existing = {x["id"]: x for x in m.read_jsonl(gen_path)}

print(f"rows={len(rows)} existing={len(existing)}", flush=True)

for i, row in enumerate(rows, 1):
    item_id = row["id"]

    if item_id in existing:
        print(f"[{i}/30] SKIP {item_id}", flush=True)
        continue

    print(f"[{i}/30] {item_id}", flush=True)
    result = m.request_one(base_url, row["entry"]["question"])

    raw_score = None
    adapted_score = None
    adapted_answer = ""
    scorer_error = ""

    if not result["error"]:
        scorer = reasoning_gym.get_score_answer_fn("course_schedule")
        try:
            raw_score = float(
                scorer(answer=result["completion"], entry=row["entry"])
            )
            adapted_answer = m.extract_answer(
                "course_schedule",
                result["completion"],
                row["entry"],
            )
            adapted_score = float(
                scorer(answer=adapted_answer, entry=row["entry"])
            )
        except Exception as exc:
            scorer_error = f"{type(exc).__name__}: {exc}"

    record = {
        "id": item_id,
        "benchmark": row["benchmark"],
        "task": row["task"],
        "difficulty": row["difficulty"],
        "generator_index": row["generator_index"],
        "generator_seed": row["generator_seed"],
        "parser_version": "rg4_parser_v4",
        "completion": result["completion"],
        "reasoning": result["reasoning"],
        "adapted_answer": adapted_answer,
        "raw_native_score": raw_score,
        "adapted_native_score": adapted_score,
        "full_credit": (
            adapted_score is not None and adapted_score >= 0.999999
        ),
        "finish_reason": result["finish_reason"],
        "truncated": result["finish_reason"] == "length",
        "usage": result["usage"],
        "latency_s": result["latency_s"],
        "attempts": result["attempts"],
        "error": result["error"],
        "scorer_error": scorer_error,
    }

    m.append_jsonl(gen_path, record)
    existing[item_id] = record

# Summary is label-stratified because the 15-course pilot showed a large
# solvable/unsolvable runtime asymmetry.
groups = defaultdict(list)
data_by_id = {row["id"]: row for row in rows}

for item_id, g in existing.items():
    row = data_by_id[item_id]
    gold = str(row["entry"]["answer"])
    groups[(row["difficulty"], gold)].append(g)

metrics = {
    "benchmark": "course_cal30_v1",
    "rows": 30,
    "completed": len(existing),
    "parser_version": "rg4_parser_v4",
    "by_difficulty_gold": {},
}

for (level, gold), xs in sorted(groups.items()):
    scores = [
        x["adapted_native_score"]
        for x in xs
        if x["adapted_native_score"] is not None
    ]
    lats = [
        x["latency_s"]
        for x in xs
        if x["latency_s"] is not None
    ]
    toks = [
        (x.get("usage") or {}).get("completion_tokens")
        for x in xs
        if (x.get("usage") or {}).get("completion_tokens") is not None
    ]

    metrics["by_difficulty_gold"][f"{level}::{gold}"] = {
        "n": len(xs),
        "accuracy": (
            sum(s >= 0.999999 for s in scores) / len(scores)
            if scores else None
        ),
        "truncated": sum(bool(x["truncated"]) for x in xs),
        "mean_latency_s": (
            sum(lats) / len(lats) if lats else None
        ),
        "mean_completion_tokens": (
            sum(toks) / len(toks) if toks else None
        ),
    }

(out_dir / "metrics.json").write_text(
    json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
PY

cat "$OUT/metrics.json"
echo "COURSE CAL30 COMPLETE"
