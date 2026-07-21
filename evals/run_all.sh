#!/usr/bin/env bash
# Serial roster driver: serve -> smoke -> eval -> score -> tear down, per model.
# A model that fails to serve is skipped; the roster continues.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
cd "$REPO" || exit 1

TASKSET_ID="${TASKSET_ID:?TASKSET_ID required (directory name under outputs/tasksets)}"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d-%H%M%S)}"
STATE="$REPO/outputs/sweeps/$SWEEP_ID"
mkdir -p "$STATE"

MODELS=(
  qwen3.6-27b
  qwen3.5-27b
  gemma-4-31b-it
  qwen3.5-9b
  qwen3.5-4b
  llama-3.1-8b-instruct
)

echo "sweep $SWEEP_ID | taskset $TASKSET_ID | ${#MODELS[@]} models"
echo "$TASKSET_ID" > "$STATE/taskset_id"

for m in "${MODELS[@]}"; do
  echo
  echo "############################################################"
  echo "### $m  ($(date -Is))"
  echo "############################################################"

  if [ -f "$STATE/$m.done" ]; then
    echo "already done, skipping"
    continue
  fi

  if ! bash "$REPO/evals/serve.sh" start "$m"; then
    echo "SERVE FAILED for $m -- skipping" | tee "$STATE/$m.failed"
    bash "$REPO/evals/serve.sh" stop
    continue
  fi

  run_dir=$("$PY" - "$m" "$SWEEP_ID" <<'EOF'
import sys, datetime, pathlib
m, sweep = sys.argv[1], sys.argv[2]
ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(pathlib.Path("outputs/runs") / f"{ts}__{m}__pilot")
EOF
)

  echo "--- eval -> $run_dir"
  if "$PY" -m evals.runner \
        model="$m" \
        taskset_id="$TASKSET_ID" \
        sweep_id="$SWEEP_ID" \
        hydra.run.dir="$run_dir" 2>&1 | tee "$STATE/$m.log"; then
    echo "--- score"
    "$PY" -m evals.scoring "$run_dir" \
        --taskset-dir "outputs/tasksets/$TASKSET_ID" 2>&1 | tee -a "$STATE/$m.log"
    echo "$run_dir" > "$STATE/$m.done"
  else
    echo "EVAL FAILED for $m" | tee "$STATE/$m.failed"
  fi

  bash "$REPO/evals/serve.sh" stop
done

echo
echo "=== sweep $SWEEP_ID finished at $(date -Is)"
"$PY" -m evals.report --runs-glob "outputs/runs/*__pilot" --sweep "$SWEEP_ID" || true
