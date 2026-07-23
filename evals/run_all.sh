#!/usr/bin/env bash
# Serial roster driver: serve -> smoke -> eval -> score -> tear down, per model.
# A model that fails to serve is skipped; the roster continues.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
cd "$REPO" || exit 1

# The container name serve.sh uses. Defined here too so the `docker logs -f`
# server-log capture below resolves it -- serve.sh runs in a separate shell, so
# its CONTAINER does not carry over, and under `set -u` an unbound reference
# would abort the capture and silently drop every model's server.log.
CONTAINER=arggym-vllm

TASKSET_ID="${TASKSET_ID:?TASKSET_ID required (directory name under data/tasksets)}"
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

  # Always capture the server log, so a runtime crash after health-ready (which
  # serve.sh no longer watches) is diagnosable despite the container's --rm.
  docker logs -f "$CONTAINER" >"$STATE/$m.server.log" 2>&1 &
  LOGPID=$!

  # Optional: evaluate only a slice of the canonical taskset (e.g.
  # EVAL_LEVELS='[1,3,5]' for low difficulty) instead of building a derived one.
  FILTER_ARG=()
  [ -n "${EVAL_LEVELS:-}" ] && FILTER_ARG+=("eval_filter.levels=$EVAL_LEVELS")

  echo "--- eval -> $run_dir"
  if "$PY" -m evals.runner \
        model="$m" \
        taskset_id="$TASKSET_ID" \
        sweep_id="$SWEEP_ID" \
        "${FILTER_ARG[@]}" \
        hydra.run.dir="$run_dir" 2>&1 | tee "$STATE/$m.log"; then
    echo "--- score"
    "$PY" -m evals.scoring "$run_dir" \
        --taskset-dir "data/tasksets/$TASKSET_ID" 2>&1 | tee -a "$STATE/$m.log"
    # A run dominated by API errors means the server died mid-run, not that the
    # model is bad. The runner exits 0 either way (one bad sample must not kill a
    # run), so exit code cannot tell them apart -- gate on the error rate.
    err=$("$PY" -c "import json;print(json.load(open('$run_dir/metrics.json'))['overall'].get('api_error_rate',1))" 2>/dev/null || echo 1)
    if "$PY" -c "import sys;sys.exit(0 if $err<=0.20 else 1)" 2>/dev/null; then
      echo "$run_dir" > "$STATE/$m.done"
    else
      echo "$run_dir api_error_rate=$err" | tee "$STATE/$m.crashed"
      echo "REJECTED $m: api_error_rate=$err -- server died mid-run, not marked done"
    fi
  else
    echo "EVAL FAILED for $m" | tee "$STATE/$m.failed"
  fi

  kill "$LOGPID" 2>/dev/null
  bash "$REPO/evals/serve.sh" stop
done

echo
echo "=== sweep $SWEEP_ID finished at $(date -Is)"
"$PY" -m evals.report --runs-glob "outputs/runs/*__pilot" --sweep "$SWEEP_ID" || true
