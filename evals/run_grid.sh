#!/usr/bin/env bash
# Grid driver: for each model, serve once and run EACH difficulty level as its
# own standalone eval (separate run dir + scoring), then tear down. Serving each
# model once (rather than per level) avoids re-loading weights for every level,
# while keeping per-level runs independent so levels can be compared and a single
# level's failure does not touch the others.
#
#   TASKSET_ID=<dir> SWEEP_ID=<id> LEVELS="1 3 5 10 15" MODELS="qwen3.6-27b ..." \
#     bash evals/run_grid.sh
#
# SCORER selects the benchmark's scorer: evals.scoring for v1, evals.v2_scoring
# for ArgGYM_v2. The two are different benchmarks with different scorers, so the
# taskset and the scorer must be chosen together -- pointing one at the other's
# taskset produces numbers that mean nothing.
#
# Markers under outputs/sweeps/<id>/: <model>.L<lvl>.done / .crashed, <model>.failed
# (serve failure). A completed <model>.L<lvl>.done is skipped on re-run (resume).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
cd "$REPO" || exit 1

# serve.sh's container name; defined here too so the server-log capture resolves
# under set -u (serve.sh runs in a separate shell, so its CONTAINER is not shared).
CONTAINER=arggym-vllm

TASKSET_ID="${TASKSET_ID:?TASKSET_ID required (directory name under data/tasksets)}"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d-%H%M%S)}"
STATE="$REPO/outputs/sweeps/$SWEEP_ID"
mkdir -p "$STATE"

# Levels to run, each as a standalone eval. Space-separated.
read -ra LEVELS <<< "${LEVELS:-1 3 5 10 15}"

# Scorer module, paired with the taskset. v1: evals.scoring. v2: evals.v2_scoring.
SCORER="${SCORER:-evals.scoring}"

# Suffix on each run directory, so runs from different benchmarks never
# land in one namespace and get globbed into a single report.
RUN_TAG="${RUN_TAG:-pilot}"

if [ -n "${MODELS:-}" ]; then
  read -ra MODELS <<< "$MODELS"
else
  MODELS=(
    qwen3.6-27b
    qwen3.5-27b
    gemma-4-31b-it
    qwen3.5-9b
    qwen3.5-4b
    gemma-4-E4B-it
    llama-3.1-8b-instruct
  )
fi

echo "grid $SWEEP_ID | taskset $TASKSET_ID | scorer $SCORER | ${#MODELS[@]} models | levels: ${LEVELS[*]}"
echo "$TASKSET_ID" > "$STATE/taskset_id"

for m in "${MODELS[@]}"; do
  echo
  echo "############################################################"
  echo "### $m  ($(date -Is))"
  echo "############################################################"

  # Skip the model entirely only if every requested level is already done.
  all_done=1
  for lvl in "${LEVELS[@]}"; do
    [ -f "$STATE/$m.L$lvl.done" ] || { all_done=0; break; }
  done
  if [ "$all_done" -eq 1 ]; then
    echo "all levels done, skipping"
    continue
  fi

  if ! bash "$REPO/evals/serve.sh" start "$m"; then
    echo "SERVE FAILED for $m -- skipping" | tee "$STATE/$m.failed"
    bash "$REPO/evals/serve.sh" stop
    continue
  fi

  # Capture the server log for the whole model (covers every level) so a crash
  # after health-ready is diagnosable despite the container's --rm.
  docker logs -f "$CONTAINER" >"$STATE/$m.server.log" 2>&1 &
  LOGPID=$!

  for lvl in "${LEVELS[@]}"; do
    if [ -f "$STATE/$m.L$lvl.done" ]; then
      echo "--- $m L$lvl already done, skipping"
      continue
    fi

    run_dir=$("$PY" - "$m" "$lvl" "$RUN_TAG" <<'EOF'
import sys, datetime, pathlib
m, lvl = sys.argv[1], sys.argv[2]
ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(pathlib.Path("outputs/runs") / f"{ts}__{m}__L{int(lvl):02d}__{sys.argv[3]}")
EOF
)

    echo "--- eval $m L$lvl -> $run_dir"
    if "$PY" -m evals.runner \
          model="$m" \
          taskset_id="$TASKSET_ID" \
          sweep_id="$SWEEP_ID" \
          "eval_filter.levels=[$lvl]" \
          hydra.run.dir="$run_dir" 2>&1 | tee "$STATE/$m.L$lvl.log"; then
      echo "--- score $m L$lvl"
      "$PY" -m "$SCORER" "$run_dir" \
          --taskset-dir "data/tasksets/$TASKSET_ID" 2>&1 | tee -a "$STATE/$m.L$lvl.log"
      # Gate on API error rate: a run dominated by API errors means the server
      # died mid-run, not that the model is bad (runner exits 0 either way).
      err=$("$PY" -c "import json;print(json.load(open('$run_dir/metrics.json'))['overall'].get('api_error_rate',1))" 2>/dev/null || echo 1)
      if "$PY" -c "import sys;sys.exit(0 if $err<=0.20 else 1)" 2>/dev/null; then
        echo "$run_dir" > "$STATE/$m.L$lvl.done"
      else
        echo "$run_dir api_error_rate=$err" | tee "$STATE/$m.L$lvl.crashed"
        echo "REJECTED $m L$lvl: api_error_rate=$err -- server died mid-run"
        # The server is likely dead; stop looping this model's remaining levels
        # rather than hammering a broken endpoint. Remaining levels resume later.
        break
      fi
    else
      echo "EVAL FAILED for $m L$lvl" | tee "$STATE/$m.L$lvl.failed"
    fi
  done

  kill "$LOGPID" 2>/dev/null
  bash "$REPO/evals/serve.sh" stop
done

echo
echo "=== grid $SWEEP_ID finished at $(date -Is)"
echo "run dirs: outputs/runs/*__L*__$RUN_TAG (sweep $SWEEP_ID)"
