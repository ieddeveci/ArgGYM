#!/usr/bin/env bash
# Launch the ArgGYM_v2 model sweep.
#
# A thin wrapper over run_grid.sh that pins the three things which must agree
# for a v2 run: the v2 scorer, v2's level set, and a run-directory tag that
# keeps v2 runs out of the v1 namespace. Pointing one benchmark's scorer at the
# other's taskset produces numbers that mean nothing, so they are set together
# here rather than left to be remembered at the call site.
#
#   TASKSET_ID=v2-pilot-<stamp>-<hash> bash evals/run_v2_sweep.sh
#
# Optional: MODELS="a b c" to override the roster, LEVELS to override the ladder.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKSET_ID="${TASKSET_ID:?TASKSET_ID required (directory name under data/tasksets)}"

# ArgGYM_v2's curriculum is 3/6/9/12/15, not v1's 1/3/5/10/15. Each level runs
# as a standalone eval so per-level scores stay directly comparable and a
# crashed model resumes at level granularity.
export LEVELS="${LEVELS:-3 6 9 12 15}"
export SCORER="evals.v2_scoring"
export RUN_TAG="v2"
export SWEEP_ID="${SWEEP_ID:-v2-$(date +%Y%m%d-%H%M%S)}"
export TASKSET_ID

manifest="$REPO/data/tasksets/$TASKSET_ID/manifest.json"
if [ ! -f "$manifest" ]; then
  echo "no manifest at $manifest" >&2
  exit 1
fi
# Refuse to score a v1 taskset with the v2 scorer. The two produce plausible
# numbers for each other's data, so the mismatch would not announce itself.
if ! grep -q '"benchmark": "ArgGYM_v2"' "$manifest"; then
  echo "taskset $TASKSET_ID is not an ArgGYM_v2 taskset (manifest lacks benchmark=ArgGYM_v2)" >&2
  echo "use evals/run_grid.sh for v1 tasksets" >&2
  exit 1
fi

echo "ArgGYM_v2 sweep $SWEEP_ID"
echo "  taskset : $TASKSET_ID"
echo "  scorer  : $SCORER"
echo "  levels  : $LEVELS"
exec bash "$REPO/evals/run_grid.sh"
