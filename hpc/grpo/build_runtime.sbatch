#!/usr/bin/env bash
#SBATCH --job-name=arggym-grpo-build
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH -p kolyoz-cuda
#SBATCH -A romer
#SBATCH -C H200
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00

set -euo pipefail
: "${ARGGYM_ROOT:?export ARGGYM_ROOT}"
cd "$ARGGYM_ROOT"
RUNTIME_DIR="${ARGGYM_GRPO_RUNTIME_DIR:-/arf/scratch/$USER/arggym_grpo_runtime}"
IMAGE="$RUNTIME_DIR/arggym-grpo-vllm0.28.0-trl1.13.0.sif"
mkdir -p "$RUNTIME_DIR" outputs/slurm

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_BIN=singularity
else
    echo 'ERROR: Apptainer/Singularity unavailable' >&2
    exit 1
fi

exec 9>"$RUNTIME_DIR/build.lock"
flock 9
CACHE_ROOT="${APPTAINER_CACHEDIR:-/arf/scratch/$USER/apptainer-cache}"
TMP_ROOT="${APPTAINER_TMPDIR:-/arf/scratch/$USER/apptainer-tmp/${SLURM_JOB_ID:-$$}}"
mkdir -p "$CACHE_ROOT" "$TMP_ROOT"
export APPTAINER_CACHEDIR="$CACHE_ROOT"
export APPTAINER_TMPDIR="$TMP_ROOT"

TMP_IMAGE="$RUNTIME_DIR/.arggym-grpo-${SLURM_JOB_ID:-$$}.sif"
TMP_DEF="$RUNTIME_DIR/.arggym-grpo-${SLURM_JOB_ID:-$$}.def"
trap 'rm -f "$TMP_IMAGE" "$TMP_DEF"; rm -rf "$TMP_ROOT"' EXIT

# Apptainer resolves %files on the host; materialize absolute checkout paths.
sed \
  -e "s#^    hpc/grpo/requirements-grpo.txt #    $ARGGYM_ROOT/hpc/grpo/requirements-grpo.txt #" \
  -e "s#^    pyproject.toml #    $ARGGYM_ROOT/pyproject.toml #" \
  hpc/grpo/arggym-grpo.def > "$TMP_DEF"

"$CONTAINER_BIN" build --fakeroot "$TMP_IMAGE" "$TMP_DEF"
"$CONTAINER_BIN" exec --bind "$ARGGYM_ROOT:$ARGGYM_ROOT" --pwd "$ARGGYM_ROOT" \
    "$TMP_IMAGE" python3 - <<'PY'
import importlib.metadata as md
import arggym, evals.prompt, rl.reward
for name in ('vllm','trl','peft','datasets','accelerate','python-argumentation'):
    print(name, md.version(name))
assert md.version('vllm').split('+',1)[0] == '0.28.0'
assert md.version('trl').split('+',1)[0] == '1.13.0'
assert md.version('python-argumentation') == '2.0.2'
print('new GRPO runtime validation: PASS')
PY
mv -f "$TMP_IMAGE" "$IMAGE"
sha256sum "$IMAGE" | tee "$IMAGE.sha256"
echo "Runtime ready: $IMAGE"
