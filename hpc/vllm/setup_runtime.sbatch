#!/usr/bin/env bash
#SBATCH --job-name=arggym-runtime
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

: "${ARGGYM_ROOT:?ARGGYM_ROOT was not exported by submit_truba.sh}"
cd "$ARGGYM_ROOT"

SCRIPT_DIR="$ARGGYM_ROOT/hpc/vllm"
RUNTIME_DIR="${ARGGYM_RUNTIME_DIR:-/arf/scratch/$USER/arggym_runtime}"
IMAGE="${ARGGYM_IMAGE:-$RUNTIME_DIR/arggym-vllm-0.29.0.sif}"
STAMP="${ARGGYM_FINGERPRINT_FILE:-$RUNTIME_DIR/fingerprint.sha256}"

mkdir -p "$RUNTIME_DIR" outputs/slurm

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_BIN=singularity
else
    echo "ERROR: Apptainer/Singularity is not available on this compute node." >&2
    exit 1
fi
export CONTAINER_BIN

# Prevent two simultaneous submissions from rebuilding the same runtime.
exec 9>"$RUNTIME_DIR/build.lock"
flock 9

if "$SCRIPT_DIR/check_runtime.sh" --fast >/dev/null 2>&1; then
    echo "ArgGYM runtime is already current; setup job has nothing to do."
    exit 0
fi

CACHE_ROOT="${APPTAINER_CACHEDIR:-/arf/scratch/$USER/apptainer-cache}"
TMP_ROOT="${APPTAINER_TMPDIR:-/arf/scratch/$USER/apptainer-tmp/${SLURM_JOB_ID:-$$}}"
mkdir -p "$CACHE_ROOT" "$TMP_ROOT"
export APPTAINER_CACHEDIR="$CACHE_ROOT"
export APPTAINER_TMPDIR="$TMP_ROOT"

TMP_IMAGE="$RUNTIME_DIR/.arggym-vllm-${SLURM_JOB_ID:-$$}.sif"
TMP_DEF="$RUNTIME_DIR/.arggym-vllm-${SLURM_JOB_ID:-$$}.def"
rm -f "$TMP_IMAGE" "$TMP_DEF"
cleanup() {
    rm -f "$TMP_IMAGE" "$TMP_DEF"
    rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

echo "Building ArgGYM runtime with $CONTAINER_BIN on $(hostname)"
echo "Output image: $IMAGE"
echo "Cache: $APPTAINER_CACHEDIR"
echo "Temporary build area: $APPTAINER_TMPDIR"

# Materialize absolute host paths in %files so the build does not depend on
# how a particular Apptainer version resolves relative host paths.
sed \
    -e "s#^    hpc/vllm/requirements-vllm.txt #    $ARGGYM_ROOT/hpc/vllm/requirements-vllm.txt #" \
    -e "s#^    pyproject.toml #    $ARGGYM_ROOT/pyproject.toml #" \
    "$SCRIPT_DIR/arggym-vllm.def" > "$TMP_DEF"

# TRUBA documents fakeroot container construction. Building occurs in this
# scheduled CPU job, never on the UI/login node.
"$CONTAINER_BIN" build --fakeroot \
    "$TMP_IMAGE" \
    "$TMP_DEF"

# Validate the new image before it can become the current runtime.
"$CONTAINER_BIN" exec \
    --bind "$ARGGYM_ROOT:$ARGGYM_ROOT" \
    --pwd "$ARGGYM_ROOT" \
    "$TMP_IMAGE" \
    python3 - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 required, found {sys.version.split()[0]}")

import importlib.metadata as md

import hydra
import openai
import torch
import tqdm
import transformers
import vllm
import yaml
import arggym
import evals.run
import evals.score

def base_version(version: str) -> str:
    return version.split("+", 1)[0]

vllm_version = md.version("vllm")
transformers_version = md.version("transformers")
torch_version = md.version("torch")

if base_version(vllm_version) != "0.29.0":
    raise SystemExit(f"vLLM mismatch: {vllm_version}")
if base_version(transformers_version) != "5.17.0":
    raise SystemExit(f"transformers mismatch: {transformers_version}")

print("new runtime validation: PASS")
print("Python", sys.version.split()[0])
print("vLLM", vllm_version)
print("Transformers", transformers_version)
print("PyTorch", torch_version)
PY

"$CONTAINER_BIN" exec \
    --bind "$ARGGYM_ROOT:$ARGGYM_ROOT" \
    --pwd "$ARGGYM_ROOT" \
    "$TMP_IMAGE" \
    python3 hpc/vllm/verify_bundle.py

mv -f "$TMP_IMAGE" "$IMAGE"
EXPECTED="$($SCRIPT_DIR/check_runtime.sh --print-fingerprint)"
printf '%s\n' "$EXPECTED" > "$STAMP.tmp"
mv -f "$STAMP.tmp" "$STAMP"

"$SCRIPT_DIR/check_runtime.sh" --deep
sha256sum "$IMAGE" > "$RUNTIME_DIR/image.sha256"
echo "Runtime ready: $IMAGE"
