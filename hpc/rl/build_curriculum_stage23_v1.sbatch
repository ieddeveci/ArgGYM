#!/bin/bash
#SBATCH -A romer
#SBATCH -p kolyoz-cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=arggym-build-stage23
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/arf/scratch/futan/ArgGYM/logs/rl/build_curriculum_stage23_%j.out
#SBATCH --error=/arf/scratch/futan/ArgGYM/logs/rl/build_curriculum_stage23_%j.err

set -eo pipefail

ROOT=/arf/scratch/futan/ArgGYM
SIF=/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif

OUTDIR=$ROOT/data/rl/curriculum_stage23_build_v1

cd "$ROOT"

if [ -e "$OUTDIR" ]; then
    echo "REFUSING: staging directory already exists:"
    echo "$OUTDIR"
    exit 20
fi

echo "================================================================================"
echo "PRE-GENERATION CANONICAL STATE"
echo "================================================================================"

sha256sum \
  data/rl/curriculum/stage1.jsonl \
  data/rl/curriculum/dev.jsonl \
  data/taskset.jsonl

echo
echo "================================================================================"
echo "GENERATING FULL CURRICULUM INTO ISOLATED STAGING DIRECTORY"
echo "================================================================================"

apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env PYTHONPATH="$ROOT" \
  python3 "$ROOT/rl/generate.py" \
    curriculum \
    --spec "$ROOT/tasksets/standard.yaml" \
    --frozen "$ROOT/data/taskset.jsonl" \
    --curriculum "$ROOT/configs/rl/curriculum.yaml" \
    --out-dir "$OUTDIR" \
    --max-candidates-per-item 80

echo
echo "================================================================================"
echo "GENERATED FILES"
echo "================================================================================"

find "$OUTDIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\n' \
  | sort

echo
echo "================================================================================"
echo "GENERATED DATA HASHES"
echo "================================================================================"

sha256sum \
  "$OUTDIR/stage1.jsonl" \
  "$OUTDIR/stage2.jsonl" \
  "$OUTDIR/stage3.jsonl" \
  "$OUTDIR/dev.jsonl"

echo
echo "================================================================================"
echo "REPRODUCIBILITY CHECK: EXISTING STAGE1"
echo "================================================================================"

OLD_STAGE1=$(sha256sum "$ROOT/data/rl/curriculum/stage1.jsonl" | awk '{print $1}')
NEW_STAGE1=$(sha256sum "$OUTDIR/stage1.jsonl" | awk '{print $1}')

echo "existing stage1: $OLD_STAGE1"
echo "staged   stage1: $NEW_STAGE1"

if [ "$OLD_STAGE1" != \
"1e08153b108369121e2f3f1d0b052af8179fe87e8e7fd8897b6cfc13be49a235" ]; then
    echo "REFUSING: existing canonical Stage1 changed"
    exit 21
fi

if [ "$NEW_STAGE1" != "$OLD_STAGE1" ]; then
    echo "REFUSING: generator does not reproduce canonical Stage1 byte-for-byte"
    exit 22
fi

echo "STAGE1 REPRODUCIBILITY: PASS"

echo
echo "================================================================================"
echo "DEV COMPARISON"
echo "================================================================================"

OLD_DEV=$(sha256sum "$ROOT/data/rl/curriculum/dev.jsonl" | awk '{print $1}')
NEW_DEV=$(sha256sum "$OUTDIR/dev.jsonl" | awk '{print $1}')

echo "existing dev: $OLD_DEV"
echo "staged   dev: $NEW_DEV"

if [ "$OLD_DEV" = "$NEW_DEV" ]; then
    echo "DEV REPRODUCIBILITY: EXACT MATCH"
else
    echo "DEV REPRODUCIBILITY: DIFFERENT -- DO NOT PROMOTE ANYTHING YET"
fi

echo
echo "================================================================================"
echo "FROZEN TASKSET POST-GENERATION"
echo "================================================================================"

sha256sum "$ROOT/data/taskset.jsonl"

echo
echo "CURRICULUM STAGING BUILD COMPLETE"
