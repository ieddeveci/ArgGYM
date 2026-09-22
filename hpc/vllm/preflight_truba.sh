#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGGYM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ARGGYM_ROOT"

fail=0
for cmd in sbatch squeue; do
    if command -v "$cmd" >/dev/null 2>&1; then
        echo "PASS command: $cmd -> $(command -v "$cmd")"
    else
        echo "FAIL command missing: $cmd" >&2
        fail=1
    fi
done

if command -v apptainer >/dev/null 2>&1; then
    echo "PASS container runtime on UI: $(apptainer --version 2>/dev/null || true)"
elif command -v singularity >/dev/null 2>&1; then
    echo "PASS container runtime on UI: $(singularity --version 2>/dev/null || true)"
else
    echo "WARN: Apptainer/Singularity not visible on this UI node; compute jobs will check again."
fi

for f in \
    pyproject.toml \
    hpc/vllm/arggym-vllm.def \
    hpc/vllm/check_runtime.sh \
    hpc/vllm/setup_runtime.sbatch \
    hpc/vllm/submit_truba.sh \
    hpc/vllm/truba_vllm.sbatch \
    hpc/vllm/run_vllm_arggym.py \
    hpc/vllm/requirements-vllm.txt; do
    if [[ -f "$f" ]]; then
        echo "PASS file: $f"
    else
        echo "FAIL file missing: $f" >&2
        fail=1
    fi
done

PROFILE_COUNT="$(find hpc/vllm/models -maxdepth 1 -name 'hf-*.yaml' -type f | wc -l | tr -d ' ')"
echo "Model profiles: $PROFILE_COUNT"
if [[ "$PROFILE_COUNT" -lt 1 ]]; then
    fail=1
fi

# The authoritative ARF-ACC submission point is cuda-ui. A missing partition is
# a hard stop because the bundle targets H100/H200 on kolyoz-cuda.
if command -v scontrol >/dev/null 2>&1; then
    if scontrol show partition=kolyoz-cuda >/dev/null 2>&1; then
        echo "PASS Slurm partition: kolyoz-cuda is visible"
    else
        echo "FAIL: kolyoz-cuda is not visible. Connect to TRUBA cuda-ui before submitting." >&2
        fail=1
    fi
fi

if [[ -x hpc/vllm/check_runtime.sh ]] && hpc/vllm/check_runtime.sh --fast >/dev/null 2>&1; then
    echo "Runtime image: READY"
else
    echo "Runtime image: not ready; first submission will schedule setup_runtime.sbatch"
fi

if (( fail != 0 )); then
    echo "TRUBA preflight: FAILED" >&2
    exit 1
fi

echo "TRUBA preflight: PASS"
