# ArgGYM HF/vLLM — TRUBA Apptainer bundle

The vLLM serving profiles for the `hf-*` eval configs, and the scripts that serve them on the TRUBA cluster's H100 and H200 nodes. `docs/evaluation.md` ("Reproducing the results") covers serving the same profiles on your own machine.

## Run

On TRUBA `cuda-ui`:

```bash
cd /arf/scratch/$USER/ArgGYM
chmod +x hpc/vllm/*.sh hpc/vllm/*.py
./hpc/vllm/preflight_truba.sh
export HF_TOKEN=...        # only for gated models
export ACCOUNT=...         # optional; defaults to $USER
GPU_TYPE=H100 ./hpc/vllm/submit_truba.sh hf-qwen3.8-27b-medium
```

The job evaluates `data/taskset-lite.jsonl`, the default in `evals/conf/config.yaml`: 120 items per task are enough for a per-task ranking, at a fifth of the standard taskset's cost.

If the Apptainer runtime is missing or stale, the submission helper first schedules a one-GPU runtime-build job on `kolyoz-cuda` and makes the evaluation depend on its successful completion. Later submissions reuse the image.

See [truba-directive.md](truba-directive.md) for the exact sequence and checks.
