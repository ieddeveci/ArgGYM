# ArgGYM HF/vLLM — TRUBA Apptainer bundle

This is an overlay for an existing ArgGYM repository. It preserves the benchmark/scoring code and adds the H100/H200 model profiles plus TRUBA deployment automation.

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

If the Apptainer runtime is missing or stale, the submission helper first schedules a one-GPU runtime-build job on `kolyoz-cuda` and makes the evaluation depend on its successful completion. Later submissions reuse the image.

See `TRUBA_DIRECTIVE.md` for the exact sequence and checks.
