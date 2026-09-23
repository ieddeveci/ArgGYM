# ArgGYM on TRUBA — exact execution directive

## What this bundle does

This bundle runs ArgGYM models on TRUBA H100/H200 through `cuda-ui` and `kolyoz-cuda`.
It does **not** create a venv or install Python packages on `/arf`. The Python/vLLM stack is a single Apptainer image.

The first submission behaves as follows:

```text
submit model
  -> runtime image missing/stale
  -> submit one setup job on kolyoz-cuda
  -> build + verify Apptainer image
  -> afterok
  -> submit/run model evaluation
```

Later submissions reuse the image and go directly to model evaluation.

Why the setup job also requests one GPU: TRUBA documents ARF and ARF-ACC as separate Slurm systems, while H100/H200 submission is through `cuda-ui`. Using the same `kolyoz-cuda` controller makes the `afterok` dependency valid. The setup script itself does not use CUDA.

## 1. Put the bundle in the ArgGYM repository

Your TRUBA tree must look like:

```text
/arf/scratch/$USER/ArgGYM/
  pyproject.toml
  arggym/
  evals/
  hpc/vllm/
```

Do not overwrite the corrected `evals/client.py` with an older version. Provider failures must return `Attempt(error=...)`, not `None`.

Then run:

```bash
cd /arf/scratch/$USER/ArgGYM
sed -i 's/\r$//' hpc/vllm/*.sh hpc/vllm/*.sbatch
chmod +x hpc/vllm/*.sh hpc/vllm/*.py
```

## 2. Connect to the correct TRUBA UI

H100/H200 ARF-ACC jobs are submitted through `cuda-ui`:

```bash
ssh -l $USER 172.16.6.16
cd /arf/scratch/futan/ArgGYM
```

Run:

```bash
./hpc/vllm/preflight_truba.sh
```

Do not continue unless it ends with:

```text
TRUBA preflight: PASS
```

A message saying the runtime image is not ready is normal on the first run.

## 3. Export credentials

For a gated Hugging Face model:

```bash
export HF_TOKEN='AIzaSyD6mJDiA4jy8JPl53rkm-VNC4Y4Whvk8Cc'
```

The Slurm account defaults to `$USER`. If your project uses a different account:

```bash
export ACCOUNT='YOUR_PROJECT_ACCOUNT'
```

## 4. Submit one model

Example, H100:

```bash
GPU_TYPE=H100 ./hpc/vllm/submit_truba.sh hf-qwen3.8-27b-medium
```

Example, H200:

```bash
GPU_TYPE=H200 ./hpc/vllm/submit_truba.sh hf-gemma4-31b-it
```

The argument is an eval config under `evals/conf/model/`. A model with
reasoning-effort levels (Qwen3.8, gpt-oss) has one config per level and no
level-less one, so name the level: `hf-qwen3.8-27b-low`, `-medium` or `-xhigh`.
The serving profile is the name without the level.

A run that outlasts the 3-day wall time loses only its in-flight generations.
Submit the same config again with `RESUME=1` to continue from the rows already
written.

The script automatically:

1. reads the model profile;
2. chooses its H100/H200 GPU count;
3. requests 16 CPU cores per GPU;
4. checks the Apptainer runtime fingerprint;
5. builds the runtime in a scheduled compute job if needed;
6. submits the evaluation with `afterok:<setup_job>` when setup is needed;
7. starts vLLM inside Apptainer with `--nv`;
8. resolves and records the immutable Hugging Face revision;
9. runs ArgGYM generation and scoring;
10. records the container SHA-256 in run provenance.

No `uv`, host venv, host `pip install`, or manual vLLM startup is required.

## 5. Monitor

```bash
squeue -u $USER
```

Logs:

```bash
ls -lh outputs/slurm/
tail -f outputs/slurm/arggym-runtime-<JOBID>.out
tail -f outputs/slurm/arggym-vllm-<JOBID>.out
```

vLLM logs:

```bash
ls -lh outputs/vllm_logs/
```

## 6. Verify a completed run

```bash
find outputs/runs -maxdepth 2 -type f | sort
cat outputs/runs/<RUN>/run.json
cat outputs/runs/<RUN>/metrics.json
```

`run.json` should include an `hf_runtime` record containing the model ID, exact HF revision, BF16 configuration, tensor-parallel size, vLLM/Python versions, Slurm job ID, GPU type, container path and container SHA-256.

## Operational facts encoded in the scripts

- UI/controller: `cuda-ui` / ARF-ACC
- GPU partition: `kolyoz-cuda`
- GPU type: H100 or H200
- CPU request: 16 cores per requested GPU
- Default evaluation walltime: 3 days, the kolyoz-cuda maximum; a run that needs more is resubmitted with `RESUME=1`, which continues from the rows already written
- Runtime/cache: `/arf/scratch/$USER/arggym_runtime`
- HF cache: `/arf/scratch/$USER/arggym_hf_cache`
- Runtime base: `vllm/vllm-openai:v0.29.0-x86_64-cu129`
- Runtime Python: 3.12
- vLLM: 0.29.0
- Transformers: 5.17.0

`/arf/scratch` is temporary storage. If TRUBA removes the cached SIF later, the next submission automatically rebuilds it.
