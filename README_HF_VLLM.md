# ArgGYM Hugging Face / vLLM bundle

Copy this bundle into the repository root. Do not replace `evals/client.py`, `evals/run.py`, prompting, or scoring.

## Placement
- `evals/conf/model/hf-*.yaml`: existing evaluator endpoint configs.
- `hpc/vllm/models/hf-*.yaml`: local serving/HPC profiles.
- `hpc/vllm/run_vllm_arggym.py`: starts vLLM, resolves immutable HF revision, calls existing ArgGYM generation and scorer, records provenance.
- `hpc/vllm/submit_truba.sh`: Slurm submission helper.
- `hpc/vllm/truba_vllm.sbatch`: generic TRUBA job.
- `hpc/vllm/requirements-vllm.txt`: separate vLLM environment.

## Install serving env
```bash
python3 -m venv ~/venvs/arggym-vllm
source ~/venvs/arggym-vllm/bin/activate
python -m pip install -U pip uv
uv pip install -r hpc/vllm/requirements-vllm.txt --torch-backend=auto
deactivate
```
Keep the normal ArgGYM environment pinned by its existing `uv.lock`.

## Run
```bash
export HF_TOKEN=...   # only when needed
GPU_TYPE=H100 ./hpc/vllm/submit_truba.sh hf-qwen3.8-27b
GPU_TYPE=H100 ./hpc/vllm/submit_truba.sh hf-qwen2.5-3b-instruct
GPU_TYPE=H200 TP_SIZE=1 ./hpc/vllm/submit_truba.sh hf-qwen3.6-35b-a3b
```

All profiles hard-code BF16 weights. No FP8/AWQ/GPTQ/GGUF checkpoints are used. The runner refuses profiles above 40B total parameters.

## RL note
For later RLVR/R1-Zero-style work, prefer base checkpoints where available (for example Qwen2.5 3B/7B or Mistral 7B base) if the research question is whether reward induces the behavior. Keep instruction checkpoints for evaluation and for RL experiments whose goal is refinement rather than induction. Start RL at 3B-8B; 27B+ changes the systems problem substantially.


## Gemma 4 additions

- `google/gemma-4-31B-it` — dense 31B instruction model.
- `google/gemma-4-26B-A4B-it` — 26B-total MoE with about 4B active parameters.

Both profiles use BF16, Google's recommended sampling (`temperature=1.0`,
`top_p=0.95`, `top_k=64`), vLLM's `gemma4` reasoning parser, and
`reasoning_effort: medium`.

TRUBA defaults:
- Gemma 4 31B: TP2 on H100; `TP_SIZE=1` is the preferred H200 override.
- Gemma 4 26B-A4B: TP1 on H100/H200.

Examples:

```bash
GPU_TYPE=H100 ./hpc/vllm/submit_truba.sh hf-gemma4-31b-it
GPU_TYPE=H200 TP_SIZE=1 ./hpc/vllm/submit_truba.sh hf-gemma4-31b-it
GPU_TYPE=H100 ./hpc/vllm/submit_truba.sh hf-gemma4-26b-a4b-it
```


## Audited v3 corrections

This bundle supersedes v2. Important corrections:

- Serving stack pinned to vLLM 0.29.0 and Transformers 5.17.0.
- `--generation-config vllm` is forced so Hugging Face `generation_config.json`
  cannot silently alter benchmark sampling defaults.
- Ministral 3 Reasoning now uses Mistral tokenizer/config/load modes and the
  `mistral` reasoning parser.
- Qwen3.6 general-thinking sampling now includes `presence_penalty: 1.5`.
- Gemma 4 thinking is explicit; special reasoning delimiters are preserved.
- H100 defaults use TP2 for Qwen3-32B, Qwen3.6-27B, Qwen3.6-35B-A3B, and
  Gemma 4 31B. H200 defaults use TP1; `TP_SIZE` still overrides.
- Slurm log directories are created before `sbatch`.
- Each Slurm job receives a distinct localhost port to avoid collisions on
  shared GPU nodes.
- vLLM logs include the Slurm job ID.
- Clean runs refuse to reuse an existing output directory. Set `RESUME=1` only
  to resume an exact run with matching model revision provenance.
- Hugging Face revision is recorded before generation begins and checked again
  on resume.
- `/v1/models` readiness now verifies the expected served model ID.

Run the static audit before submitting:

```bash
uv run python hpc/vllm/verify_bundle.py
bash -n hpc/vllm/submit_truba.sh
bash -n hpc/vllm/truba_vllm.sbatch
uv run python -m py_compile hpc/vllm/run_vllm_arggym.py
```

Recommended vLLM environment creation:

```bash
python3 -m venv ~/venvs/arggym-vllm
source ~/venvs/arggym-vllm/bin/activate
python -m pip install -U pip uv
uv pip install "vllm==0.29.0" --torch-backend=auto
uv pip install "transformers==5.17.0" "PyYAML>=6.0,<7"
deactivate
```
