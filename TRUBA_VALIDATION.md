# TRUBA deployment validation record

## Scope

This validates the deployment overlay itself. It does not claim a live TRUBA H100/H200 execution, because this build environment cannot submit to TRUBA.

## Verified statically

- 14 serving profiles and 14 evaluator endpoint configs pair one-to-one.
- Every serving profile uses BF16 and is <= 40B total parameters.
- H100/H200 tensor-parallel counts are 1..4 and are translated to 16 CPU cores per requested GPU.
- Runtime base is `vllm/vllm-openai:v0.29.0-x86_64-cu129`.
- Runtime enforces Python 3.12, vLLM 0.29.0 and Transformers 5.17.0.
- No TRUBA host venv or host pip installation is used.
- The GPU job contains no package installation.
- `run_vllm_arggym.py` uses `sys.executable`, not host `uv run`.
- Shell syntax (`bash -n`) passes for all `.sh` and `.sbatch` deployment files.
- Python compilation passes for the runner and bundle verifier.
- `verify_bundle.py` reports all 14 model profiles/endpoints valid.

## Slurm orchestration mocked

Missing runtime:

1. setup job submitted to `kolyoz-cuda` with 1 GPU, 16 CPUs, H100/H200 constraint;
2. evaluation job submitted to `kolyoz-cuda`;
3. evaluation includes `--dependency=afterok:<setup_job>`.

Ready runtime:

- setup is skipped;
- evaluation is submitted directly.

Profile resource examples verified:

- `hf-qwen3.8-27b`, H100 -> 1 GPU / 16 CPUs;
- `hf-gemma4-31b-it`, H100 -> 2 GPUs / 32 CPUs;
- `hf-gemma4-31b-it`, H200 -> 1 GPU / 16 CPUs.

Fingerprint invalidation was also tested: dependency-definition changes make an existing runtime stale.

## TRUBA-specific design correction

ARF and ARF-ACC are separate Slurm systems. H100/H200 jobs are submitted through `cuda-ui` to `kolyoz-cuda`. Therefore this final design does not try to chain an ARF CPU-partition setup job to an ARF-ACC GPU job. First-time setup instead requests one `kolyoz-cuda` GPU so setup and evaluation share the same Slurm controller and `afterok` is valid.

## Live gates intentionally left to TRUBA

The first real submission will still test facts that cannot be emulated here:

- the user's authorization/account for `kolyoz-cuda`;
- Docker Hub egress from the allocated compute node;
- Apptainer `--fakeroot` availability for that account/node;
- host NVIDIA driver compatibility with the pinned vLLM CUDA image;
- actual download/access for gated Hugging Face models;
- model-specific load and memory behavior on the allocated H100/H200.

The scripts stop on each of these failures before ArgGYM evaluation proceeds. A successful first setup/evaluation is therefore the final live certification gate.
