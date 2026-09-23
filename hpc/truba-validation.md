# TRUBA deployment validation record

## Scope

This validates the deployment overlay itself. It does not claim a live TRUBA H100/H200 execution, because this build environment cannot submit to TRUBA.

## Verified statically

- Serving profiles and eval configs pair by file stem (`hpc/vllm/pairing.py`).
- Every serving profile uses BF16.
- H100/H200 tensor-parallel counts are 1..4 and are translated to 16 CPU cores per requested GPU.
- Runtime enforces the Python, vLLM and Transformers versions pinned in
  `hpc/vllm/arggym-vllm.def` and `hpc/vllm/requirements-vllm.txt`.
- No TRUBA host venv or host pip installation is used.
- The GPU job contains no package installation.
- `run_vllm_arggym.py` uses `sys.executable`, not host `uv run`.
- Shell syntax (`bash -n`) passes for all `.sh` and `.sbatch` deployment files.
- Python compilation passes for the runner and bundle verifier.
- `verify_bundle.py` passes.

## Slurm orchestration mocked

Missing runtime:

1. setup job submitted to `kolyoz-cuda` with 1 GPU, 16 CPUs, H100/H200 constraint;
2. evaluation job submitted to `kolyoz-cuda`;
3. evaluation includes `--dependency=afterok:<setup_job>`.

Ready runtime:

- setup is skipped;
- evaluation is submitted directly.

GPU counts come from each profile's `tensor_parallel_size_h100` /
`tensor_parallel_size_h200`, at 16 CPUs per GPU.

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
