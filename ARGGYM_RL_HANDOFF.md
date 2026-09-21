# ArgGYM RL Experiment — Current State Handoff

_Last updated: 2026-09-21, late evening (Europe/Istanbul)_

> Canonical state handoff for the ArgGYM RL/article work. New chats should read this before proposing changes. Treat it as a current snapshot, not a substitute for inspecting the actual repository. Never invent repository paths or interfaces; verify them first.

## Research aim and deadline

Submission deadline: **morning of 2026-09-26**.

Base model:
- `Qwen/Qwen3-14B`
- revision `40c069824f4251a91eefaf281ebe4c544efd3e18`
- local snapshot `/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18`

Paper evidence plan:
1. Frozen ArgGYM PRE vs POST.
2. External transfer PRE vs POST on exactly six official benchmarks: Multi-LogiEval, RuleArena, LogiQA 2.0, MMLU-Pro, BBH, Reasoning Gym.
3. ArgGYM difficulty change over L1–L15 and bands L1–5 / L6–10 / L11–15.

GSM8K and FineReason were earlier pilots only and are **not** part of the official six-benchmark transfer suite. ArgBench is dropped.

Transfer results must never influence RL training hyperparameters, curriculum, checkpoint selection, or early stopping. Do not compute a cross-benchmark grand average.

## Frozen ArgGYM benchmark

Canonical benchmark: `data/taskset.jsonl`

- 1,440 instances = 12 tasks × 15 levels × 4 orderings × 2 instances/cell
- manifest hash `121f452f2744ef0a6022c4731097b6b1`
- never train on or overwrite this file
- ignore misleading `evals/taskset.jsonl` with 3 rows

Canonical interfaces:
- `arggym.create(...)`
- `arggym.score_row(...)`

RL reward reuses the native task-specific ArgGYM verifier exactly.

## Frozen ArgGYM PRE baseline

Pinned Qwen3-14B baseline:
- BF16
- context 40960
- max completion 32768
- temperature 0.6
- top_p 0.95
- top_k 20
- min_p 0
- thinking enabled

Results:
- overall `202/1440 = 14.0%`
- L1–5 `28.7%`
- L6–10 `11.5%`
- L11–15 `1.9%`
- last-link `23.2%`
- weakest-link `4.9%`
- democratic `13.2%`
- elitist `14.9%`
- truncation `8/1440`
- no-answer `19/1440`

POST must use the same generation cap and scoring protocol.

## RL method

GRPO/DAPO-style RL with native ArgGYM verifier rewards.

Core settings:
- BF16, no quantization
- LoRA `r=32`, `alpha=16`, `dropout=0`, all-linear; `lm_head` excluded
- G=8
- temperature .6, top_p .95, top_k20, min_p0
- thinking enabled
- prompt cap 8192
- context 40960
- completion cap 32768
- `scale_rewards: none`
- epsilon low/high `.20/.28`
- beta `0`
- `num_iterations: 1`
- `mask_truncated_completions: true`
- native reward `[0,1]`, no reward shaping
- AdamW fused, betas `.9/.999`, eps `1e-8`, wd `0`, grad norm `1`
- LR `5e-6`, constant with warmup
- gradient checkpointing enabled

Effective update:
- batch1/device
- gradacc8
- 3 trainer GPUs
- G8
- effective completion batch24
- 3 unique prompt groups/update

Importance sampling:
```yaml
importance_sampling_level: token
vllm_importance_sampling_correction: true
vllm_importance_sampling_mode: token_truncate
vllm_importance_sampling_clip_min: null
vllm_importance_sampling_clip_max: 3.0
```
Do not disable IS correction.

## Memory-safe trainer/runtime

Trainer file: `rl/memory_safe_grpo.py`

Class: `MemorySafeGRPOTrainer(GRPOTrainer)`

- chunked old-logprob path
- token chunk size `256`
- synthetic comparison passed `allclose`
- expected diagnostic: `ArgGYM memory-safe old-logprob path active: token_chunk_size=256`

SIF:
`/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif`

SHA256:
`a6317095fc2cb21d225bdd5a6d1f7822a4d57ac6430e5dfecd6206aa2aa8a957`

Runtime stack:
- Python 3.12.3
- torch 2.13 + cu129
- transformers 5.15.1
- vLLM 0.28
- TRL 1.13
- PEFT 0.21

External Liger:
`/arf/scratch/futan/arggym_grpo_runtime/liger-kernel-0.8.2-site`

The original SIF remains untouched.

## Curriculum and RL_DEV

Config: `configs/rl/curriculum.yaml`
Seed: `arggym-rl-article-v1`

Stage distributions:
- Stage1 easy720 / medium504 / hard216
- Stage2 easy576 / medium432 / hard432
- Stage3 easy288 / medium432 / hard720
- each stage = 1,440 prompts

RL_DEV:
- namespace `RL_DEV`
- 720 prompts
- one per task × level × ordering
- used for checkpoint selection/evaluation only

Original Stage1:
- `data/rl/curriculum/stage1.jsonl`
- SHA256 `1e08153b108369121e2f3f1d0b052af8179fe87e8e7fd8897b6cfc13be49a235`

## Critical Stage1 ordering issue

Original Stage1 file is task-major. Because the trainer intentionally uses `shuffle_dataset:false`, runtime followed file order exactly:
- steps 1–20 attack/easy
- 21–34 attack/medium
- 35–40 attack/hard
- 41–60 attack_defense/easy
- 61+ attack_defense/medium

Thus the 40-step warmup was effectively attack-only. This is an unintended task curriculum, not the intended level curriculum.

The intended order is:
- level curriculum L1 → L15
- tasks/orderings mixed within each level
- keep `shuffle_dataset:false`; encode curriculum in file order
- no regeneration and no new examples

## Corrected Stage1 level-mixed V2

Dataset:
`data/rl/curriculum/stage1_levelmixed_v2.jsonl`

SHA256:
`ddabdb9ff221899234100a93d07a2a6eb5c809423f6df803270fc3111c0d789c`

Ordering manifest:
`data/rl/curriculum/stage1_levelmixed_v2.manifest.json`

Ordering manifest SHA256:
`310bcea949e00ffd25ff5330777a244f2320ab4dbdedc01c1673dc33fc590638`

V2 sidecar:
`data/rl/curriculum/stage1_levelmixed_v2.jsonl.manifest.json`
SHA256 `84ada1e863e6e10f06c954695e86bcd53931e540df2fdb3eea32ad7b027bc51d`

Original sidecar:
`data/rl/curriculum/stage1.jsonl.manifest.json`
SHA256 `cee6c23480130d8eb1aaa86676e1c220f4ca6b07e11a99584939d4b6bf556ef6`

Validation:
- rows 1440
- exact same canonical payload multiset/signature as original Stage1: PASS
- monotonic levels: PASS
- every task 120
- optimizer groups have unique tasks: 480/480
- max consecutive same task: 2
- band counts unchanged: easy720 / medium504 / hard216
- cross-level optimizer groups: 274, 341, 375, 423, 437, 452, 466 (1-indexed)
- seven cross-level groups are mathematically unavoidable with the inherited uneven per-level counts and groups of 3
- do not change the dataset again

Level counts inherited from probabilistic Stage1 subsampling:
`{1:144,2:144,3:144,4:144,5:144,6:100,7:101,8:101,9:101,10:101,11:43,12:43,13:44,14:43,15:43}`

Warmup implication:
- L1 has 144 rows
- 3 prompts/update
- 48 optimizer steps at L1
- warmup40 is therefore entirely mixed-task L1, as intended

V2 config:
`configs/rl/qwen3_14b_stage1_levelmixed_v2.yaml`
SHA256 `b44dedb641cf854b0444aeeaa78a3b5988b0a3ee0f54015cd700e68cee01f311`

Original config SHA256:
`29b84049d45545360a459b95d2e0a2c6acbe472da8bc14b5ed55220601f9bb7f`

V2 preflight:
`rl/preflight_stage1_levelmixed_v2.py`
SHA256 `0e43b9dd09ba05551a8f170f0db42160f57ea4b1d9a9905b74e3abda54cf2a35`

Host and SIF full preflights: PASS.
Tokenizer audit: 2160 prompts, max 4066 tokens.

## Corrected Stage2/Stage3 level-mixed V2 — VERIFIED COMPLETE

The original full curriculum build was recovered intact at:
`data/rl/curriculum_build_20260920/`

Its Stage1 and dev artifacts match the retained canonical files byte-for-byte, establishing the original lineage for Stage2/Stage3. No new examples were generated. Canonical Stage2/Stage3 were promoted additively into `data/rl/curriculum/`.

Canonical Stage2:
- `data/rl/curriculum/stage2.jsonl`
- SHA256 `9622ea1ebaa5d74f6493867cd22c0724f922ae2e7f1808bf2da53faad05e9c79`
- rows 1440
- namespace `RL_TRAIN_STAGE_2`
- band counts easy576 / medium432 / hard432
- level counts `{1:115,2:116,3:115,4:115,5:115,6:86,7:86,8:87,9:87,10:86,11:86,12:86,13:86,14:87,15:87}`

Stage2 level-mixed V1:
- `data/rl/curriculum/stage2_levelmixed_v1.jsonl`
- SHA256 `8a105877311c48f2aeb79fbb2a437b14f54772a6dccc66c8be724f8ffb9ecde8`
- V1 ordering manifest SHA256 `b4a0cbc671422f89faba8c2232c1e54695e4259b8b957a42c1c225ca66149119`

Stage2 level-mixed V2:
- `data/rl/curriculum/stage2_levelmixed_v2.jsonl`
- SHA256 `d3af47d5293d2c6f66ca7075b7bff587103316cce6c2bdf12a1a6c45fe634385`
- ordering manifest `data/rl/curriculum/stage2_levelmixed_v2.manifest.json`
- ordering manifest SHA256 `77a953744e99bbc41492e25afa1e1e8297bf18365b8cf0061fd466fb93f95f29`
- V2 sidecar SHA256 `02071910f3e7e37f9eb9a9d0cdef5ca96dd08ff18436f012425e7e8c54cb36c3`
- initial bad optimizer groups `[154]` (1-indexed)
- repairs performed 1
- optimizer groups with 3 unique tasks 480/480
- cross-level optimizer groups `[39,116,154,221,250,279,308,365,394]`
- max consecutive same task 2

Canonical Stage3:
- `data/rl/curriculum/stage3.jsonl`
- SHA256 `546b2374bc2922a42bf07bc5ff9f50aa9d70274d80e3b48b5ad21d4ee4449de9`
- rows 1440
- namespace `RL_TRAIN_STAGE_3`
- band counts easy288 / medium432 / hard720
- level counts `{1:57,2:57,3:58,4:58,5:58,6:86,7:86,8:86,9:87,10:87,11:144,12:144,13:144,14:144,15:144}`

Stage3 level-mixed V1:
- `data/rl/curriculum/stage3_levelmixed_v1.jsonl`
- SHA256 `68aa9a6f31041bb07b061e9be10aa90a5e2340379ca54dacbe02e779a4997402`
- V1 ordering manifest SHA256 `49358fdb9a4731275a9f03107ee81c8e662c880de743f0775c07efddc46731d8`

Stage3 level-mixed V2:
- `data/rl/curriculum/stage3_levelmixed_v2.jsonl`
- SHA256 `304d0a16b819c07ef6f1f047bb188db2dc218d1f1748ea7922c72ba5b408cf0f`
- ordering manifest `data/rl/curriculum/stage3_levelmixed_v2.manifest.json`
- ordering manifest SHA256 `78a98c00b29fb42df9c498482e3b21b73cad7b696cbf32a681b9c2e0de19490e`
- V2 sidecar SHA256 `33335f3f365b8457a8a0664ddab36cfda105780dd91e208791fd31e45b48b0d7`
- initial bad optimizer groups `[182]` (1-indexed)
- repairs performed 1
- optimizer groups with 3 unique tasks 480/480
- cross-level optimizer groups `[58,77,125,154]`
- max consecutive same task 1

Final Stage2/Stage3 audit:
- canonical → V1 → V2 payload multiset identity: PASS
- structural signature identity: PASS
- V1/V2 level sequence identical and monotonic L1→L15: PASS
- task counts 120 each, ordering counts 360 each, task×ordering 30 each: PASS
- all 480 global 3-row optimizer groups contain 3 distinct tasks: PASS
- V2 sidecar differs from canonical sidecar only in `data_sha256`; `cells` identical: PASS
- frozen-taskset provenance hash remains `121f452f2744ef0a6022c4731097b6b1`: PASS
- no new instances generated; frozen ArgGYM untouched

Builder used:
`rl/build_stage23_levelmixed_v2.py`

Scientific status:
- Stage2/Stage3 data preparation is complete.
- Preserve canonical, V1, V2, and manifests as provenance artifacts.
- Do not regenerate or reorder them again unless new evidence shows a defect.

## Smaller-model Qwen3 scale-out — CONFIGURED, DOWNLOADED, QUEUED

The article now includes a controlled smaller-model scale-out using **non-Base / post-trained Qwen3 checkpoints only**. The selected models are:

- `Qwen/Qwen3-8B` — revision `b968826d9c46dd6066d109eabc6255188de91218`
- `Qwen/Qwen3-4B` — revision `1cfa9a7208912126459214e8b04321603b3df60c`
- `Qwen/Qwen3-1.7B` — revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

These are the instruction-following / thinking-capable Qwen3 checkpoints, **not** the `*-Base` variants. Pinned checkpoint metadata was queried at the exact revisions and verified:
- architecture `Qwen3ForCausalLM`
- `torch_dtype=bfloat16`
- `max_position_embeddings=40960`
- generation config temperature `.6`, top-p `.95`, top-k `20`
- `min_p` is absent/`None` in the upstream generation config; the ArgGYM article protocol explicitly uses `min_p: 0.0` to preserve the already-frozen 14B protocol

Local snapshots:
- `/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`
- `/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`
- `/arf/scratch/futan/arggym_hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

All three snapshots were downloaded successfully and contain `config.json`.

### Smaller-model frozen-ArgGYM evaluation profiles

Additive evaluation profiles were created; the older conservative `hf-qwen3-8b.yaml` was deliberately left untouched.

Eval endpoint YAMLs:
- `evals/conf/model/hf-qwen3-8b-arggym-40k.yaml`
- `evals/conf/model/hf-qwen3-4b-arggym-40k.yaml`
- `evals/conf/model/hf-qwen3-1.7b-arggym-40k.yaml`

vLLM serving YAMLs:
- `hpc/vllm/models/hf-qwen3-8b-arggym-40k.yaml`
- `hpc/vllm/models/hf-qwen3-4b-arggym-40k.yaml`
- `hpc/vllm/models/hf-qwen3-1.7b-arggym-40k.yaml`

Frozen evaluation protocol for all three:
- benchmark `data/taskset.jsonl`, 1,440 rows
- BF16
- context `40960`
- max output `32768`
- temperature `.6`
- top-p `.95`
- top-k `20`
- min-p `0.0`
- Qwen3 thinking enabled
- Qwen3 reasoning parser
- tensor parallel size 1 on H200

`hpc/vllm/verify_bundle.py` passed after the additions: **27 model profile/endpoint pairs validated** and the TRUBA deployment closure passed statically. A separate semantic audit verified all three new profiles, including exact revision, model ID, context, sampling settings, reasoning parser, and H200 TP=1. `git diff --check` returned 0 for the touched/new files.

`hpc/vllm/run_vllm_arggym.py` was generalized additively so a serving profile may contain `revision:`. It now prefers that pinned revision and otherwise falls back to resolving the Hub revision. Existing profiles without `revision` preserve their previous behavior. vLLM is invoked with explicit `--revision`.

Security repair:
- the hard-coded Hugging Face token fallback was removed from `hpc/vllm/submit_truba.sh`
- syntax check passed
- a scan confirmed no hard-coded token variable or literal `hf_...` token remains in that script
- because the previous token appeared in terminal/chat output, it should be considered exposed; revocation/rotation should be confirmed separately if not already done

### Smaller-model RL configs

New Stage1 V2 configs:
- `configs/rl/qwen3_8b_stage1_levelmixed_v2.yaml`
- `configs/rl/qwen3_4b_stage1_levelmixed_v2.yaml`
- `configs/rl/qwen3_1p7b_stage1_levelmixed_v2.yaml`

A protocol-identity audit passed for all three: relative to `configs/rl/qwen3_14b_stage1_levelmixed_v2.yaml`, `data`, `peft`, and `generation` are identical, and all training fields are identical except model-specific `output_dir` / `run_name`. The frozen taskset hash is identical. Thus model size is the manipulated variable while the article RL protocol remains fixed.

`rl/preflight.py` was generalized from a single hard-coded 14B revision to an explicit reviewed model→revision mapping containing exactly the 14B, 8B, 4B, and 1.7B Qwen3 checkpoints above. Unknown models still fail closed. The 14B revision pin remains unchanged. BF16/FP16 validation remains mandatory.

The production GRPO topology was re-verified from `hpc/grpo/run_node.sh`:
- GPU0: vLLM rollout server
- GPUs1–3: 3 data-parallel LoRA/GRPO trainer processes
- preflight invoked with `--world-size 3`
- Accelerate launches 3 trainer processes

This preserves the audited Stage1/2/3 V2 optimizer-group assumption: 3 trainer GPUs × batch1 × gradacc8 / G8 = **3 unique prompts per optimizer update**.

### Current smaller-model Slurm state

Latest user-provided `squeue` snapshot:

Frozen ArgGYM PRE evaluations:
- jobs `1572303`, `1572304`, `1572305` — all `PD (Priority)`

Stage1 V2 RL trainings:
- job `1572306` — `PD (Resources)`
- jobs `1572307`, `1572308` — `PD (Priority)`

The jobs were submitted for the intended model order 8B → 4B → 1.7B, but the exact job→model mapping has **not yet been independently verified with `scontrol` in the handoff**. Verify before quoting a specific job as a specific model.

Existing 14B jobs remain live in the same queue snapshot:
- `1570871` RUNNING, elapsed `20:35:32`, node `kolyoz51`
- `1571672` RUNNING, elapsed `08:19:55`, node `kolyoz49`

Do not cancel either 14B job without explicit user instruction.

### What we are waiting for now

1. The three smaller-model PRE evaluation jobs must start and finish. Confirm terminal state with `sacct`, not only disappearance from `squeue`.
2. For each smaller model, verify that the frozen ArgGYM run committed/scored all **1,440** benchmark rows and preserve the raw generations plus score artifacts. Do not infer completion from runner progress lines alone.
3. Preserve the PRE aggregate plus task/level/ordering breakdowns before interpreting RL results.
4. The three smaller-model Stage1 V2 RL jobs may run concurrently with PRE because both start from immutable pinned base snapshots and write to separate outputs; this does not contaminate PRE.
5. When each RL job starts, verify from its logs: correct model snapshot/revision, preflight PASS, correct Stage1 V2 dataset, frozen taskset hash, GPU0 vLLM + GPUs1–3 trainer topology, 3 trainer ranks, G8, 40960 context, 32768 completion cap, BF16, Liger, IS correction, and no startup/OOM/NCCL error.
6. The first meaningful smaller-model RL result is RL_DEV checkpoint evaluation, not online training reward.

## Stage1 training jobs

Current `squeue` only confirms that both 14B jobs are still running; the detailed optimizer-step numbers below come from earlier health audits and must not be extrapolated from elapsed wall time. Latest queue snapshot: `1570871` RUNNING at `20:35:32` on `kolyoz51`; `1571672` RUNNING at `08:19:55` on `kolyoz49`.

### Old task-major production / diagnostic job

Job: `1570871`

Latest verified state from the RL health audit:
- **RUNNING**
- elapsed `18:48:13`
- node `kolyoz51`
- latest emitted `epoch=0.25` → exactly **step 120/480**
- no fatal numerical/runtime errors in the checked logs

Resources:
- account romer
- partition kolyoz-cuda
- 4×H200
- 64 CPU
- 875G
- 3-day limit

Submission command path:
`/arf/scratch/futan/ArgGYM/hpc/grpo/truba_grpo.sbatch`

Logs:
- `outputs/slurm/arggym-grpo-1570871.out`
- `outputs/slurm/arggym-grpo-1570871.err`

Scientific status:
- this is the **unintended task-major** Stage1 run
- keep it as a diagnostic/ablation artifact, not the primary paper run
- it has now reached the first configured save/eval boundary at step120
- at the time of the audit, no second eval record or `checkpoint-120` directory had yet appeared; do not claim the post-step120 RL_DEV result until it is actually emitted
- do **not** infer current optimizer progress from wall-clock time; inspect the latest log/epoch

Late-step behavior at the step120 boundary is scientifically informative. In the last 30 logged optimizer steps, all on `claim_chain`, difficulty progressed from easy→medium→hard and training efficiency deteriorated sharply. Approximate 10-step windows:
- first 10: mean reward ≈ `.171`, success ≈ `.158`, mean length ≈ `3067`, truncation ≈ `0.4%`, mean step time ≈ `149s`, mean grad norm ≈ `.00351`
- middle 10: mean reward ≈ `.166`, success ≈ `.138`, mean length ≈ `4290`, truncation ≈ `1.3%`, mean step time ≈ `302s`, mean grad norm ≈ `.00271`
- last 10: mean reward ≈ `.0375`, success ≈ `.0208`, mean length ≈ `6871`, truncation ≈ `5.0%`, mean step time ≈ `550s`, mean grad norm ≈ `.00098`

Observed late old-run pathologies are **curriculum inefficiency, not numerical instability**:
- hard claim-chain batches reach mean lengths around `10–11k` tokens
- per-step completion clipping reaches `8–17%` on some batches
- multiple groups have `reward_std=0`, `frac_reward_zero_std=1`, `loss=0`, `grad_norm=0`; this is expected when all 8 completions in a prompt group receive identical reward, so GRPO has no within-group advantage signal
- importance-sampling means stay near 1, but tails become broader; isolated `sampling_logp_difference/max` reached `9.702`, IS minima reached `6.116e-05`, and maxima sometimes hit the configured cap `3`
- policy `clip_ratio` remains very small, so these tails have not produced a numerical blow-up

Do not cancel this job unless the user explicitly decides to do so.

### Corrected level-mixed V2 primary RL job

Job: `1571672`
JobName: `arggym-grpo-lvlmix-v2`

Latest verified state from the RL health audit:
- **RUNNING**
- elapsed `06:32:36`
- node `kolyoz49`
- latest emitted `epoch=0.02708` → approximately **step 13/480**
- still inside the 40-step LR warmup
- no fatal numerical/runtime errors in the checked logs

This is the **primary production RL run for the paper**. The user explicitly chose to let it coexist with `1570871`; leave both alone unless explicitly instructed otherwise.

Resources:
- partition kolyoz-cuda
- account romer
- feature H200
- 4×H200
- 64 CPU
- 875G
- time limit `2-23:30`

Logs:
- `outputs/slurm/arggym-grpo-lvlmix-v2-1571672.out`
- analogous `.err`

Startup already verified:
- exact SIF SHA `a6317095...`
- tokenizer audit 2160 prompts, max4066
- V2 preflight PASS
- frozen ArgGYM taskset hash `121f452f2744ef0a6022c4731097b6b1`, rows1440
- training pool explicitly `data/rl/curriculum/stage1_levelmixed_v2.jsonl`, rows1440, SHA starts `ddabdb9f...`
- RL_DEV720
- exact pinned Qwen3-14B revision
- BF16 true / FP16 false
- context prompt8192 / completion32768 / model40960
- effective completion batch24, G8, 3 unique prompt groups/update
- max_steps480
- vLLM ready
- IS correction token-truncate, clip max3
- Liger enabled with `liger_loss_compiled=False`
- old-logprob token chunk256

Initial V2 RL_DEV evaluation was emitted at **epoch 0 before training**:
- eval reward `0.2889`
- reward std `0.2807`
- success `0.1222`
- easy `0.3558`
- medium `0.2746`
- hard `0.2318`
- answer-region rate `0.9778`
- mean completion length `6656`
- clipped ratio `0.006944`
- IS ratio min/mean/max `0.5085 / 0.9846 / 1.624`
- eval clip ratio `0`
- eval runtime `2.141e4 s` ≈ **5h57m**

This `0.2889` is the correct **within-run V2 pre-training reference**. It is not a post-RL result and is operationally consistent with the earlier baseline `0.2913`; do not call the difference a regression.

V2 task-level epoch-0 rewards:
- attack `.2644`
- defeat_diagnosis `.3085`
- perturbation `.3002`
- claim_chain `.2340`
- status_query `.4070`
- attack_defense `.2286`
- counter_argument_strict `.2555`
- counter_argument `.2560`
- formalization `.2633`
- semantics_query `.4126`
- preference_construction `.3459`
- defence `.2589`

First 13 V2 optimizer steps are technically clean:
- mean online training reward ≈ `.562`
- mean online success ≈ `.455`
- mean completion length ≈ `4157`
- observed completion truncation `0%` in these 13 steps
- mean grad norm ≈ `.00231`
- mean IS ratio ≈ `.9905`
- mean policy clip ratio ≈ `.00126`
- no NaN/Inf/OOM/NCCL/fatal exception
- LR progression matches the intended 40-step warmup (`0` at step1, then increments of `1.25e-7`; ~`1.5e-6` by step13 toward target `5e-6`)

Do **not** compare V2 online L1 training reward (~`.56` early) directly with the full RL_DEV reward `.2889`; these are different difficulty distributions.

### V2 L1 optimizer-group audit — verified balanced blocked curriculum

A direct audit of the first 48 optimizer groups (all L1) confirms:
- 144 L1 rows = 48 optimizer updates × 3 unique prompts/update
- all **12 task families** appear within every consecutive 4-step cycle
- each task appears exactly 12 times over L1
- task co-occurrence is deterministic in four recurring triplets:
  1. `defence / counter_argument_strict / claim_chain`
  2. `preference_construction / semantics_query / attack_defense`
  3. `attack / status_query / perturbation`
  4. `defeat_diagnosis / formalization / counter_argument`
- within each 16-step block, every task receives each of the four aggregation orderings exactly once
- the 16-step task×ordering structure repeats three times, so every task×ordering cell occurs exactly 3 times over L1
- the first 40 warmup steps contain ten complete all-task 4-step cycles, so LR warmup is not task-biased

Interpret this as a **deterministic blocked, balanced level curriculum**, not a random shuffle. The periodic task triplets are real, but current evidence does not justify restarting or changing V2.

Important logging interpretation:
- the per-step emitted `arggym_reward_task_*` field can name only one task even though the optimizer step contains three different prompt groups/tasks
- therefore do **not** infer actual optimizer-group composition from the single task name in a training log line; audit the frozen dataset grouping when task mixture matters

### RL evaluation interpretation / checkpoint selection

Earlier independent RL_DEV baseline:
- eval reward `0.2913`
- reward std `0.2737`
- completion length mean `6718`
- min `4304`, max `9257`
- clipped `0.00833`
- answer-region fraction `0.9736`
- easy `0.3545`
- medium `0.2722`
- hard `0.2422`
- success `0.1236`

Use the V2 epoch-0 `0.2889` as the primary within-run comparison for future V2 checkpoints. The first meaningful evidence of RL improvement will be the **V2 step120 RL_DEV evaluation**, not online training reward.

At step120, V2 will have completed L1 and L2 and entered L3. A plausible healthy early signature would be easy improvement with overall improvement while medium/hard remain stable or improve modestly. Do not require hard-band gains this early, but watch for medium/hard collapse as a forgetting/specialization signal.

The old run's step120 RL_DEV evaluation is also scientifically useful as a task-major diagnostic. Once it emits a second eval record, compare it against its own epoch-0/pre-training evaluation rather than against online training reward.

## Official external transfer suite — V4

Exactly six official benchmarks:
1. Multi-LogiEval
2. RuleArena
3. LogiQA 2.0
4. MMLU-Pro
5. BBH
6. Reasoning Gym

Shared protocol:
- pinned Qwen3-14B revision above
- BF16
- context 40960
- thinking enabled
- Qwen3 reasoning parser where vLLM is used
- temperature .6
- top_p .95
- top_k20
- min_p0
- seed1234
- max generation 16384
- require `prompt + 16384 <= 40960`
- PRE and POST identical except model checkpoint
- individual Slurm jobs, normally 1×H200, 16 CPU, account romer, partition kolyoz-cuda, `-C H200`
- no shared writable generation directories
- preserve raw generations and provenance

V4 scientific design decision:
- Multi-LogiEval, RuleArena, LogiQA 2.0, MMLU-Pro, and BBH have frozen PRE protocols/results
- Reasoning Gym is the exception: the original 10-task/1000-row PRE run exposed runtime/scorer-validity problems and is being redesigned before the final official PRE/POST freeze
- do **not** report the failed/incomplete original Reasoning Gym run as the official PRE score

Transfer never affects RL/checkpoint/curriculum. RL_DEV is for checkpoint selection. No transfer grand average.

## Multi-LogiEval official PRE

Source:
`/arf/scratch/futan/arggym_transfer_sources/multilogieval`
HEAD starts `6d55ade...`

Frozen rows: 1556.
The source repository count differs from a paper-reported 1552; document the discrepancy rather than silently dropping rows.

Original parser had 12 formatting variants and was invalid for reporting.
Formatting-only parser repair v1 applied to the same saved generations:
- `1139/1556 = 73.2005%`
- unparsed `0`
- one parser-conflict case remained wrong

Manifest SHA starts `d88d69...`.
POST must use the exact same parser v1.
Status: CLOSED.

## LogiQA 2.0 official PRE

Rows: 1572.

Original parser produced 1340 `None` values and was invalid.
Formatting-only parser v1 applied to the same raw generations:
- `1226/1572 = 77.9898%`
- unparsed `0`
- conflicts `0`

Manifest SHA starts `7ba324...`.
POST must use the exact same parser v1.
Status: CLOSED.

## MMLU-Pro official PRE

Official PRE:
- `8093/12032 = 67.26%`

Use the same frozen V4 harness for POST.

## BBH official PRE

Official PRE:
- `5978/6511 = 91.81%`

Use the same frozen V4 harness for POST.

## Reasoning Gym — redesign and current calibration

### Original 10-task V4 run: historical/incomplete, not the official headline PRE

Original frozen artifact:
- 10 task families × 100 = 1000 rows
- data `data/transfer/v4/frozen/reasoning_gym/reasoning_gym_v4_1000.jsonl`
- data SHA256 `c005eca94a9bd9c8ba1c2a3660349e300daddd101cf427bb61f573f5ccd31ca6`
- pinned Reasoning Gym revision `49b07130b3fcd12f2d064bba7c43869543a0e7e7`
- native-gold self-score `1000/1000`: PASS

Original PRE job `1571321` **TIMED OUT** after `15:00:13` with only `715/1000` generations committed. It completed through all 100-row blocks for circuit_logic, cryptarithm, family_relationships, knights_knaves, self_reference, shortest_path, zebra_puzzles plus 15 sudoku; n_queens and polynomial_equations were not reached. There is no valid official aggregate score from this run.

A read-only audit of the 715 rows showed three distinct failure classes:
1. genuine reasoning failures,
2. generation-budget failures,
3. correct final answers rejected by strict whole-completion scorers.

This motivated a protocol redesign rather than extending the original 1000-row run. Preserve the original artifacts as historical provenance; do not overwrite them.

### 8-task qualitative redesign pilot

The first redesign considered:
- `arc_1d`
- `family_relationships`
- `course_schedule`
- `knights_knaves`
- `propositional_logic`
- `self_reference`
- `syllogism`
- `zebra_puzzles`

Known upstream validity issues discovered during this stage:
- `family_relationships`: pinned generator has directionally incorrect in-law labels and directionally suspect niece/nephew branches; these categories were excluded
- `syllogism`: pinned validity checker can accept invalid arguments because it omits a required middle-term-distribution condition; a separate finite-model validator was used for the pilot
- `propositional_logic`: pinned generator can occasionally emit an `example_answer` that scores only `0.05` under its own verifier; such candidates must be skipped deterministically

Final 24-row qualitative artifact:
- `data/transfer/pilots/rg8/rg8_qual24_v4.jsonl`
- rows24
- SHA256 `7fbfb2aeb690c23e9f5365f748886439ccbd2728f2b56fde63580b6f08077911`
- job `1571800` COMPLETED, ExitCode `0:0`, elapsed `00:24:53`, node `kolyoz43`

Qualitative conclusion:
- keep as core candidates: `propositional_logic`, `knights_knaves`, `course_schedule`, `zebra_puzzles`
- `family_relationships`: control-only / ceiling
- `syllogism`: control-only / too easy and upstream checker problematic
- `self_reference`: expensive, 3/3 correct in pilot, weak information-per-GPU-hour
- `arc_1d`: medium/hard settings degenerated into length-cap failures

### Parser evolution for selected four tasks

Scoring policy for the selected four tasks:
1. preserve raw model completion,
2. deterministically extract/normalize only the submitted final answer,
3. pass that extracted answer to the **unchanged native Reasoning Gym scorer**,
4. store both raw-native and adapted-native scores,
5. never use gold-aware extraction.

Parser-v2 repaired ordinary Markdown/LaTeX formatting, but had an operator-order bug: `strip_markdown_math()` removed `\\left` before logical-operator normalization, corrupting `\\leftrightarrow` into `rightarrow`.

Parser-v3 fixes the order: normalize logical TeX operators first, then strip `\\left`/`\\right`. Verified corrections on saved generations:
- `U \\leftrightarrow \\neg Q` → `U ↔ ¬ Q`, native score `0.05 → 1.0`
- `S \\land (Q \\leftrightarrow P)` → `S ∧ (Q ↔ P)`, native score `0.05 → 1.0`

Parser-v3 file:
`transfer/pilots/rg4/rg4_pilot_v3.py`

Parser-v4 was then created for `knights_knaves` after finding another real extraction bug. In `rg4_cal120_v1::knights_knaves::easy::006`, the model's explicit final answer was fully correct, including `Owen is an **altruist**`, but parser-v3 failed to match the Markdown-bold role and fell back to earlier reasoning where `Owen is an egoist` appeared inside a quoted proposition. This produced a false partial score `0.86`.

Parser-v4 changes for knights/knaves:
- normalize/strip Markdown wrappers before role matching,
- if an explicit final-answer region exists, **never fall back to the reasoning body** when that region fails to parse,
- incomplete explicit finals return empty rather than being “rescued” from intermediate hypothetical assignments.

Parser-v4 file:
`transfer/pilots/rg4/rg4_pilot_v4.py`

Retrospective v4 rescore over all 30 knights/knaves rows changed **exactly one row**:
- `easy::006`: `0.86 → 1.0`
- all other 29 rows unchanged

The remaining partial-credit knights rows are genuine model errors, not parser artifacts:
- `medium::003 = 0.533333...`
- `medium::007 = 0.65`
- `hard::008 = 0.90`
- `hard::003 = 0.0`, truncated

The native knights scorer's partial-credit behavior observed here is consistent with `0.3 + 0.7 × correct_assignments/n`.

Do not restart GPU generation because of parser defects. Saved completions are sufficient for retrospective rescoring. The live `1571906` process loaded its earlier parser at launch; generation outputs remain valid, while final reporting must use the latest validated retrospective parser.

Pinned propositional-logic scorer semantics verified during calibration:
- `1.0`: entailed, non-trivial conclusion
- `0.25`: entailed but scorer-classified trivial/atomic conclusion
- `0.05`: parseable but non-entailed conclusion
- `0.0`: malformed/unscorable or otherwise zero under the native scorer

Therefore do not label every `0.25` row as a reasoning failure.

### Four-task 120-row calibration pilot: final state

Dataset/namespace: `rg4_cal120_v1`

Frozen data:
- `data/transfer/pilots/rg4/rg4_cal120_v1.jsonl`
- 120 rows = 4 tasks × 3 bands × 10 examples
- SHA256 `df68bb45ec08273550eb3a8b3e08a6f4798dd97f62604fc56fb0afa09675499c`
- structural preflight PASS

Calibration ranges:
- propositional_logic: 4 vars/4 premises/complexity2; 6/6/3; 8/8/4
- knights_knaves: 5 people depth2 width3; 6 people depth3 width3; 7 people depth3 width4
- course_schedule: original 15 / 25 / 35 courses
- zebra_puzzles: 4×4 / 5×5 / 6×6

Selection rules were deterministic and did **not** use model outcomes.

Job `1571906` (`rg4-cal120-v1`) is now formally verified:
- `COMPLETED`
- `ExitCode 0:0`
- elapsed `02:34:52`
- node `kolyoz27`
- **120/120 generations committed**

Runner log:
`logs/transfer/pilots/rg4_cal120_v1_1571906.runner.log`

Generations:
`outputs/transfer/pilots/rg4_cal120_v1/generations.jsonl`

The runner's final aggregate JSON reports parser-v2 because that process loaded the earlier parser at launch. Final scientific interpretation for propositional logic and knights must use the validated retrospective parser-v3/v4 rescoring described below; saved generations remain valid.

#### Propositional logic: completed 30/30 under parser-v3/v4 semantics

No truncations.

By difficulty:
- easy: 4×`1.0`, 6×`0.25`, 0 invalid → **10/10 entailed**, native mean `0.55`
- medium: 6×`1.0`, 4×`0.25`, 0 invalid → **10/10 entailed**, native mean `0.70`
- hard: 6×`1.0`, 2×`0.25`, 2×`0.05` → **8/10 entailed**, native mean `0.66`

The two hard `0.05` answers were genuine non-entailment errors, not parser failures. The 4/6/8 ladder is viable: no generation-budget failure and genuine reasoning headroom at hard.

#### Knights/knaves: completed 30/30 under parser-v4

Corrected results:
- easy: `10/10` full credit, no truncation, mean `1.000`
- medium: `8/10` full credit + two genuine partials (`0.5333`, `0.65`), no truncation, mean `0.9183`
- hard: `8/10` full credit + one genuine partial (`0.90`) + one truncation (`0.0`), mean `0.8900`
- overall native mean `0.9361`

This task is a strong final-suite candidate. Parser-v4 changed only the known Markdown/fallback contamination row.

#### Course schedule: size-based ladder rejected near the token wall

The course parser passed the explicit-final-region audit; the observed failures are not parser artifacts.

Original **15-course** cell:
- gold False: `3/5` correct, `2/5` truncated, mean completion `13,933`
- gold True: `0/5` correct, `4/5` truncated, mean completion `16,281`
- overall: `3/10` correct, `6/10` truncated

Therefore original 15/25/35 is rejected as a size-driven generation-budget failure.

`course_cal30_v1` (5/8/12) frozen dataset:
- `data/transfer/pilots/course_schedule/course_cal30_v1.jsonl`
- SHA256 `c83f8c90683aa1cbc244cd5ab9caa3bfc9a0f196704a49f3487306571ef9f0b8`
- 30/30 generations committed
- all 30 correct, zero truncations

5 courses:
- True `5/5`, mean tokens `3576.6`, max `5015`
- False `5/5`, mean `5428.4`, max `8242`

8 courses:
- True `5/5`, mean `5898.8`, max `8427`
- False `5/5`, mean `6846.0`, max `8750`

12 courses:
- True `5/5`, mean `12714.8`, max `14642`
- False `5/5`, mean `10172.8`, max `11585`

This proves 5/8/12 is safe but ceilinged; 12 is already near the token-budget boundary.

`course_boundary20_v1` (13/14) frozen dataset:
- `data/transfer/pilots/course_schedule/course_boundary20_v1.jsonl`
- SHA256 `ae10673ea7c48b2b748749a0b163cb067d8c2348749994d1a81b5caccf3c3770`
- job `1572181`
- formally `COMPLETED`, ExitCode `0:0`, elapsed `00:42:20`, node `kolyoz31`
- **20/20 generations committed**

Results:
- n13 False: `4/5` correct, `1/5` truncated, mean tokens `12437.2`
- n13 True: `2/5` correct, `3/5` truncated, mean `13558.0`
- n14 False: `5/5` correct, `0/5` truncated, mean `11178.2`
- n14 True: `3/5` correct, `2/5` truncated, mean `14590.6`

Interpretation:
- 13 courses overall `6/10` with `4/10` truncations
- 14 courses overall `8/10` with `2/10` truncations
- non-monotonic aggregate accuracy shows instance structure dominates at this small boundary sample
- both 13 and 14 are still materially contaminated by generation-budget failure, especially on solvable/True instances
- therefore **do not choose 13 or 14 as final hard cells simply from aggregate accuracy**
- treat `num_courses` as having reached a sharp token-budget transition between 12 and 15, not a clean difficulty axis

The pinned `CourseScheduleConfig` exposes independent supported axes:
- `num_courses`
- prerequisites per course
- cycle length

The next controlled pilot therefore keeps `num_courses=12` and cycle length `3–5` fixed and changes **prerequisite density only**.

New frozen calibration dataset `course_density20_v1`:
- `data/transfer/pilots/course_schedule/course_density20_v1.jsonl`
- SHA256 `9248f45f391657ba6a0ce0cca4568fc0d70c9554c02f347114952bc1eb890be6`
- rows20
- `p34`: 12 courses, 3–4 prerequisites/course, cycle3–5, 10 rows = 5 True + 5 False
- `p45`: 12 courses, 4–5 prerequisites/course, cycle3–5, 10 rows = 5 True + 5 False
- selection deterministic; no model outcomes used
- job `1572208` (`course-density20-v1`) was last verified **RUNNING** on `kolyoz31`; preserve exact state as stale until rechecked

Files:
- builder `transfer/pilots/course_schedule/build_course_density20_v1.py`
- runner `transfer/pilots/course_schedule/run_course_density20_v1.py`
- Slurm `hpc/transfer/pilots/course_schedule/pre_rl_course_density20_v1.sbatch`

#### Zebra puzzles: parser validated; 6×6 rejected as a pure token-budget cliff

Retrospective parser-v4 rescore over all 30 saved zebra rows:
- **changed v2→v4: 0/30**
- easy 4×4: `10/10` full credit, `0/10` truncated, mean `1.0`
- medium 5×5: `10/10` full credit, `0/10` truncated, mean `1.0`
- hard 6×6: `4/10` full credit, `6/10` truncated, mean `0.4`

All six non-full-credit 6×6 rows are `finish_reason=length` with empty extracted final answer. There are **no observed non-truncated hard zebra reasoning errors**. Therefore 6×6 is rejected as a generation-budget condition rather than a useful hard reasoning cell.

Pinned `ZebraConfig` supports independent axes:
- `num_people` 2–7
- `num_characteristics` 2–7

The next controlled zebra pilot keeps safe `num_people=5` and raises **characteristics only**.

New frozen calibration dataset `zebra_chars20_v1`:
- `data/transfer/pilots/zebra_puzzles/zebra_chars20_v1.jsonl`
- SHA256 `d1215d163587181b496cd1b5b546c48a8978950c42c80bee616a551b45c54148`
- rows20
- `c6`: 5 people × 6 characteristics, 10 rows
- `c7`: 5 people × 7 characteristics, 10 rows
- selection deterministic; no model outcomes used
- job `1572209` (`zebra-chars20-v1`) was last verified **RUNNING** on `kolyoz34`; preserve exact state as stale until rechecked

Files:
- builder `transfer/pilots/zebra_puzzles/build_zebra_chars20_v1.py`
- runner `transfer/pilots/zebra_puzzles/run_zebra_chars20_v1.py`
- Slurm `hpc/transfer/pilots/zebra_puzzles/pre_rl_zebra_chars20_v1.sbatch`

For both new calibration jobs, committed `generations.jsonl` row count is authoritative progress; a runner `[n/20]` line means that row has started, not necessarily committed.

## RuleArena official PRE — final decision

Job: `1571319`
JobName: `arggym-v4-pre-rule`
Status: **COMPLETED**, ExitCode `0:0`
Start `2026-09-20 23:03:34`
End `2026-09-21 09:21:37`
Node `kolyoz55`

Outputs:
- `outputs/transfer/pre_rl_qwen3_14b_v4/rulearena/generations.jsonl`
- `outputs/transfer/pre_rl_qwen3_14b_v4/rulearena/samples.jsonl`
- `outputs/transfer/pre_rl_qwen3_14b_v4/rulearena/metrics.json`
- `outputs/transfer/pre_rl_qwen3_14b_v4/rulearena/prompts.jsonl`
- `outputs/transfer/pre_rl_qwen3_14b_v4/rulearena/run.json`

Freeze manifest SHA256:
`975c1d31e43cff2ed3354695aaf13844129e5231dca027e44d3d3cf09b9d714b`

Official RuleArena PRE score:
- **199/816 = 24.3872549%**
- 617 incorrect
- 0 missing
- 0 infrastructure errors
- 0 scorer exceptions
- 2 truncated

Domain breakdown:
- airline `44/300 = 14.6667%`
- NBA `73/216 = 33.7963%`
- tax `82/300 = 27.3333%`

Complexity breakdown previously verified:
- airline c0 `26/100=.26`; c1 `10/100=.10`; c2 `8/100=.08`
- NBA c0 `31/81=.382716`; c1 `28/89=.314607`; c2 `14/46=.304348`
- tax c0 `68/100=.68`; c1 `13/100=.13`; c2 `1/100=.01`

The two truncated rows are:
- `rulearena:tax:0:36`
- `rulearena:tax:1:21`
Both score 0.

### RuleArena scorer audit

A detailed forensic audit was performed because the upstream scorer is permissive/brittle in places.

Verified upstream behavior:
- NBA: score is exact target-string containment in the completion (`target_response in response`)
- Airline: upstream looks for exact literal `The total cost is`, parses from there, and uses equality/substring fallback
- Tax: upstream regexes the first `The total tax owed/overpaid is $...` and compares numerically with `np.isclose`

Audit findings:
- NBA upstream score 73/216 can overcredit responses containing multiple categorical answers if the gold string appears anywhere
- a “last explicit answer” diagnostic gave 67/216
- a strict “single distinct explicit answer” diagnostic gave 55/216
- these are **diagnostic only**, not final reported metrics
- Airline narrow final-answer audit remained exactly `44/300`; all 300 had a match; two rows had multiple explicit final-cost statements but both remained wrong
- Tax last-explicit diagnostic gave `79/300`; three upstream-correct rows first stated the precise correct value and then explicitly replaced it with an incorrectly rounded final answer despite the prompt saying not to round

Audit artifacts are retained under:
`outputs/transfer/pre_rl_qwen3_14b_v4/rulearena/parser_audit/`

### Final RuleArena scientific decision

Use the **official upstream RuleArena scorer as the primary PRE/POST metric**.

Reason:
- unlike the Multi-LogiEval and LogiQA2 repairs, a RuleArena “repair” is no longer a simple formatting normalization; it requires domain-specific semantic policy about which of multiple model statements counts as final
- several plausible corrected policies produce materially different NBA scores
- changing the evaluator after seeing PRE generations would introduce evaluator-design discretion

Therefore freeze/report:
- RuleArena PRE = **199/816 = 24.3873%**
- use the same unchanged official scorer for POST
- preserve the audit as a limitation/provenance analysis, not as a competing headline metric

Important reporting nuance:
`parser_errors=0` means no parser exceptions; it does not mean every response semantically yielded one unambiguous answer.

## Transfer results currently available

Official PRE scores known:
- Multi-LogiEval: **1139/1556 = 73.2005%** (formatting-only parser v1)
- RuleArena: **199/816 = 24.3873%** (official scorer)
- LogiQA 2.0: **1226/1572 = 77.9898%** (formatting-only parser v1)
- MMLU-Pro: **8093/12032 = 67.26%**
- BBH: **5978/6511 = 91.81%**
- Reasoning Gym: **official PRE not yet frozen/reported**; task-family calibration is still being finalized

Current RG sub-status:
- propositional_logic: calibrated/viable at the 4/6/8 variables-complexity ladder under parser-v3/v4
- knights_knaves: calibrated/viable at 5/6/7 people under parser-v4
- course_schedule: size axis 15/25/35 rejected; 5/8/12 ceiling-confirmed; 13/14 rejected as token-budget-contaminated; density-only 12-course pilot `1572208` (`p34` vs `p45`) is the current calibration
- zebra_puzzles: parser-v4 validated with 0/30 changes; 4×4 and 5×5 ceiling, 6×6 rejected because all six failures truncate; characteristics-only 5-person pilot `1572209` (`c6` vs `c7`) is the current calibration

Do not substitute GSM8K/FineReason pilot numbers into this official six-benchmark table.

## Immediate next operational priorities

1. Monitor jobs `1572303–1572308` with `squeue`; use `sacct` for terminal state. Do not interpret `PD (Priority)` or `PD (Resources)` as an error.
2. Verify exact job→model mapping for the six smaller-model jobs with `scontrol` before attaching scientific results to job IDs.
3. For PRE jobs `1572303–1572305`, require 1,440 committed/scored frozen ArgGYM rows per model and preserve raw generation/score artifacts before calling a PRE run complete.
4. For RL jobs `1572306–1572308`, inspect startup logs as soon as each enters RUNNING: pinned snapshot, model revision, preflight PASS, Stage1 V2 pool, RL_DEV, GPU topology, vLLM readiness, 3 Accelerate ranks, memory-safe old-logprob diagnostic, IS correction, Liger, no OOM/NCCL/fatal error.
5. Leave both 14B jobs (`1570871`, `1571672`) alone unless the user explicitly changes that decision. Their exact optimizer-step progress must be re-read from logs when needed.
6. Primary 14B V2 job `1571672`: use epoch-0 `eval_reward=0.2889` as its within-run baseline. The next decisive result remains its step120 RL_DEV evaluation.
7. Do not alter Stage1/2/3 V2 data ordering or regenerate those datasets. Stage2/Stage3 preparation is final.
8. Smaller-model PRE and RL can overlap safely because both are initialized/served from immutable pinned base-model snapshots and write separate artifacts; never train on `data/taskset.jsonl`.
9. For the official six-benchmark transfer suite, Reasoning Gym remains the unfinished calibration item; do not let smaller-model scale-out change the already frozen 14B transfer protocol.
10. Once a checkpoint is selected for POST, evaluate it on the exact frozen ArgGYM benchmark with the same model-specific PRE decoding protocol; transfer results do not select checkpoints.

## Non-negotiable scientific/engineering rules

- never contaminate or overwrite `data/taskset.jsonl`
- never train on the frozen ArgGYM benchmark
- never silently regenerate V2 Stage1 or alter its examples/order now that it is verified
- verify actual repo paths/interfaces before persistent edits
- preserve raw generations, prompts, manifests, hashes, and scores
- do not modify transfer protocols based on whether PRE scores look convenient
- parser repairs are allowed only when they are narrowly formatting-only, deterministic, applied to saved generations, versioned, and frozen identically for POST
- RuleArena is the exception decided above: use official upstream scorer unchanged
- transfer results do not select checkpoints
- no global `set -euo pipefail` in the interactive SSH shell
- avoid destructive git/file commands in the dirty repository

## Common failure modes and mandatory checks for the next model

These are **observed mistakes from this project**, not hypothetical advice. Before giving the user commands or claiming code is ready, explicitly check the relevant items below.

### 1. Never invent repository structure or interfaces

Observed failure: claiming a nonexistent `benchmarks/` folder and proposing code before inspecting the actual ArgGYM tree.

Mandatory check:
- inspect the actual repo (`pwd`, `find`, `ls`, `grep`) before naming persistent paths
- reuse verified interfaces such as `arggym.create(...)` and `arggym.score_row(...)` rather than recreating them
- if a path/interface has not been inspected in the current repo snapshot, label it unverified

### 2. Use the correct Python/runtime for the job

Observed failure: mixing host Python, SIF Python, and the dedicated Reasoning Gym venv.

For current Reasoning Gym builder/scorer work, the known-good pattern is:
```bash
apptainer exec \
  --bind /arf/scratch/futan:/arf/scratch/futan \
  "$SIF" \
  env PYTHONPATH="$RG:$ROOT" \
  "$RGVENV/bin/python" ...
```

Mandatory check:
- identify which environment owns each dependency before executing
- do not substitute host `python3` for the pinned venv when `reasoning_gym` is required
- do not install packages into the host or canonical SIF to paper over an environment mismatch
- preserve the original SIF; use additive external environments when needed

### 3. Do not assume `git` exists inside the SIF

Observed failure: a pilot builder called `git` from inside the container and crashed with `FileNotFoundError`.

Mandatory pattern:
- run `git -C "$RG" rev-parse HEAD` on the **host**
- pass the verified revision into the container through an environment variable such as `ARGGYM_RG_REVISION`
- inside the Python builder, compare against the exact expected revision; do not shell out to git

### 4. `py_compile` is not a runtime validation

Observed failure: code was declared “syntax-checked” but had a missing `import re`; `py_compile` cannot detect undefined runtime names.

Mandatory check before telling the user to run code:
- syntax compile
- import the module in the target environment
- execute the relevant function/path on a small real or synthetic example
- for parser code, include unit cases matching actual model outputs
- do not call a check “runtime-validated” unless the relevant code path actually ran

### 5. Check imports explicitly after edits

Observed failure: repeated missing `import re` in generated pilot scripts.

Mandatory check:
- inspect imports after every code rewrite
- exercise regex/parser branches at runtime
- when feasible, run a linter/static undefined-name check in addition to compilation

### 6. Do not run unnecessary exhaustive CPU work on the login node

Observed failure: a pilot selector performed expensive exhaustive truth-table filtering and attempted to build multiple datasets in one invocation, exceeding login-node CPU limits.

Mandatory check:
- bound combinatorial validation (`2^N`) to small N
- do not build multiple large pilot variants in one login-node run
- prefer deterministic bounded candidate filtering
- use allocated compute/container runtime for heavier work

### 7. Do not assume every RG task stores gold in `entry["answer"]`

Observed failure: generic self-check code broke on `propositional_logic`, where the generated witness is in `entry["metadata"]["example_answer"]`.

Mandatory check:
- inspect the pinned task/scorer implementation for each selected family
- use task-specific gold/witness access where necessary
- native-gold self-score every candidate before freezing it

### 8. A generator can emit internally invalid examples

Observed failure: `propositional_logic` hard candidate idx13 passed our structural filter but its generated `example_answer` scored `0.05` under RG’s own native scorer.

Mandatory response:
- do not abort the entire deterministic build for this task
- skip verifier-invalid candidates deterministically and record that policy
- never use model performance to decide which candidates survive

### 9. Audit upstream benchmark semantics, not just code execution

Observed issues:
- `family_relationships`: wrong-direction in-law labels and suspect niece/nephew direction
- `syllogism`: checker can validate invalid arguments because of missing middle-term distribution
- RuleArena: permissive/brittle official scoring in NBA/Tax

Mandatory check:
- self-score golds
- manually inspect representative examples
- if semantics are suspect, use an independent validity check for protocol development or drop the task
- do not silently “fix” an official benchmark after seeing outcomes; version and justify any repair

### 10. Parser transformations must be order-safe

Observed failure: parser-v2 stripped `\left` before normalizing `\leftrightarrow`, producing `rightarrow`.

Mandatory check:
- normalize complete TeX operator commands first (`\leftrightarrow`, `\rightarrow`, `\land`, etc.)
- only then remove wrappers such as `\left` / `\right`
- unit-test every supported operator and actual failure string
- compare old/new extraction and native score on saved generations before freezing a parser version

### 11. Raw whole-completion scoring can be scientifically misleading

Observed failure: correct knights/knaves, self-reference, zebra, course-schedule, and propositional answers could score zero because explanations contaminated strict scorers.

Mandatory policy for RG selected tasks:
- preserve raw completion and raw-native score
- deterministically extract only the final submitted answer
- pass extraction to the untouched native scorer
- preserve adapted answer and adapted-native score
- keep extraction gold-blind

### 12. Do not equate every non-1.0 propositional score with “wrong”

Pinned scorer semantics verified during this work:
- `1.0`: valid non-trivial conclusion
- `0.25`: valid but trivial/atomic conclusion
- `0.05`: parseable non-entailed conclusion
- `0.0`: malformed/unscorable/zero

Mandatory reporting:
- report native mean if desired, but also separate non-trivial-valid, entailed-including-trivial, invalid, zero, and truncation counts

### 13. Parser bugs do not require rerunning GPU generation

Observed good recovery: parser-v3 rescored saved `1571906` completions and changed only the two corrupted `\leftrightarrow` rows from `0.05` to `1.0`.

Mandatory response:
- keep generation artifacts immutable
- create a new parser version additively
- retrospectively rescore saved completions
- restart GPU inference only if generation itself was invalid, not merely parsing

### 14. Do not edit a running protocol in place

During `1571906`, parser-v3 was created as a separate file rather than changing the module already loaded by the live process.

Mandatory check:
- version fixes additively (`v2`, `v3`, etc.)
- record which parser the live job actually loaded
- use retrospective rescoring to reconcile results

### 15. Do not claim a row is completed merely because it appears in the runner log

The runner prints `[n/N] ID` **before** generation. Therefore that line means the item has started, not necessarily committed.

Mandatory check:
- use `wc -l generations.jsonl` or inspect the row itself for committed-count claims

### 16. Do not patch scientific files with blind text replacement

Observed risk: repeated manual `sed`/copy operations can silently alter the wrong block.

Mandatory pattern:
- assert the expected old text/token exists exactly once
- abort if it does not
- write additive versioned files when possible
- re-run syntax/runtime checks after the patch
- verify hashes/row counts before `sbatch`

### 17. Do not infer current Slurm state from an old handoff

Mandatory check:
- `squeue` for live state
- `sacct` for completed/failed/timeout state
- inspect stdout/stderr before declaring a run healthy
- exact optimizer-step/progress claims must come from logs, not elapsed time

### 18. Never cancel the old RL run or replace the corrected run without explicit user instruction

The user intentionally kept `1570871` and `1571672` running simultaneously. Preserve that decision.

### 19. Preserve the dirty repository and provenance

Mandatory constraints:
- no `git add .`
- no broad `git reset`, `git clean`, or destructive cleanup
- no overwriting frozen data/manifests
- retain old versions, hashes, raw generations, manifests, and audit outputs

### 20. Keep the official transfer suite exactly six benchmarks

Do not reintroduce ArgBench, GSM8K, or FineReason into the official table. They are dropped/pilot-only. The six are Multi-LogiEval, RuleArena, LogiQA2, MMLU-Pro, BBH, Reasoning Gym.

### 21. Do not select final benchmark examples based on PRE outcomes

Calibration can choose **difficulty ranges and protocol validity** based on runtime, truncation, and aggregate behavior, but the final official suite should use a new deterministic seed/namespace and should not cherry-pick individual examples the base model gets wrong/right. Prefer not to reuse pilot rows.

### 22. Interactive-shell safety differs from sbatch safety

Do not prepend global `set -euo pipefail` to ad-hoc SSH commands. It can terminate the user’s shell flow unexpectedly. `set -eo pipefail` inside a dedicated sbatch script is acceptable when intentionally scoped to that script.

### 23. Validate generator configuration against the pinned task implementation before building a pilot

Observed failure: the first `course_cal30_v1` builder used `min_cycle_length=2`, but the pinned `course_schedule` config enforces `min_cycle_length >= 3`, causing an assertion failure at dataset creation.

Mandatory check:
- inspect the pinned task config/`validate()` implementation before choosing generator ranges,
- run a one-instance `reasoning_gym.create_dataset(...)` smoke test in the pinned RG environment before launching a full deterministic build,
- do not assume seemingly reasonable parameter ranges are accepted by the pinned revision.

### 24. Never claim an artifact, hash, or validation result unless the tool/filesystem actually confirmed it

Observed failure in this session: an artifact-generation tool call reset/failed, yet the subsequent response presented bundle hashes and validation claims as though the call had succeeded.

Mandatory rule:
- if a tool says execution failed/reset, assume **no files and no side effects**,
- re-run the creation/validation or inspect the exact filesystem path before linking or quoting a hash,
- never fabricate or carry forward hashes from an unsuccessful execution attempt.

### 25. Do not patch an uninspected script using brittle multiline assumptions

Observed failure: the boundary Slurm patch assumed exact multiline formatting and exact occurrence counts that were not verified from the actual file. Two patch attempts aborted (`len(rows) != 30` occurred twice, and the expected difficulty block formatting did not match).

Mandatory response:
- inspect the actual file first (`sed`, `grep`, or `diff`),
- if a new protocol is small, prefer writing a new explicit additive file over repeatedly patching an unknown formatting variant,
- if patching, count/print target occurrences before mutation and keep the transformation minimal,
- do not tell the user a patch is robust until it has been tested against the actual file.

### 26. Remember that a single Slurm file may contain duplicate protocol constants in preflight and runner code

Observed failure: assuming there was one `if len(rows) != 30:` check when the validated Slurm script contained two: one in preflight and one in the embedded runner wrapper.

Mandatory check:
- search the whole file for every protocol constant (`row count`, namespace, progress denominator, metrics rows, difficulty labels),
- distinguish preflight copies from runtime copies,
- after edits, search for stale old constants and abort if any remain unintentionally.

### 27. For course-schedule calibration, always stratify runtime and truncation by gold label

Observed scientific issue: at 15 courses, aggregate behavior concealed a severe class asymmetry.
- False: `3/5` correct, `2/5` truncation
- True: `0/5` correct, `4/5` truncation

Mandatory reporting/calibration:
- report True and False separately for accuracy, truncation, completion tokens, and latency,
- do not accept a difficulty setting based only on aggregate accuracy or aggregate truncation,
- treat systematic label-dependent budget exhaustion as a protocol confound, not merely “harder reasoning.”

### 28. Zero truncation is not enough; calibration also needs accuracy headroom

Observed result: `course_cal30_v1` at 5/8/12 had `30/30` correct and zero truncations. This proved the protocol was safe but too easy for a useful PRE/POST transfer measure.

Mandatory decision logic:
- reject cells dominated by truncation/resource exhaustion,
- also reject cells that are pure ceiling if the goal is to measure improvement,
- use small boundary pilots (here 13/14) instead of rerunning broad grids once the transition region is localized.

### 29. Do not interpret an empty final-answer string on a length-capped reasoning-model response as zero generated tokens

Observed course rows had `finish_reason=length`, `completion_tokens=16384`, and an empty stored `completion` string because the reasoning parser separates reasoning from final answer.

Mandatory check:
- inspect `finish_reason`, token usage, and the stored `reasoning` field,
- distinguish “no final answer emitted” from “no generation occurred,”
- classify these as generation-budget failures when the cap was reached.

### 30. Do not call a Slurm job COMPLETED merely because it disappeared from `squeue`

Observed state: `1572137` disappeared from the latest `squeue` after producing 30/30 rows, but no `sacct` terminal state was pasted in this conversation.

Mandatory check:
- use `sacct -j <id> --format=JobID,State,ExitCode,Elapsed,...` for terminal provenance,
- phrase interim state as “30/30 generations committed; no longer in latest `squeue`” until `sacct` confirms completion.

### 31. When a parser fix is validated, quantify exactly what changed

Good practice established here:
- parser-v3 changed exactly two propositional rows, both known `\\leftrightarrow` corruption cases,
- parser-v4 changed exactly one knights row, the known Markdown/fallback contamination case,
- all other rows remained unchanged.

Mandatory check:
- compare old/new extracted answers and native scores over **all saved rows**, not only the motivating example,
- inspect every changed row before freezing the parser,
- unexpected extra changes mean the parser is not yet validated.

### 32. Do not infer optimizer-group task composition from a single emitted task metric

Observed interpretive trap: V2 training log lines emitted one `arggym_reward_task_*` field per optimizer step, which initially made early training look like only four task families were cycling. A direct audit of the frozen V2 dataset showed that each optimizer step actually contains three unique tasks, and every four consecutive L1 steps cover all 12 tasks.

Mandatory check:
- when curriculum/task mixing matters, inspect the actual frozen rows grouped by effective prompts-per-update,
- treat per-step task metric labels as logging summaries, not authoritative evidence of all tasks present in that optimizer update,
- distinguish deterministic blocked balance from random shuffling when describing the curriculum in the paper.

## Working style for future chats

- distinguish verified facts from proposals
- inspect actual files before modifying anything
- do not invent paths, folders, interfaces, hashes, or job states
- do not rerun expensive jobs when saved generations can be rescored
- if a command fails because of environment mismatch, use the exact production SIF rather than installing packages on the host
- keep old and corrected artifacts additive; do not overwrite scientific provenance
- when status matters, re-check Slurm/logs instead of assuming from this handoff

Repository root:
`/arf/scratch/futan/ArgGYM`

# Operations Manual — Evaluating and RL-Training Models with ArgGYM

This manual separates **ArgGYM scientific logic** from **TRUBA-specific scheduling/runtime wrappers**. The repository paths below are authoritative for this checkout, but a different server does not need Slurm, TRUBA partitions, or the TRUBA SIF as long as it provides compatible Python/CUDA/vLLM/TRL/PEFT dependencies and sufficient GPU memory.

## A. Core safety rule: evaluation data and RL data are different

Frozen benchmark for evaluation only:
- `data/taskset.jsonl`
- 1,440 frozen rows
- frozen manifest hash `121f452f2744ef0a6022c4731097b6b1`
- **never train on this file, never overwrite it, never add procedurally generated training rows to it**

RL datasets:
- Stage1 canonical: `data/rl/curriculum/stage1.jsonl`
- Stage1 article ordering: `data/rl/curriculum/stage1_levelmixed_v2.jsonl`
- Stage2 canonical: `data/rl/curriculum/stage2.jsonl`
- Stage2 article ordering: `data/rl/curriculum/stage2_levelmixed_v2.jsonl`
- Stage3 canonical: `data/rl/curriculum/stage3.jsonl`
- Stage3 article ordering: `data/rl/curriculum/stage3_levelmixed_v2.jsonl`
- validation/checkpoint-selection split: `data/rl/curriculum/dev.jsonl`

Every RL JSONL has a companion `.jsonl.manifest.json` used by preflight/training integrity checks. Keep the manifests with the datasets.

## B. Frozen ArgGYM evaluation

### B1. Relevant files

Evaluation harness:
- `evals/run.py` — evaluation execution
- `evals/client.py` — OpenAI-compatible chat endpoint client
- `evals/taskset.py` — frozen taskset loading/integrity
- `evals/score.py` — ArgGYM scoring/reporting path
- `evals/report.py` — reporting utilities
- `evals/conf/config.yaml` — Hydra/default evaluation configuration
- `evals/conf/model/*.yaml` — model endpoint + decoding profiles

vLLM/local-serving layer:
- `hpc/vllm/models/*.yaml` — serving profiles: model ID, optional pinned revision, BF16, TP, max model length, memory utilization, reasoning parser, extra vLLM args
- `hpc/vllm/run_vllm_arggym.py` — repository orchestrator that loads a serving profile, starts vLLM with explicit model/revision settings, runs ArgGYM evaluation, and writes into `outputs/runs/`
- `hpc/vllm/verify_bundle.py` — validates pairing of serving profiles with `evals/conf/model` endpoint profiles

TRUBA-only wrappers:
- `hpc/vllm/submit_truba.sh`
- `hpc/vllm/truba_vllm.sbatch`
- `hpc/vllm/setup_runtime.sbatch`
- `hpc/vllm/check_runtime.sh`

The `hpc/` directory name does **not** mean the underlying evaluator is TRUBA-only. The shell/sbatch wrappers are platform-specific; the Python evaluator and vLLM orchestrator are the reusable parts.

### B2. Create or select a model profile

A model normally has two matched YAMLs sharing the same profile name. Example article profile:

`evals/conf/model/hf-qwen3-8b-arggym-40k.yaml` contains the endpoint/model ID and decoding protocol, including fields such as:
- `name`
- `model`
- `base_url`
- `sampling.max_tokens`
- `sampling.temperature`
- `sampling.top_p`
- optional `extra_body` fields such as top-k, min-p, thinking controls

`hpc/vllm/models/hf-qwen3-8b-arggym-40k.yaml` contains serving controls, including:
- `endpoint_config` — must match the eval YAML name
- `hf_model` — must match the eval YAML model ID
- optional `revision` — strongly recommended for reproducible experiments
- `dtype`
- `tensor_parallel_size` and optional GPU-type-specific TP fields
- `max_model_len`
- `gpu_memory_utilization`
- optional `reasoning_parser`
- `trust_remote_code`
- `extra_args`

For scientific PRE/POST comparisons, pin the exact model revision and do not silently change decoding between PRE and POST.

Before launch:
```bash
python3 hpc/vllm/verify_bundle.py
```

### B3. TRUBA evaluation launch

On TRUBA, after the model snapshot/runtime is available:

```bash
cd /arf/scratch/futan/ArgGYM
export HF_HOME=/arf/scratch/futan/arggym_hf_cache
export GPU_TYPE=H200          # or the intended supported GPU type
export ACCOUNT=romer          # site/account specific
export TEMPLATE=xml_tags
export ELICITATION=cot
export RESUME=0
export SKIP_SCORE=0

./hpc/vllm/submit_truba.sh hf-qwen3-8b-arggym-40k
```

This wrapper is TRUBA-specific because it uses `sbatch`, TRUBA accounts/partitions, and the repository's TRUBA runtime setup. Do not copy those scheduler flags blindly to another cluster.

### B4. Non-TRUBA / generic GPU server evaluation

On a normal Linux GPU server with the required Python environment and vLLM installed, the closest repository-native entrypoint is:

```bash
cd /path/to/ArgGYM
python3 hpc/vllm/run_vllm_arggym.py <profile-name> \
  --template xml_tags \
  --elicitation cot
```

The orchestrator itself starts the local vLLM server from the selected `hpc/vllm/models/<profile>.yaml`, passes the explicit pinned revision when present, and then runs the evaluation harness. `--resume` and `--skip-score` are available when their semantics are desired; do not use them casually for a clean official run.

If the server cannot run this orchestrator unchanged, reproduce its two logical components rather than its TRUBA plumbing:
1. start an OpenAI-compatible model server with the same model/revision/dtype/context/decoding support;
2. point the matching `evals/conf/model/<profile>.yaml` at that endpoint and run the ArgGYM evaluation harness.

The important scientific invariants are the frozen `data/taskset.jsonl`, exact model revision, exact decoding settings, and unchanged native ArgGYM scoring—not the scheduler brand.

### B5. Evaluation outputs and completion criteria

`hpc/vllm/run_vllm_arggym.py` creates a deterministic run directory:

`outputs/runs/<endpoint_config>__<template>__<elicitation>`

For the current smaller Qwen models:
- `outputs/runs/hf-qwen3-8b-arggym-40k__xml_tags__cot`
- `outputs/runs/hf-qwen3-4b-arggym-40k__xml_tags__cot`
- `outputs/runs/hf-qwen3-1.7b-arggym-40k__xml_tags__cot`

A scheduler job exiting successfully is not enough. Before calling a frozen ArgGYM evaluation complete:
- confirm terminal job/process exit success
- confirm the run metadata records the intended model/profile/revision
- confirm **1,440 committed/scored benchmark rows**
- preserve raw generations as well as scores
- preserve task/level/ordering breakdowns needed for analysis
- do not infer committed row count from a progress line printed before generation is written

For PRE→POST, use the same frozen benchmark and the same generation/scoring protocol for that model.

## C. RL fine-tuning with ArgGYM stage datasets

### C1. Relevant files

Core training code:
- `rl/train_grpo.py` — GRPO/DAPO training entrypoint
- `rl/memory_safe_grpo.py` — memory-safe trainer implementation used by the article protocol
- `rl/preflight.py` — provenance, data, model-revision, curriculum, context, batch, and tokenizer checks
- `rl/common.py` — shared RL data/hash utilities

Current article config template:
- `configs/rl/qwen3_14b_stage1_levelmixed_v2.yaml`

Current smaller-model Stage1 configs:
- `configs/rl/qwen3_8b_stage1_levelmixed_v2.yaml`
- `configs/rl/qwen3_4b_stage1_levelmixed_v2.yaml`
- `configs/rl/qwen3_1p7b_stage1_levelmixed_v2.yaml`

Curriculum definition/provenance:
- `configs/rl/curriculum.yaml`
- `data/rl/curriculum/*.jsonl`
- `data/rl/curriculum/*.jsonl.manifest.json`

TRUBA launch layer:
- `hpc/grpo/submit_truba.sh`
- `hpc/grpo/truba_grpo.sbatch`
- `hpc/grpo/run_node.sh`

Article container/runtime on TRUBA:
- `/arf/scratch/futan/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif`
- external Liger site `/arf/scratch/futan/arggym_grpo_runtime/liger-kernel-0.8.2-site`

The SIF is a reproducible TRUBA runtime, not a logical requirement of ArgGYM itself. Another server may use a compatible container/venv with the required CUDA/PyTorch/Transformers/vLLM/TRL/PEFT stack.

### C2. Choose the correct training dataset

For the article protocol, prefer the audited `*_levelmixed_v2.jsonl` files because their file order encodes the intended level curriculum while mixing tasks inside optimizer groups. Do not regenerate them.

Stage meanings:
- Stage1: easy-heavy curriculum
- Stage2: more balanced easy/medium/hard
- Stage3: hard-heavy curriculum

`dev.jsonl` is evaluation/checkpoint-selection data, not a training stage.

Never replace a stage dataset with `data/taskset.jsonl`. That would contaminate the frozen benchmark.

### C3. Create/adapt a model-specific RL config

Start from an audited config close to the intended protocol, not from scratch. At minimum verify:
- `experiment.frozen_taskset_hash` remains the canonical hash
- `experiment.baseline_model_revision` equals the pinned base checkpoint revision
- `model.name` and `model.revision` identify the exact initialization checkpoint
- `data.train` points to the intended RL stage file
- `data.dev` points to `data/rl/curriculum/dev.jsonl`
- output directory/run name are unique
- context budget is valid: `max_model_length >= max_prompt_length + max_completion_length`
- PEFT/LoRA settings are intentional
- optimizer/LR/batching/G/GRPO-DAPO settings are intentional
- `shuffle_dataset:false` when using the article's ordered curriculum

For a new base model, `rl/preflight.py` intentionally fails closed unless the model/revision pair has been explicitly reviewed and added to `SUPPORTED_BASELINE_REVISIONS`. Do not weaken this check just to make a run start.

### C4. Always run preflight before expensive training

Set `ARGGYM_MODEL_PATH` to the exact local Hugging Face snapshot directory, not merely a model name. Example:

```bash
export ARGGYM_MODEL_PATH=/path/to/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/<exact-sha>

python3 -m rl.preflight \
  --config configs/rl/qwen3_8b_stage1_levelmixed_v2.yaml \
  --world-size 3
```

Run preflight in the **same software environment** that will train the model whenever possible. Do not use `--skip-token-audit` for an official run unless there is a specific documented reason.

Preflight is expected to catch problems before GPUs are consumed, including wrong local revision, wrong/faulty manifests, frozen-data collisions, curriculum structure, context overflow, incompatible training settings, and tokenizer-budget issues.

### C5. The current article topology is not arbitrary

The V2 datasets were audited against the production update geometry:
- 1 rollout/vLLM GPU
- 3 trainer GPUs
- per-device train batch 1
- gradient accumulation 8
- G=8 generations
- therefore 3 unique prompts per optimizer update

`hpc/grpo/run_node.sh` implements this as GPU0 for vLLM and GPUs1–3 for three Accelerate trainer ranks.

This matters for portability. A different server can reproduce the protocol if it supplies the equivalent topology, but if you change trainer world size, batch size, gradient accumulation, or G, the **optimizer-group boundaries change**. The V2 files remain valid datasets, but the verified “three task-unique prompts per update” property may no longer apply. Treat such a change as a new training protocol and re-audit the grouping/batch math; do not claim exact article-protocol equivalence.

### C6. TRUBA RL launch

On TRUBA, with the exact local model snapshot already downloaded:

```bash
cd /arf/scratch/futan/ArgGYM

export ARGGYM_ROOT=/arf/scratch/futan/ArgGYM
export ARGGYM_GRPO_RUNTIME_DIR=/arf/scratch/futan/arggym_grpo_runtime
export ARGGYM_LIGER_SITE=/arf/scratch/futan/arggym_grpo_runtime/liger-kernel-0.8.2-site
export HF_HOME=/arf/scratch/futan/arggym_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ACCOUNT=romer

ARGGYM_MODEL_PATH=/exact/snapshot/path \
./hpc/grpo/submit_truba.sh \
  configs/rl/<model-specific-config>.yaml
```

The TRUBA wrapper submits `hpc/grpo/truba_grpo.sbatch`, which requests the site-specific resources and then executes `hpc/grpo/run_node.sh`. Do not reuse its account, partition, feature, CPU, or walltime settings on another cluster without adapting them to that site.

### C7. Non-TRUBA / generic GPU server RL launch

If the machine has four suitable GPUs and the required dependencies installed, `hpc/grpo/run_node.sh <config>` is the closest reusable node-level launcher. It assumes the article topology and explicitly assigns GPU0 to vLLM and GPUs1–3 to three trainer processes. Set the same environment variables it expects, especially `ARGGYM_ROOT`, `ARGGYM_MODEL_PATH`, and the relevant cache/runtime paths.

If the machine has a different GPU topology, do not blindly edit the datasets. Instead:
1. keep the frozen RL stage JSONLs and manifests unchanged;
2. design an equivalent rollout/trainer launch for that machine;
3. run `rl.preflight.py` with the actual trainer world size;
4. recompute effective prompts/update from world size, per-device batch, gradient accumulation, and G;
5. re-audit curriculum grouping if the effective prompt-group size differs from 3;
6. record the deviation as a different protocol.

The scientific core is native ArgGYM procedural training data + native verifier reward + contamination isolation. Slurm/TRUBA is only one execution environment.

### C8. What to verify when an RL job starts

Do not treat `RUNNING` as sufficient validation. Check the startup logs for:
- exact model ID and pinned snapshot/revision
- expected config path
- frozen taskset hash
- expected RL train pool and its SHA/manifest
- RL_DEV path/count
- tokenizer audit PASS
- BF16 / FP16=false
- context and completion caps
- vLLM server ready
- expected trainer world size / Accelerate ranks
- LoRA settings
- DAPO/GRPO settings
- memory-safe old-logprob diagnostic
- importance-sampling correction
- Liger setting if used
- no OOM, NCCL, NaN/Inf, or fatal exception

For learning claims, use RL_DEV checkpoint evaluations. Online training reward is curriculum-dependent and is not directly comparable to full RL_DEV or frozen ArgGYM PRE accuracy.

### C9. Stage progression

Stage2/Stage3 datasets are already finalized, but model-specific Stage2/Stage3 training configs should be created only by adapting the actual audited training interfaces/config schema. Preserve the same canonical/V1/V2 datasets and manifests. Do not regenerate them simply to continue training.

A paper run does not automatically need all three stages. Stage progression should be based on the predeclared research plan/checkpoint evidence, not on transfer-benchmark outcomes. Transfer results must not select RL checkpoints or tune the curriculum.

### C10. POST evaluation after RL

After selecting a checkpoint using the permitted RL_DEV logic, evaluate it on the exact frozen ArgGYM benchmark:
- same `data/taskset.jsonl`
- same task scoring
- same relevant PRE generation protocol for that model
- preserve raw outputs and aggregate/task/level/order metrics

Never move frozen benchmark rows into training after seeing PRE errors. That would invalidate PRE→POST contamination claims.

