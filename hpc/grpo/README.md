# ArgGYM Qwen3-14B GRPO on TRUBA

This directory is a **separate training runtime**. Do not modify or rebuild the
existing `hpc/vllm/` evaluation image. The article baseline was produced with
vLLM 0.29.0; GRPO training uses a distinct vLLM 0.28.0 + TRL 1.13.0 image.

## Frozen protocol values already fixed

- Base: `Qwen/Qwen3-14B`
- Revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`
- BF16, no quantization
- LoRA r=32, alpha=16, all-linear, dropout=0
- Thinking mode on; temperature=.6, top_p=.95, top_k=20, min_p=0
- G=8
- DAPO loss aggregation
- `scale_rewards=none`
- clip epsilon=.20, epsilon_high=.28
- beta=0, one policy iteration
- prompt safety ceiling 8,192 tokens; prompts are never silently truncated
- smoke completion cap 20,480; main cap is frozen only after the smoke/CDF audit
- vLLM context 40,960 (same context ceiling used by the frozen baseline)

## Phase A — smoke only

The generator depends on the pinned ArgGYM/PyArg runtime. Do **not** assume the
TRUBA login environment has `python-argumentation==2.0.2`. Build the separate
GRPO image first, then use that same image for generation and preflight.

From the ArgGYM repository root:

```bash
export ARGGYM_ROOT=$PWD
export ACCOUNT=romer                 # change only if your TRUBA project differs
export ARGGYM_MODEL_PATH=/arf/scratch/$USER/.../models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18

./hpc/grpo/build_only.sh
```

Wait until that Slurm build job finishes successfully. Then confirm the image
exists:

```bash
ls -lh /arf/scratch/$USER/arggym_grpo_runtime/arggym-grpo-vllm0.28.0-trl1.13.0.sif
```

Generate the disposable smoke data **inside the new image**:

```bash
./hpc/grpo/exec_runtime.sh python3 -m rl.generate smoke \
  --spec tasksets/standard.yaml \
  --frozen data/taskset.jsonl \
  --out-dir data/rl/smoke
```

Expected generated data: 24 smoke-train rows and 12 smoke-dev rows. Every task is
represented, and the small design rotates over low/medium/high levels and all
four orderings. Smoke data uses independent domain-separated seeds and is never
reused in the article training stages.

Run preflight in the same image:

```bash
./hpc/grpo/exec_runtime.sh python3 -m rl.preflight \
  --config configs/rl/qwen3_14b_smoke.yaml \
  --world-size 3
```

Only after preflight prints `ArgGYM GRPO preflight: PASS`, submit the smoke:

```bash
./hpc/grpo/submit_truba.sh configs/rl/qwen3_14b_smoke.yaml
```

The submit helper reuses the already-built image. Topology on one 4xH200 node:

```text
GPU 0    vLLM rollout server
GPU 1-3  three data-parallel Qwen3-14B LoRA/GRPO trainer processes
```

The smoke run is exactly 2 optimizer steps. It is not a paper result.

### Smoke gates

Do not unlock the main run unless all hold:

- no verifier/engine exceptions;
- finite loss/gradients and an actual LoRA update;
- vLLM accepts updated policy weights after optimizer steps;
- reward remains finite and in [0,1];
- `frac_reward_zero_std` is not pathological;
- completion clipping at 20,480 is acceptable or motivates a larger cap;
- generated-dev evaluation completes;
- a checkpoint is saved and can be resumed;
- no OOM on rollout or trainer GPUs.

Archive the `.out`, `.err`, `vllm-server.log`, trainer metrics and checkpoint.

## Phase B — PRE-RL length audit and pilot decision

The existing Qwen3-14B frozen run can determine the real completion tail without
regenerating anything:

```bash
python -m rl.audit_baseline_lengths /path/to/qwen3-14b/frozen/run
```

This prints P97.5/P99/P99.5 and fractions above 16,384 / 20,480 / 24,576 /
32,768. Use it together with smoke clipping to freeze the main completion cap.

The only planned optimizer pilot is `5e-6` versus `1e-5`, and only if the smoke
is stable enough to justify it. The pilot must use a fresh generated namespace
and generated dev only. Never inspect frozen ArgGYM or transfer scores when
choosing LR or token cap.

## Phase C — generate the article curriculum (after smoke review)

The exact three-stage curriculum is encoded in `configs/rl/curriculum.yaml`:

```text
stage1  1440: L1-5=720, L6-10=504, L11-15=216   (50/35/15)
stage2  1440: L1-5=576, L6-10=432, L11-15=432   (40/30/30)
stage3  1440: L1-5=288, L6-10=432, L11-15=720   (20/30/50)
```

Each stage simultaneously enforces:

- 120 examples per task;
- 360 examples per exact ordering;
- 30 examples per task x ordering stratum;
- maximal balance among the five levels inside every band;
- independent deterministic seed namespace;
- no question/public/theory/coordinate overlap with frozen ArgGYM or another split.

Generate all stages + the 720-row held-out dev set inside the pinned runtime:

```bash
./hpc/grpo/exec_runtime.sh python3 -m rl.generate curriculum \
  --spec tasksets/standard.yaml \
  --frozen data/taskset.jsonl \
  --curriculum configs/rl/curriculum.yaml \
  --out-dir data/rl/curriculum
```

`configs/rl/qwen3_14b_main.yaml` is deliberately locked. After the smoke/pilot
review, freeze `max_completion_length` and `learning_rate`, then change:

```yaml
experiment:
  main_run_unlocked: true
```

Run preflight again inside the pinned runtime before submission:

```bash
./hpc/grpo/exec_runtime.sh python3 -m rl.preflight \
  --config configs/rl/qwen3_14b_main.yaml \
  --world-size 3
```

The main trainer concatenates stage1 -> stage2 -> stage3 **without shuffling**.
With 3 trainer GPUs, batch=1, accumulation=8 and G=8, it consumes three unique
prompt groups per optimizer update. Therefore each 1,440-row stage is exactly
480 updates and the complete 4,320-prompt curriculum is exactly 1,440 updates.
Optimizer and scheduler state are continuous across stage boundaries.

Main eval/save cadence is 240 steps: halfway and end of each stage. Checkpoint
selection uses the 720-row generated `RL_DEV` only.

## Transfer evaluation boundary

Before the main RL run, freeze and run PRE-RL evaluation for exactly:
Multi-LogiEval, RuleArena, GSM8K, LogiQA 2.0, FineReason, MMLU-Pro and BIG-Bench
Hard. Do not use any of these results during training or checkpoint selection.
After generated-dev checkpoint selection, run the same frozen transfer protocol
and the untouched 1,440-row ArgGYM benchmark once on the selected checkpoint.
