# 001: Qwen3-4B, levels 1-4

Status: stopped at step 21 of 120, 2026-10-15.
wandb: `bdsaglam/arggym-rl/z3t6exhg`. Config: `rl/configs/qwen3-4b-l1-4.yaml` at
commit `2d0d6a1` (removed from the branch later).

## Question

Does the pipeline learn anything on the easiest levels with a small model and
the guide's default settings?

## Card

| | |
|---|---|
| Model | `Qwen/Qwen3-4B`, BF16, thinking on |
| Data | All 12 tasks, levels 1-4; training pool 4,800 rows. Dev: 2 rows per (task, level) = 96 prompts x 4 samples |
| Batch | 8 prompts x 16 completions = 128 rollouts per step |
| Sampling | T 1.0, top_p 1.0, top_k off, no penalty; cap 16,384 tokens |
| Loss and optimizer | GRPO with DAPO loss, no KL, clip 0.2 / 0.28, no std scaling; LoRA r32 all-linear, LR 1e-5 |
| Reward | Strict success (0/1); truncated completions return `None` |
| Hardware | vLLM on 1 A100, trainer on 2; about 16 min per step |

## Results

| Dev strict success | Step 0 (base) | Step 20 |
|---|---|---|
| Overall (384 samples) | 0.068 | 0.081 |

The difference is +0.013 with a standard error of about 0.013: noise.

Training side, over 21 steps:

| | Range | Mean |
|---|---|---|
| Train success | 0.00 - 0.23 | 0.09 |
| Groups with no signal | 0.38 - 1.00 | 0.69 |
| Truncated completions | 0 | 0 |

Train success showed no trend.

## Discussion

The base model is too weak on this benchmark for GRPO to get a gradient. On
the lite eval, levels 1-4, Qwen3-4B scores 0.068 strict success, and 45% of its
answers on levels 1-8 have no `<answer>` region at all; two tasks
(status_query, semantics_query) never use the tags. With success near 0.07, a
group of 16 is all-wrong most of the time, and 69% of groups gave no signal.
The thinking-off variant scores 0.010 and was not tried.

What stopped the run was the comparison with Qwen3.5-4B, which scores 0.445
on the same levels in the stored evals. A model that succeeds about half the
time gives the most signal per group, so the next experiment switched models
rather than tuning this one.
