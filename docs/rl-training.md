# RL training on ArgGYM

`rl/` trains a model on ArgGYM with GRPO, through TRL's `GRPOTrainer` and a vLLM
server for rollouts. It is outside the `arggym` package and its dependencies are
an opt-in group, so `pip install arggym` is unchanged. The practices behind the
choices below are in [rl-best-practices.html](rl-best-practices.html) (the
guide); this page is how to run the code and what it decides. Why this is new
code rather than the `experiments-phase` branch is in
[rl-prior-branch.md](rl-prior-branch.md).

```
uv sync --group rl
uv run --group rl python -m rl.launch rl/configs/smoke.yaml                    # pipeline check
uv run --group rl python -m rl.launch rl/configs/qwen3.5-4b-curriculum.yaml    # the current run
uv run --group rl python -m rl.launch rl/configs/qwen3.5-4b-curriculum.yaml grpo.learning_rate=2e-5
```

One command builds the training pool if needed, starts `vllm serve` on one GPU,
runs the trainer on the others with `accelerate`, and merges the selected
checkpoint at the end. It tears vLLM down on every exit path. Outputs go under
`outputs/rl/` (gitignored). Metrics are tracked with
[trackio](https://github.com/gradio-app/trackio), locally, in
`outputs/rl/trackio`; `trackio show --project arggym-rl` opens the dashboard.
The configs assume 3 GPUs of 80 GB: GPU 0 serves rollouts, GPUs 1-2 train.

## Layout

| File | What it does |
|---|---|
| `rl/config.py` | One YAML per run; `key.sub=value` overrides. Unknown keys are refused. |
| `rl/pool.py` | Generates the training pool and the dev split from fresh seed windows, disjoint from the eval tasksets by seed window and by question and theory hash. |
| `rl/reward.py` | The reward: strict success (0/1) from `evals.score.score_one`, the function that scores every eval run. |
| `rl/curriculum.py` | The pass-rate curriculum sampler (below). |
| `rl/gsm8k.py` | The GSM8K positive control's data, prompt, reward and metrics (below). |
| `rl/train.py` | The trainer: TRL's `GRPOTrainer` plus dev evaluation, selection and a few fixes listed in `ArgGymGRPOTrainer`. |
| `rl/export.py` | Merges the adapter into a copy of the base checkpoint and checks that every trained tensor differs from base. |
| `rl/launch.py` | The one command above. |
| `rl/configs/` | `smoke.yaml`, `qwen3-4b*.yaml`, `qwen3.5-4b.yaml`, `qwen3.5-4b-curriculum.yaml`, `gsm8k-qwen3-0.6b.yaml`. Every hyperparameter is in the file, with the reason next to it. |

Tests are in `tests/rl/` and run on CPU; the ones that need a GPU or a live vLLM
server skip by default.

## What the code guarantees

Each of these has a test.

- **Training prompts are the eval prompts.** A training row renders to the same
  messages, and the same token ids, as `evals/` sends for that row with the same
  model config, template and elicitation.
- **Training data is disjoint from evaluation.** Pools come from seed windows that
  do not overlap the eval specs, and any row whose question or theory also
  appears in an eval taskset is dropped.
- **The reward is the eval's strict success.** Truncated completions return
  `None`, which takes them out of the group baseline as well as the loss.
  Partial credit is logged, never paid.
- **Checkpoints are selected on dev strict success**, never on reward, and
  `selection.json` records whether the best checkpoint beat the base model's
  step-0 dev score.
- **The served model is the trained model.** `rl/export.py` fails if any trained
  tensor equals base after the merge.
- **Dev evaluation samples as the eval does**: the eval config's sampling, its
  presence penalty and its 65,536-token cap.

## Main decisions

**Algorithm.** GRPO with the DAPO token-level loss, no KL term (`beta: 0`), clip
0.2 / 0.28, group-mean baseline without std scaling, 8 completions per prompt,
one gradient step per batch. TRL's importance-sampling correction for the
vLLM/trainer mismatch is on, per token, truncated at 2. These are the guide's
starting defaults. The loss is averaged over the tokens of completions with a
nonzero advantage only: a group scored all 0 or all 1 gives no gradient, and
with its tokens in the denominator it would shrink every update by its share
of the batch, about half in experiment 002.

**LoRA.** Rank 32 on all linear layers, LR 1e-5, constant after a short warmup.
On Qwen3.5-4B that is 64.9M parameters, the Gated DeltaNet projections included.

**Model.** Qwen3.5-4B with thinking on, at its published BF16 weights. Qwen3-4B
was tried first and gave almost no learning signal: base strict success 0.068 on
lite levels 1-4, and most groups had all 8 completions wrong. Qwen3.5-4B scores
0.445 on the same levels.

**Rollout sampling.** Training rollouts use the model card's settings,
`top_p 0.95, top_k 20, presence_penalty 1.5`, the same as the eval. The guide's
default is to sample at T=1 with no truncation so the trainer computes the same
distribution vLLM sampled from. Here that default cost too much: without the
card's penalty, 42-58% of Qwen3.5-4B's rollouts looped into the 40,960-token cap
and were masked out. The trainer scores the full-vocabulary unpenalised
distribution, and the truncated importance weights correct for the gap. Watch
`sampling/importance_sampling_ratio/*` and `sampling/sampling_logp_difference/*`.

**Curriculum by pass rate.** A group whose completions all succeed or all fail
teaches nothing. On Qwen3.5-4B the level number predicts that badly: base
success is about 1.0 on claim_chain L1-3, 0.0 on perturbation L2-4, and flat on
preference_construction across all 15 levels. So the sampler keeps a pass rate
per (task, level) cell, starting from the base eval and updated by every
training group, and draws cells in proportion to p(1 - p). Mid-difficulty cells
come first and harder cells come in as easier ones saturate. A small floor keeps
every cell in play. An outcome reaches the sampler two steps later. Details are
in `rl/curriculum.py`.

**Data.** The curriculum config trains on levels 1-8, all 12 tasks. The dev
split is one row per (task, level), 96 prompts, sampled twice. Dev success is
logged for levels 1-4 and 5-8 separately.

## Results so far

| Run | Outcome |
|---|---|
| Qwen3-4B, levels 1-4, 20 steps | Dev 0.068 → 0.081, within noise. Train success flat; 60-90% of groups gave no signal. Stopped. |
| Qwen3.5-4B, levels 1-4, smoke | Pipeline verified end to end. ~47 min per step (generation ~21 min). 42-52% of rollouts truncated without the presence penalty. |
| Qwen3.5-4B curriculum, smoke | 2 steps, curriculum working; ~50 min per step. Truncation still 42-58% without the penalty, which led to the change above. |

No run has yet shown an improvement on the eval taskset. A trained model counts
as better only after `rl.export` and an `evals.run` on the lite taskset,
compared with the stored base result; dev success picks the checkpoint but is
too small a sample to report.

## GSM8K positive control

No ArgGYM run has raised reward yet, which leaves two explanations: the
pipeline is broken, or ArgGYM is hard for these models. GRPO is known to raise
GSM8K accuracy within tens of steps, so a run that does not points at the
pipeline.

```
uv run --group rl python -m rl.launch rl/configs/gsm8k-qwen3-0.6b.yaml
uv run --group rl python -m rl.launch rl/configs/gsm8k-qwen3-0.6b.yaml grpo.learning_rate=1e-4
```

`task: gsm8k` in the config swaps four things: the rows (HF `openai/gsm8k`,
`main`; training from `train`, dev from 200 seeded `test` problems), the prompt
(the question and "put your final answer within `\boxed{}`", thinking on), the
reward (1 when the last `\boxed{}` is the gold number, compared as numbers;
`None` when truncated) and the selection metric (`eval_gsm8k_success_rate`).
The trainer, vLLM server, weight sync, IS correction, DAPO loss, LoRA and dev
evaluation are the ArgGYM runs' own; a config without `task` is an ArgGYM run,
unchanged. No pool is built. Qwen3-0.6B's dev evaluation samples as
`evals/conf/model/hf-qwen3-0.6b.yaml` says, the card's settings at a 32,768 cap.

What to read: `gsm8k_success_rate` per step and `eval_gsm8k_success_rate` every
10 steps, against the step-0 dev score in `selection.json`. The 8,192-token
completion cap is a guess, twice TRL's example; read `gsm8k_truncated_rate` at
step 0 before trusting the run, and lower or raise it from the CLI
(`grpo.max_completion_length=4096`).

## Known limits

- **Wall time.** A step takes 45-50 minutes, mostly generation at 30-40k-token
  completions. 100 steps is about 4 days on 3 GPUs. A dev eval adds 45-90
  minutes.
- **`causal-conv1d` is not installed** (no wheel for this torch). Its PyTorch
  fallback costs about 3% of a forward-backward pass; the trainer allows that
  one fallback and stops on any other slow-kernel warning.
- **Revision pin.** `qwen3.5-4b*.yaml` pin the one Qwen3.5-4B snapshot that was
  cached on the machine that produced the stored eval. That it is the snapshot
  the eval used is inferred, not recorded.
