# Why `rl/` is new code rather than `experiments-phase`

The `experiments-phase` branch already had GRPO training code (commit `5f1c2a1`,
2026-09-21). We reviewed it against current `main`, TRL 1.15 and the
[practices guide](rl-best-practices.html), then wrote `rl/` fresh and kept its
sound ideas.

## What stopped a direct reuse

1. **It does not run on current `main`.** The pool generator, preflight and
   tests require a 1,440-row frozen suite with a fixed hash and
   `tasksets/standard.yaml`. `main` has neither; it ships the 7,200-row
   `data/taskset.jsonl`.
2. **The memory-saving trainer fails on current TRL.** `rl/memory_safe_grpo.py`
   requires TRL's Liger loss, which TRL 1.14 removed, so it raises at start-up.
   TRL now chunks the log-prob computation itself, which is what the file did.
3. **Truncated completions stayed in the group baseline.** They were masked from
   the loss but scored 0 and kept in the group mean. Every finished sibling then
   got a higher advantage, whatever it answered. TRL 1.13+ drops a completion
   from the baseline when the reward returns `None`; `rl/reward.py` does that.
4. **Checkpoints were selected on reward, the partial-credit score.** Constant
   answers earn a floor of partial credit on several tasks, so a run could look
   better by learning them. `rl/` selects on strict success.
5. **The configuration starved the gradient.** It took 3 prompts per optimizer
   step, where the guide cites 32-512. At about 12% base success most groups
   were all wrong and gave no signal, and some steps had zero loss and zero
   gradient. LR 5e-6 is low for LoRA. Completions were not logged. Each dev eval
   took about 6 hours.
6. **It is tied to the old cluster.** About 2,800 lines of curriculum scripts
   reorder data files by hard-coded SHA-256 hashes, and those files exist only
   on that cluster. The launch path assumes one 4xH200 node and an Apptainer
   image.
7. **No run result to build on.** No reward curve or post-training eval was
   committed. The handoff notes record one 14B job at step 120 of 480 and
   another at step 13.

## What we kept

- The reward reads only the last `<answer>` region and scores it with ArgGYM's
  own scorer.
- The GRPO settings: DAPO loss, no KL term, clip 0.2 / 0.28, no std scaling,
  8 completions per prompt, LoRA on all linear layers.
- vLLM in server mode with `--logprobs-mode processed_logprobs`, and TRL's
  importance-sampling correction.
- Training prompts composed by the eval harness's own code.
- Pools checked against the eval tasksets by hash, not only by seed.

The level curriculum became a pass-rate curriculum (`rl/curriculum.py`): on
Qwen3.5-4B the level number predicts poorly which groups will carry signal.
