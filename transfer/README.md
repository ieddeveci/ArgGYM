# Frozen external transfer evaluation

The transfer suite is fixed at seven benchmarks:

1. Multi-LogiEval
2. RuleArena
3. GSM8K
4. LogiQA 2.0
5. FineReason
6. MMLU-Pro
7. BIG-Bench Hard

Evaluate each benchmark exactly twice for the article protocol:

- PRE-RL: the pinned Qwen3-14B base model.
- POST-RL: the checkpoint selected using ArgGYM `RL_DEV` only.

Transfer results must never be used for learning-rate choice, completion-cap
choice, curriculum changes, checkpoint selection, or early stopping. Before the
main RL run, freeze each benchmark's exact dataset source/revision, split/subset,
prompt, answer parser/scorer, decoding parameters, max tokens and seeds. The
training process must not import this package.
