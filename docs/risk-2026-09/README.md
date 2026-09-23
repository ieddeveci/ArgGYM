# Risk assessments, September 2026

Two analyses of what the open defects would do to results, written before the next
evaluation sweep and after RL training started. They answer different questions and
reach almost opposite rankings, which is the point of having both.

| file | question it answers | assessed against |
|---|---|---|
| [eval-risk-2026-09-21.md](eval-risk-2026-09-21.md) | which open issues would make an **evaluation** mislead a reader | `main` at `e809626` |
| [rl-risk-2026-09-22.md](rl-risk-2026-09-22.md) | which defects would corrupt **RL training** | `experiments-phase` at `5f1c2a1` |

## Why two

An evaluation *reports* a defect; a policy *optimises against* it. Sorting the same
backlog by "what would a policy learn" instead of "what would a number say" inverts the
ranking almost completely:

- The format cliffs — the bloat gate, the stray-token zero — are the largest evaluation
  risk and an RL non-issue. The rule is stated in the prompt, the policy has a
  `<thought>` block for prose, and answer-region rate was 0.978 at epoch 0.
- The dense partial-credit floors are an evaluation footnote and the RL catastrophe. At
  the hard band a policy that never reads the theory scores 0.2204 against Qwen3-14B's
  0.2318 at epoch 0.

Read whichever matches what you are about to run. If you are doing both, read both — a
fix that helps one can be irrelevant to the other.

## Reproducing the numbers

Every measured claim names the command that produced it. The scripts behind the RL
analysis are in [`probes/`](probes/) and run from the repository root:

```
uv run python docs/risk-2026-09/probes/<name>.py
```

Two of them (`floors.py`, `measure_rl.py`) write their output to `/tmp` and take several
minutes. The `.jsonl` dumps they produce are not committed — they are derived data and
regenerating them is a better check than trusting a committed copy.

These are evidence, not tools. They were written to answer one question each and are not
maintained.

## Status

Both documents describe the tree as it stood on their date. Issue and PR numbers are
cited throughout; check whether those are still open before acting on a recommendation.
Neither document is a plan — the ordered lists at the end of each are what their author
would do, not what has been decided.
