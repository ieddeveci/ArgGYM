# RL experiments

One file per experiment, numbered in the order they ran. Each holds the card
(what was run), the results, and what we concluded. How to run the code is in
[docs/rl-training.md](../../docs/rl-training.md).

| # | Experiment | Status | Outcome |
|---|---|---|---|
| 001 | [Qwen3-4B, levels 1-4](001-qwen3-4b-levels-1-4.md) | Stopped at step 21 | No gain; the model rarely succeeds, so most groups carried no signal |
| 002 | [Qwen3.5-4B, pass-rate curriculum, levels 1-8](002-qwen3.5-4b-curriculum.md) | Running | No gain at step 20; step 40 pending |

## Conventions

- **Selection metric**: strict success (`arggym_success_rate`) on the dev split,
  which is generated from seeds disjoint from training and from the eval
  tasksets. Dev picks the checkpoint; it is too small to report as a result.
- **A result** is the selected checkpoint merged (`rl.export`) and run through
  `evals.run` on the lite taskset, compared with the stored base result for the
  same model, template and elicitation.
- **Noise**: quote the standard error with every dev number. For a success rate
  p over n samples it is sqrt(p(1-p)/n), about 0.034 at p = 0.34, n = 192.

## Card template

```markdown
# NNN: <model>, <what changes>

Status: running | stopped at step N | finished. Dates.
wandb: <run path>. Config: `rl/configs/<file>.yaml` at commit <sha>.

## Question
The one thing this run is meant to tell us.

## Card
| | |
|---|---|
| Model | |
| Data | tasks, levels, pool size, dev split |
| Batch | prompts x completions per step |
| Sampling | temperature, top_p, top_k, penalties, length cap |
| Loss and optimizer | |
| What differs from the previous experiment | |

## Results
Dev table by eval step, then training-side metrics in windows.

## Discussion
What the numbers say, what they do not, and what changes next.
```
