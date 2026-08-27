# Evidence ledger

## Repository snapshots

The fresh clone in `repo/` and the outer repository's `origin/main` both
resolved to owner-main commit
`1fdc9db03ddcc415abd82bef0f230ab31b8a6892`. The exact ladder-fix commit was
checked in the detached worktree `fix-df453dc/` at
`df453dc04e9b510ae3595132e457c5f13d858842`.

`df453dc` is not an ancestor of owner main. The historical fixes therefore
cannot be treated as upstream fixes merely because they exist in the local
repository. Git blame attributes owner main's conjunction implementation to the
owner's independent commit
`3feace110d0d93e243ed2717f538cfef260b7186`.

Owner main has no `evals/` directory. B1 and B2 describe the local evaluation
harness, not a component currently shipped by the owner's repository.

## Reproducible generator and harness probes

Run:

```bash
cd workspace/v2-ladder-coverage-audit-2026-08-27
repo/.venv/bin/python probes.py
```

The probe found:

```text
A1 pilot shared conflicts 0 / 200
A1 pilot mean goal-minus-min gap last_link_elitist {3: 0, 6: 0, 9: 0, 12: 1, 15: 0.8}
A1 pilot mean goal-minus-min gap weakest_link_elitist {3: -2.4, 6: -7.55, 9: -9.35, 12: -10.8, 15: -15.6}
A1 fixed mean goal-minus-min gap last_link_elitist {3: 0, 6: 1, 9: 1, 12: 2.25, 15: 3.3}
A1 fixed mean goal-minus-min gap weakest_link_elitist {3: -2.4, 6: -6.5, 9: -8.35, 12: -11.35, 15: -8.9}
A2/A3 pilot defence recipes {6: {(3, 0, 3, 3, 1, 1)}, 9: {(4, 1, 4, 4, 1, 2)}, 12: {(5, 1, 5, 5, 2, 3)}, 15: {(5, 1, 5, 5, 2, 3)}}
A4 pilot mean directives {3: 3.5, 6: 3}
A5 pilot conjunctive DSL occurrences across all tasks 0
A5 pilot conjunctive rules {'status_query': 0, 'formalization': 0}
A5 fixed conjunctive DSL occurrences across all tasks 691
A5 fixed conjunctive rules {'status_query': 400, 'formalization': 291}
A1 current shared flag {3: False, 6: False, 9: False, 12: False, 15: False}
A2/A3 current defence recipes {3: (2, 0, 2, 2, 0, 0), 6: (3, 0, 3, 3, 1, 1), 9: (4, 1, 4, 4, 1, 2), 12: (5, 1, 5, 5, 2, 3), 15: (5, 1, 5, 5, 2, 3)}
A4 current 8-seed last-link mean directives {3: 3, 6: 2.5}
A5 current L15 conjunctive rules status_query 152 across 8 items
A5 current L15 conjunctive rules formalization 40 across 8 items
B1 old payload keys ['max_tokens', 'messages', 'model', 'stream', 'temperature']
B1 fixed payload keys ['max_tokens', 'messages', 'model', 'repetition_penalty', 'stream', 'temperature']
B2 old driver health check False breaks after high-error gate True
B2 fixed driver health check True higherror marker True
```

The current source explains each result:

- `ArgGYM_v2/tasks/preference_construction.py:91-104` enables the shared case
  only when `level % 3 == 2`; none of levels 3, 6, 9, 12, or 15 satisfy it.
- `ArgGYM_v2/tasks/attack_defense.py:256-275` derives attacker count, strictness,
  both depths, and strict-rule decoys from adjacent level schedules. Its caps
  make L12 and L15 equal, while the L6-to-L9 step changes several variables
  together.
- `ArgGYM_v2/tasks/counter_argument.py:77-86` permits depth 2 at L3 and enables
  the middle target only from L6. Lines 155-171 require depth at least 3 before
  choosing the cheaper middle target.
- `ArgGYM_v2/tasks/status_query.py:114-133` and
  `ArgGYM_v2/tasks/formalization.py:152-169` now create two-antecedent rules.

At the exact historical repair commit, the regression suite passes:

```text
$ repo/.venv/bin/python -m pytest tests/test_v2_difficulty_ladder.py -q
.......                                                                  [100%]
7 passed in 36.11s
```

This confirms that the proposed A fixes work on their branch. It does not make
them part of owner main.

## Stored evaluation evidence and corrections

The stored 2026-08-11 report gives Gemma 4 31B's `defence` means as 0.575 at
L3, 0.475 at L6, and 0.025 at L9. The earlier note's claimed L6-to-L9 drop of
0.575 to 0.025 uses the L3 value as its starting point. The correct comparison
is 0.475 to 0.025. This remains a sharp correlation, but it does not by itself
prove which bundled schedule change caused the drop.

The same report gives Gemma 4 31B's `counter_argument_strict` means as 0.125 at
L3 and 0.600 at L6. These are one model's results, despite the note's plural
wording. The taskset-level minimum-directive inversion is independent of that
model result: 3.5 at L3 and 3.0 at L6.

For preference construction, all 27 defined `mean_efficiency` values among 33
reported run rows equal 1.0. The earlier note calls these “all 30 cells.” More
important, its statement that every correct answer was automatically minimal is
false: even the pilot taskset has positive mean goal-minus-minimum gaps at
last-link L12 and L15. The dead shared-conflict switch is real, and the observed
efficiency metric was inert, but the universal causal claim should be removed.

The 2026-08-11 sweep contains 33 runs, 14,495 generations, and 4,312 truncated
generations. B1's old request builder forwards only `temperature`, `top_p`, and
`top_k`; commit `b7df560` forwards repetition controls and rejects unknown
keys. I did not find a stored classification that establishes the note's claim
that exactly half of the truncations ended in repetition loops. The sweep
configuration also did not request a repetition penalty, so this defect would
have dropped such a setting if configured, but it did not silently alter this
particular sweep.

Qwen 3.6 27B's L9 run reports an API-error rate of 0.218. Its 96 recorded errors
are timeouts, and the stored run list has no L12 or L15 cells. The old
`run_grid.sh:111-121` broke the level loop solely from the error rate. Commit
`9fc9556` adds a health request before deciding whether the server died. The v2
wrapper calls this repaired driver. The generic `evals/run_all.sh:85-94` still
uses the old heuristic, but it is outside the v2 sweep path.

## Limits C1 and C2

C1 is directionally sound but too absolute. In the fixed taskset, weakest-link
mean goal-minus-minimum gaps are -2.4, -6.5, -8.35, -11.35, and -8.9, so the
minimum already requires many more preferences than one per goal. Efficiency
has little room to distinguish redundant successful answers. It is not
mathematically pass/fail: the scorer can still penalize extra legal directives.
It is a low-sensitivity design limit, and it describes the branch-fixed
taskset, not current owner main.

The four-arm sampler script reproduces C2's table exactly:

```text
penalty             truncation   no answer   four-task macro
0.0                   71.9%        57.5%          0.116
1.0                   40.6%        23.8%          0.262
1.5                   28.1%        15.0%          0.341
2.0                   16.9%         7.5%          0.356
```

`counter_argument` remains at 0.0 for all four arms while its no-answer rate
falls. The control supports “the penalty improves answer production more than
reasoning correctness.” The configuration wording needs one correction: the
9B config inherits the Qwen 3.5 27B family generation settings; the 9B model
does not publish its own `generation_config.json`. Qwen's published 27B config
uses temperature 0.6, top-p 0.95, and top-k 20. DeepSeek-R1 recommends
temperature 0.5-0.7 and does not specify a repetition penalty. The paper cited
as arXiv:2504.20131 reports residual degenerate repetition under standard
penalties, so it supports the limitation rather than a new code defect.

Sources:

- Qwen 3.5 27B generation config:
  <https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/generation_config.json>
- DeepSeek-R1 model card:
  <https://huggingface.co/deepseek-ai/DeepSeek-R1>
- LZ Penalty paper: <https://arxiv.org/abs/2504.20131v4>

## GitHub overlap check

The live issue ledger contained issues 1-29 on 2026-08-27. GitHub searches over
titles, bodies, and comments returned zero matches for `shared conflict`,
`strict attacker`, `duplicate rungs`, `difficulty ladder`, `conjunctive rules`,
`repetition_penalty`, and `sampling settings`. The only `health check` match was
pull request 14, not an issue.

Broad term searches found four issues worth manual comparison:

- #28 concerns uncapped subset-search runtime. It shares the
  `preference_construction` and `counter_argument` files but not these ladder
  defects.
- #23 concerns missing export registry entries, not defence generation.
- #21 concerns scorer treatment of illegal directive kinds, not generator
  schedules or conjunction coverage.
- #12 is a broad tracking issue for the older benchmark audit. It does not state
  any of A1-A5 or B1-B2.

Issues #22-#29 from the current audit cover dependency metadata, export
coverage, silent cell loss, answer ordering, leaked noise labels, redundant
semantics computation, construction runtime, and the `overruled` contract.
None has the same cause, symptom, or remedy as the three surviving ladder
findings.

## What was not checked

- I did not re-run any GPU model evaluation. Stored generations and reports were
  sufficient to check the numerical claims.
- I did not manually classify all 4,312 truncated generations, so the “half are
  repetition loops” claim remains unverified.
- I did not run a causal ablation separating strictness, attacker count, and
  depth for the Gemma drop. The generator defect is established; its share of
  that model effect is not.
- I did not publish, comment on, close, or reopen any GitHub issue.
