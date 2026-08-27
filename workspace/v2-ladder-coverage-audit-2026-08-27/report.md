# Audit of the prior V2 ladder and coverage findings

The earlier note is mostly sound, but its unit of action is wrong: it yields
three current-main issues, not seven. A2 and A3 have one cause and need one
repair; A5 is already fixed on owner main; B1 and B2 concern an unpublished eval
harness that owner main does not contain.

Audited owner main at
[`1fdc9db`](https://github.com/ieddeveci/ArgGYM/tree/1fdc9db03ddcc415abd82bef0f230ab31b8a6892)
and the live issue ledger through issue 29 on 2026-08-27.

| Finding | Legitimate? | Owner main | Overlap | Disposition |
|---|---|---|---|---|
| A1: shared preference conflicts never run | Yes, with narrower impact wording | Still present | None | Published as #30 |
| A2: defence L12 and L15 are identical | Yes | Still present | None | Published with A3 as #31 |
| A3: defence L6-to-L9 bundles several changes | Yes; one score is wrong and causality is unproven | Still present | None | Published with A2 as #31 |
| A4: strict counter-argument L3 is harder than L6 | Yes; quoted scores belong to one model | Still present | None | Published as #32 |
| A5: no conjunction coverage | Valid for the pilot | Fixed on owner main | None | Do not file |
| B1: sampling keys were silently dropped | Valid for the local harness; impact partly overstated | Harness absent from owner main; local fix exists | None | Do not file upstream |
| B2: high error rate was treated as server death | Valid for the local v2 harness and recorded incident | Harness absent from owner main; v2 path fixed locally | None | Do not file upstream |
| C1: weakest-link efficiency has low sensitivity | Useful limitation, not an invariant | Applies to the branch-fixed taskset | None | Report caveat only |
| C2: sampler penalty is a control, not a benchmark fix | Supported, with vendor-config wording corrected | No code defect | None | Report caveat only |

## The three current-main issues

### 1. The shared-conflict branch misses every evaluated level

[`preference_construction.py:91-104`](https://github.com/ieddeveci/ArgGYM/blob/1fdc9db03ddcc415abd82bef0f230ab31b8a6892/ArgGYM_v2/tasks/preference_construction.py#L91-L104)
sets `shared` only when `level % 3 == 2`. The benchmark evaluates levels 3, 6,
9, 12, and 15, so the branch is false in every official cell. The pilot confirms
zero shared conflicts among all 200 preference-construction items.

The issue should not repeat the note's claim that this makes every correct
answer automatically minimal. Pilot last-link L12 and L15 already have positive
mean gaps between goal count and the exact minimum. The supported impact is:
the intended shared-conflict feature was never evaluated, and every defined
preference-construction efficiency mean in the stored sweep was 1.0.

### 2. The defence schedule has a duplicate top rung and a bundled middle step

These are one issue. The same schedule in
[`attack_defense.py:256-275`](https://github.com/ieddeveci/ArgGYM/blob/1fdc9db03ddcc415abd82bef0f230ab31b8a6892/ArgGYM_v2/tasks/attack_defense.py#L256-L275)
causes both symptoms:

```text
      attackers  strict attackers  support depth  attacker depth  decoys  strict-rule decoys
L6        3              0               3               3           1             1
L9        4              1               4               4           1             2
L12       5              1               5               5           2             3
L15       5              1               5               5           2             3
```

L12 and L15 are identical, while L6-to-L9 changes attacker count, strictness,
both depths, and decoy strictness together. Separate issues would prescribe the
same schedule-table fix and could conflict.

The note's model evidence has a transcription error. Gemma 4 31B's defence
score falls from 0.475 at L6 to 0.025 at L9. The quoted 0.575 is its L3 score.
The drop is evidence that the step is sharp; without a factor ablation, it is
not proof that the bundled schedule caused the whole drop.

### 3. `counter_argument_strict` has an inverted first difficulty step

[`counter_argument.py:77-86`](https://github.com/ieddeveci/ArgGYM/blob/1fdc9db03ddcc415abd82bef0f230ab31b8a6892/ArgGYM_v2/tasks/counter_argument.py#L77-L86)
allows depth 2 at L3 but activates middle targets only from L6.
[`counter_argument.py:155-171`](https://github.com/ieddeveci/ArgGYM/blob/1fdc9db03ddcc415abd82bef0f230ab31b8a6892/ArgGYM_v2/tasks/counter_argument.py#L155-L171)
also requires depth at least 3 to use that cheaper target. The frozen pilot
requires 3.5 directives on average at L3 and 3.0 at L6. A fresh eight-seed
current-main probe reproduced the inversion at 3.0 versus 2.5.

The 0.125 and 0.600 scores in the earlier note are specifically Gemma 4 31B's
L3 and L6 means, not a result established across models.

## Findings that should not become new issues

A5 was valid for the pilot: no conjunctive DSL rule occurs anywhere in its
2,196 items. Owner main now generates two-antecedent rules in
[`status_query.py:114-133`](https://github.com/ieddeveci/ArgGYM/blob/1fdc9db03ddcc415abd82bef0f230ab31b8a6892/ArgGYM_v2/tasks/status_query.py#L114-L133)
and
[`formalization.py:152-169`](https://github.com/ieddeveci/ArgGYM/blob/1fdc9db03ddcc415abd82bef0f230ab31b8a6892/ArgGYM_v2/tasks/formalization.py#L152-L169).
Blame traces both blocks to owner commit
[`3feace1`](https://github.com/ieddeveci/ArgGYM/commit/3feace110d0d93e243ed2717f538cfef260b7186).
Focused L15 generation produced 152 conjunctive rules in eight status-query
items and 40 in eight formalization items.

B1 and B2 are real historical harness defects. The old request builder omitted
repetition-control keys, and the old v2 grid driver broke the remaining level
loop when API errors exceeded 20%. The latter did remove Qwen 3.6 27B's L12 and
L15 cells after 96 L9 timeouts. Local commits `b7df560` and `9fc9556` repair
them, and the v2 wrapper uses the repaired grid driver. Owner main has no
`evals/` directory, so neither is a current upstream repository issue. The
claim that half of 4,312 truncations were repetition loops was not backed by a
stored classification and should not be published as measured fact.

C1 and C2 belong in evaluation documentation. Under weakest-link, successful
preference answers often require many more directives than the number of goals,
which makes the efficiency term insensitive but not mathematically pass/fail.
The Qwen sampler control does show that penalties recover parseable answers
faster than correctness. The 9B setup inherits the Qwen 3.5 27B family settings;
calling them the 9B model's own published settings is too strong.

## Overlap result

No current issue has the same cause and remedy as any of the three candidates.
Exact GitHub searches over issue titles, bodies, and comments found no ladder,
shared-conflict, strict-attacker, conjunction, or sampling-key issue.

The broad matches are adjacent but distinct:

- [#28](https://github.com/ieddeveci/ArgGYM/issues/28) is about uncapped subset
  searches, not the shape of either difficulty ladder.
- [#23](https://github.com/ieddeveci/ArgGYM/issues/23) is about missing export
  registry entries, not defence generation.
- [#21](https://github.com/ieddeveci/ArgGYM/issues/21) is about scoring illegal
  directive kinds, not schedule or conjunction coverage.
- [#12](https://github.com/ieddeveci/ArgGYM/issues/12) is a broad older tracking
  issue and does not describe these defects.

After the audit, the user approved publication. The findings were posted as
[#30](https://github.com/ieddeveci/ArgGYM/issues/30),
[#31](https://github.com/ieddeveci/ArgGYM/issues/31), and
[#32](https://github.com/ieddeveci/ArgGYM/issues/32) on 2026-08-27. No existing
issue needed a comment or body update because none contained the same finding.
