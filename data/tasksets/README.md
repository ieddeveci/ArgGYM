# ArgGYM tasksets

A **taskset** is a frozen set of evaluation items drawn from the ArgGYM
generator. ArgGYM is procedural — it can produce unlimited tasks — so comparing
models requires pinning a fixed sample and shipping it, which is what lives here.

## What is in a taskset

```
pilot-650388ac/
  taskset.jsonl.gz   # the items, one JSON object per line
  config.yaml        # the grid it was generated from
  manifest.json      # provenance: hashes, per-cell counts, checks
```

Each row:

| Field | Meaning |
|---|---|
| `sample_id` | `<task>__<mode>__L<level>__<idx>`, e.g. `attack__content__L05__003` |
| `task`, `mode`, `level`, `idx` | grid coordinates (`mode` is `symbolic` or `content`) |
| `cell_seed` | RNG seed for this (task, mode, level) cell |
| `prompt` | the complete prompt to send a model — already includes the formalism intro, the theory, and the answer-format block |
| `entry` | the ArgGYM entry: `question`, `answer` (gold), and `metadata` (ops, ordering, kind, task-specific fields) |

The directory name is `<name>-<hash8>`, where the hash covers **every prompt plus
the `kb.json` they were drawn from**. A taskset built against a different KB
therefore lands in a different directory and cannot be silently confused with
this one.

## Using it

```python
from evals.taskset import load_taskset
from aspic_gym import score_answer

rows = load_taskset("data/tasksets/pilot-650388ac")   # handles .jsonl and .jsonl.gz

for row in rows:
    answer = my_model(row["prompt"])                  # prompt needs no further assembly
    score  = score_answer(answer, row["entry"])       # float in [0, 1]
```

Gold answers are computed symbolically by PyArg under grounded semantics — no
LLM judges anything. `score_answer` is the only scorer; partial credit is
task-specific (per-claim accuracy, F1 over sets, and so on).

**Scoring caveat.** `aspic_gym._answer_region` matches the **first**
`[answer]` tag in the text. Reasoning models routinely draft an answer
mid-chain-of-thought and then revise it, so score the model's *final* output
only — never chain-of-thought concatenated with the answer. In one measured
sample of 13, that distinction flipped a score from 0.00 to 1.00.

## The `pilot` taskset

| | |
|---|---|
| Samples | 1200 |
| Grid | 12 tasks × 2 modes × 5 levels × 10 samples = 120 cells |
| Tasks | every registered task except `syntax`, which is excluded from the benchmark |
| Modes | `symbolic` (DSL atoms) and `content` (plain language, drawn from `kb.json`) |
| Levels | 1, 3, 5, 10, 15 |
| Base seed | 20260721 |

Levels are not evenly spaced on purpose. `levels.py` unlocks every *qualitative*
feature by level 5 — conflict and negation at 2, axioms, side branches and
derived attacks at 3, strict rules and attack chains at 4, undercuts at 5 —
while levels 6–15 only scale knobs (more atoms, more rules) without adding new
phenomena. So 1/3/5 walk the feature ladder and 10/15 isolate pure scale at a
saturated feature set.

## How it is generated

```bash
python -m evals.taskset                              # uses evals/conf/taskset/pilot.yaml
python -m evals.taskset taskset.n=20 taskset.name=big # override any field
```

Output goes to `outputs/tasksets/<name>-<hash8>/` (gitignored). To publish one,
gzip `taskset.jsonl` into `data/tasksets/<id>/` alongside its `manifest.json`
and `config.yaml`.

The generator walks every (task, mode, level) cell and pulls `n` items:

1. **Seeding.** Each cell gets `cell_seed = blake2b(base_seed, task, mode, level)`.
   A hash rather than a running counter, so adding or removing a task from the
   grid does not renumber every other cell.
2. **Sampling.** One `ASPICDataset` per cell. Because `ASPICDataset.__getitem__`
   derives its RNG from `idx` alone, **raising `n` extends a cell rather than
   reshuffling it** — the first `n` items stay byte-identical, so you can grow a
   taskset without invalidating results already collected on it.
3. **Skips.** A few (task, level) combinations occasionally fail to produce a
   valid theory at a given index even after the generator's own 60 internal
   retries. Those indices are skipped deterministically and counted in
   `manifest.json` under `generator_skips`, rather than shrinking the cell. In
   `pilot`, 2 indices were skipped, both in `counter_argumentation/symbolic/L15`.
4. **Gold self-check.** Every item's own gold answer is scored against itself and
   must return exactly 1.0. A single failure aborts the build. If the parsers
   cannot read their own gold, no model score from that taskset would mean
   anything. All 1200 items in `pilot` pass.

## Reproducing and verifying

Generation is deterministic given the same seed, generator code, and `kb.json`:

```bash
python -m evals.verify_taskset data/tasksets/pilot-650388ac
```

This regenerates the grid from the taskset's own `config.yaml` and compares the
content hash, reporting which prompts drifted if they do not match. Exit codes:
`0` match, `1` mismatch, `2` local `kb.json` differs from the one the taskset was
built against (regeneration is impossible; the shipped file is still usable).

Content-mode prompts are drawn from `kb.json`, so **widening the KB changes
them**. That is intended: a new KB produces a new taskset id, old results stay
attributable to the KB that produced them, and results from the two are never
pooled by accident.

## Known issue

Content-mode tasks that ask the model to *construct* directives — `attack`,
`counter_argumentation`, `evidence_construction`, `preference_construction` —
currently under-score. The prompt gives no syntax for labelling a newly added
rule, yet the gold answer requires referring to it by an auto-assigned ordinal
(`Rule 5`), and any explicit label causes the whole answer to fail parsing.
Scores on those cells reflect notation rather than reasoning. See
`workspace/model-eval-2026-07-21/findings.md` (F1). Tasks that only name
existing elements, such as `attackers_of` and `claim_identification`, are
unaffected.
